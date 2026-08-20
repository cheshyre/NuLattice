from functools import partial
from typing import Tuple, Literal

import jax
import jax.numpy as jnp

from NuLattice.utils._jax_types import ShardingManager

from .davidson import davidson_eigh

Array = jax.Array
EigenSolver = Literal["dense", "davidson"]

def _adjoint(x):
    return jnp.swapaxes(jnp.conj(x), -1, -2)

def _make_hermitian(x):
    return 0.5 * (x + _adjoint(x))

def init_density(number_of_states: int, hole_indices: Tuple[int], dtype=None):
    density = jnp.zeros((number_of_states, number_of_states), dtype=dtype)
    hole_indices = jnp.array(hole_indices)
    density = density.at[hole_indices, hole_indices].set(1.0)
    return density

@jax.jit
def contract_2nf_fused(indices: Array, values: Array, density: Array) -> Array:
    """Contract the sparse two-body interaction with a one-body density."""
    p, q, r, s = (indices[:, i] for i in range(4))
    n = density.shape[0]
    dtype = jnp.result_type(values.dtype, density.dtype)
    res = jnp.zeros((n, n), dtype=dtype)
    res = res.at[p, r].add(+values * density[q, s])
    res = res.at[q, r].add(-values * density[p, s])
    res = res.at[p, s].add(-values * density[q, r])
    res = res.at[q, s].add(+values * density[p, r])
    return res


@jax.jit
def contract_3nf_fused(indices: Array, values: Array, density: Array) -> Array:
    """Contract the sparse three-body interaction with two densities."""
    a, b, c, d, e, f = (indices[:, i] for i in range(6))
    n = density.shape[0]
    dtype = jnp.result_type(values.dtype, density.dtype)
    v2 = values * 2.0
    res = jnp.zeros((n, n), dtype=dtype)

    res = res.at[a, d].add(v2 * (density[b, e] * density[c, f] - density[c, e] * density[b, f]))
    res = res.at[b, d].add(v2 * (density[c, e] * density[a, f] - density[a, e] * density[c, f]))
    res = res.at[c, d].add(v2 * (density[a, e] * density[b, f] - density[b, e] * density[a, f]))

    res = res.at[a, e].add(v2 * (density[b, f] * density[c, d] - density[c, f] * density[b, d]))
    res = res.at[b, e].add(v2 * (density[c, f] * density[a, d] - density[a, f] * density[c, d]))
    res = res.at[c, e].add(v2 * (density[a, f] * density[b, d] - density[b, f] * density[a, d]))

    res = res.at[a, f].add(v2 * (density[b, d] * density[c, e] - density[c, d] * density[b, e]))
    res = res.at[b, f].add(v2 * (density[c, d] * density[a, e] - density[a, d] * density[c, e]))
    res = res.at[c, f].add(v2 * (density[a, d] * density[b, e] - density[b, d] * density[a, e]))
    return res

def build_2b_and_3b_fock_matrices(
    density: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array = None,
    w3_val: Array = None,
) -> tuple[Array, Array]:
    fock_2b = _make_hermitian(contract_2nf_fused(v2_idx, v2_val, density))
    fock_3b = None
    if (w3_idx is not None) and (w3_val is not None):
        fock_3b = _make_hermitian(contract_3nf_fused(w3_idx, w3_val, density))
    return fock_2b, fock_3b


def build_full_fock_matrix(
    h1: Array,
    fock_2b: Array,
    fock_3b: Array = None,
) -> Array:
    if fock_3b is None:
        return _make_hermitian(h1 + fock_2b)
    return _make_hermitian(h1 + fock_2b + 0.5 * fock_3b)

def compute_hf_energy_from_fock_matrices(
    density: Array,
    h1: Array,
    fock_2b: Array,
    fock_3b: Array = None,
) -> Array:

    e_h1 = jnp.einsum("ij,ji->", h1, density)
    e_2b = jnp.einsum("ij,ji->", fock_2b, density)
    e_3b = jnp.asarray(0, dtype=jnp.real(density[0]).dtype)
    if fock_3b is not None:
        e_3b = jnp.einsum("ij,ji->", fock_3b, density)
    return jnp.real(e_h1 + 0.5 * e_2b + (1.0 / 6.0) * e_3b)



@partial(jax.jit, static_argnames=("npart", "diagonalizer", ))
def iterate_hf_equations(
    density, h1, v2_idx, v2_val, w3_idx, w3_val, number_of_particles, mixing_param, prev_vecs,
    diagonalizer, davidson_max_iter
):
    fock_2b, fock_3b = build_2b_and_3b_fock_matrices(density, v2_idx, v2_val, w3_idx, w3_val)
    fock = build_full_fock_matrix(h1, fock_2b, fock_3b)
    energy = compute_hf_energy_from_fock_matrices(density, h1, fock_2b, fock_3b)

    _, orbitals = (
        jnp.linalg.eigh(fock)
        if diagonalizer == "dense"
        else davidson_eigh(fock, number_of_particles, prev_vecs, davidson_max_iter)
    )
    occupied_orbitals = orbitals[:, :number_of_particles]

    new_density = occupied_orbitals @ _adjoint(occupied_orbitals)

    mixed_density = (1.0 - mixing_param) * density + mixing_param * new_density
    residual_density = jnp.sum(jnp.abs(mixed_density - density))

    return occupied_orbitals, energy, mixed_density, residual_density

def prepare_inputs(op1, op2, op3, density: Array, sm: ShardingManager, dtype=jnp.float64):
    has_three_body = op3 is not None and len(op3) > 0

    if sm is not None:
        assert sm.num_nodes == 1 or sm.num_gpus == 1, "HF expects 1D mesh, ensure sm.num_nodes or sm.num_gpus is 1"
        h1 = sm.prepare(op1.to_dense(), rank=0)
        density = sm.prepare(density, rank=0)
        v2_idx = sm.prepare(op2.indices)
        v2_val = sm.prepare(op2.values)
        if has_three_body:
            w3_idx = sm.prepare(op3.indices)
            w3_val = sm.prepare(op3.values)
        else:
            w3_idx = None
            w3_val = None
    else:
        h1 = jnp.asarray(op1.to_dense())
        v2_idx = jnp.asarray(op2.indices)
        v2_val = jnp.asarray(op2.values)
        if has_three_body:
            w3_idx = jnp.asarray(op3.indices)
            w3_val = jnp.asarray(op3.values)
        else:
            w3_idx = None
            w3_val = None
        density = jnp.asarray(density)

    return h1, v2_idx, v2_val, w3_idx, w3_val, density

def solve_HF(
    op1,
    op2,
    op3,
    density: Array,
    mix: float =0.5,
    eps: float =1e-8,
    max_iter: int = 100,
    davidson_max_iter: int = 10,
    verbose: bool = False,
    sm: ShardingManager = None,
    diagonalizer: EigenSolver = "davidson",
    keep_all_orbitals: bool = True,
):

    if diagonalizer not in {"davidson", "dense"}:
        raise ValueError("diagonalizer must be 'davidson' or 'dense'")

    h1_dense, v2_idx, v2_val, w3_idx, w3_val, _density = prepare_inputs(
        op1, op2, op3, density, sm
    )

    prev_energy = 0.0
    converged = False
    npart = int(jnp.real(jnp.trace(_density)).round())

    occupied_orbitals = guess_occupied_orbitals_from_density(_density, npart)

    for i in range(max_iter):
        occupied_orbitals, energy, _density, diff_density = iterate_hf_equations(
            _density, h1_dense, v2_idx, v2_val, w3_idx, w3_val, npart, mix, occupied_orbitals,
            diagonalizer, davidson_max_iter,
        )

        dE = jnp.abs(energy - prev_energy)

        if verbose:
            print(f"Iter {i}: E={energy:.8f}, dE={dE:.6e}, dRho={diff_density:.6e}")

        if (diff_density < eps):
            converged = True
            break

        prev_energy = energy

    if keep_all_orbitals:
        fock_2b, fock_3b = build_2b_and_3b_fock_matrices(_density, v2_idx, v2_val, w3_idx, w3_val)
        fock = build_full_fock_matrix(h1_dense, fock_2b, fock_3b)
        _, orbs = jnp.linalg.eigh(fock)
    else:
        orbs = occupied_orbitals


    return energy, orbs, converged

def guess_occupied_orbitals_from_density(density: jax.Array, npart: int) -> jax.Array:
    # We generate a random matrix P that serve as trial basis to get the eigenvectors of dens
    # P has the size npart + CONDITION_NUMBER, where CONDITION_NUMBER ensures numerical stability
    dim = len(density)
    key = jax.random.key(42)
    CONDITION_NUMBER = 5
    CONDITION_NUMBER = min(CONDITION_NUMBER, dim - npart) # Make sure we stay in bounds
    P = jax.random.normal(key, shape=(dim, npart + CONDITION_NUMBER), dtype=density.dtype)

    # We project out the eigenbasis of the density from P
    # This works because the density is a projector,
    # specifically a projector onto the space of occupied states
    X = density @ P

    # We take the resulting vectors and perform a QR decomposition
    # to get the orthogonal vectors Q
    Q, _ = jnp.linalg.qr(X)

    # # This code will shuffle the vectors in Q
    # # This can be used to test that the filter below is working
    # perm = jax.random.permutation(key, npart + CONDITION_NUMBER)
    # Q = Q[:, perm]

    # We compute the eigenvalues of the density matrix from Q
    # They are on the diagonal of the resulting matrix
    eigenvalues = jnp.diag(jnp.conjugate(jnp.transpose(Q)) @ density @ Q)
    # print(eigenvalues)

    # We loop over the eigenvalues once and find the ones that are nonzero
    indices = jnp.zeros(shape=(npart), dtype=int)
    count = 0
    for i in range(npart + CONDITION_NUMBER):
        if eigenvalues[i] > 0.75:
            if count >= npart:
                print(f"Warning: Found more than npart={npart} eigenvectors of density matrix with nonzero eigenvalues.")
                print("Something might be wrong!")
                break
            indices = indices.at[count].set(i)
            count += 1
    if count != npart:
        print(f"Warning: Found only {count} (which is less than npart={npart}) eigenvectors of density matrix with nonzero eigenvalues.")
        print("Something might be wrong!")

    # We get the occupied orbitals from the eigenvalues that are nonzero
    occupied_orbitals = jnp.asarray(Q[:, indices])

    # # This code can be used to double-check that we actually get the eigenvalues we think
    # vals_final = jnp.conjugate(jnp.transpose(occupied_orbitals)) @ density @ occupied_orbitals
    # print(jnp.diag(vals_final))

    return occupied_orbitals


def contract_3nf(op3, density):
    w3_idx = jnp.asarray(op3.indices)
    w3_val = jnp.asarray(op3.values)

    return contract_3nf_fused(w3_idx, w3_val, density)


def contract_2nf(op2, density):
    v2_idx = jnp.asarray(op2.indices)
    v2_val = jnp.asarray(op2.values)

    return contract_2nf_fused(v2_idx, v2_val, density)


def HF_energy(op1, op2, op3, density):
    f_1b = jnp.zeros_like(density)
    f_1b += jnp.asarray(op1.to_dense())
    f_1b += 0.5 * contract_2nf(op2, density)
    f_1b += (1.0 / 6.0) * contract_3nf(op3, density)

    energy = jnp.einsum("ij,ji", f_1b, density)

    if jnp.abs(jnp.imag(energy)) > 1e-4:
        print(f"Warning: Computed energy is complex: {energy}")
        print("Something is probably wrong!")

    return jnp.real(energy)


def HF_iter(op1, op2, op3, density, mix=0.5):
    npart = round(jnp.real(jnp.trace(density)))

    energy = HF_energy(op1, op2, op3, density)
    fock = make_HF_ham(op1, op2, op3, density)
    _, orbitals = jnp.linalg.eigh(fock)
    occupied_orbitals = orbitals[:, 0:npart]
    new_density = occupied_orbitals @ jnp.conjugate(jnp.transpose(occupied_orbitals))

    mixed_density = mix * new_density + (1.0 - mix) * density

    return energy, mixed_density, orbitals


def make_HF_ham(op1, op2, op3, density):
    fock = jnp.asarray(op1.to_dense().astype(density.dtype))
    fock += contract_2nf(op2, density)
    fock += 0.5 * contract_3nf(op3, density)
    return fock
