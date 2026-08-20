"""
functions to perform a Hartree-Fock computation on the lattice, implemented in JAX

The one-body part of the Hamiltonian is passed as an operator that can be
converted to a dense matrix via op1.to_dense(), while the two- and three-body
parts are passed as sparse operators carrying the attributes indices and
values, i.e., a list of index tuples [p,q,r,s] or [a,b,c,d,e,f] and the
associated matrix elements.
"""
__authors__   =  "Thomas Papenbrock, Vivek Booshan, Matthias Heinz"
__credits__   =  ["Thomas Papenbrock", "Vivek Booshan", "Matthias Heinz"]
__copyright__ = "(c) Thomas Papenbrock, Vivek Booshan, Matthias Heinz"
__license__   = "BSD-3-Clause"
__date__      = "2026"

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

from NuLattice.utils._jax_types import ShardingManager, OneBodyOperator, TwoBodyOperator, ThreeBodyOperator

from .davidson import davidson_eigh

Array = jax.Array
EigenSolver = Literal["dense", "davidson"]


def init_density(number_of_states: int, hole_indices: list[int], dtype=None) -> Array:
    """
    creates a density matrix of dimension number_of_states x number_of_states given the hole information

    :param number_of_states: dimension of single-particle basis
    :type number_of_states:  int
    :param hole_indices:     list of occupied single-particle states, as numbers from 0 ... A-1
    :type hole_indices:      list[int]
    :param dtype:            data type of returned object
    :type dtype:             jax.numpy.dtype, i.e., jnp.float64 or jnp.complex128
    :return:                 density matrix where hole states are occupied (1) and all others not (0)
    :rtype:                  jax.Array((number_of_states,number_of_states), dtype=float)
    """
    density = jnp.zeros((number_of_states, number_of_states), dtype=dtype)
    hole_indices = jnp.array(hole_indices)
    density = density.at[hole_indices, hole_indices].set(1.0)
    return density


def HF_energy(
    op1: OneBodyOperator | None,
    op2: TwoBodyOperator | None,
    op3: ThreeBodyOperator | None,
    density: Array,
) -> Array:
    """
    Computes the Hartree-Fock energy for a given density and Hamiltonian consisting
    of one-body term op1, two-body term op2, and three-body term op3

    Warns if the expectation value acquires a sizeable imaginary part, which
    signals an inconsistent density matrix or Hamiltonian.

    Any operator given as None is replaced by an empty operator of the
    appropriate rank, i.e., it does not contribute to the energy.

    :param op1:     one-body operator, convertible to a dense matrix via op1.to_dense()
    :type op1:      OneBodyOperator or None
    :param op2:     sparse two-body operator with attributes indices and values
    :type op2:      TwoBodyOperator or None
    :param op3:     sparse three-body operator with attributes indices and values
    :type op3:      ThreeBodyOperator or None
    :param density: density matrix (same shape as the dense form of op1)
    :type density:  jax.Array((:,:), dtype=float or complex)
    :return:        Hartree-Fock energy
    :rtype:         jax.Array((), dtype=float)
    """
    if op1 is None:
        op1 = OneBodyOperator.empty(len(density))
    if op2 is None:
        op2 = TwoBodyOperator.empty(len(density))
    if op3 is None:
        op3 = ThreeBodyOperator.empty(len(density))
    energy = evaluate_slater_determinant_expectation_value(
        op1, op2, op3, density, force_real=False
    )

    if jnp.abs(jnp.imag(energy)) > 1e-4:
        print(f"Warning: Computed energy is complex: {energy}")
        print("Something is probably wrong!")

    return jnp.real(energy)


def evaluate_slater_determinant_expectation_value(
    op1: OneBodyOperator | None,
    op2: TwoBodyOperator | None,
    op3: ThreeBodyOperator | None,
    density: Array,
    force_real: bool = False,
) -> Array:
    """
    evaluates the expectation value of the Hamiltonian in the Slater determinant
    defined by the density matrix

    The one-, two-, and three-body pieces enter with the weights 1, 1/2, and
    1/6, respectively, so that the trace against the density gives the energy
    and not the Hartree-Fock Hamiltonian (see make_HF_ham for the latter).

    Any operator given as None is replaced by an empty operator of the
    appropriate rank, i.e., it does not contribute to the expectation value.

    :param op1:        one-body operator, convertible to a dense matrix via op1.to_dense()
    :type op1:         OneBodyOperator or None
    :param op2:        sparse two-body operator with attributes indices and values
    :type op2:         TwoBodyOperator or None
    :param op3:        sparse three-body operator with attributes indices and values
    :type op3:         ThreeBodyOperator or None
    :param density:    density matrix (same shape as the dense form of op1)
    :type density:     jax.Array((:,:), dtype=float or complex)
    :param force_real: if True, the imaginary part of the result is discarded
    :type force_real:  bool
    :return:           expectation value of the Hamiltonian
    :rtype:            jax.Array((), dtype=float or complex)
    """
    if op1 is None:
        op1 = OneBodyOperator.empty(len(density))
    if op2 is None:
        op2 = TwoBodyOperator.empty(len(density))
    if op3 is None:
        op3 = ThreeBodyOperator.empty(len(density))

    f_1b = jnp.zeros_like(density)
    f_1b += jnp.asarray(op1.to_dense())
    f_1b += 0.5 * _contract_2nf(op2, density)
    f_1b += (1.0 / 6.0) * _contract_3nf(op3, density)

    exp_val = jnp.einsum("ij,ji", f_1b, density)

    if force_real:
        return jnp.real(exp_val)
    return exp_val


def make_HF_ham(
    op1: OneBodyOperator | None,
    op2: TwoBodyOperator | None,
    op3: ThreeBodyOperator | None,
    density: Array,
) -> Array:
    """
    takes Hamiltonian consisting of one-body operator op1, two-body operator op2,
    and three-body operator op3, and the density matrix and returns the Hartree-Fock Hamiltonian

    Any operator given as None is replaced by an empty operator of the
    appropriate rank, i.e., it does not contribute to the Hamiltonian.

    :param op1:     one-body operator, convertible to a dense matrix via op1.to_dense()
    :type op1:      OneBodyOperator or None
    :param op2:     sparse two-body operator with attributes indices and values
    :type op2:      TwoBodyOperator or None
    :param op3:     sparse three-body operator with attributes indices and values
    :type op3:      ThreeBodyOperator or None
    :param density: density matrix (same shape as the dense form of op1)
    :type density:  jax.Array((:,:), dtype=float or complex)
    :return:        matrix in the shape of op1 and density that is the Hartree-Fock Hamiltonian
    :rtype:         jax.Array((:,:), dtype=float or complex)
    """
    if op1 is None:
        op1 = OneBodyOperator.empty(len(density))
    if op2 is None:
        op2 = TwoBodyOperator.empty(len(density))
    if op3 is None:
        op3 = ThreeBodyOperator.empty(len(density))

    fock = jnp.asarray(op1.to_dense().astype(density.dtype))
    fock += _contract_2nf(op2, density)
    fock += 0.5 * _contract_3nf(op3, density)
    return fock


def solve_HF(
    op1: OneBodyOperator | None,
    op2: TwoBodyOperator | None,
    op3: ThreeBodyOperator | None,
    density: Array,
    mix: float = 0.5,
    eps: float = 1e-8,
    max_iter: int = 100,
    davidson_max_iter: int = 10,
    verbose: bool = False,
    sm: ShardingManager | None = None,
    diagonalizer: EigenSolver = "davidson",
    keep_all_orbitals: bool = True,
) -> tuple[Array, Array, bool]:
    """
    Solve the Hartree-Fock problem

    The iteration is stopped once the change of the density matrix drops below
    eps. With diagonalizer="davidson" only the npart lowest orbitals are
    computed in each iteration, warm-started from the previous ones; the full
    set of orbitals is then recovered by one dense diagonalization at the end
    if keep_all_orbitals is True.

    Any operator given as None is replaced by an empty operator of the
    appropriate rank, i.e., it does not contribute to the Hamiltonian.

    :param op1:               one-body operator, convertible to a dense matrix via op1.to_dense()
    :type op1:                OneBodyOperator or None
    :param op2:               sparse two-body operator with attributes indices and values
    :type op2:                TwoBodyOperator or None
    :param op3:               sparse three-body operator with attributes indices and values
    :type op3:                ThreeBodyOperator or None
    :param density:           initial density matrix (same shape as the dense form of op1)
    :type density:            jax.Array((:,:), dtype=float or complex)
    :param mix:               parameter used in the mixing: mix*new_density + (1-mix)*old_density
    :type mix:                float
    :param eps:               convergence criterion for the change of the density matrix
    :type eps:                float
    :param max_iter:          maximum number of HF iterations
    :type max_iter:           int
    :param davidson_max_iter: number of subspace expansion steps taken by the Davidson solver
    :type davidson_max_iter:  int
    :param verbose:           if True, print energy and residuals for every iteration
    :type verbose:            bool
    :param sm:                sharding manager used to distribute the arrays, or None for a single device
    :type sm:                 ShardingManager or None
    :param diagonalizer:      eigensolver used in each iteration, either "dense" or "davidson"
    :type diagonalizer:       EigenSolver
    :param keep_all_orbitals: if True, return all orbitals; if False, return only the occupied ones
    :type keep_all_orbitals:  bool
    :returns:                 * **energy** (*jax.Array((), dtype=float)*) -- Hartree-Fock energy of
                                the final iteration
                              * **orbs** (*jax.Array((:,:), dtype=float or complex)*) -- transformation
                                matrix that diagonalizes the HF Hamiltonian; the first A columns are
                                occupied. Only the occupied columns are returned if keep_all_orbitals
                                is False
                              * **converged** (*bool*) -- True if the change of the density matrix
                                dropped below eps
    :rtype:                   tuple[jax.Array, jax.Array, bool]
    :raises ValueError:       if diagonalizer is neither "dense" nor "davidson"
    """

    if diagonalizer not in {"davidson", "dense"}:
        raise ValueError("diagonalizer must be 'davidson' or 'dense'")

    if op1 is None:
        op1 = OneBodyOperator.empty(len(density))
    if op2 is None:
        op2 = TwoBodyOperator.empty(len(density))
    if op3 is None:
        op3 = ThreeBodyOperator.empty(len(density))

    h1_dense, v2_idx, v2_val, w3_idx, w3_val, _density = _prepare_inputs(
        op1, op2, op3, density, sm
    )

    prev_energy = 0.0
    converged = False
    npart = int(jnp.real(jnp.trace(_density)).round())

    occupied_orbitals = _guess_occupied_orbitals_from_density(_density, npart)

    for i in range(max_iter):
        occupied_orbitals, energy, _density, diff_density = _iterate_hf_equations(
            _density,
            h1_dense,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            npart,
            mix,
            occupied_orbitals,
            diagonalizer,
            davidson_max_iter,
        )

        dE = jnp.abs(energy - prev_energy)

        if verbose:
            print(f"Iter {i}: E={energy:.8f}, dE={dE:.6e}, dRho={diff_density:.6e}")

        if diff_density < eps:
            converged = True
            break

        prev_energy = energy

    if keep_all_orbitals:
        fock_2b, fock_3b = _build_2b_and_3b_fock_matrices(
            _density, v2_idx, v2_val, w3_idx, w3_val
        )
        fock = _build_full_fock_matrix(h1_dense, fock_2b, fock_3b)
        _, orbs = jnp.linalg.eigh(fock)
    else:
        orbs = occupied_orbitals

    return energy, orbs, converged


# Functions below this are private, they are not guaranteed to be stable between versions of NuLattice


def _adjoint(x: Array) -> Array:
    """
    takes the hermitian conjugate of a matrix, i.e., transposes the last two axes and conjugates

    :param x: matrix (or stack of matrices) to be conjugated
    :type x:  jax.Array((...,:,:), dtype=float or complex)
    :return:  hermitian conjugate of x
    :rtype:   jax.Array((...,:,:), dtype=float or complex)
    """
    return jnp.swapaxes(jnp.conj(x), -1, -2)


def _make_hermitian(x: Array) -> Array:
    """
    symmetrizes a matrix, i.e., returns its hermitian part

    :param x: matrix to be symmetrized
    :type x:  jax.Array((:,:), dtype=float or complex)
    :return:  (x + x^dagger)/2
    :rtype:   jax.Array((:,:), dtype=float or complex)
    """
    return 0.5 * (x + _adjoint(x))


@jax.jit
def _contract_2nf_fused(indices: Array, values: Array, density: Array) -> Array:
    """
    Contract the sparse two-body interaction with a one-body density.

    Each stored matrix element <pq|v|rs> contributes the four antisymmetric
    combinations obtained from the permutations P(pq) and P(rs).

    :param indices: index tuples [p,q,r,s] of the stored two-body matrix elements
    :type indices:  jax.Array((num_ele,4), dtype=int)
    :param values:  two-body matrix elements belonging to indices
    :type values:   jax.Array((num_ele,), dtype=float or complex)
    :param density: square density matrix
    :type density:  jax.Array((:,:), dtype=float or complex)
    :return:        one-body operator of the same shape as the density matrix
    :rtype:         jax.Array((:,:), dtype=float or complex)
    """
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
def _contract_3nf_fused(indices: Array, values: Array, density: Array) -> Array:
    """
    Contract the sparse three-body interaction with two densities.

    Each stored matrix element contributes the 36 antisymmetric combinations of
    the ket (abc) and bra (def) single-particle states. Of these, pairs are
    identical because the two densities commute under relabeling, so only 18
    terms are written out and the matrix elements are multiplied by two.

    :param indices: index tuples [a,b,c,d,e,f] of the stored three-body matrix elements
    :type indices:  jax.Array((num_ele,6), dtype=int)
    :param values:  three-body matrix elements belonging to indices
    :type values:   jax.Array((num_ele,), dtype=float or complex)
    :param density: square density matrix
    :type density:  jax.Array((:,:), dtype=float or complex)
    :return:        one-body operator of the same shape as the density matrix
    :rtype:         jax.Array((:,:), dtype=float or complex)
    """
    a, b, c, d, e, f = (indices[:, i] for i in range(6))
    n = density.shape[0]
    dtype = jnp.result_type(values.dtype, density.dtype)
    v2 = values * 2.0
    res = jnp.zeros((n, n), dtype=dtype)

    res = res.at[a, d].add(
        v2 * (density[b, e] * density[c, f] - density[c, e] * density[b, f])
    )
    res = res.at[b, d].add(
        v2 * (density[c, e] * density[a, f] - density[a, e] * density[c, f])
    )
    res = res.at[c, d].add(
        v2 * (density[a, e] * density[b, f] - density[b, e] * density[a, f])
    )

    res = res.at[a, e].add(
        v2 * (density[b, f] * density[c, d] - density[c, f] * density[b, d])
    )
    res = res.at[b, e].add(
        v2 * (density[c, f] * density[a, d] - density[a, f] * density[c, d])
    )
    res = res.at[c, e].add(
        v2 * (density[a, f] * density[b, d] - density[b, f] * density[a, d])
    )

    res = res.at[a, f].add(
        v2 * (density[b, d] * density[c, e] - density[c, d] * density[b, e])
    )
    res = res.at[b, f].add(
        v2 * (density[c, d] * density[a, e] - density[a, d] * density[c, e])
    )
    res = res.at[c, f].add(
        v2 * (density[a, d] * density[b, e] - density[b, d] * density[a, e])
    )
    return res


def _build_2b_and_3b_fock_matrices(
    density: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array,
    w3_val: Array,
) -> tuple[Array, Array]:
    """
    contracts the two- and three-body interactions with the density to get the
    corresponding one-body (Fock) matrices

    Both results are explicitly symmetrized to remove the small hermiticity
    violations caused by the sparse storage and the scatter-adds.

    :param density: square density matrix
    :type density:  jax.Array((:,:), dtype=float or complex)
    :param v2_idx:  index tuples [p,q,r,s] of the two-body matrix elements
    :type v2_idx:   jax.Array((:,4), dtype=int)
    :param v2_val:  two-body matrix elements belonging to v2_idx
    :type v2_val:   jax.Array((:,), dtype=float or complex)
    :param w3_idx:  index tuples [a,b,c,d,e,f] of the three-body matrix elements
    :type w3_idx:   jax.Array((:,6), dtype=int)
    :param w3_val:  three-body matrix elements belonging to w3_idx
    :type w3_val:   jax.Array((:,), dtype=float or complex)
    :returns:       * **fock_2b** (*jax.Array((:,:), dtype=float or complex)*) -- two-body
                      interaction contracted with one density
                    * **fock_3b** (*jax.Array((:,:), dtype=float or complex)*) -- three-body
                      interaction contracted with two densities
    :rtype:         tuple[jax.Array, jax.Array]
    """
    fock_2b = _make_hermitian(_contract_2nf_fused(v2_idx, v2_val, density))
    fock_3b = _make_hermitian(_contract_3nf_fused(w3_idx, w3_val, density))
    return fock_2b, fock_3b


def _build_full_fock_matrix(
    h1: Array,
    fock_2b: Array,
    fock_3b: Array,
) -> Array:
    """
    assembles the Hartree-Fock Hamiltonian from its one-, two-, and three-body pieces

    :param h1:      dense one-body matrix elements
    :type h1:       jax.Array((:,:), dtype=float or complex)
    :param fock_2b: two-body interaction contracted with one density
    :type fock_2b:  jax.Array((:,:), dtype=float or complex)
    :param fock_3b: three-body interaction contracted with two densities
    :type fock_3b:  jax.Array((:,:), dtype=float or complex)
    :return:        hermitian Hartree-Fock Hamiltonian
    :rtype:         jax.Array((:,:), dtype=float or complex)
    """
    return _make_hermitian(h1 + fock_2b + 0.5 * fock_3b)


def _compute_hf_energy_from_fock_matrices(
    density: Array,
    h1: Array,
    fock_2b: Array,
    fock_3b: Array,
) -> Array:
    """
    computes the Hartree-Fock energy from the already contracted Fock matrices

    The pieces enter with the weights 1, 1/2, and 1/6, so that no double
    counting of the interaction energy occurs.

    :param density: square density matrix
    :type density:  jax.Array((:,:), dtype=float or complex)
    :param h1:      dense one-body matrix elements
    :type h1:       jax.Array((:,:), dtype=float or complex)
    :param fock_2b: two-body interaction contracted with one density
    :type fock_2b:  jax.Array((:,:), dtype=float or complex)
    :param fock_3b: three-body interaction contracted with two densities
    :type fock_3b:  jax.Array((:,:), dtype=float or complex)
    :return:        Hartree-Fock energy
    :rtype:         jax.Array((), dtype=float)
    """

    e_h1 = jnp.einsum("ij,ji->", h1, density)
    e_2b = jnp.einsum("ij,ji->", fock_2b, density)
    e_3b = jnp.einsum("ij,ji->", fock_3b, density)
    return jnp.real(e_h1 + 0.5 * e_2b + (1.0 / 6.0) * e_3b)


@partial(
    jax.jit,
    static_argnames=(
        "number_of_particles",
        "diagonalizer",
    ),
)
def _iterate_hf_equations(
    density: Array,
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array,
    w3_val: Array,
    number_of_particles: int,
    mixing_param: float,
    prev_vecs: Array,
    diagonalizer: EigenSolver,
    davidson_max_iter: int,
) -> tuple[Array, Array, Array, Array]:
    """
    Performs one iteration of the Hartree-Fock procedure

    The returned energy is the one belonging to the incoming density, i.e., it
    is evaluated before the Fock matrix is diagonalized and the density is
    updated.

    :param density:             square density matrix
    :type density:              jax.Array((:,:), dtype=float or complex)
    :param h1:                  dense one-body matrix elements
    :type h1:                   jax.Array((:,:), dtype=float or complex)
    :param v2_idx:              index tuples [p,q,r,s] of the two-body matrix elements
    :type v2_idx:               jax.Array((:,4), dtype=int)
    :param v2_val:              two-body matrix elements belonging to v2_idx
    :type v2_val:               jax.Array((:,), dtype=float or complex)
    :param w3_idx:              index tuples [a,b,c,d,e,f] of the three-body matrix elements
    :type w3_idx:               jax.Array((:,6), dtype=int)
    :param w3_val:              three-body matrix elements belonging to w3_idx
    :type w3_val:               jax.Array((:,), dtype=float or complex)
    :param number_of_particles: number of occupied single-particle states
    :type number_of_particles:  int
    :param mixing_param:        returned density is mixing_param*new_density
                                + (1-mixing_param)*old_density
    :type mixing_param:         float
    :param prev_vecs:           occupied orbitals of the previous iteration, used to warm-start Davidson
    :type prev_vecs:            jax.Array((:,number_of_particles), dtype=float or complex)
    :param diagonalizer:        eigensolver to be used, either "dense" or "davidson"
    :type diagonalizer:         EigenSolver
    :param davidson_max_iter:   number of subspace expansion steps taken by the Davidson solver
    :type davidson_max_iter:    int
    :returns:                   * **occupied_orbitals** (*jax.Array((:,number_of_particles),
                                  dtype=float or complex)*) -- the number_of_particles lowest
                                  orbitals of the Fock matrix
                                * **energy** (*jax.Array((), dtype=float)*) -- HF energy of the
                                  incoming density
                                * **mixed_density** (*jax.Array((:,:), dtype=float or complex)*) --
                                  updated density after mixing with the incoming one
                                * **residual_density** (*jax.Array((), dtype=float)*) -- summed
                                  absolute change of the density
    :rtype:                     tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    """
    fock_2b, fock_3b = _build_2b_and_3b_fock_matrices(
        density, v2_idx, v2_val, w3_idx, w3_val
    )
    fock = _build_full_fock_matrix(h1, fock_2b, fock_3b)
    energy = _compute_hf_energy_from_fock_matrices(density, h1, fock_2b, fock_3b)

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


def _prepare_inputs(
    op1: OneBodyOperator,
    op2: TwoBodyOperator,
    op3: ThreeBodyOperator,
    density: Array,
    sm: ShardingManager | None,
    dtype=jnp.float64,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    converts the operators and the density into the plain arrays used by the
    Hartree-Fock iteration and, if requested, distributes them over devices

    :param op1:     one-body operator, convertible to a dense matrix via op1.to_dense()
    :type op1:      OneBodyOperator
    :param op2:     sparse two-body operator with attributes indices and values
    :type op2:      TwoBodyOperator
    :param op3:     sparse three-body operator with attributes indices and values
    :type op3:      ThreeBodyOperator
    :param density: square density matrix
    :type density:  jax.Array((:,:), dtype=float or complex)
    :param sm:      sharding manager used to distribute the arrays, or None for a single device
    :type sm:       ShardingManager or None
    :param dtype:   data type used for the arrays
    :type dtype:    jax.numpy.dtype
    :returns:       * **h1** (*jax.Array((:,:), dtype=float or complex)*) -- dense one-body
                      matrix elements
                    * **v2_idx** (*jax.Array((:,4), dtype=int)*) -- index tuples of the two-body
                      matrix elements
                    * **v2_val** (*jax.Array((:,), dtype=float or complex)*) -- two-body matrix
                      elements belonging to v2_idx
                    * **w3_idx** (*jax.Array((:,6), dtype=int)*) -- index tuples of the three-body
                      matrix elements
                    * **w3_val** (*jax.Array((:,), dtype=float or complex)*) -- three-body matrix
                      elements belonging to w3_idx
                    * **density** (*jax.Array((:,:), dtype=float or complex)*) -- density matrix
    :rtype:         tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    """
    if sm is not None:
        assert (
            sm.num_nodes == 1 or sm.num_gpus == 1
        ), "HF expects 1D mesh, ensure sm.num_nodes or sm.num_gpus is 1"
        h1 = sm.prepare(op1.to_dense(), rank=0)
        density = sm.prepare(density, rank=0)
        v2_idx = sm.prepare(op2.indices)
        v2_val = sm.prepare(op2.values)
        w3_idx = sm.prepare(op3.indices)
        w3_val = sm.prepare(op3.values)
    else:
        h1 = jnp.asarray(op1.to_dense())
        v2_idx = jnp.asarray(op2.indices)
        v2_val = jnp.asarray(op2.values)
        w3_idx = jnp.asarray(op3.indices)
        w3_val = jnp.asarray(op3.values)
        density = jnp.asarray(density)

    return h1, v2_idx, v2_val, w3_idx, w3_val, density


def _guess_occupied_orbitals_from_density(density: Array, npart: int) -> Array:
    """
    extracts a set of npart occupied orbitals from a density matrix

    The density is a projector onto the occupied space, so applying it to a
    random trial basis and orthonormalizing the result gives vectors that span
    that space. The vectors with eigenvalue close to one are kept; a few extra
    trial vectors (CONDITION_NUMBER) are used to keep the QR decomposition well
    conditioned. These orbitals serve as the initial guess for the Davidson
    solver.

    :param density: square density matrix, assumed to be a projector
    :type density:  jax.Array((:,:), dtype=float or complex)
    :param npart:   number of occupied single-particle states
    :type npart:    int
    :return:        orthonormal orbitals spanning the occupied space of density
    :rtype:         jax.Array((:,npart), dtype=float or complex)
    """
    # We generate a random matrix P that serve as trial basis to get the eigenvectors of dens
    # P has the size npart + CONDITION_NUMBER, where CONDITION_NUMBER ensures numerical stability
    dim = len(density)
    key = jax.random.key(42)
    CONDITION_NUMBER = 5
    CONDITION_NUMBER = min(CONDITION_NUMBER, dim - npart)  # Make sure we stay in bounds
    P = jax.random.normal(
        key, shape=(dim, npart + CONDITION_NUMBER), dtype=density.dtype
    )

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
                print(
                    f"Warning: Found more than npart={npart} eigenvectors of density matrix with nonzero eigenvalues."
                )
                print("Something might be wrong!")
                break
            indices = indices.at[count].set(i)
            count += 1
    if count != npart:
        print(
            f"Warning: Found only {count} (which is less than npart={npart}) eigenvectors of density matrix with nonzero eigenvalues."
        )
        print("Something might be wrong!")

    # We get the occupied orbitals from the eigenvalues that are nonzero
    occupied_orbitals = jnp.asarray(Q[:, indices])

    # # This code can be used to double-check that we actually get the eigenvalues we think
    # vals_final = jnp.conjugate(jnp.transpose(occupied_orbitals)) @ density @ occupied_orbitals
    # print(jnp.diag(vals_final))

    return occupied_orbitals


def _contract_3nf(op3: ThreeBodyOperator, density: Array) -> Array:
    """
    takes a sparse three-body operator and contracts it with two densities to get a one-body operator

    :param op3:     sparse three-body operator with attributes indices and values
    :type op3:      operator with indices (:,6) and values (:,)
    :param density: square density matrix
    :type density:  jax.Array((:,:), dtype=float or complex)
    :return:        one-body operator of the same shape as the density matrix
    :rtype:         jax.Array((:,:), dtype=float or complex)
    """
    w3_idx = jnp.asarray(op3.indices)
    w3_val = jnp.asarray(op3.values)

    return _contract_3nf_fused(w3_idx, w3_val, density)


def _contract_2nf(op2: TwoBodyOperator, density: Array) -> Array:
    """
    takes a sparse two-body operator and contracts it with the density to get a one-body operator

    :param op2:     sparse two-body operator with attributes indices and values
    :type op2:      operator with indices (:,4) and values (:,)
    :param density: square density matrix
    :type density:  jax.Array((:,:), dtype=float or complex)
    :return:        one-body operator of the same shape as the density matrix
    :rtype:         jax.Array((:,:), dtype=float or complex)
    """
    v2_idx = jnp.asarray(op2.indices)
    v2_val = jnp.asarray(op2.values)

    return _contract_2nf_fused(v2_idx, v2_val, density)


def _HF_iter(
    op1: OneBodyOperator | None,
    op2: TwoBodyOperator | None,
    op3: ThreeBodyOperator | None,
    density: Array,
    mix: float = 0.5,
) -> tuple[Array, Array, Array]:
    """
    Performs one iteration of the Hartree-Fock procedure using dense diagonalization

    This is the straightforward reference implementation kept for testing; the
    production path goes through _iterate_hf_equations.

    Any operator given as None is replaced by an empty operator of the
    appropriate rank by HF_energy and make_HF_ham, i.e., it does not contribute.

    :param op1:     one-body operator, convertible to a dense matrix via op1.to_dense()
    :type op1:      OneBodyOperator or None
    :param op2:     sparse two-body operator with attributes indices and values
    :type op2:      TwoBodyOperator or None
    :param op3:     sparse three-body operator with attributes indices and values
    :type op3:      ThreeBodyOperator or None
    :param density: density matrix (same shape as the dense form of op1)
    :type density:  jax.Array((:,:), dtype=float or complex)
    :param mix:     returned density matrix is mix*new_density + (1-mix)*old_density
    :type mix:      float
    :returns:       * **energy** (*jax.Array((), dtype=float)*) -- HF energy of the incoming
                      density
                    * **mixed_density** (*jax.Array((:,:), dtype=float or complex)*) -- updated
                      density after mixing with the incoming one
                    * **orbitals** (*jax.Array((:,:), dtype=float or complex)*) -- transformation
                      matrix that diagonalizes the HF Hamiltonian
    :rtype:         tuple[jax.Array, jax.Array, jax.Array]
    """
    npart = round(jnp.real(jnp.trace(density)))

    energy = HF_energy(op1, op2, op3, density)
    fock = make_HF_ham(op1, op2, op3, density)
    _, orbitals = jnp.linalg.eigh(fock)
    occupied_orbitals = orbitals[:, 0:npart]
    new_density = occupied_orbitals @ jnp.conjugate(jnp.transpose(occupied_orbitals))

    mixed_density = mix * new_density + (1.0 - mix) * density

    return energy, mixed_density, orbitals
