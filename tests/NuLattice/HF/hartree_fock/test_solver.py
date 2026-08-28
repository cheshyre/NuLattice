"""Tests for the public Hartree-Fock solver."""

import jax
import numpy as np
import pytest

from NuLattice import references
from NuLattice.HF.hartree_fock import init_density, solve_HF
from NuLattice._reference.HF import hartree_fock as reference_hf
from NuLattice.utils._jax_types import OneBodyOperator, ShardingManager


NUCLEI = [("He4", references.ref_4He_gs), ("O16", references.ref_16O_gs)]


def projector(orbitals, particle_count):
    """Compare occupied subspaces without fixing phases or degenerate vectors."""

    occupied = np.asarray(orbitals)[:, :particle_count]
    return occupied @ occupied.conj().T


@pytest.mark.parametrize(("nucleus", "reference"), NUCLEI)
def test_dense_solver_matches_reference_when_not_converged(
    small_hamiltonian, nucleus, reference
):
    """Match the first-iteration energy for nonconverged He4 and O16 runs."""

    del nucleus
    holes = references.reference_to_holes(reference, small_hamiltonian.basis)
    density = init_density(small_hamiltonian.nstat, list(holes))
    actual_energy, actual_orbitals, actual_converged = solve_HF(
        *small_hamiltonian.operators,
        density,
        mix=0.7,
        eps=1e-7,
        max_iter=1,
        diagonalizer="dense",
    )
    expected_energy, _, expected_converged = reference_hf.solve_HF(
        *(operator.to_list() for operator in small_hamiltonian.operators),
        np.asarray(density),
        mix=0.7,
        eps=1e-7,
        max_iter=1,
    )

    np.testing.assert_allclose(actual_energy, expected_energy, rtol=1e-11, atol=1e-11)
    assert actual_orbitals.shape == (small_hamiltonian.nstat, small_hamiltonian.nstat)
    assert actual_converged is expected_converged is False


@pytest.mark.parametrize(("nucleus", "reference"), NUCLEI)
@pytest.mark.parametrize("keep_all_orbitals", [True, False])
def test_dense_solver_matches_converged_reference(
    small_hamiltonian, nucleus, reference, keep_all_orbitals
):
    """Match converged energies and occupied subspaces for He4 and O16."""

    del nucleus
    holes = references.reference_to_holes(reference, small_hamiltonian.basis)
    particle_count = len(holes)
    density = init_density(small_hamiltonian.nstat, list(holes))
    actual_energy, actual_orbitals, actual_converged = solve_HF(
        *small_hamiltonian.operators,
        density,
        mix=0.7,
        eps=1e-7,
        max_iter=100,
        diagonalizer="dense",
        keep_all_orbitals=keep_all_orbitals,
    )
    expected_energy, expected_orbitals, expected_converged = reference_hf.solve_HF(
        *(operator.to_list() for operator in small_hamiltonian.operators),
        np.asarray(density),
        mix=0.7,
        eps=1e-7,
        max_iter=100,
    )

    expected_columns = small_hamiltonian.nstat if keep_all_orbitals else particle_count
    assert actual_orbitals.shape == (small_hamiltonian.nstat, expected_columns)
    assert actual_converged is expected_converged is True
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        projector(actual_orbitals, particle_count),
        projector(expected_orbitals, particle_count),
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(actual_orbitals).conj().T @ np.asarray(actual_orbitals),
        np.eye(expected_columns),
        atol=2e-9,
    )


@pytest.mark.parametrize("keep_all_orbitals", [True, False])
def test_davidson_solver_and_retained_orbitals(keep_all_orbitals):
    """Solve a nondegenerate model with Davidson and both retention modes."""

    nstat = 12
    particle_count = 4
    diagonal = np.linspace(-2.0, 3.0, nstat)
    indices = np.column_stack((np.arange(nstat), np.arange(nstat)))
    operator = OneBodyOperator(indices, diagonal, nstat)
    density = init_density(nstat, list(range(particle_count)))

    energy, orbitals, converged = solve_HF(
        operator,
        None,
        None,
        density,
        eps=1e-10,
        max_iter=10,
        davidson_max_iter=3,
        diagonalizer="davidson",
        keep_all_orbitals=keep_all_orbitals,
    )

    expected_columns = nstat if keep_all_orbitals else particle_count
    np.testing.assert_allclose(energy, diagonal[:particle_count].sum(), atol=1e-10)
    np.testing.assert_allclose(
        projector(orbitals, particle_count), density, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(orbitals).conj().T @ np.asarray(orbitals),
        np.eye(expected_columns),
        atol=2e-9,
    )
    assert orbitals.shape == (nstat, expected_columns)
    assert converged is True


def test_single_device_sharding_matches_unsharded(small_hamiltonian):
    """Preserve dense solver results on a one-device sharding mesh."""

    density = init_density(small_hamiltonian.nstat, [0, 1, 2, 3])
    kwargs = dict(max_iter=1, diagonalizer="dense")
    expected = solve_HF(*small_hamiltonian.operators, density, **kwargs)
    actual = solve_HF(
        *small_hamiltonian.operators,
        density,
        sm=ShardingManager(num_nodes=1, num_gpus=1),
        **kwargs,
    )

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        projector(actual[1], 4), projector(expected[1], 4), atol=1e-10
    )
    assert len(actual[1].sharding.device_set) == 1
    assert actual[2] is expected[2]


def test_verbose_solver_reports_iteration_residuals(small_hamiltonian, capsys):
    """Print energy and density residuals when verbose output is enabled."""

    density = init_density(small_hamiltonian.nstat, [0, 1, 2, 3])
    solve_HF(
        *small_hamiltonian.operators,
        density,
        max_iter=1,
        diagonalizer="dense",
        verbose=True,
    )
    output = capsys.readouterr().out
    assert "Iter 0: E=" in output
    assert "dE=" in output
    assert "dRho=" in output


def test_invalid_diagonalizer_raises(small_hamiltonian):
    """Reject diagonalizer names outside the public literal choices."""

    density = init_density(small_hamiltonian.nstat, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="diagonalizer must be 'davidson' or 'dense'"):
        solve_HF(
            *small_hamiltonian.operators,
            density,
            diagonalizer="invalid",
        )


def test_solver_returns_jax_arrays(small_hamiltonian):
    """Keep the public numerical outputs as JAX arrays."""

    density = init_density(small_hamiltonian.nstat, [0, 1, 2, 3])
    energy, orbitals, _ = solve_HF(
        *small_hamiltonian.operators,
        density,
        max_iter=1,
        diagonalizer="dense",
    )
    assert isinstance(energy, jax.Array)
    assert isinstance(orbitals, jax.Array)

