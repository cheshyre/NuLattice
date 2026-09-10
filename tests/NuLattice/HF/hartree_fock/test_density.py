"""Tests for Hartree-Fock density initialization."""

import jax.numpy as jnp
import numpy as np
import pytest

from NuLattice import references
from NuLattice.HF.hartree_fock import init_density
from NuLattice._reference.HF.hartree_fock import init_density as reference_init_density


@pytest.mark.parametrize(
    ("nucleus", "reference"),
    [("He4", references.ref_4He_gs), ("O16", references.ref_16O_gs)],
)
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_init_density_matches_reference(small_hamiltonian, nucleus, reference, dtype):
    """Match legacy initial densities for He4 and O16 in both dtypes."""

    del nucleus
    holes = references.reference_to_holes(reference, small_hamiltonian.basis)
    actual = init_density(small_hamiltonian.nstat, list(holes), dtype=dtype)
    expected = reference_init_density(
        small_hamiltonian.nstat, holes, dtype=np.dtype(dtype)
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype


def test_init_density_is_particle_number_projector(small_hamiltonian):
    """Produce a Hermitian projector with trace equal to the hole count."""

    holes = [0, 3, 7, 11]
    density = np.asarray(init_density(small_hamiltonian.nstat, holes))
    np.testing.assert_allclose(density, density.conj().T)
    np.testing.assert_allclose(density @ density, density)
    np.testing.assert_allclose(np.trace(density), len(holes))


def test_shared_density_states_have_physical_one_body_bounds(density_cases):
    """Keep all reused densities Hermitian with valid occupations and trace."""

    for case in density_cases:
        eigenvalues = np.linalg.eigvalsh(case.density)
        np.testing.assert_allclose(case.density, case.density.conj().T, atol=1e-12)
        np.testing.assert_allclose(np.trace(case.density), case.particle_count, atol=1e-12)
        assert eigenvalues.min() >= -1e-12
        assert eigenvalues.max() <= 1.0 + 1e-12
        if case.stage in {"initial", "converged"}:
            np.testing.assert_allclose(
                case.density @ case.density, case.density, atol=2e-8
            )
        else:
            assert not np.allclose(case.density @ case.density, case.density)

