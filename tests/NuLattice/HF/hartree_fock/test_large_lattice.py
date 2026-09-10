"""Large-lattice regressions for example-derived Hartree-Fock interactions."""

import numpy as np
import pytest

from NuLattice import references
from NuLattice.HF.hartree_fock import HF_energy, init_density, make_HF_ham, solve_HF
from NuLattice._reference.HF import hartree_fock as reference_hf


pytestmark = pytest.mark.large_lattice


LARGE_CASES = ["interaction_2017_l3", "interaction_2016b_l4"]
NUCLEI = [("He4", references.ref_4He_gs), ("O16", references.ref_16O_gs)]


def density_for(case, reference):
    """Initialize a reference nucleus in a large-case basis."""

    holes = references.reference_to_holes(reference, case.basis)
    return init_density(case.nstat, list(holes), dtype=complex)


@pytest.mark.parametrize("fixture_name", LARGE_CASES)
@pytest.mark.parametrize(("nucleus", "reference"), NUCLEI)
def test_large_energy_and_fock_match_reference(request, fixture_name, nucleus, reference):
    """Match He4/O16 energies and Fock matrices for L=3 and L=4 models."""

    del nucleus
    case = request.getfixturevalue(fixture_name)
    density = density_for(case, reference)
    expected_energy = reference_hf.HF_energy(
        *case.legacy_operators, np.asarray(density)
    )
    expected_fock = reference_hf.make_HF_ham(
        *case.legacy_operators, np.asarray(density)
    )

    np.testing.assert_allclose(
        HF_energy(*case.operators, density), expected_energy, rtol=2e-11, atol=2e-9
    )
    np.testing.assert_allclose(
        make_HF_ham(*case.operators, density),
        expected_fock,
        rtol=2e-11,
        atol=2e-9,
    )


@pytest.mark.parametrize(
    ("fixture_name", "reference"),
    [
        ("interaction_2017_l3", references.ref_4He_gs),
        ("interaction_2016b_l4", references.ref_16O_gs),
    ],
)
def test_large_single_iteration_matches_reference(request, fixture_name, reference):
    """Match cost-appropriate nonconverged solver energies on both large models."""

    case = request.getfixturevalue(fixture_name)
    density = density_for(case, reference)
    actual_energy, orbitals, actual_converged = solve_HF(
        *case.operators,
        density,
        mix=0.7,
        eps=1e-8,
        max_iter=1,
        diagonalizer="dense",
    )
    expected_energy, _, expected_converged = reference_hf.solve_HF(
        *case.legacy_operators,
        np.asarray(density),
        mix=0.7,
        eps=1e-8,
        max_iter=1,
    )

    np.testing.assert_allclose(actual_energy, expected_energy, rtol=2e-11, atol=2e-9)
    np.testing.assert_allclose(
        np.asarray(orbitals).conj().T @ np.asarray(orbitals),
        np.eye(case.nstat),
        atol=2e-9,
    )
    assert actual_converged is expected_converged is False


@pytest.mark.parametrize("fixture_name", LARGE_CASES)
def test_large_fock_eigenspace_is_stationary(request, fixture_name):
    """Check that occupied Fock eigenspaces commute with their source Fock matrix."""

    case = request.getfixturevalue(fixture_name)
    density = density_for(case, references.ref_4He_gs)
    fock = np.asarray(make_HF_ham(*case.operators, density))
    _, orbitals = np.linalg.eigh(fock)
    occupied = orbitals[:, :4]
    fock_density = occupied @ occupied.conj().T
    commutator = fock @ fock_density - fock_density @ fock

    np.testing.assert_allclose(commutator, 0.0, rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(fock_density @ fock_density, fock_density, atol=2e-10)
    np.testing.assert_allclose(np.trace(fock_density), 4.0, atol=2e-10)

