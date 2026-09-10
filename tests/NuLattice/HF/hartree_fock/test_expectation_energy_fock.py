"""Tests for Hartree-Fock expectation values, energies, and Fock matrices."""

import jax.numpy as jnp
import numpy as np
import pytest

from NuLattice.HF.hartree_fock import (
    HF_energy,
    evaluate_slater_determinant_expectation_value,
    make_HF_ham,
)
from NuLattice._reference.HF import hartree_fock as reference_hf
from NuLattice.utils._jax_types import OneBodyOperator


OPERATOR_MASKS = [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


def select_operators(operators, mask):
    """Replace disabled Hamiltonian terms by the public None sentinel."""

    return tuple(operator if enabled else None for operator, enabled in zip(operators, mask))


def legacy_lists(operators):
    """Convert public operators to reference inputs, preserving absent terms."""

    return tuple([] if operator is None else operator.to_list() for operator in operators)


@pytest.mark.parametrize("density_index", range(6))
@pytest.mark.parametrize("operator_mask", OPERATOR_MASKS)
def test_expectation_and_energy_match_reference(
    small_hamiltonian, density_cases, density_index, operator_mask
):
    """Match every nonempty rank combination at all shared density stages."""

    density = density_cases[density_index].density
    operators = select_operators(small_hamiltonian.operators, operator_mask)
    expected = reference_hf.HF_energy(*legacy_lists(operators), density)

    expectation = evaluate_slater_determinant_expectation_value(*operators, density)
    energy = HF_energy(*operators, density)

    np.testing.assert_allclose(expectation, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(energy, np.real(expected), rtol=1e-11, atol=1e-11)
    assert not jnp.iscomplexobj(energy)


@pytest.mark.parametrize("density_index", range(6))
@pytest.mark.parametrize("operator_mask", OPERATOR_MASKS)
def test_fock_matrix_matches_reference(
    small_hamiltonian, density_cases, density_index, operator_mask
):
    """Match reference Fock matrices for every rank and density stage."""

    density = density_cases[density_index].density
    operators = select_operators(small_hamiltonian.operators, operator_mask)
    expected = reference_hf.make_HF_ham(*legacy_lists(operators), density)
    actual = make_HF_ham(*operators, density)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_absent_operators_contribute_zero(small_hamiltonian):
    """Treat three absent Hamiltonian terms as correctly shaped zeros."""

    density = np.eye(small_hamiltonian.nstat)[:4]
    density = density.T @ density
    expectation = evaluate_slater_determinant_expectation_value(
        None, None, None, density
    )
    np.testing.assert_array_equal(expectation, 0.0)
    np.testing.assert_array_equal(HF_energy(None, None, None, density), 0.0)
    np.testing.assert_array_equal(
        make_HF_ham(None, None, None, density), np.zeros_like(density)
    )


def test_complex_density_matches_reference(small_hamiltonian, density_cases):
    """Preserve complex Hermitian density contractions in energy and Fock APIs."""

    density = density_cases[1].density.astype(complex)
    phases = np.exp(1j * np.linspace(0.0, 0.7, small_hamiltonian.nstat))
    density = phases[:, None] * density * phases.conj()[None, :]
    legacy = tuple(operator.to_list() for operator in small_hamiltonian.operators)

    expected_energy = reference_hf.HF_energy(*legacy, density)
    expected_fock = reference_hf.make_HF_ham(*legacy, density)
    expectation = evaluate_slater_determinant_expectation_value(
        *small_hamiltonian.operators, density
    )

    assert jnp.iscomplexobj(expectation)
    np.testing.assert_allclose(expectation, expected_energy, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        make_HF_ham(*small_hamiltonian.operators, density),
        expected_fock,
        rtol=1e-11,
        atol=1e-11,
    )
    assert not jnp.iscomplexobj(HF_energy(*small_hamiltonian.operators, density))


def test_force_real_and_energy_warning(capsys):
    """Expose complex expectations while energy warns and returns the real part."""

    operator = OneBodyOperator([[0, 0]], [1.0j], 2)
    density = np.diag([1.0, 0.0])
    expectation = evaluate_slater_determinant_expectation_value(
        operator, None, None, density
    )
    forced = evaluate_slater_determinant_expectation_value(
        operator, None, None, density, force_real=True
    )
    energy = HF_energy(operator, None, None, density)

    np.testing.assert_allclose(expectation, 1.0j)
    np.testing.assert_allclose(forced, 0.0)
    np.testing.assert_allclose(energy, 0.0)
    assert jnp.iscomplexobj(expectation)
    assert not jnp.iscomplexobj(forced)
    assert "Warning: Computed energy is complex" in capsys.readouterr().out

