"""Shared inputs for Hartree-Fock public-method tests."""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from NuLattice import constants_NLEFT, lattice, references
from NuLattice._reference.HF import hartree_fock as reference_hf
from NuLattice.operators import one_body_operators, two_body_operators
from NuLattice.utils._jax_types import (
    OneBodyOperator,
    ThreeBodyOperator,
    TwoBodyOperator,
)


@dataclass(frozen=True)
class HamiltonianCase:
    """Hold matching JAX operators and legacy operator lists."""

    nstat: int
    basis: list
    operators: tuple
    legacy_operators: tuple


@dataclass(frozen=True)
class DensityCase:
    """Identify a nuclear density and its iteration stage."""

    nucleus: str
    stage: str
    particle_count: int
    density: np.ndarray


@dataclass(frozen=True)
class LargeHamiltonianCase:
    """Hold an example-derived large-lattice regression Hamiltonian."""

    style: str
    size: int
    nstat: int
    basis: list
    operators: tuple
    legacy_operators: tuple


def occupied_projector(orbitals, particle_count):
    """Build the density projector while removing orbital phase freedom."""

    occupied = np.asarray(orbitals)[:, :particle_count]
    return occupied @ occupied.conj().T


def sampled_short_range(sites, size, s_local, s_nonlocal, strength, op1b=None):
    """Build one representative site of an expensive smeared contact channel."""

    density = one_body_operators.get_smeared_dens(
        sites,
        size,
        s_local,
        s_nonlocal,
        op1b=op1b,
        sites=[sites[0]],
    )[0]
    return two_body_operators.rho_mult_NO(density, density, strength)


def large_case(style, size, one_body_list, two_body_matrix):
    """Create matching public and legacy forms of a large Hamiltonian."""

    basis = lattice.get_sp_basis(size)
    nstat = len(basis)
    op1 = OneBodyOperator.from_list(one_body_list, nstat)
    op2 = TwoBodyOperator.from_scipy_csr(two_body_matrix, nstat)
    operators = (op1, op2, None)
    return LargeHamiltonianCase(
        style,
        size,
        nstat,
        basis,
        operators,
        (op1.to_list(), op2.to_list(), []),
    )


@pytest.fixture(scope="session")
def small_hamiltonian():
    """Build the full L=2 kinetic, contact, and three-body model once."""

    size = 2
    sites = lattice.get_lattice(size)
    basis = lattice.get_sp_basis(size)
    nstat = len(basis)
    legacy = (
        lattice.Tkin(sites, size),
        lattice.contacts(-1.5, -1.0, sites, size),
        lattice.NNNcontact(0.25, sites, size),
    )
    operators = (
        OneBodyOperator.from_list(legacy[0], nstat),
        TwoBodyOperator.from_list(legacy[1], nstat),
        ThreeBodyOperator.from_list(legacy[2], nstat),
    )
    return HamiltonianCase(nstat, basis, operators, legacy)


@pytest.fixture(scope="session")
def density_cases(small_hamiltonian):
    """Reuse initial, mixed nonconverged, and converged He4/O16 densities."""

    cases = []
    references_by_nucleus = {
        "He4": references.ref_4He_gs,
        "O16": references.ref_16O_gs,
    }
    for nucleus, reference in references_by_nucleus.items():
        holes = references.reference_to_holes(reference, small_hamiltonian.basis)
        particle_count = len(holes)
        initial = reference_hf.init_density(small_hamiltonian.nstat, holes)
        _, nonconverged, _ = reference_hf.HF_iter(
            *small_hamiltonian.legacy_operators, initial, mix=0.35
        )
        _, converged_orbitals, converged = reference_hf.solve_HF(
            *small_hamiltonian.legacy_operators,
            initial,
            mix=0.7,
            eps=1e-7,
            max_iter=100,
        )
        assert converged
        converged_density = occupied_projector(converged_orbitals, particle_count)
        cases.extend(
            [
                DensityCase(nucleus, "initial", particle_count, initial),
                DensityCase(nucleus, "nonconverged", particle_count, nonconverged),
                DensityCase(nucleus, "converged", particle_count, converged_density),
            ]
        )
    return cases


@pytest.fixture(scope="session")
def interaction_2017_l3():
    """Build the L=3 2017 kinetic, OPE, and sampled smeared contact model."""

    size = 3
    spacing = 1.0 / 100.0
    sites = lattice.get_lattice(size)
    kinetic = one_body_operators.tKin(
        size, 3, spacing, mass=constants_NLEFT.mass
    )
    ope = two_body_operators.onePionEx(
        size,
        0.7,
        spacing,
        sites,
        g_A=constants_NLEFT.g_A,
        f_pi=constants_NLEFT.f_pi,
        m_pi_0=constants_NLEFT.m_pi_0,
    )
    contact = sampled_short_range(
        sites, size, 0.08, 0.08, -0.185 / spacing
    )
    return large_case("2017", size, kinetic, ope + contact)


@pytest.fixture(scope="session")
def interaction_2016b_l4():
    """Build the L=4 2016B model with representative contact channels."""

    size = 4
    spacing = 1.0 / 100.0
    sites = lattice.get_lattice(size)
    kinetic = one_body_operators.tKin(
        size, 3, spacing, mass=constants_NLEFT.mass
    )
    interaction = two_body_operators.onePionEx(size, 0.7, spacing, sites)

    c_nonlocal = -0.1171 / spacing
    c_isospin_nonlocal = 0.02607 / spacing
    interaction += sampled_short_range(sites, size, 0.0, 0.077, c_nonlocal)
    tau_z = one_body_operators.list_to_sparse1b(
        one_body_operators.pauli_tau_z(sites, size)
    )
    interaction += sampled_short_range(
        sites, size, 0.0, 0.077, c_isospin_nonlocal, op1b=tau_z
    )

    c_local = -0.01013 / spacing
    channel_strength = -c_local / 3.0
    spin_z = one_body_operators.list_to_sparse1b(
        one_body_operators.pauli_spin_z(sites, size)
    )
    interaction += sampled_short_range(sites, size, 0.81, 0.0, c_local)
    interaction += sampled_short_range(
        sites, size, 0.81, 0.0, channel_strength, op1b=spin_z
    )
    interaction += sampled_short_range(
        sites, size, 0.81, 0.0, channel_strength, op1b=tau_z
    )
    interaction += sampled_short_range(
        sites,
        size,
        0.81,
        0.0,
        channel_strength,
        op1b=spin_z @ tau_z,
    )
    return large_case("2016B", size, kinetic, interaction)
