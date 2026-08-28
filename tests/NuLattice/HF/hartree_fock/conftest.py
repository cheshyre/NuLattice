"""Shared inputs for Hartree-Fock public-method tests."""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from NuLattice import lattice, references
from NuLattice._reference.HF import hartree_fock as reference_hf
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


def occupied_projector(orbitals, particle_count):
    """Build the density projector while removing orbital phase freedom."""

    occupied = np.asarray(orbitals)[:, :particle_count]
    return occupied @ occupied.conj().T


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

