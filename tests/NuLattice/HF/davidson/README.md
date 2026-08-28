# Davidson tests

Run the focused suite with JAX compilation enabled by default:

```sh
python -m pytest tests/NuLattice/HF/davidson
```

The suite covers real, complex, diagonal, coupled, clustered, and degenerate
Hermitian problems in the 64-bit JAX configuration enabled by `NuLattice`.
It checks helper boundaries, eigenvalue ordering, orthogonality, occupied-space
projectors, normalized eigenpair residuals, warm starts, iteration counts, and
repeated calls to an explicitly lowered and compiled solver.

Converged perturbed-start tolerances are set from the normalized residual
`||H v - lambda v|| / max(||H||_2, 1)`. The largest accepted residual is
`2e-5` for the restarted two-root complex problem; real problems use `1e-6` or
tighter tolerances.

Run the Hartree-Fock integration tests without the expensive lattice cases:

```sh
python -m pytest tests/NuLattice/HF/hartree_fock -m "not large_lattice"
```

Run all non-large-lattice tests with:

```sh
python -m pytest -m "not large_lattice"
```
