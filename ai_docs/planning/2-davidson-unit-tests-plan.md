@../../AGENTS.md
@planning.md
@../development-style.md
@../unit-tests.md

# Davidson unit-test plan

## Goal

Add focused pytest coverage for `NuLattice/HF/davidson.py`. Verify the public
solver against dense Hermitian diagonalization and test its numerical helpers
at their important boundaries. Compare eigenspaces rather than individual
eigenvectors when phases or degeneracies make vectors non-unique.

The tests should expose two current correctness risks: zero-padded initial
subspaces can introduce spurious Ritz values, and returned eigenvalues and
eigenvectors can describe different iteration states.

## Dependencies

None.

## Implementation steps

### 1. Establish deterministic matrices and comparison utilities

Create `tests/NuLattice/HF/davidson/` with reusable small real and complex
Hermitian problems. Include diagonal, coupled, clustered, and degenerate
spectra with exact and perturbed warm-start vectors. Add concise helpers for
eigenpair residuals, orthogonality, and occupied-subspace projectors.

Commit: `Add Davidson test inputs and comparison utilities`

### 2. Test numerical helper functions

Cover `_adjoint` for real, complex, and batched arrays. Test denominator
regularization on negative, zero, positive, threshold, and unchanged entries.
Verify that `_cholesky_qr` and `_cqr2` preserve the column space and produce
finite, orthonormal columns for well-conditioned and nearly dependent
full-rank inputs.

Commit: `Test Davidson numerical helpers`

### 3. Test public eigenpair correctness

Compare `davidson_eigh` with `jax.numpy.linalg.eigh` for the lowest one and
multiple roots of the deterministic problems. Check eigenvalue ordering,
output shapes and dtypes, orthogonality, invariant subspaces, and
`H V - V Lambda` residuals. Include positive-definite diagonal matrices and
shifted copies of the same matrix to guard against spurious zero Ritz values.

Commit: `Test Davidson eigenpair correctness`

### 4. Cover iteration and numerical edge behavior

Exercise exact and perturbed warm starts, coincident diagonal/eigenvalue
preconditioner entries, clustered and degenerate roots, complex Hermitian
inputs, and representative iteration counts. Require finite outputs and
ensure each returned eigenvalue is paired with its returned eigenvector.
Document the supported dimensional and iteration preconditions rather than
testing inputs outside the function contract.

Commit: `Test Davidson iteration and edge behavior`

### 5. Verify compiled and Hartree-Fock use

Run the focused suite with JAX compilation enabled and confirm stable results
across repeated calls. Retain the existing Hartree-Fock Davidson integration
coverage and add only a targeted regression there if a module-level failure
cannot be represented by the focused tests. Make the smallest solver changes
needed for all correctness tests to pass.

Commit: `Verify Davidson tests and integration`

### 6. Document and run the suites

Document the focused pytest command and numerical coverage. Run the Davidson
suite, the Hartree-Fock tests, and the full non-large-lattice suite. Confirm
that tolerances are justified by eigenpair residuals and the configured JAX
precision.

Commit: `Document Davidson unit-test coverage`
