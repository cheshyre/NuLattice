@../../AGENTS.md
@planning.md
@../development-style.md
@../unit-tests.md

# Hartree-Fock public-method unit-test plan

## Goal

Add pytest coverage for the public functions in
`NuLattice/HF/hartree_fock.py`:

- `init_density`
- `evaluate_slater_determinant_expectation_value`
- `HF_energy`
- `make_HF_ham`
- `solve_HF`

Compare results with `NuLattice/_reference/HF/hartree_fock.py` where an
analogue exists, using `to_list()` to supply legacy operator inputs. Cover
He4 and O16 with initial, nontrivial nonconverged, and converged densities.

Include fast tests on small lattices and larger regression cases at `L=3` and
`L=4`. The Hamiltonians should range from the kinetic/contact model in
`Example_Hartree_Fock.py` to the more complicated interactions in
`Example_interaction_2017.py` and `Example_interaction_2016B.py`.

## Dependencies

None.

## Implementation steps

### 1. Add shared test inputs and density coverage

**COMPLETED**

Establish reusable Hamiltonian and density inputs for He4 and O16. Include
simple `L=2` inputs and the larger example-based interactions. Test
`init_density` against the reference implementation and validate the physical
properties of the generated densities.

Commit: `Add Hartree-Fock test inputs and density tests`

### 2. Test expectation values, energies, and Fock matrices

Compare the JAX and reference results for one-, two-, and three-body terms,
their combinations, and absent operators. Exercise real and complex densities,
including the public real/complex behavior of the expectation and energy
functions.

Commit: `Test Hartree-Fock energies and Fock matrices`

### 3. Test solver behavior on small lattices

Compare dense `solve_HF` results with the reference solver for converged and
nonconverged He4 and O16 cases. Cover the Davidson option, retained-orbital
options, single-device sharding, verbose output, and invalid diagonalizer
handling. Compare orbital results in a way that accounts for phase and
degenerate-subspace freedom.

Commit: `Test Hartree-Fock solver behavior on small lattices`

### 4. Add large-lattice interaction regressions

Test the 2017-style interaction at `L=3` and the 2016B-style interaction at
`L=4`. Compare energies and Fock matrices with the reference implementation
for He4 and O16, and include cost-appropriate solver comparisons and
self-consistency checks.

Mark the expensive cases so the fast and large-lattice suites can be run
separately.

Commit: `Add large-lattice Hartree-Fock regression tests`

### 5. Verify and document the test suites

Run the fast tests, large-lattice tests, and complete pytest suite. Confirm
that expensive inputs are reused, tests are order-independent, and numerical
tolerances are appropriate for JAX/reference comparisons. Document how to run
the two test tiers and note their expected resource requirements.

Commit: `Document and verify Hartree-Fock test coverage`
