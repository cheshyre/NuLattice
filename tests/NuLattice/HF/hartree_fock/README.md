# Hartree-Fock public-method tests

Run the fast suite during development:

```sh
python -m pytest tests/NuLattice/HF/hartree_fock -m "not large_lattice"
```

Run the `L=3` and `L=4` example-interaction regressions separately:

```sh
python -m pytest tests/NuLattice/HF/hartree_fock -m large_lattice
```

Run both tiers by omitting the marker expression. The large tier constructs
complex OPE and smeared-contact sparse matrices once per pytest session. On a
single CPU device it requires approximately one minute and substantially more
memory than the fast `L=2` suite; runtime and memory depend on the JAX backend.

