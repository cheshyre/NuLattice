# Plan to migrate to NuLattice v2

1. Move the existing code to a NuLattice/_reference directory
2. Update the HF and CCM modules with your JAX updates
3. Add NuLattice/distributed with the sharding manager and the HF and CCM modules that support sharding
4. Add unit tests
5. Publish
