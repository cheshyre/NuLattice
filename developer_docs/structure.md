# Structure of the repository

NuLattice contains separated reference implementations
(found in `NuLattice/_reference`)
and optimized implementations
(found in the base `NuLattice` directory).
Every feature in NuLattice should
have a complete and correct implementation in `NuLattice/_reference`.
The actual implementation in `NuLattice` may be optimized
or accelerated in some way
or it may simply be the same as in `NuLattice/_reference`.
It is important that the two implementations (reference vs. optimized) 
be tested against each other
and yield numerically equal results.

When developers add to NuLattice, 
their first correct implementation
should land in `NuLattice/_reference`.
This implementation may then be copied to a corresponding module
in `NuLattice` and further optimized in necessary.
If bugs are identified
they should be fixed in all implementations.
