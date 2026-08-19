import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.operators.one_body_operators as obops
import NuLattice.operators.two_body_operators as tbops
import NuLattice.lattice as lat
import NuLattice.constants_NLEFT as nleftConsts
import NuLattice.HF.hartree_fock as hf
import NuLattice.references as ref
import NuLattice.operators.jax_adapters as jax_help
if __name__ == '__main__':
    thisL = 3
    a = 1.0 / 100.0
    my_basis = lat.get_sp_basis(thisL)
    lattice = lat.get_lattice(thisL)

    myTkin=obops.tKin(thisL, 3, a,mass=nleftConsts.mass)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    verbose = True
    v_OPE = tbops.onePionEx(thisL, bpi, a, lattice, verbose=verbose, g_A=nleftConsts.g_A, f_pi = nleftConsts.f_pi, m_pi_0=nleftConsts.m_pi_0)
    sNL = 0.08
    sL = 0.08
    c0 = -0.185 / a

    v_0=tbops.shortRangeV_2body(lattice, thisL, sL, sNL, c0, verbose=verbose)

    my_VNN = tbops.sparse_to_list_2body(v_0+v_OPE, thisL)
    print("number of two-body matrix elements", len(my_VNN))

    # We compute oxygen-16
    my_ref = ref.ref_16O_gs
    hole = ref.reference_to_holes(my_ref,my_basis)
    hnum = len(hole)

    # Density must be defined as complex because Hamiltonian is complex Hermitian
    dens = hf.init_density(len(my_basis),hole,dtype=complex)

    eps=1.e-3
    mix = 0.7
    max_iter=500
    verbose = True
    myTkin = jax_help.one_body_from_list(myTkin, len(my_basis))
    my_VNN = jax_help.two_body_from_sparse(v_0 + v_OPE, len(my_basis))
    my_V3N = jax_help.empty_three_body(len(my_basis))
    erg, trafo, conv = hf.solve_HF(myTkin, my_VNN, my_V3N, dens,
                                mix=mix, eps=eps, max_iter=max_iter, verbose=verbose)

    if conv:
        print("HF energy (MeV) = ", erg)
    else:
        print("HF did not converge")
