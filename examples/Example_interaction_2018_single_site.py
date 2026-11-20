import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.operators.one_body_operators as obops
import NuLattice.operators.two_body_operators as twbops
import NuLattice.operators.three_body_operators as thbops
import NuLattice.lattice as lat
import NuLattice.constants_NLEFT as nleftConsts
import NuLattice.constants as consts
if __name__ == '__main__':
    thisL = 4
    a = 1.0 / 150.0
    # Lu et al., PLB 797, 134863 (2019) quote a = (150 MeV)^-1 = 1.32 fm, which needs the CODATA
    # hbar*c (197.3269804 MeV fm -> a = 1.3155 fm), not nleftConsts.hbarc = 197 MeV fm (1.3133 fm).
    # hbar*c only enters conversions to fm, e.g., densities; the energies do not depend on it.
    hbarc = consts.hbarc
    print(f"lattice spacing a = {a * hbarc:.4f} fm")
    lattice = lat.get_lattice(thisL)
    myTkin=obops.tKin(thisL, 3, a,mass=nleftConsts.mass)
    verbose = True
    sL = 0.061
    sNL = 0.5
    c3 = -1.4e-14 / (a ** 6)
    c2 = -3.41e-7 / (a ** 3)
    bpi = 0.7
    v_OPE = twbops.onePionEx(thisL, bpi, a, lattice, verbose=verbose, g_A=nleftConsts.g_A, f_pi = nleftConsts.f_pi, m_pi_0=nleftConsts.m_pi_0)
    # We get the interaction on a single site
    # We choose [2,2,2] without loss of generality
    site = [[2, 2, 2]]
    site_2body = twbops.shortRangeV_2body(lattice, thisL, sL, sNL, c2,verbose=verbose, sites=site)
    site_2body = twbops.sparse_to_list_2body(site_2body, thisL)
    site_3body = thbops.shortRangeV_3body(lattice, thisL, sL, sNL, c3, verbose=verbose, max_mem=1e9, sites=site)