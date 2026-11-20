"""
Provides useful constants to be used in the rest of the code as defined in the NLEFT papers
"""

# hbar*c only converts lattice spacings between MeV^-1 and fm. The NLEFT papers with
# a = (100 MeV)^-1 quote a = 1.97 fm, i.e., hbar*c = 197 MeV fm. For a = (150 MeV)^-1,
# Lu et al., PLB 797, 134863 (2019) quote a = 1.32 fm, which needs the CODATA value
# NuLattice.constants.hbarc (see examples/Example_interaction_2018_single_site.py).
hbarc = 197  # MeV fm
mass  = 938.92
# Elhatisari et al., PRL 117, 132501 (2016) and PRL 119, 222505 (2017)
g_A = 1.29
m_pi_0 = 134.98 # MeV
m_pi_c = 139.57 #MeV
f_pi = 92.2 #MeV
