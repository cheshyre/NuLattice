"""
This module provides functions to define the 3D lattice   
"""
__authors__   =  ["Thomas Papenbrock", "Maxwell Rothman", "Ben Johnson"]
__credits__   =  ["Thomas Papenbrock", "Maxwell Rothman", "Ben Johnson"]
__copyright__ = "(c) Thomas Papenbrock and Maxwell Rothman and Ben Johnson"
__license__   = "BSD-3-Clause"
__date__      = "2025-07-26"

import copy
import numpy as np

from NuLattice.constants import hbarc, mass


def phys_unit(a_lat):
    """
    returns the energy unit from basic units

    :param a_lat:   lattice spacing in fm
    :type a_lat:    float
    :return:    factor to scale the lattice units to energy units
    :rtype:     float
    """
    return 0.5*hbarc**2/(mass*a_lat**2)


def get_sp_basis(myL, spin=2, isospin=2):
    """
    builds a 3D lattice for nucleons with spin isospin degrees of freedom

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin:    Optional; number of isospin degrees of freedom
    :type isospin:     int
    :return:    List of integer list [i,j,k,tz,sz] where lattice sites are 
                labelled by i, j, k (from 0 to myL-1) in direction 1, 2, 3; 
                tz=0, 1 and sz=0,1 correspond to isospin tz-1/2 and spin sz-1/2, 
                respectively
    :rtype: list[(int, int, int, int, int)]
    """
    sp_basis = []
    for i in range(myL):
        for j in range(myL):
            for k in range(myL):
                for iso in range(isospin):
                    for sz in range(spin):
                        sp_basis.append([i, j, k, iso, sz])
    return sp_basis


def state2index(state, myL, spin=2, isospin=2):
    """
    given a state list [i,j,k,tz,sz] this function returns the
    index of that state in the list returned by get_sp_basis
    
    :param state:   the list [i,j,k,tz,sz]
    :type state:    list[(int, int, int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:    index as an integer
    :rtype: int
    """
    i = state[0]
    j = state[1]
    k = state[2]
    tz = state[3]
    sz = state[4]
    index = i*myL**2*isospin*spin + j*myL*isospin*spin + k*isospin*spin + tz*spin + sz
    return index

def index2state(ind, myL, spin=2, isospin=2):
    """
    given an index of a state in the lattice, return a state list
    [i, j, k, tz, sz]

    :param ind:     index of the state
    :type ind:      int
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:        state as a list [i,j,k,tz,sz]
    :rtype:         list[(int, int, int, int, int)]
    """
    sz = ind % spin
    tz = ((ind - sz) // spin) % isospin
    site = (ind - sz - tz * spin) // (spin * isospin)
    k = site % myL
    j = ((site - k) // myL) % myL
    i = (((site - k) // myL) - j) // myL
    return [i, j, k, tz, sz]

def get_lattice(myL):
    """
    builds a 3D lattice
    
    :param myL: number of lattice sites in each direction
    :type myL:  int    
    :return:    List of integer lists [i,j,k] of lattice sites are labelled 
                by i, j, k (from 0 to myL-1) in direction 1, 2, 3
    :rtype:     list[(int, int, int)]
    """
    lattice = []
    for i in range(myL):
        for j in range(myL):
            for k in range(myL):
                lattice.append([i, j, k])
    return lattice


def site2index(site, myL):
    """
    given a site list [i,j,k] this function returns the index of that state in the list 
    returned by get_lattice
    
    :param state:   the list [i,j,k]
    :type state:    list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int    
    :return:        index as an integer
    :rtype:         int
    """
    i = site[0]
    j = site[1]
    k = site[2]
    index = i*myL**2 + j*myL + k
    return index


def right(site, myL):
    """
    moves a site to the right in 1D, respecting periodic boundary conditions
    
    :param site:    integer location of the site
    :type site:     int
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :return:        index of site one to the right of site with index site
    :rtype:         int
    """
    if site + 1 < myL:
        res = site + 1
    else:
        res = 0
    return res


def left(site, myL):
    """
    moves a site to the left in 1D, respecting periodic boundary conditions
    
    :param site:    integer location of the site
    :type site:     int
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :return:        index of site one to the left of site with index site
    :rtype:         int
    """
    if site-1 >= 0:
        res = site-1
    else:
        res = myL-1
    return res


def Tkin(lattice, myL, spin=2, isospin=2):
    """
    computes 1-body kinetic energy matrix elements. Really: the negative dimensionless laplacian 

    :param lattice: list of lattice sites returned by get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:    list of tuples [i, j, value] where i and j are indices in the single-particle 
                basis, and value is the value of the matrix element Tij
    :rtype:     list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #diagonal element from each direction
        val = 2.0 * 3
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx1, val])
        #
        #hop to the right in x
        r = right(i, myL=myL) #r,j,k
        val = -1.0
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [r, j, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, val]) #adds a hop-to-the left matrix element
        #
        #hop to the right in y
        r = right(j, myL=myL) #i,r,k
        val = -1.0
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, r, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, val]) #adds a hop-to-the left matrix element
        #
        #hop to the right in z
        r = right(k, myL=myL) #i,j,r
        val = -1.0
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, j, r, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, val]) #adds a hop-to-the left matrix element
    #
    return mat


def contacts(vT1, vS1, lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 2-body onsite contacts
    
    :param vT1:     strength of T=1 coupling
    :type vT1:      float
    :param vS1:     strength of S=1 coupling
    :type vS1:      float
    :param lattice: list of lattice sites returned by get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:    list of lists [i, j, k, l, value] where i, j and k, l are indices of two particles
                in the single-particle basis, and value is the value of the matrix element <ij||kl>. 
                All elements have i<j and k<l
    :rtype:         list[(int, int, int, int, float)]
    """
    valueT1 = vT1  #isospin triplet strength
    valueS1 = vS1  #spin triplet strength
    matele = []
    for site in lattice:
        for tz1 in range(isospin):
            for sz1 in range(spin):
                stat1 = copy.deepcopy(site)
                stat1.append(tz1)
                stat1.append(sz1)
                indx1 = state2index(stat1, myL=myL, spin=spin, isospin=isospin)
                for tz2 in range(isospin):
                    for sz2 in range(spin):
                        if tz1 == tz2 and sz1 == sz2: #not asymetric under exchange
                            continue
                        stat2 = copy.deepcopy(site)
                        stat2.append(tz2)
                        stat2.append(sz2)
                        indx2 = state2index(stat2, myL=myL, spin=spin, isospin=isospin)
                        if indx2 <= indx1: #we only keep properly ordered two-body states
                            continue
                        for tz3 in range(isospin):
                            for sz3 in range(spin):
                                stat3 = copy.deepcopy(site)
                                stat3.append(tz3)
                                stat3.append(sz3)
                                indx3 = state2index(stat3, myL=myL, spin=spin, isospin=isospin)
                                for tz4 in range(isospin):
                                    if tz1 + tz2 != tz3 + tz4: #Tz is not conserved
                                        continue
                                    for sz4 in range(spin):
                                        if sz1+sz2 != sz3+sz4: #Sz is not conserved
                                            continue
                                        if tz3 == tz4 and sz3 == sz4: #not asymetric under exchange
                                            continue
                                        stat4 = copy.deepcopy(site)
                                        stat4.append(tz4)
                                        stat4.append(sz4)
                                        indx4 = state2index(stat4, myL=myL, spin=spin, isospin=isospin)

                                        if indx4 <= indx3: #we only keep properly ordered two-body states
                                            continue   
                                        if tz1 == tz2: #|Tz|=1, T=1, and S=Sz=0 from antisymmetry
                                            #all isospins are equal
                                            matele.append([indx1, indx2, indx3, indx4, valueT1])
                                        elif sz1 == sz2: #|Sz|=1, S=1, and T=Tz=0 from antisymmetry      
                                            #all spins are equal
                                            matele.append([indx1, indx2, indx3, indx4, valueS1])
                                        else: #Tz=0 and Sz=0 
                                            if indx1 in (indx3, indx4):
                                                matele.append([indx1, indx2, indx3, indx4,
                                                               (valueS1+valueT1)*0.5])
                                            else:
                                                matele.append([indx1, indx2, indx3, indx4,
                                                               (valueS1-valueT1)*0.5])
    #
    return matele


def NNNcontact(v3NF, lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for three-body onsite contact
    
    :param v3NF:        strength of the 3 nucleon force
    :type v3NF:         float
    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:    list of tuples [i1, i2, i3, j1, j2, j3, value] where i1, i2, i3
                and j1, j2, j3 are indices of three particles in the
                single-particle basis, and value is one (unit strength) for the
                matrix element <i1 i2 i3||j1 j2 j3>.
                All elements have i1<i2<i3 and j1<j2<j3
    :rtype:             list[(int, int, int, int, int, int, float)]
    """
    value = v3NF
    matele = []
    for site in lattice:
        for tz1 in range(isospin):
            for sz1 in range(spin):
                stat1 = copy.deepcopy(site)
                stat1.append(tz1)
                stat1.append(sz1)
                indx1 = state2index(stat1, myL=myL, spin=spin, isospin=isospin)
                for tz2 in range(isospin):
                    for sz2 in range(spin):
                        if tz1 == tz2 and sz1 == sz2: #not asymetric under exchange
                            continue
                        stat2 = copy.deepcopy(site)
                        stat2.append(tz2)
                        stat2.append(sz2)
                        indx2 = state2index(stat2, myL=myL, spin=spin, isospin=isospin)
                        if indx2 <= indx1:
                            continue
                        for tz3 in range(isospin):
                            for sz3 in range(spin):
                                if tz1 == tz3 and sz1 == sz3: #not asymetric under exchange
                                    continue
                                if tz3 == tz2 and sz3 == sz2: #not asymetric under exchange
                                    continue
                                stat3 = copy.deepcopy(site)
                                stat3.append(tz3)
                                stat3.append(sz3)
                                indx3 = state2index(stat3, myL=myL, spin=spin, isospin=isospin)
                                if indx3 <= indx2:
                                    continue
                                
                                matele.append([indx1, indx2, indx3, indx1, indx2, indx3, value])

    return matele


def p_x(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body momentum operator p_x. Really: -i times d_x
    
    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle 
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #hop to the right in x
        r = right(i, myL=myL) #r,j,k
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [r, j, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, -val]) #adds a hop-to-the left matrix element
    #
    return mat


def p_y(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body momentum operator p_y. Really: -i times d_y
    
    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int    
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle 
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
            #
        #hop to the right in y
        r = right(j, myL=myL) #i,r,k
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, r, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, -val]) #adds a hop-to-the left matrix element
    #
    return mat


def p_z(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body momentum operator p_z. Really: -i times d_z
    
    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int    
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle 
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #
        #hop to the right in z
        r = right(k, myL=myL) #i,j,r
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, j, r, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, -val]) #adds a hop-to-the left matrix element
    #
    return mat

def states2PHSpace(holeList, myL):
    """
    Takes a list of hole states and returns the hole and particle spaces
    
    :param holeList:    list of holes and their sites as [i, j, k, tz, sz]
    :type holeList:     list[(int, int, int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :return:            hole and particle space
    :rtype:             tuple(int), tuple(int)
    """
    holes = []
    for h in holeList:
        holes.append(state2index(h, myL))
    
    parts = tuple(np.delete(np.arange(myL ** 3 * 4), holes))
    holes = tuple(holes)
    return holes, parts

def makeState(x, y, z, tz, sz):
    """
    Takes position in x, y, and z on the lattice as well as the spin and isospin and returns a state

    :param x:   x position in lattice
    :type x:    int
    :param y:   y position in lattice
    :type y:    int
    :param z:   z position in lattice
    :type z:    int
    :param tz:  isospin
    :type tz:   0.5 | -0.5
    :param sz:  spin
    :type sz:   0.5 | -0.5
    :return:    a particle state on the lattice as a list
    :rtype:     list[(int, int, int, int, int)]
    """
    return [x, y, z, int(tz + 0.5), int(sz + 0.5)]

