"""
functions to perform a Hartree-Fock computation on the lattice
"""
__authors__   =  "Thomas Papenbrock"
__credits__   =  ["Thomas Papenbrock, Jifeng Fan"]
__copyright__ = "(c) Thomas Papenbrock"
__license__   = "BSD-3-Clause"
__date__      = "2025-07-26"

import numpy as np
from opt_einsum import contract
#from numba import njit

def _result_dtype(dens, *ops):
    """
    returns the dtype needed to hold the density matrix and all matrix elements,
    i.e., complex as soon as the density or any operator is complex

    :param dens: density matrix
    :type dens:  numpy.array((:,:))
    :param ops:  operators as lists of matrix elements [..., value] or as scipy.sparse arrays
    :type ops:   list[list[int, ..., float or complex]] or scipy.sparse.csr_array
    :return:     data type for one-body operators built from dens and ops
    :rtype:      numpy.dtype
    """
    dtypes = [dens.dtype]
    for op in ops:
        if hasattr(op, "dtype"):  # scipy.sparse array
            dtypes.append(op.dtype)
        elif len(op) > 0:
            dtypes.append(np.asarray([mat_ele[-1] for mat_ele in op]).dtype)
    return np.result_type(*dtypes)

def get_1body_matrix(myTkin,nstat,dtype=float):
    """
    takes the list of one-body matrix elements and turns it into a square matrix
    
    :param nstat:  dimension of matrix, i.e. the number of 1-body states
    :type nstat:   int
    :param dtype:  data type of returned object
    :type dtype:   numpy.dtype, i.e. np.float64 or float or np.complex128 or complex
    :param myTkin: list of one-body matrix elements [[p1,q1,value1], [p2,q2,value2], ...]
    :type myTkin:  list[list[int,int, float]]
    :return:       nstat x nstat matrix of the list of matrix elements
    :rtype:        numpy.array((:,:), dtype=float)
    """
    op1 = np.zeros((nstat,nstat),dtype=dtype)
    for [a, b, val] in myTkin:
        op1[a,b]+=val
    return op1

def contract_2nf(v2,dens):
    """
    takes list of two-body matrix elements and contracts them with the density to get a one-body operator

    :param v2:   list of two-body matrix elements [p,q,r,s,value] 
    :type v2:    list[list[int,int,int,int, float]]
    :param dens: square density matrix
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     one-body operator of the same shape as the density matrix dens
    :rtype:      numpy.array((:,:), dtype=float)
    """
    res = np.zeros(dens.shape, dtype=_result_dtype(dens, v2))
    num_ele = len(v2)
    for i in range(num_ele):
        [a, b, c, d, val] = v2[i]
        # res^a_c += V^{ab}_{cd} rho^d_b, with dens[p,q] = rho^p_q = <a^dag_q a_p>
        res[a,c] += val*dens[d,b] #1
        res[b,c] -= val*dens[d,a] #P(ab)
        res[a,d] -= val*dens[c,b] #P(cd)
        res[b,d] += val*dens[c,a] #P(ab)P(cd)
    return res

def contract_3nf(w3,dens):
    """
    takes list of three-body matrix elements and contracts them with the density to get a one-body operator

    :param w3:   list of two-body matrix elements [p,q,r,s,value] 
    :type w3:    list[list[int,int,int,int,int,int, float]]
    :param dens: square density matrix
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     one-body operator of the same shape as the density matrix dens
    :rtype:      numpy.array((:,:), dtype=float)
    """
    res = np.zeros(dens.shape, dtype=_result_dtype(dens, w3))
    for mat_ele in w3:  # we need all 36 antisymmetric combinations of the ket (abc) and bra (def) single-particle states
        [a, b, c, d, e, f, val] = mat_ele
        # res^p_q += W^{prs}_{qtu} rho^t_r rho^u_s, with dens[p,q] = rho^p_q = <a^dag_q a_p>
        res[a,d] += val*( dens[e,b]*dens[f,c]  # (abc), (def), antisym last two pairs
                         -dens[e,c]*dens[f,b]
                         -dens[f,b]*dens[e,c]
                         +dens[f,c]*dens[e,b] )
        res[b,d] += val*( dens[e,c]*dens[f,a]  # (bca), (def), antisym last two pairs
                         -dens[e,a]*dens[f,c]
                         -dens[f,c]*dens[e,a]
                         +dens[f,a]*dens[e,c] )        
        res[c,d] += val*( dens[e,a]*dens[f,b]  # (cab), (def), antisym last two pairs
                         -dens[e,b]*dens[f,a]
                         -dens[f,a]*dens[e,b]
                         +dens[f,b]*dens[e,a] )
        res[a,e] += val*( dens[f,b]*dens[d,c]  # (abc), (efd), antisym last two pairs
                         -dens[f,c]*dens[d,b]
                         -dens[d,b]*dens[f,c]
                         +dens[d,c]*dens[f,b] )
        res[b,e] += val*( dens[f,c]*dens[d,a]  # (bca), (efd), antisym last two pairs
                         -dens[f,a]*dens[d,c]
                         -dens[d,c]*dens[f,a]
                         +dens[d,a]*dens[f,c] )        
        res[c,e] += val*( dens[f,a]*dens[d,b]  # (cab), (efd), antisym last two pairs
                         -dens[f,b]*dens[d,a]
                         -dens[d,a]*dens[f,b]
                         +dens[d,b]*dens[f,a] )
        res[a,f] += val*( dens[d,b]*dens[e,c]  # (abc), (fde), antisym last two pairs
                         -dens[d,c]*dens[e,b]
                         -dens[e,b]*dens[d,c]
                         +dens[e,c]*dens[d,b] )
        res[b,f] += val*( dens[d,c]*dens[e,a]  # (bca), (fde), antisym last two pairs
                         -dens[d,a]*dens[e,c]
                         -dens[e,c]*dens[d,a]
                         +dens[e,a]*dens[d,c] )        
        res[c,f] += val*( dens[d,a]*dens[e,b]  # (cab), (fde), antisym last two pairs
                         -dens[d,b]*dens[e,a]
                         -dens[e,a]*dens[d,b]
                         +dens[e,b]*dens[d,a] )
    return res

def make_HF_ham(op1,op2,op3,dens):
    """
    takes Hamiltonian consisting of one-body operator op1, two-body operator op2,
    and three-body operator op3, and the density matrix and returns the Hartree-Fock Hamiltonian.

    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:    list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     matrix in the shape of op1 and dens that is the Hartree-Fock Hamiltonian
    :rtype:      numpy.array((:,:), dtype=float)
    """
    nstat = len(dens)
    denstype=dens.dtype
    hf_op = get_1body_matrix(op1,nstat).astype(denstype)
    hf_op += contract_2nf(op2,dens)
    hf_op += 0.5*contract_3nf(op3,dens)
    return hf_op

def init_density(nstat,hole,dtype=float):
    """
    creates a density matrix of dimension nstat x nstat given the hole information

    :param nstat: dimension of single-particle basis
    :type nstat:  int
    :param dtype: data type of returned object
    :type dtype:  numpy.dtype, i.e. np.float64 or float or np.complex128 or complex
    :param hole:  tuple of occupied single-particle states, as numbers from 0 ... A-1
    :type hole:   tuple(int, int, ... )
    :return:      density matrix where hole states are occupied (1) and all others not (0)
    :rtype:       numpy.array((nstat,nstat), dtype = float)
    """
    dens = np.zeros((nstat,nstat),dtype=dtype)
    for i in hole:
        dens[i,i] = 1.0
    return dens

def HF_iter(op1, op2, op3, dens, mix=0.5):
    """
    Performs one iteration of the Hartree-Fock procedure

    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:   list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :param mix:  returned density matrix is mix*new_density + (1-mix)*old_density
    :type mix:   float 
    :return:     energy, density, vecs as the current HF energy, current density
                 matrix, and orthogonal transformation matrix that diagonalizes
                 the HF Hamiltonian
    :rtype:      float, numpy.array((:,:), dtype=float), numpy.array((:,:), dtype=float)
    """
    npart=round(np.real(np.trace(dens))) # rounds to nearest integer
    erg = HF_energy(op1, op2, op3, dens)
    hf = make_HF_ham(op1, op2, op3, dens)
    vals, vecs = np.linalg.eigh(hf)
    new_dens=contract("pi,qi->pq", vecs[:,0:npart], np.conjugate(vecs[:,0:npart]) )
    res_dens = mix*new_dens + (1.0-mix)*dens
    return erg, res_dens, vecs

def solve_HF(op1, op2, op3, dens, mix=0.5, eps=1.e-8, max_iter=100, verbose=False):
    """
    Solve the Hartree-Fock problem

    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:   list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :param mix:  parameter used in the mixing: mix*new_density + (1-mix)*old_density
    :type mix:   float
    :param eps:  converegence of energy
    :type eps:   float
    :param max_iter:  maximum number of HF iterations
    :type max_iter:   float
    :return:     energy, orthogonal transformation matrix that diagonalizes
                 the HF Hamiltonian (the first A columns are occupied), converged
    :rtype:      float, numpy.array((:,:), dtype=float), boolean
    """
    converged = False
    my_dens=dens.copy()
    erg0 = HF_energy(op1, op2, op3, my_dens)
    for i in range(max_iter):
        erg, new_dens, vecs = HF_iter(op1, op2, op3, my_dens, mix=mix)
        diff = np.abs(erg-erg0)
        diff_dens = np.sum(np.abs(new_dens-my_dens))
        if verbose:
            print(i, "E=", erg, ", Delta E=", diff, ", Delta rho =", diff_dens)
        if diff_dens < eps and i > 1:
            converged = True
            break
        else:
            erg0 = erg
            my_dens = new_dens.copy()
    return erg, vecs, converged

def HF_energy(op1, op2, op3, dens, w3_sparse=False):
    """
    Computes the Hartree-Fock energy for a given density dens and Hamiltonian consisting
    of one-body term op1, two-body term op2, and three-body term op3
    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:   list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :param w3_sparse: 
    :return:     Hartree-Fock energy
    :rtype:      float
    """
    nstat = len(dens)
    data_type=_result_dtype(dens, op1, op2, op3)
    dum = get_1body_matrix(op1,nstat,dtype=data_type)
    dum += 0.5*contract_2nf(op2,dens)
    if w3_sparse:
        dum += (1.0/6.0)*contract_3nf_sparse(op3,dens)
    else:
        dum += (1.0/6.0)*contract_3nf(op3,dens)
        
    erg = contract("ij,ji",dum,dens)
    return np.real_if_close(erg)

def contract_3nf_sparse(csr,dens):
    """
    takes list of three-body matrix elements and contracts them with the density to get a one-body operator
    :param csr:  matrix of three-body elements
    :type csr:   scipy.sparse.csr_array
    :param dens: square density matrix
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     one-body operator of the same shape as the density matrix dens
    :rtype:      numpy.array((:,:), dtype=float)
    """
    dim = len(dens)
    res = np.zeros(dens.shape, dtype=_result_dtype(dens, csr))
    rows, cols = csr.shape
    for i in range(rows):
    # Get start and end indices for current row i
        start = csr.indptr[i]
        end = csr.indptr[i+1]

        row_values = csr.data[start:end]
        row_indices = csr.indices[start:end]

        for j, val in zip(row_indices, row_values):
            # process row i, col j
            # index = a + b * dim + c*dim**2
            a = i % dim
            b = ((i - a) % dim**2) // dim
            c = (i - a - b*dim) // dim**2
            d = j % dim
            e = ((j - d) % dim**2) // dim
            f = (j - d - e*dim) // dim**2
            # res^p_q += W^{prs}_{qtu} rho^t_r rho^u_s, with dens[p,q] = rho^p_q = <a^dag_q a_p>
            res[a,d] += val*( dens[e,b]*dens[f,c]  # (abc), (def), antisym last two pairs
                             -dens[e,c]*dens[f,b]
                             -dens[f,b]*dens[e,c]
                             +dens[f,c]*dens[e,b] )
            res[b,d] += val*( dens[e,c]*dens[f,a]  # (bca), (def), antisym last two pairs
                             -dens[e,a]*dens[f,c]
                             -dens[f,c]*dens[e,a]
                             +dens[f,a]*dens[e,c] )        
            res[c,d] += val*( dens[e,a]*dens[f,b]  # (cab), (def), antisym last two pairs
                             -dens[e,b]*dens[f,a]
                             -dens[f,a]*dens[e,b]
                             +dens[f,b]*dens[e,a] )
            res[a,e] += val*( dens[f,b]*dens[d,c]  # (abc), (efd), antisym last two pairs
                             -dens[f,c]*dens[d,b]
                             -dens[d,b]*dens[f,c]
                             +dens[d,c]*dens[f,b] )
            res[b,e] += val*( dens[f,c]*dens[d,a]  # (bca), (efd), antisym last two pairs
                             -dens[f,a]*dens[d,c]
                             -dens[d,c]*dens[f,a]
                             +dens[d,a]*dens[f,c] )        
            res[c,e] += val*( dens[f,a]*dens[d,b]  # (cab), (efd), antisym last two pairs
                             -dens[f,b]*dens[d,a]
                             -dens[d,a]*dens[f,b]
                             +dens[d,b]*dens[f,a] )
            res[a,f] += val*( dens[d,b]*dens[e,c]  # (abc), (fde), antisym last two pairs
                             -dens[d,c]*dens[e,b]
                             -dens[e,b]*dens[d,c]
                             +dens[e,c]*dens[d,b] )
            res[b,f] += val*( dens[d,c]*dens[e,a]  # (bca), (fde), antisym last two pairs
                             -dens[d,a]*dens[e,c]
                             -dens[e,c]*dens[d,a]
                             +dens[e,a]*dens[d,c] )        
            res[c,f] += val*( dens[d,a]*dens[e,b]  # (cab), (fde), antisym last two pairs
                             -dens[d,b]*dens[e,a]
                             -dens[e,a]*dens[d,b]
                             +dens[e,b]*dens[d,a] )
    return res
