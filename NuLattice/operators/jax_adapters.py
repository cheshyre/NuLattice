"""
adapters that build the JAX operator types from legacy list and scipy.sparse input
"""
__authors__   =  "Vivek Booshan"
__credits__   =  ["Vivek Booshan"]
__copyright__ = "(c) Vivek Booshan"
__license__   = "BSD-3-Clause"
__date__      = "2026"

import numpy as np

from NuLattice.utils._jax_types import (
    OneBodyOperator,
    ThreeBodyOperator,
    TwoBodyOperator,
)


def one_body_from_list(elements, nstat):
    """
    builds a one-body operator from a legacy list of matrix elements

    An empty list yields an operator with no stored matrix elements rather
    than an error, so that a vanishing one-body term can be passed on.

    :param elements: list of one-body matrix elements [[p1,q1,value1], [p2,q2,value2], ...]
    :type elements:  list[list[int,int, float]]
    :param nstat:    dimension of the single-particle basis
    :type nstat:     int
    :return:         one-body operator holding the index pairs [p,q] and their matrix elements
    :rtype:          OneBodyOperator
    """
    if len(elements) == 0:
        return OneBodyOperator(
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.float64),
            nstat,
        )

    indices = np.asarray([row[:2] for row in elements], dtype=np.int32)
    values = np.asarray([row[2] for row in elements])
    return OneBodyOperator(indices, values, nstat)


def two_body_from_sparse(matrix, nstat):
    """
    builds a two-body operator from a sparse matrix in the composite two-body basis

    The row index of the matrix is read as p + q*nstat and the column index as
    r + s*nstat, so that the stored element becomes the matrix element
    <pq|v|rs> of the resulting operator. The matrix is converted to coordinate
    format first, so any scipy.sparse format is accepted.

    :param matrix: sparse matrix of dimension nstat**2 x nstat**2
    :type matrix:  scipy.sparse matrix or array
    :param nstat:  dimension of the single-particle basis
    :type nstat:   int
    :return:       two-body operator holding the index tuples [p,q,r,s] and their matrix elements
    :rtype:        TwoBodyOperator
    """
    matrix = matrix.tocoo()

    p = matrix.row % nstat
    q = matrix.row // nstat
    r = matrix.col % nstat
    s = matrix.col // nstat
    indices = np.column_stack((p, q, r, s)).astype(np.int32, copy=False)

    return TwoBodyOperator(indices, matrix.data, nstat)


def three_body_from_sparse(matrix, nstat):
    """
    builds a three-body operator from a sparse matrix in the composite three-body basis

    The row index of the matrix is read as p + q*nstat + r*nstat**2 and the
    column index as s + t*nstat + u*nstat**2, so that the stored element
    becomes the matrix element <pqr|w|stu> of the resulting operator. The
    matrix is converted to coordinate format first, so any scipy.sparse format
    is accepted.

    :param matrix: sparse matrix of dimension nstat**3 x nstat**3
    :type matrix:  scipy.sparse matrix or array
    :param nstat:  dimension of the single-particle basis
    :type nstat:   int
    :return:       three-body operator holding the index tuples [p,q,r,s,t,u] and
                   their matrix elements
    :rtype:        ThreeBodyOperator
    """
    matrix = matrix.tocoo()
    nstat2 = nstat * nstat

    p = matrix.row % nstat
    q = (matrix.row // nstat) % nstat
    r = matrix.row // nstat2
    s = matrix.col % nstat
    t = (matrix.col // nstat) % nstat
    u = matrix.col // nstat2
    indices = np.column_stack((p, q, r, s, t, u)).astype(np.int32, copy=False)

    return ThreeBodyOperator(indices, matrix.data, nstat)


def empty_three_body(nstat, dtype=np.float64):
    """
    creates a three-body operator without any stored matrix elements

    This is the convenient way to run a computation without a three-body
    force, since the resulting operator contracts to zero.

    :param nstat: dimension of the single-particle basis
    :type nstat:  int
    :param dtype: data type of the (empty) array of matrix elements
    :type dtype:  numpy.dtype, i.e., np.float64 or np.complex128
    :return:      three-body operator with no stored matrix elements
    :rtype:       ThreeBodyOperator
    """
    return ThreeBodyOperator.empty(nstat, dtype)
