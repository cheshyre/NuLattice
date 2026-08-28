"""
JAX-backed operator types and sharding helpers
"""
__authors__   =  "Vivek Booshan"
__credits__   =  ["Vivek Booshan"]
__copyright__ = "(c) Vivek Booshan"
__license__   = "BSD-3-Clause"
__date__      = "2026"

import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.sparse import BCOO
from jax.sharding import NamedSharding, PartitionSpec as P

class _OperatorBase:
    """
    common storage and conversions for sparse operators of a fixed rank

    Do not instantiate this class directly: the rank check and the rank used
    by to_bcoo and empty come from _get_expected_rank, which only the concrete
    subclasses implement.
    """

    def __init__(self, indices: jnp.ndarray, values: jnp.ndarray, nstat: int):
        """
        stores the index tuples and matrix elements as JAX arrays

        A one-dimensional array of indices is promoted to shape (num_ele, 1),
        so that indices always carries an explicit rank axis. Note that the
        indices are cast to integers, i.e., truncated rather than rounded, and
        that they are not checked against nstat.

        :param indices: index tuples of the stored matrix elements, cast to int32
        :type indices:  array-like of shape (num_ele, rank)
        :param values:  matrix elements belonging to indices
        :type values:   array-like of shape (num_ele,)
        :param nstat:   dimension of the single-particle basis
        :type nstat:    int
        """
        self.nstat = nstat
        # JAX arrays are the primary backend here
        self.indices = jnp.asarray(indices, dtype=jnp.int32)
        self.values = jnp.asarray(values)

        if self.indices.ndim == 1:
            self.indices = self.indices[:, jnp.newaxis]

    def __len__(self):
        """
        gives the number of stored matrix elements

        :return: number of stored matrix elements
        :rtype:  int
        """
        return len(self.values)

    def to_jax_indices_and_values(self):
        """
        gives the underlying index tuples and matrix elements

        This is the accessor to use in computational routines, so that they do
        not depend on the attribute names of this class. The arrays are
        returned as they are stored, i.e., without a copy.

        :returns: * **indices** (*jax.Array((num_ele,rank), dtype=int32)*) -- index tuples
                    of the stored matrix elements
                  * **values** (*jax.Array((num_ele,), dtype=float or complex)*) -- matrix
                    elements belonging to indices
        :rtype:   tuple[jax.Array, jax.Array]
        """
        return self.indices, self.values

    def to_list(self):
        """
        converts the operator into the legacy list-of-lists representation

        Each entry is one index tuple followed by the associated matrix
        element, i.e., the format accepted by from_list. Indices and matrix
        elements are converted to Python scalars.

        :return: list of matrix elements [[p, q, ..., val], ...], empty if
                 nothing is stored
        :rtype:  list[list]
        """
        if len(self) == 0:
            return []

        out_list = []
        for i in range(len(self.values)):
            row = self.indices[i].tolist()
            row.append(self.values[i].item())
            out_list.append(row)
        return out_list

    def to_bcoo(self, sm: "ShardingManager | None" = None):
        """Converts operator to a JAX BCOO sparse array, optionally sharded.

        The result has shape (nstat,) * rank, with the rank taken from
        _get_expected_rank. If a sharding manager is given, both arrays are
        placed on its mesh with ShardingManager.prepare, i.e., split along the
        axis holding the stored matrix elements.

        :param sm: sharding manager used to distribute the stored elements, or
                   None to leave the arrays where they are
        :type sm:  ShardingManager or None
        :return:   sparse representation of the operator
        :rtype:    jax.experimental.sparse.BCOO
        """
        rank = self._get_expected_rank()
        shape = (self.nstat,) * rank

        # If a sharding manager is provided, shard the NNZ dimension
        if sm is not None:
            indices = sm.prepare(self.indices)
            data = sm.prepare(self.values)
            return BCOO((data, indices), shape=shape)

        return BCOO((self.values, self.indices), shape=shape)

    def to_dense(self, sm: "ShardingManager | None" = None):
        """
        builds the dense array of the operator by scattering the stored elements

        Elements sharing an index tuple are summed, while elements whose
        indices fall outside the basis are dropped by the scatter. The rank is
        taken from the shape of the stored indices rather than from
        _get_expected_rank, so this also works on the base class.

        :param sm: sharding manager used to distribute the leading axis of the
                   result, or None to leave the array where it is
        :type sm:  ShardingManager or None
        :return:   dense array of shape (nstat,) * rank
        :rtype:    jax.Array
        """
        rank = self.indices.shape[1]
        shape = (self.nstat,) * rank
        mat = jnp.zeros(shape, dtype=self.values.dtype)
        mat = mat.at[tuple(self.indices[:, i] for i in range(rank))].add(self.values)
        if sm is not None:
            return sm.prepare(mat)
        return mat

    @classmethod
    def from_list(
        cls,
        operator_list,
        nstat: int,
    ):
        """Operator from a legacy list of lists [[p, q, ..., val], ...]

        The list is read into a single float array, so the indices are rounded
        before they are cast to integers. An empty list gives an operator with
        no stored matrix elements.

        :param operator_list: list of matrix elements [[p, q, ..., val], ...]
        :type operator_list:  list[list]
        :param nstat:         dimension of the single-particle basis
        :type nstat:          int
        :return:              operator of the class this is called on
        :rtype:               _OperatorBase subclass
        """
        if not operator_list:
            rank = cls._get_expected_rank()
            return cls(
                jnp.empty((0, rank), dtype=jnp.int32),
                jnp.empty((0,), dtype=jnp.float64),
                nstat,
            )

        data = jnp.array(operator_list, dtype=jnp.float64)
        indices = jnp.round(data[:, :-1]).astype(jnp.int32)
        values = data[:, -1]

        return cls(indices, values, nstat)

    @classmethod
    def _get_expected_rank(cls) -> int:
        """
        gives the number of single-particle indices per stored matrix element

        :return:                     rank of the operator, i.e., 2, 4, or 6
        :rtype:                      int
        :raises NotImplementedError: always, since only the subclasses define a rank
        """
        raise NotImplementedError("Do not use the base class, Operator. Instead use OneBodyOperator, TwoBodyOperator, ThreeBodyOperator or similar.")

    @classmethod
    def empty(cls, nstat: int, dtype=np.float64):
        """
        creates an operator without any stored matrix elements

        Such an operator contracts to zero, which is the way to switch off a
        term of the Hamiltonian without special-casing it downstream.

        :param nstat: dimension of the single-particle basis
        :type nstat:  int
        :param dtype: data type of the (empty) array of matrix elements
        :type dtype:  numpy.dtype, i.e., np.float64 or np.complex128
        :return:      operator of the class this is called on, of length zero
        :rtype:       _OperatorBase subclass
        """
        return cls(
            np.empty((0, cls._get_expected_rank()), dtype=np.int32),
            np.empty(0, dtype=dtype),
            nstat,
        )


class OneBodyOperator(_OperatorBase):
    """
    sparse one-body operator, storing index pairs [p,q] and matrix elements <p|o|q>
    """

    def __init__(self, indices, values, nstat):
        """
        stores the index pairs and matrix elements, checking the rank

        :param indices:     index pairs [p,q] of the stored matrix elements
        :type indices:      array-like of shape (num_ele, 2)
        :param values:      matrix elements belonging to indices
        :type values:       array-like of shape (num_ele,)
        :param nstat:       dimension of the single-particle basis
        :type nstat:        int
        :raises ValueError: if a non-empty index array does not have two columns
        """
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        """
        gives the number of single-particle indices per stored matrix element

        :return: 2
        :rtype:  int
        """
        return 2

class TwoBodyOperator(_OperatorBase):
    """
    sparse two-body operator, storing index tuples [p,q,r,s] and matrix elements <pq|v|rs>
    """

    def __init__(self, indices, values, nstat):
        """
        stores the index tuples and matrix elements, checking the rank

        :param indices:     index tuples [p,q,r,s] of the stored matrix elements
        :type indices:      array-like of shape (num_ele, 4)
        :param values:      matrix elements belonging to indices
        :type values:       array-like of shape (num_ele,)
        :param nstat:       dimension of the single-particle basis
        :type nstat:        int
        :raises ValueError: if a non-empty index array does not have four columns
        """
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        """
        gives the number of single-particle indices per stored matrix element

        :return: 4
        :rtype:  int
        """
        return 4

    @classmethod
    def from_scipy_coo(
        cls,
        matrix,
        nstat: int,
    ):
        """
        builds a two-body operator from a sparse matrix in the composite two-body basis

        The row index is read as p + q*nstat and the column index as
        r + s*nstat, so that the stored element becomes <pq|v|rs>. The matrix
        must already be in coordinate format, i.e., carry the attributes row,
        col, and data; see from_scipy_csr for other formats.

        :param matrix: sparse matrix of dimension nstat**2 x nstat**2
        :type matrix:  scipy.sparse.coo_matrix or coo_array
        :param nstat:  dimension of the single-particle basis
        :type nstat:   int
        :return:       two-body operator holding the index tuples [p,q,r,s]
        :rtype:        TwoBodyOperator
        """
        p = matrix.row % nstat
        q = matrix.row // nstat
        r = matrix.col % nstat
        s = matrix.col // nstat

        indices = np.column_stack((p, q, r, s)).astype(np.int32, copy=False)

        return cls(indices, matrix.data, nstat)

    @classmethod
    def from_scipy_csr(
        cls,
        matrix,
        nstat: int,
    ):
        """
        builds a two-body operator from a sparse matrix in any scipy format

        The matrix is converted to coordinate format and handed to
        from_scipy_coo, so the index convention described there applies.

        :param matrix: sparse matrix of dimension nstat**2 x nstat**2
        :type matrix:  scipy.sparse matrix or array
        :param nstat:  dimension of the single-particle basis
        :type nstat:   int
        :return:       two-body operator holding the index tuples [p,q,r,s]
        :rtype:        TwoBodyOperator
        """
        matrix = matrix.tocoo()
        return cls.from_scipy_coo(matrix, nstat)

class ThreeBodyOperator(_OperatorBase):
    """
    sparse three-body operator, storing index tuples [p,q,r,s,t,u] and matrix
    elements <pqr|w|stu>
    """

    def __init__(self, indices, values, nstat):
        """
        stores the index tuples and matrix elements, checking the rank

        :param indices:     index tuples [p,q,r,s,t,u] of the stored matrix elements
        :type indices:      array-like of shape (num_ele, 6)
        :param values:      matrix elements belonging to indices
        :type values:       array-like of shape (num_ele,)
        :param nstat:       dimension of the single-particle basis
        :type nstat:        int
        :raises ValueError: if a non-empty index array does not have six columns
        """
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 6:
            raise ValueError(f"Expected (N, 6) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        """
        gives the number of single-particle indices per stored matrix element

        :return: 6
        :rtype:  int
        """
        return 6

    @classmethod
    def from_scipy_coo(cls, matrix, nstat):
        """
        builds a three-body operator from a sparse matrix in the composite three-body basis

        The row index is read as p + q*nstat + r*nstat**2 and the column index
        as s + t*nstat + u*nstat**2, so that the stored element becomes
        <pqr|w|stu>. The matrix must already be in coordinate format, i.e.,
        carry the attributes row, col, and data; see from_scipy_csr for other
        formats.

        :param matrix: sparse matrix of dimension nstat**3 x nstat**3
        :type matrix:  scipy.sparse.coo_matrix or coo_array
        :param nstat:  dimension of the single-particle basis
        :type nstat:   int
        :return:       three-body operator holding the index tuples [p,q,r,s,t,u]
        :rtype:        ThreeBodyOperator
        """
        nstat2 = nstat * nstat

        p = matrix.row % nstat
        q = (matrix.row // nstat) % nstat
        r = matrix.row // nstat2
        s = matrix.col % nstat
        t = (matrix.col // nstat) % nstat
        u = matrix.col // nstat2

        indices = np.column_stack((p, q, r, s, t, u)).astype(np.int32, copy=False)

        return cls(indices, matrix.data, nstat)

    @classmethod
    def from_scipy_csr(cls, matrix, nstat):
        """
        builds a three-body operator from a sparse matrix in any scipy format

        The matrix is converted to coordinate format and handed to
        from_scipy_coo, so the index convention described there applies.

        :param matrix: sparse matrix of dimension nstat**3 x nstat**3
        :type matrix:  scipy.sparse matrix or array
        :param nstat:  dimension of the single-particle basis
        :type nstat:   int
        :return:       three-body operator holding the index tuples [p,q,r,s,t,u]
        :rtype:        ThreeBodyOperator
        """
        matrix = matrix.tocoo()
        return cls.from_scipy_coo(matrix, nstat)

class ShardingManager:
    """
    builds a device mesh and places arrays on it

    The mesh has the two axes "nodes" and "gpus". Arrays are distributed along
    their leading axis by default, with the partition spec chosen from the
    rank of the array, so that callers only have to say what they want placed.
    """

    def __init__(self, num_nodes=1, num_gpus=1):
        """
        creates the two-dimensional device mesh

        :param num_nodes: extent of the mesh axis "nodes"
        :type num_nodes:  int
        :param num_gpus:  extent of the mesh axis "gpus"
        :type num_gpus:   int
        """
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus
        self.mesh = jax.make_mesh(axis_shapes=(num_nodes, num_gpus), axis_names=("nodes", "gpus"))

    def prepare(self, arr, rank: int | None = None, spec: P | None = None):
        """
        places an array on the mesh, sharded along its leading axis

        The partition spec follows the rank: a scalar is replicated, a
        one-dimensional array is split across both mesh axes, and a
        higher-dimensional array is split along its leading axis only, using
        both mesh axes if the mesh is effectively one-dimensional. Large numpy
        arrays are sliced on the host and moved device by device, so that no
        full copy has to fit on a single device.

        :param arr:   array to be placed on the mesh
        :type arr:    numpy.ndarray or jax.Array
        :param rank:  rank used to pick the partition spec; taken from arr.ndim if None
        :type rank:   int or None
        :param spec:  partition spec overriding the rank-based choice
        :type spec:   jax.sharding.PartitionSpec or None
        :return:      the array, placed on the mesh with the chosen sharding
        :rtype:       jax.Array
        """
        r = rank if rank is not None else arr.ndim

        if spec is None:
            if r == 0:
                spec = P() # alternatively can be used for replication
            elif r == 1:
                spec = P(("nodes", "gpus")) # 1d array should be split across everything
            elif self.num_nodes == 1 or self.num_gpus == 1:
                spec = P(("nodes", "gpus"), *([None] * (r - 1)))
            else:
                spec = P("nodes", "gpus", *([None] * (r - 2)))

        sharding = NamedSharding(self.mesh, spec)

        # cpu check
        if isinstance(arr, np.ndarray):
            if arr.nbytes > 1e9: # only if > 1 gb
                # calculate bounding box per gpu, slice on cpu, move to gpu
                return jax.make_array_from_callback(
                    arr.shape,
                    sharding,
                    lambda idx: arr[idx])
            else:
                return jax.device_put(arr, sharding)
        return jax.device_put(arr, sharding)
