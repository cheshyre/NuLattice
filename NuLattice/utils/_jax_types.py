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
    def __init__(self, indices: jnp.ndarray, values: jnp.ndarray, nstat: int):
        self.nstat = nstat
        # JAX arrays are the primary backend here
        self.indices = jnp.asarray(indices, dtype=jnp.int32)
        self.values = jnp.asarray(values)

        if self.indices.ndim == 1:
            self.indices = self.indices[:, jnp.newaxis]

    def __len__(self):
        return len(self.values)

    def to_jax_indices_and_values(self):
        return self.indices, self.values

    def to_list(self):
        if len(self) == 0:
            return []

        out_list = []
        for i in range(len(self.values)):
            row = self.indices[i].tolist()
            row.append(self.values[i])
            out_list.append(row)
        return out_list

    def to_bcoo(self, mesh=None):
        """Converts operator to a JAX BCOO sparse array, optionally sharded."""
        rank = self._get_expected_rank()
        shape = (self.nstat,) * rank

        # If a mesh is provided, shard the NNZ dimension
        if mesh:
            sharding = NamedSharding(mesh, P("data"))
            indices = jax.device_put(self.indices, sharding)
            data = jax.device_put(self.values, sharding)
            return BCOO((data, indices), shape=shape)

        return BCOO((self.values, self.indices), shape=shape)

    def to_dense(self, mesh=None):
        rank = self.indices.shape[1]
        shape = (self.nstat,) * rank
        mat = jnp.zeros(shape, dtype=self.values.dtype)
        mat = mat.at[tuple(self.indices[:, i] for i in range(rank))].add(self.values)
        if mesh:
            sharding = NamedSharding(mesh, P("data", *((None,) * (rank - 1))))
            return jax.device_put(mat, sharding)
        return mat

    @classmethod
    def from_list(
        cls,
        operator_list,
        nstat: int,
    ):
        """Operator from a legacy list of lists [[p, q, ..., val], ...]"""
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
        raise NotImplementedError("Do not use the base class, Operator. Instead use OneBodyOperator, TwoBodyOperator, ThreeBodyOperator or similar.")

    @classmethod
    def empty(cls, nstat: int, dtype=np.float64):
        return cls(
            np.empty((0, cls._get_expected_rank()), dtype=np.int32),
            np.empty(0, dtype=dtype),
            nstat,
        )


class OneBodyOperator(_OperatorBase):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 2

class TwoBodyOperator(_OperatorBase):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 4

    @classmethod
    def from_scipy_coo(
        cls,
        matrix,
        nstat: int,
    ):
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
        matrix = matrix.tocoo()
        return cls.from_scipy_coo(matrix, nstat)

class ThreeBodyOperator(_OperatorBase):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 6:
            raise ValueError(f"Expected (N, 6) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 6

    @classmethod
    def from_scipy_coo(matrix, nstat):
        nstat2 = nstat * nstat

        p = matrix.row % nstat
        q = (matrix.row // nstat) % nstat
        r = matrix.row // nstat2
        s = matrix.col % nstat
        t = (matrix.col // nstat) % nstat
        u = matrix.col // nstat2

        indices = np.column_stack((p, q, r, s, t, u)).astype(np.int32, copy=False)

        return ThreeBodyOperator(indices, matrix.data, nstat)

    def from_scipy_csr(cls, matrix, nstat):
        matrix = matrix.tocoo()
        return cls.from_scipy_coo(matrix, nstat)

class ShardingManager:
    def __init__(self, num_nodes=1, num_gpus=1):
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus
        self.mesh = jax.make_mesh(axis_shapes=(num_nodes, num_gpus), axis_names=("nodes", "gpus"))

    def prepare(self, arr, rank: int | None = None, spec: NamedSharding = None):
        r = rank if rank is not None else arr.ndim 

        if spec is not None:
            spec = spec
        else:
            if r == 0: 
                spec = P() # alternatively can be used for replication
            elif r == 1:
                spec = P(('nodes', 'gpus')) # 1d array should be split across everything
            else:
                if self.num_nodes == 1 or self.num_gpus == 1:
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
