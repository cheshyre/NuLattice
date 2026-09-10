"""Compiled-execution tests for the Davidson solver."""

import jax.numpy as jnp
import numpy as np

from NuLattice.HF.davidson import davidson_eigh


def test_lowered_compiled_solver_is_stable_across_calls(eigenpair_comparisons):
    """Reuse one compiled executable with stable eigenpair results."""

    matrix = jnp.diag(jnp.arange(1.0, 9.0))
    guess = jnp.eye(8)[:, :2]
    compiled_solver = davidson_eigh.lower(matrix, 2, guess, 3).compile()

    first_values, first_vectors = compiled_solver(matrix, guess, 3)
    repeated_values, repeated_vectors = compiled_solver(matrix, guess, 3)
    shifted_values, shifted_vectors = compiled_solver(
        matrix + 5.0 * jnp.eye(8), guess, 3
    )

    np.testing.assert_array_equal(repeated_values, first_values)
    np.testing.assert_allclose(
        eigenpair_comparisons.projector(repeated_vectors),
        eigenpair_comparisons.projector(first_vectors),
        atol=1e-12,
    )
    np.testing.assert_allclose(first_values, [1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(shifted_values, [6.0, 7.0], atol=1e-12)
    np.testing.assert_allclose(
        eigenpair_comparisons.residuals(
            matrix + 5.0 * jnp.eye(8), shifted_values, shifted_vectors
        ),
        0.0,
        atol=1e-12,
    )
