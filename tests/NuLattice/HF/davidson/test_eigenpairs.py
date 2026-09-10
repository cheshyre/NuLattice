"""Public eigenpair correctness tests for the Davidson solver."""

import jax.numpy as jnp
import numpy as np
import pytest

from NuLattice.HF.davidson import davidson_eigh


@pytest.mark.parametrize("root_count", [1, 2], ids=["lowest-root", "multiple-roots"])
def test_davidson_matches_dense_hermitian_eigensolver(
    hermitian_problem,
    root_count,
    eigenpair_comparisons,
):
    """Match dense lowest eigenpairs for deterministic Hermitian problems."""

    matrix = jnp.asarray(hermitian_problem.matrix)
    guess = jnp.asarray(hermitian_problem.exact_guess(root_count))
    expected_values, expected_vectors = jnp.linalg.eigh(matrix)

    actual_values, actual_vectors = davidson_eigh(
        matrix, root_count, guess, max_iter=2
    )

    assert actual_values.shape == (root_count,)
    assert actual_vectors.shape == (matrix.shape[0], root_count)
    assert actual_values.dtype == jnp.real(matrix).dtype
    assert actual_vectors.dtype == matrix.dtype
    assert np.all(np.diff(np.asarray(actual_values)) >= 0.0)
    np.testing.assert_allclose(
        actual_values, expected_values[:root_count], rtol=1e-11, atol=1e-11
    )
    eigenpair_comparisons.assert_orthonormal(actual_vectors)
    np.testing.assert_allclose(
        eigenpair_comparisons.residuals(matrix, actual_values, actual_vectors),
        0.0,
        atol=1e-11,
    )

    boundary_is_degenerate = (
        root_count < matrix.shape[0]
        and abs(
            float(expected_values[root_count] - expected_values[root_count - 1])
        )
        < 1e-10
    )
    if not boundary_is_degenerate:
        np.testing.assert_allclose(
            eigenpair_comparisons.projector(actual_vectors),
            eigenpair_comparisons.projector(expected_vectors[:, :root_count]),
            atol=1e-10,
        )


@pytest.mark.parametrize("shift", [0.0, 10.0], ids=["positive", "shifted"])
def test_positive_diagonal_matrix_has_no_zero_ritz_values(
    shift,
    eigenpair_comparisons,
):
    """Exclude spurious zeros from padded initial Davidson subspaces."""

    diagonal = jnp.arange(1.0, 9.0) + shift
    matrix = jnp.diag(diagonal)
    guess = jnp.eye(8)[:, :2]

    actual_values, actual_vectors = davidson_eigh(matrix, 2, guess, max_iter=3)

    np.testing.assert_allclose(actual_values, diagonal[:2], atol=1e-12)
    np.testing.assert_allclose(
        eigenpair_comparisons.residuals(matrix, actual_values, actual_vectors),
        0.0,
        atol=1e-12,
    )
