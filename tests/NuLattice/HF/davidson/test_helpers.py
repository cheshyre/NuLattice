"""Tests for private Davidson numerical helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from NuLattice.HF.davidson import (
    _adjoint,
    _cholesky_qr,
    _cqr2,
    _regularize_denominator,
)


@pytest.mark.parametrize(
    "matrix",
    [
        np.arange(6.0).reshape(2, 3),
        np.arange(6.0).reshape(2, 3) + 1j * np.arange(6.0, 12.0).reshape(2, 3),
    ],
    ids=["real", "complex"],
)
def test_adjoint_matches_conjugate_transpose(matrix):
    """Conjugate and transpose the final two axes."""

    actual = _adjoint(jnp.asarray(matrix))

    np.testing.assert_array_equal(actual, matrix.conj().T)


def test_adjoint_preserves_batch_axes():
    """Apply the adjoint independently to batched matrices."""

    matrices = np.arange(24.0).reshape(2, 3, 4)
    matrices = matrices + 1j * np.flip(matrices, axis=-1)

    actual = _adjoint(jnp.asarray(matrices))

    np.testing.assert_array_equal(actual, matrices.conj().swapaxes(-1, -2))


def test_regularize_denominator_preserves_sign_and_threshold_entries():
    """Bound only entries strictly inside the signed shift threshold."""

    denominator = jnp.array([-2.0, -0.1, -0.05, 0.0, 0.05, 0.1, 2.0])

    actual = _regularize_denominator(denominator, shift=0.1)

    np.testing.assert_allclose(actual, [-2.0, -0.1, -0.1, 0.1, 0.1, 0.1, 2.0])


@pytest.mark.parametrize(
    ("orthonormalize", "orthogonality_tolerance"),
    [(_cholesky_qr, 1e-4), (_cqr2, 1e-10)],
    ids=["cholesky-qr", "cholesky-qr2"],
)
@pytest.mark.parametrize("nearly_dependent", [False, True], ids=["regular", "near-rank-loss"])
def test_cholesky_qr_preserves_full_rank_column_space(
    orthonormalize,
    orthogonality_tolerance,
    nearly_dependent,
    eigenpair_comparisons,
):
    """Preserve the space and finite orthogonality for full-rank inputs."""

    epsilon = 1e-4 if nearly_dependent else 0.5
    matrix = np.array(
        [
            [1.0 + 0.2j, 1.0 + 0.2j],
            [0.0, epsilon * (1.0 - 0.1j)],
            [1.0, 1.0 + epsilon],
            [2.0 - 0.1j, 2.0 - epsilon - 0.1j],
        ]
    )

    actual = np.asarray(orthonormalize(jnp.asarray(matrix)))

    eigenpair_comparisons.assert_orthonormal(
        actual, atol=orthogonality_tolerance
    )
    expected_projector = matrix @ np.linalg.pinv(matrix)
    np.testing.assert_allclose(
        eigenpair_comparisons.projector(actual),
        expected_projector,
        atol=orthogonality_tolerance,
    )
