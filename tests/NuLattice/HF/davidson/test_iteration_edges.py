"""Iteration and numerical edge tests for the Davidson solver."""

import jax.numpy as jnp
import numpy as np
import pytest

from NuLattice.HF.davidson import davidson_eigh


@pytest.mark.parametrize("max_iter", [0, 1, 4])
def test_exact_warm_start_is_stable_across_iteration_counts(
    hermitian_problem,
    max_iter,
    eigenpair_comparisons,
):
    """Keep exact warm-start eigenpairs stable, including zero iterations."""

    root_count = 2
    matrix = jnp.asarray(hermitian_problem.matrix)
    guess = jnp.asarray(hermitian_problem.exact_guess(root_count))

    values, vectors = davidson_eigh(matrix, root_count, guess, max_iter)

    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(
        values, hermitian_problem.eigenvalues[:root_count], atol=1e-10
    )
    np.testing.assert_allclose(
        eigenpair_comparisons.residuals(matrix, values, vectors),
        0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("max_iter", [0, 1, 3])
def test_each_returned_value_matches_its_returned_vector(
    hermitian_problems,
    max_iter,
):
    """Return eigenvalues from the same Ritz extraction as the vectors."""

    problem = hermitian_problems["coupled"]
    root_count = 2
    matrix = jnp.asarray(problem.matrix)
    guess = jnp.asarray(problem.perturbed_guess(root_count))

    values, vectors = davidson_eigh(matrix, root_count, guess, max_iter)

    rayleigh_values = jnp.real(jnp.diag(vectors.conj().T @ matrix @ vectors))
    np.testing.assert_allclose(values, rayleigh_values, rtol=1e-11, atol=1e-11)


def test_coincident_preconditioner_entries_remain_finite(
    eigenpair_comparisons,
):
    """Regularize exact diagonal/eigenvalue denominator coincidences."""

    diagonal = jnp.array([-2.0, -0.5, 0.75, 2.0, 4.0, 7.0])
    matrix = jnp.diag(diagonal)
    guess = jnp.eye(6)[:, :2]

    values, vectors = davidson_eigh(matrix, 2, guess, max_iter=4)

    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(vectors))
    np.testing.assert_allclose(values, diagonal[:2], atol=1e-12)
    np.testing.assert_allclose(
        eigenpair_comparisons.residuals(matrix, values, vectors), 0.0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("problem_name", "root_count", "residual_tolerance"),
    [
        ("coupled", 2, 1e-6),
        ("clustered", 3, 1e-8),
        ("degenerate", 2, 5e-7),
        ("complex", 2, 2e-5),
    ],
)
def test_perturbed_warm_starts_converge_for_numerical_edges(
    hermitian_problems,
    problem_name,
    root_count,
    residual_tolerance,
    eigenpair_comparisons,
):
    """Converge clustered, degenerate, coupled, and complex warm starts."""

    problem = hermitian_problems[problem_name]
    matrix = jnp.asarray(problem.matrix)
    guess = jnp.asarray(problem.perturbed_guess(root_count))

    values, vectors = davidson_eigh(matrix, root_count, guess, max_iter=20)

    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(vectors))
    eigenpair_comparisons.assert_orthonormal(vectors)
    np.testing.assert_allclose(
        eigenpair_comparisons.residuals(matrix, values, vectors),
        0.0,
        atol=residual_tolerance,
    )
    np.testing.assert_allclose(
        values,
        problem.eigenvalues[:root_count],
        atol=residual_tolerance,
    )

    if problem_name == "degenerate":
        np.testing.assert_allclose(
            eigenpair_comparisons.projector(vectors),
            eigenpair_comparisons.projector(problem.eigenvectors[:, :root_count]),
            atol=2e-5,
        )
