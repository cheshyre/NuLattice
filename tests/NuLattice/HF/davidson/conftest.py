"""Shared deterministic inputs and comparisons for Davidson tests."""

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class HermitianProblem:
    """Store a deterministic Hermitian matrix and its exact eigensystem."""

    name: str
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def exact_guess(self, root_count):
        """Return exact vectors for the requested lowest roots."""

        return self.eigenvectors[:, :root_count]

    def perturbed_guess(self, root_count, scale=0.02):
        """Return a deterministic full-rank perturbation of exact vectors."""

        guess = self.exact_guess(root_count)
        guess = guess + scale * np.roll(guess, shift=1, axis=0)
        return np.linalg.qr(guess, mode="reduced")[0]


class EigenpairComparisons:
    """Compare eigenpairs without fixing arbitrary phases or bases."""

    @staticmethod
    def residuals(matrix, eigenvalues, eigenvectors):
        """Return one normalized residual for each eigenpair."""

        matrix = np.asarray(matrix)
        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors)
        residual = matrix @ eigenvectors - eigenvectors * eigenvalues[None, :]
        scale = max(np.linalg.norm(matrix, ord=2), 1.0)
        return np.linalg.norm(residual, axis=0) / scale

    @staticmethod
    def projector(vectors):
        """Return the phase- and basis-invariant column-space projector."""

        vectors = np.asarray(vectors)
        return vectors @ vectors.conj().T

    @staticmethod
    def assert_orthonormal(vectors, atol=1e-10):
        """Require finite columns with an identity overlap matrix."""

        vectors = np.asarray(vectors)
        assert np.all(np.isfinite(vectors))
        np.testing.assert_allclose(
            vectors.conj().T @ vectors,
            np.eye(vectors.shape[1]),
            atol=atol,
        )


def _rotated_problem(name, spectrum, *, complex_matrix=False):
    """Build a reproducible dense Hermitian matrix with a chosen spectrum."""

    spectrum = np.asarray(spectrum, dtype=np.float64)
    generator = np.random.default_rng(20260828)
    transform = generator.normal(size=(spectrum.size, spectrum.size))
    if complex_matrix:
        transform = transform + 1j * generator.normal(size=transform.shape)
    unitary = np.linalg.qr(transform)[0]
    matrix = unitary @ np.diag(spectrum) @ unitary.conj().T
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return HermitianProblem(name, matrix, eigenvalues, eigenvectors)


def _diagonal_problem():
    """Build the positive-definite diagonal regression problem."""

    eigenvalues = np.arange(1.0, 9.0)
    matrix = np.diag(eigenvalues)
    return HermitianProblem("diagonal", matrix, eigenvalues, np.eye(8))


PROBLEMS = (
    _diagonal_problem(),
    _rotated_problem("coupled", [-2.0, -0.7, 0.3, 1.4, 2.5, 4.0, 6.0, 8.0]),
    _rotated_problem(
        "clustered", [-1.0, -0.9995, -0.997, 0.4, 1.7, 3.0, 4.5, 6.0]
    ),
    _rotated_problem("degenerate", [-1.0, -1.0, 0.25, 0.25, 2.0, 3.0, 4.0, 5.0]),
    _rotated_problem(
        "complex", [-2.0, -0.5, 0.1, 1.0, 2.2, 3.0, 4.0, 5.0], complex_matrix=True
    ),
)


@pytest.fixture(params=PROBLEMS, ids=lambda problem: problem.name)
def hermitian_problem(request):
    """Provide deterministic diagonal, coupled, clustered, and complex cases."""

    return request.param


@pytest.fixture
def eigenpair_comparisons():
    """Provide reusable residual, overlap, and projector comparisons."""

    return EigenpairComparisons


@pytest.fixture
def hermitian_problems():
    """Provide deterministic problems by descriptive name."""

    return {problem.name: problem for problem in PROBLEMS}
