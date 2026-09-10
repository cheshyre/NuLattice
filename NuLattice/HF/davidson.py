"""
Davidson diagonalization for the lattice Hartree-Fock problem, implemented in JAX
"""
__authors__   =  "Vivek Booshan"
__credits__   =  ["Vivek Booshan"]
__copyright__ = "(c) Vivek Booshan"
__license__   = "BSD-3-Clause"
__date__      = "2026"

from functools import partial

import jax
import jax.numpy as jnp

Array = jax.Array


DIVISION_BY_ZERO_THRESHOLD = 1e-12
SHIFT_REGULARIZATION = 1e-12


@partial(jax.jit, static_argnames=("npart",))
def davidson_eigh(H: Array, npart: int, guess_vecs: Array, max_iter: int):
    """
    finds the lowest npart eigenvalues and eigenvectors of the hermitian matrix H

    The subspace is kept at the fixed size 2*npart so that the routine can be
    jit-compiled: each iteration replaces the subspace by the current Ritz
    vectors together with their preconditioned residuals. The loop runs for a
    fixed number of steps rather than to a convergence criterion, which is
    cheap when guess_vecs comes from the previous Hartree-Fock iteration.

    H must be square and Hermitian, 1 <= npart with 2*npart <= H.shape[0],
    guess_vecs must have shape (H.shape[0], npart) and full column rank, and
    max_iter must be nonnegative. These preconditions are not checked inside
    the compiled function.

    Frankensteined from https://joshuagoings.com/2013/08/23/davidsons-method/

    :param H:          hermitian matrix to be diagonalized
    :type H:           jax.Array((nstat,nstat), dtype=float or complex)
    :param npart:      number of occupied states, i.e., number of lowest roots needed
    :type npart:       int
    :param guess_vecs: initial guess vectors, e.g., the occupied orbitals of the previous solution
    :type guess_vecs:  jax.Array((nstat,npart), dtype=float or complex)
    :param max_iter:   number of subspace expansion steps (try 3-5 for warm starts)
    :type max_iter:    int
    :return:           the npart lowest eigenvalues and the corresponding eigenvectors
    :rtype:            jax.Array((npart,), dtype=float), jax.Array((nstat,npart), dtype=float or complex)
    """
    nstat = H.shape[0]
    D = jnp.diag(H) # Extract diagonal for the preconditioner

    # Initialize a static subspace V of size (nstat, 2 * npart). Reduced QR
    # completes the zero-padded columns rather than leaving null basis vectors
    # that would introduce spurious zero Ritz values.
    V = jnp.zeros((nstat, 2 * npart), dtype=H.dtype)
    V = V.at[:, :npart].set(guess_vecs)
    V = _qr_with_completion(V)

    def body_fun(i, V_sub):
        del i

        # Project into subspace: M = VT H V -> (2k, 2k)
        M = _adjoint(V_sub) @ (H @ V_sub)

        # local eigen solution
        vals, evecs = jnp.linalg.eigh(M)

        best_vals = vals[:npart]
        best_evecs = evecs[:, :npart]

        X = V_sub @ best_evecs
        HX = H @ X
        R = HX - X * best_vals[None, :]

        # preconditioner: (D - energy)^{-1} * R
        denom = D[:, None] - best_vals[None, :]
        denom = _regularize_denominator(denom, SHIFT_REGULARIZATION)
        Y = R / denom

        V_next = jnp.concatenate([X, Y], axis=1)
        V_next = _qr_with_completion(V_next)

        return V_next

    # Run fixed-iteration loop to avoid dynamic compilation tracing
    final_V = jax.lax.fori_loop(0, max_iter, body_fun, V)

    # Extract values and vectors from the same final projected problem.
    final_M = _adjoint(final_V) @ (H @ final_V)
    final_vals, final_evecs = jnp.linalg.eigh(final_M)
    vecs_out = jnp.dot(final_V, final_evecs[:, :npart])

    return final_vals[:npart], vecs_out


# Functions below this are private, they are not guaranteed to be stable between versions of NuLattice


def _adjoint(x: Array) -> Array:
    """
    takes the hermitian conjugate of a matrix, i.e., transposes the last two axes and conjugates

    :param x: matrix (or stack of matrices) to be conjugated
    :type x:  jax.Array((...,:,:), dtype=float or complex)
    :return:  hermitian conjugate of x
    :rtype:   jax.Array((...,:,:), dtype=float or complex)
    """
    return jnp.swapaxes(jnp.conj(x), -1, -2)


def _cholesky_qr(x: Array) -> Array:
    """
    orthonormalizes the columns of x via a Cholesky decomposition of the overlap matrix

    This is a QR decomposition that only needs the small (ncol x ncol) overlap
    matrix and is therefore cheap and easy to compile, but it loses accuracy
    for ill-conditioned x; see _cqr2 for the remedy.

    :param x: matrix whose columns are to be orthonormalized
    :type x:  jax.Array((nstat,ncol), dtype=float or complex)
    :return:  matrix with orthonormal columns spanning the same space as x
    :rtype:   jax.Array((nstat,ncol), dtype=float or complex)
    """
    # Compute the small overlap matrix (2k x 2k).
    S = _adjoint(x) @ x
    S += SHIFT_REGULARIZATION * jnp.eye(S.shape[0], dtype=S.dtype)
    L = jnp.linalg.cholesky(S)
    L_inv = jnp.linalg.inv(L)
    return x @ _adjoint(L_inv)


def _cqr2(V: Array) -> Array:
    """
    orthonormalizes the columns of V by applying the Cholesky QR twice

    The second pass restores orthogonality that is lost in the first pass when
    the overlap matrix is ill-conditioned ("twice is enough").

    :param V: matrix whose columns are to be orthonormalized
    :type V:  jax.Array((nstat,ncol), dtype=float or complex)
    :return:  matrix with orthonormal columns spanning the same space as V
    :rtype:   jax.Array((nstat,ncol), dtype=float or complex)
    """
    return _cholesky_qr(_cholesky_qr(V))


def _qr_with_completion(x: Array) -> Array:
    """
    orthonormalize columns and complete rank-deficient trailing columns

    Davidson residuals vanish for converged roots, so a restarted subspace can
    be rank deficient even when its Ritz vectors are valid. Reduced Householder
    QR supplies orthonormal completion vectors in those trailing positions.

    :param x: subspace matrix with at least as many rows as columns
    :type x:  jax.Array((nstat,ncol), dtype=float or complex)
    :return:  matrix with ncol finite, orthonormal columns
    :rtype:   jax.Array((nstat,ncol), dtype=float or complex)
    """
    return jnp.linalg.qr(x, mode="reduced")[0]


def _regularize_denominator(denom: Array, shift: float) -> Array:
    """
    Bound small Davidson denominators without reversing their sign.

    Entries whose magnitude falls below shift are replaced by +/-shift, keeping
    the original sign, so that the preconditioner never divides by zero.

    :param denom: denominators D_p - E_i of the Davidson preconditioner
    :type denom:  jax.Array((nstat,npart), dtype=float)
    :param shift: smallest magnitude a denominator is allowed to take
    :type shift:  float
    :return:      denominators with all magnitudes at least shift
    :rtype:       jax.Array((nstat,npart), dtype=float)
    """
    signed_shift = jnp.where(denom >= 0.0, shift, -shift)
    return jnp.where(jnp.abs(denom) < shift, signed_shift, denom)
