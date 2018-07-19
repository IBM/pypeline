# ##############################################################################
# _linalg.py
# ==========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# ##############################################################################

import numpy as np
import scipy.linalg as linalg

import pypeline.util.argcheck as chk


@chk.check(dict(A=chk.accept_any(chk.has_reals, chk.has_complex),
                B=chk.allow_None(chk.accept_any(chk.has_reals,
                                                chk.has_complex)),
                tau=chk.is_real,
                N=chk.allow_None(chk.is_integer)))
def eigh(A, B=None, tau=1, N=None):
    """
    Solve a generalized eigenvalue problem.

    Finds :math:`(D, V)`, solution of the generalized eigenvalue problem

    .. math::

       A V = B V D.

    This function is a wrapper around :py:func:`scipy.linalg.eigh` that adds energy truncation and extra output formats.

    Parameters
    ----------
    A : array-like(float or complex)
        (M, M) hermitian matrix.
        If `A` is not positive-semidefinite (PSD), its negative spectrum is discarded.
    B : array-like(float or complex), optional
        (M, M) PSD hermitian matrix.
        If unspecified, `B` is assumed to be the identity matrix.
    tau : float, optional
        Normalized energy ratio. (Default: 1)
    N : int, optional
        Number of eigenpairs to output. (Default: K, the minimum number of leading eigenpairs that account for `tau` percent of the total energy.)

        * If `N` is smaller than K, then the trailing eigenpairs are dropped.
        * If `N` is greater that K, then the trailing eigenpairs are set to 0.

    Returns
    -------
        D : :py:class:`~numpy.ndarray`
            (N,) positive real-valued eigenvalues.

        V : :py:class:`~numpy.ndarray`
            (M, N) complex-valued eigenvectors.

            The N eigenpairs are sorted in decreasing eigenvalue order.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.linalg import eigh
       import scipy.linalg as linalg

       np.random.seed(0)

       def hermitian_array(N: int) -> np.ndarray:
           '''
           Construct a (N, N) Hermitian matrix.
           '''
           D = np.arange(N)
           Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
           Q, _ = linalg.qr(Rmtx)

           A = (Q * D) @ Q.conj().T
           return A

       M = 4
       A = hermitian_array(M)
       B = hermitian_array(M) + 100 * np.eye(M)  # To guarantee PSD

    Let `A` and `B` be defined as below:

    .. doctest::

       M = 4
       A = hermitian_array(M)
       B = hermitian_array(M) + 100 * np.eye(M)  # To guarantee PSD

    Then different calls to :py:func:`~pypeline.util.math.linalg.eigh` produce different results:

    * Get all positive eigenpairs:

    .. doctest::

       >>> D, V = eigh(A, B)
       >>> print(np.around(D, 4))  # The last term is small but positive.
       [0.0296 0.0198 0.0098 0.    ]

       >>> print(np.around(V, 4))
       [[-0.0621+0.0001j -0.0561+0.0005j -0.0262-0.0004j  0.0474+0.0005j]
        [ 0.0285+0.0041j -0.0413-0.0501j  0.0129-0.0209j -0.004 -0.0647j]
        [ 0.0583+0.0055j -0.0443+0.0033j  0.0069+0.0474j  0.0281+0.0371j]
        [ 0.0363+0.0209j  0.0006+0.0235j -0.029 -0.0736j  0.0321+0.0142j]]

    * Drop some trailing eigenpairs:

    .. doctest::

       >>> D, V = eigh(A, B, tau=0.8)
       >>> print(np.around(D, 4))
       [0.0296]

       >>> print(np.around(V, 4))
       [[-0.0621+0.0001j]
        [ 0.0285+0.0041j]
        [ 0.0583+0.0055j]
        [ 0.0363+0.0209j]]

    * Pad output to certain size:

    .. doctest::

       >>> D, V = eigh(A, B, tau=0.8, N=3)
       >>> print(np.around(D, 4))
       [0.0296 0.     0.    ]

       >>> print(np.around(V, 4))
       [[-0.0621+0.0001j  0.    +0.j      0.    +0.j    ]
        [ 0.0285+0.0041j  0.    +0.j      0.    +0.j    ]
        [ 0.0583+0.0055j  0.    +0.j      0.    +0.j    ]
        [ 0.0363+0.0209j  0.    +0.j      0.    +0.j    ]]
    """
    A = np.array(A, copy=False)
    M = len(A)
    if not (chk.has_shape([M, M])(A) and np.allclose(A, A.conj().T)):
        raise ValueError('Parameter[A] must be hermitian symmetric.')

    B = np.eye(M) if (B is None) else np.array(B, copy=False)
    if not (chk.has_shape([M, M])(B) and np.allclose(B, B.conj().T)):
        raise ValueError('Parameter[B] must be hermitian symmetric.')

    if not (0 < tau <= 1):
        raise ValueError('Parameter[tau] must be in [0, 1].')

    if (N is not None) and (N <= 0):
        raise ValueError(f'Parameter[N] must be a non-zero positive integer.')

    # A: drop negative spectrum.
    Ds, Vs = linalg.eigh(A)
    idx = Ds > 0
    Ds, Vs = Ds[idx], Vs[:, idx]
    A = (Vs * Ds) @ Vs.conj().T

    # A, B: generalized eigenvalue-decomposition.
    try:
        D, V = linalg.eigh(A, B)

        # Discard near-zero D due to numerical precision.
        idx = D > 0
        D, V = D[idx], V[:, idx]
        idx = np.argsort(D)[::-1]
        D, V = D[idx], V[:, idx]
    except linalg.LinAlgError:
        raise ValueError('Parameter[B] is not PSD.')

    # Energy selection / padding
    idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= tau
    D, V = D[idx], V[:, idx]
    if N is not None:
        M, K = V.shape
        if N - K <= 0:
            D, V = D[:N], V[:, :N]
        else:
            D = np.concatenate((D, np.zeros(N - K)), axis=0)
            V = np.concatenate((V, np.zeros((M, N - K))), axis=1)

    return D, V
