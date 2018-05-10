# ##############################################################################
# linalg.py
# =========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# Revision : 0.1
# Last updated : 2018-05-10 14:07:10 UTC
# ##############################################################################

"""
Linear algebra routines.
"""

from typing import Tuple

import numpy as np
import scipy.linalg as linalg

import pypeline.util.argcheck as chk


@chk.check(dict(S=lambda _: chk.has_reals(_) or chk.has_complex(_),
                G=lambda _: chk.has_reals(_) or chk.has_complex(_),
                tau=chk.is_real,
                N=chk.allow_None(chk.is_integer)))
def eigh(S, G, tau, N=None) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Solve truncated generalized eigenvalue problem.

    Find :math:`(D, V)`, solution of the generalized eigenvalue problem

    .. math:: S V = G V D.

    :param S: [:py:class:`~numpy.ndarray`] (M, M) hermitian matrix.
    :param G: [:py:class:`~numpy.ndarray`] (M, M) PSD matrix.
    :param tau: [:py:class:`~numbers.Real` in [0, 1]] energy ratio.
    :param N: [:py:class:`~numbers.Integral` in {1, ..., M}] output shape.
    :return: Tuple (D, V) where:

        * D: [:py:class:`~numpy.ndarray`] (Q,) real-valued (positive)
          eigenvalues;
        * V: [:py:class:`~numpy.ndarray`] (M, Q) complex-valued eigenvectors.

         The Q eigenpairs are sorted in descending order.

        Q is determined as follows: let K be the minimum number of leading
        eigenpairs required to represent ``tau`` percent of the total energy.
        Then:

        * :math:`N = \text{None} \to Q = K`;

        * :math:`N \le K \to Q = N`, with the :math:`K - N` trailing
          eigenpairs being discarded;

        * :math:`N \gt K \to Q = N`, with the :math:`N - K` trailing
          eigenpairs set to 0.

    If ``S`` is not PSD, it's negative spectrum is discarded.

    .. testsetup::

       import numpy as np
       from pypeline.util.math.linalg import eigh

       np.random.seed(0)

       def hermitian_array(N: int) -> np.ndarray:
           '''
           Construct a (N, N) Hermitian matrix.
           '''
           i, j = np.triu_indices(N)
           A = np.zeros((N, N), dtype=complex)
           A[i, j] = np.random.randn(len(i)) + 1j * np.random.randn(len(i))
           A += A.conj().T
           return A

       M = 4
       S = hermitian_array(M)
       G = hermitian_array(M) + 100 * np.eye(M)  # To guarantee PSD

    :Example: Let ``S`` and ``G`` be defined as below:

    .. doctest::

       M = 4
       S = hermitian_array(M)
       G = hermitian_array(M) + 100 * np.eye(M)  # To guarantee PSD

    Then different calls to :py:func:`~pypeline.util.math.linalg.eigh` produce
    different results:

    * Get all positive eigenpairs:

    .. doctest::

       >>> D, V = eigh(S, G, tau=1)
       >>> print(np.around(D, 4))  # The last term is positive but very small.
       [0.0574 0.0397 0.002  0.    ]

       >>> print(np.around(V, 4))
       [[-0.0781-0.0007j  0.0473+0.0011j  0.017 -0.0002j -0.0436+0.0023j]
        [-0.0375+0.0351j -0.0692+0.0235j  0.0158+0.0004j -0.002 -0.0366j]
        [-0.0043+0.0047j  0.0235-0.0079j -0.002 -0.0779j  0.0321-0.0477j]
        [-0.0313+0.021j   0.0208+0.0326j -0.0514+0.0275j  0.0577+0.0089j]]

    * Drop some trailing eigenpairs:

    .. doctest::

       >>> D, V = eigh(S, G, tau=0.8)
       >>> print(np.around(D, 4))
       [0.0574]

       >>> print(np.around(V, 4))
       [[-0.0781-0.0007j]
        [-0.0375+0.0351j]
        [-0.0043+0.0047j]
        [-0.0313+0.021j ]]

    * Pad output to certain size:

    .. doctest::

       >>> D, V = eigh(S, G, tau=0.8, N=3)
       >>> print(np.around(D, 4))
       [0.0574 0.     0.    ]

       >>> print(np.around(V, 4))
       [[-0.0781-0.0007j  0.    +0.j      0.    +0.j    ]
        [-0.0375+0.0351j  0.    +0.j      0.    +0.j    ]
        [-0.0043+0.0047j  0.    +0.j      0.    +0.j    ]
        [-0.0313+0.021j   0.    +0.j      0.    +0.j    ]]
    """
    S = np.array(S, copy=False)
    G = np.array(G, copy=False)
    M = len(S)

    if not (chk.has_shape([M, M])(S) and np.allclose(S, S.conj().T)):
        raise ValueError('Parameter[S] must be hermitian symmetric.')
    if not (chk.has_shape([M, M])(G) and np.allclose(G, G.conj().T)):
        raise ValueError('Parameter[G] must be hermitian symmetric.')
    if not (0 <= tau <= 1):
        raise ValueError('Parameter[tau] must be in [0, 1].')
    if N is not None:
        if not (1 <= N <= M):
            raise ValueError(f'Parameter[N] must be in {{1, ..., {M}}}.')

    # S: drop negative spectrum.
    Ds, Vs = linalg.eigh(S)
    idx = Ds > 0
    Ds, Vs = Ds[idx], Vs[:, idx]
    S = (Vs * Ds) @ Vs.conj().T

    # S, G: generalized eigenvalue-decomposition.
    try:
        D, V = linalg.eigh(S, G)

        # Discard near-zero D due to numerical precision.
        idx = D > 0
        D, V = D[idx], V[:, idx]
        idx = np.argsort(D)[::-1]
        D, V = D[idx], V[:, idx]
    except linalg.LinAlgError:
        raise ValueError('Parameter[G] is not PSD.')

    # Energy selection / padding
    idx = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= tau
    D, V = D[idx], V[:, idx]
    if N is not None:
        M, K = len(V), np.sum(idx)
        if N - K <= 0:
            D, V = D[:N], V[:, :N]
        else:
            D = np.concatenate((D, np.zeros(N - K)), axis=0)
            V = np.concatenate((V, np.zeros((M, N - K))), axis=1)

    return D, V
