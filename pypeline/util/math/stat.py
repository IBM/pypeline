# #############################################################################
# stat.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Statistical functions not available in `SciPy <https://www.scipy.org/>`_.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats

import pypeline.util.argcheck as chk


@chk.check(dict(S=chk.accept_any(chk.has_reals, chk.has_complex),
                df=chk.is_integer))
def wishrnd(S, df):
    """
    Wishart random variable.

    Parameters
    ----------
    S : array-like(float, complex)
        (p, p) positive-semidefinite scale matrix.
    df : int
        Degrees of freedom.

    Returns
    -------
        :py:class:`~numpy.ndarray`
            (p, p) Wishart estimate.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.stat import wishrnd
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

    .. doctest::

       >>> A = hermitian_array(N=4)  # random (N, N) PSD array.
       >>> print(np.around(A, 2))
       [[ 1.85+0.j   -0.1 -0.53j -0.62+0.26j -0.63+0.46j]
        [-0.1 +0.53j  1.17+0.j    0.78+0.42j  0.21+0.21j]
        [-0.62-0.26j  0.78-0.42j  1.68+0.j    0.31-0.17j]
        [-0.63-0.46j  0.21-0.21j  0.31+0.17j  1.29+0.j  ]]

       >>> B = wishrnd(A, df=7)
       >>> print(np.around(B, 2))
       [[ 6.79+0.j   -3.13-1.96j -6.83-0.71j -0.41+2.05j]
        [-3.13+1.96j  3.15+0.j    3.13-0.94j  1.28+0.15j]
        [-6.83+0.71j  3.13+0.94j 11.03-0.j   -3.6 -5.7j ]
        [-0.41-2.05j  1.28-0.15j -3.6 +5.7j  10.79+0.j  ]]

    Notes
    -----
    The Wishart estimate is obtained using the `Bartlett Decomposition`_.

    .. _Bartlett Decomposition: https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
    """
    S = np.array(S, copy=False)
    p = len(S)

    if not (chk.has_shape([p, p])(S) and np.allclose(S, S.conj().T)):
        raise ValueError('Parameter[S] must be hermitian symmetric.')
    if not (df > p):
        raise ValueError(f'Parameter[df] must be greater than {p}.')

    Sq = linalg.sqrtm(S)
    _, R = linalg.qr(Sq)
    L = R.conj().T

    A = np.zeros((p, p))
    A[np.diag_indices(p)] = np.sqrt(stats.chi2.rvs(df=df - np.arange(p)))
    A[np.tril_indices(p, k=-1)] = stats.norm.rvs(size=p * (p - 1) // 2)

    W = L @ A
    X = W @ W.conj().T
    return X
