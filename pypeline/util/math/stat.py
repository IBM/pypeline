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

    .. doctest::

       >>> A = hermitian_array(N=4)  # random (N, N) PSD array.
       >>> print(np.around(A, 2))
       [[ 3.53+0.j    0.4 +1.45j  0.98+0.76j  2.24+0.12j]
        [ 0.4 -1.45j  3.74+0.j   -0.98+0.33j  0.95+1.49j]
        [ 0.98-0.76j -0.98-0.33j -0.3 +0.j   -0.1 +0.31j]
        [ 2.24-0.12j  0.95-1.49j -0.1 -0.31j  0.82+0.j  ]]

       >>> B = wishrnd(A, df=7)
       >>> print(np.around(B, 2))
       [[ 3.92 +0.j    6.55 +0.68j  2.95 -0.33j  3.87 -1.47j]
        [ 6.55 -0.68j 43.29 +0.j    3.8  -2.26j  7.59 +1.84j]
        [ 2.95 +0.33j  3.8  +2.26j 18.7  +0.j    4.63-16.52j]
        [ 3.87 +1.47j  7.59 -1.84j  4.63+16.52j 20.74 +0.j  ]]

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
