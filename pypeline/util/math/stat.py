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
                df=chk.is_integer,
                normalize=chk.is_boolean))
def wishrnd(S, df, normalize=True):
    """
    Wishart random variable.

    Parameters
    ----------
    S : array-like(float, complex)
        (p, p) positive-semidefinite scale matrix.
    df : int
        Degrees of freedom.
    normalize : bool
        If True, normalize estimate by `df`. (Default: True)

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
       [[0.56+0.j   0.94+0.1j  0.42-0.05j 0.55-0.21j]
        [0.94-0.1j  6.18+0.j   0.54-0.32j 1.08+0.26j]
        [0.42+0.05j 0.54+0.32j 2.67+0.j   0.66-2.36j]
        [0.55+0.21j 1.08-0.26j 0.66+2.36j 2.96+0.j  ]]

    Notes
    -----
    The Wishart estimate is obtained using the `Bartlett Decomposition`_.

    .. _Bartlett Decomposition: https://en.wikipedia.org/wiki/\
       Wishart_distribution#Bartlett_decomposition
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
    X = W @ W.conj().T / (df if normalize else 1)
    return X
