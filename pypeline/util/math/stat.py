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
import scipy.special as special
import scipy.stats as stats

import pypeline.core as core
import pypeline.util.argcheck as chk


class Distribution(core.Block):
    """
    Probability distribution.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.stat import Wishart
       import scipy.linalg as linalg

       np.random.seed(0)

       def hermitian_array(N: int) -> np.ndarray:
           '''
           Construct a (N, N) Hermitian matrix.
           '''
           D = np.arange(1, N + 1)
           Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
           Q, _ = linalg.qr(Rmtx)

           A = (Q * D) @ Q.conj().T
           return A

    .. doctest::

       >>> A = hermitian_array(N=4)  # random (N, N) PSD array.
       >>> W = Wishart(A, n=7)

       >>> np.around(W.mean, 2)
       array([[19.96+0.j  , -0.73-3.73j, -4.31+1.84j, -4.38+3.19j],
              [-0.73+3.73j, 15.21-0.j  ,  5.46+2.97j,  1.47+1.48j],
              [-4.31-1.84j,  5.46-2.97j, 18.79+0.j  ,  2.17-1.18j],
              [-4.38-3.19j,  1.47-1.48j,  2.17+1.18j, 16.04+0.j  ]])

       >>> B = hermitian_array(N=4)
       >>> W.pdf([A, B])
       array([1.06849125e-11, 5.44052225e-12])

       >>> samples = W(N_sample=2)  # 2 samples of the distribution.
       >>> np.around(samples, 2)
       array([[[ 19.92+0.j  ,  -8.21-3.72j,  -1.44-0.15j, -15.16+1.41j],
               [ -8.21+3.72j,  12.62+0.j  ,   5.91+2.06j,  13.14-2.17j],
               [ -1.44+0.15j,   5.91-2.06j,   8.47+0.j  ,  11.35-1.76j],
               [-15.16-1.41j,  13.14+2.17j,  11.35+1.76j,  31.42+0.j  ]],
       <BLANKLINE>
              [[ 32.27+0.j  ,   8.45-6.03j,  -4.68+5.54j,   2.63+6.9j ],
               [  8.45+6.03j,   7.96+0.j  ,  -3.77+1.8j ,  -1.11+3.37j],
               [ -4.68-5.54j,  -3.77-1.8j ,   7.08+0.j  ,   4.8 -2.09j],
               [  2.63-6.9j ,  -1.11-3.37j,   4.8 +2.09j,   8.79+0.j  ]]])
    """

    def __init__(self):
        """
        """
        super().__init__()

        # Buffered attributes
        self._mean = None

    @property
    def mean(self):
        """
        Mean of the distribution.
        """
        raise NotImplementedError

    def pdf(self, x):
        """
        Density of the distribution at sample points.

        Parameters
        ----------
        x : array-like
            (N, ...) values at which to determine the pdf.

        Returns
        -------
        pdf : :py:class:`~numpy.ndarray`
            (N,) densities.
        """
        raise NotImplementedError

    @chk.check('N_sample', chk.is_integer)
    def __call__(self, N_sample=1):
        """
        Generate random samples.

        Parameters
        ----------
        N_sample : int
            Number of samples to generate.

        Returns
        -------
        x : :py:class:`~numpy.ndarray`
            (N_sample, ...) samples.
        """
        raise NotImplementedError


class Wishart(Distribution):
    """
    `Wishart <https://en.wikipedia.org/wiki/Wishart_distribution>`_ distribution.
    """

    @chk.check(dict(V=chk.accept_any(chk.has_reals, chk.has_complex),
                    n=chk.is_integer))
    def __init__(self, V, n):
        """
        Parameters
        ----------
        V : array-like(float, complex)
            (p, p) positive-semidefinite scale matrix.
        n : int
            degrees of freedom.
        """
        super().__init__()

        V = np.array(V)
        p = len(V)

        if not (chk.has_shape([p, p])(V) and np.allclose(V, V.conj().T)):
            raise ValueError('Parameter[V] must be hermitian symmetric.')
        if not (n > p):
            raise ValueError(f'Parameter[n] must be greater than {p}.')

        self._V = V
        self._p = p
        self._n = n

        Vq = linalg.sqrtm(V)
        _, R = linalg.qr(Vq)
        self._L = R.conj().T

    @property
    def mean(self):
        """
        Mean of the distribution.
        """
        if self._mean is None:
            self._mean = self._n * self._V
        return self._mean

    @chk.check('x', chk.accept_any(chk.has_reals, chk.has_complex))
    def pdf(self, x):
        """
        Density of the distribution at sample points.

        Parameters
        ----------
        x : array-like
            (N, p, p) values at which to determine the pdf.

        Returns
        -------
        pdf : :py:class:`~numpy.ndarray`
            (N,) densities.
        """
        x = np.array(x, copy=False)
        if x.ndim == 2:
            x = x[np.newaxis]
        elif x.ndim == 3:
            pass
        else:
            raise ValueError('Parameter[x] must have shape (N, p, p).')

        N = len(x)
        if not (chk.has_shape([N, self._p, self._p])(x) and
                np.allclose(x, x.conj().transpose(0, 2, 1))):
            raise ValueError('Parameter[x] must be hermitian symmetric.')

        if np.linalg.matrix_rank(self._V) < self._p:
            raise linalg.LinAlgError('Wishart density is not defined when '
                                     'scale matrix V is singular.')

        # Determinants: real-valued since V,X are Hermitian.
        Vs, Vl = np.linalg.slogdet(self._V)
        dV = np.real(Vs * np.exp(Vl))
        Xs, Xl = np.linalg.slogdet(x)
        dX = np.real(Xs * np.exp(Xl))

        # Trace term
        A = np.linalg.solve(self._V, x)
        trA = np.trace(A, axis1=1, axis2=2).real

        num = (np.float_power(dX, (self._n - self._p - 1) / 2) *
               np.exp(-trA / 2))
        den = (np.float_power(2, self._n * self._p / 2) *
               np.float_power(dV, self._n / 2) *
               np.exp(special.multigammaln(self._n / 2, self._p)))

        pdf = num / den
        return pdf

    @chk.check('N_sample', chk.is_integer)
    def __call__(self, N_sample=1):
        """
        Generate random samples.

        Parameters
        ----------
        N_sample : int
            Number of samples to generate.

        Returns
        -------
        x : :py:class:`~numpy.ndarray`
            (N_sample, p, p) samples.

        Notes
        -----
        The Wishart estimate is obtained using the `Bartlett Decomposition`_.

        .. _Bartlett Decomposition: https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
        """
        if N_sample < 1:
            raise ValueError('Parameter[N_sample] must be positive.')

        A = np.zeros((N_sample, self._p, self._p))

        diag_idx = np.diag_indices(self._p)
        df = (self._n * np.ones((N_sample, 1)) - np.arange(self._p))
        A[:, diag_idx[0], diag_idx[1]] = np.sqrt(stats.chi2.rvs(df=df))

        tril_idx = np.tril_indices(self._p, k=-1)
        size = (N_sample, self._p * (self._p - 1) // 2)
        A[:, tril_idx[0], tril_idx[1]] = stats.norm.rvs(size=size)

        W = self._L @ A
        X = W @ W.conj().transpose(0, 2, 1)
        return X
