# #############################################################################
# _func.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

import warnings

import numpy as np
import scipy.interpolate as interpolate
import scipy.special as sp

import pypeline.core as core
import pypeline.util.argcheck as chk
import _pypeline_util_math_func_pybind11 as func


class SphericalDirichlet(core.Block):
    r"""
    Parameterized spherical Dirichlet kernel.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.func import SphericalDirichlet

    .. doctest::

       >>> N = 4
       >>> f = SphericalDirichlet(N)

       >>> sample_points = np.linspace(-1, 1, 25).reshape(5, 5)  # any shape
       >>> amplitudes = f(sample_points)
       >>> np.around(amplitudes, 2)
       array([[ 1.  ,  0.2 , -0.25, -0.44, -0.44],
              [-0.32, -0.13,  0.07,  0.26,  0.4 ],
              [ 0.47,  0.46,  0.38,  0.22,  0.  ],
              [-0.24, -0.48, -0.67, -0.76, -0.68],
              [-0.37,  0.27,  1.3 ,  2.84,  5.  ]])

    When only interested in kernel values close to 1, the approximation method provides significant speedups, at the cost of approximation error in values far from 1:

    .. doctest::

       N = 11
       f_exact = SphericalDirichlet(N)
       f_approx = SphericalDirichlet(N, approx=True)

       x = np.linspace(-1, 1, 2000)
       e_y = f_exact(x)
       a_y = f_approx(x)
       rel_err = np.abs((e_y - a_y) / e_y)

       fig, ax = plt.subplots(nrows=2)
       ax[0].plot(x, e_y, 'r')
       ax[0].plot(x, a_y, 'b')
       ax[0].legend(['exact', 'approx'])
       ax[0].set_title('Dirichlet Kernel')

       ax[1].plot(x, rel_err)
       ax[1].set_title('Relative Error (Exact vs. Approx)')

       fig.show()

    .. image:: _img/sph_dirichlet_example.png

    Notes
    -----
    The spherical Dirichlet function :math:`K_{N}(t): [-1, 1] \to \mathbb{R}` is defined as:

    .. math:: K_{N}(t) = \frac{P_{N+1}(t) - P_{N}(t)}{t - 1},

    where :math:`P_{N}(t)` is the `Legendre polynomial <https://en.wikipedia.org/wiki/Legendre_polynomials>`_ of order :math:`N`.
    """

    @chk.check(dict(N=chk.is_integer,
                    approx=chk.is_boolean))
    def __init__(self, N, approx=False):
        """
        Parameters
        ----------
        N : int
            Kernel order.
        approx : bool
            Approximate kernel using cubic-splines.

            This method provides extremely reliable estimates of :math:`K_{N}(t)` in the vicinity of 1 where the function's main sidelobes are found.
            Values outside the vicinity smoothly converge to 0.

            Only works for `N` greater than 10.
        """
        super().__init__()

        if N < 0:
            raise ValueError("Parameter[N] must be non-negative.")
        self._N = N

        if (approx is True) and (N <= 10):
            raise ValueError('Cannot use approximation method if '
                             'Parameter[N] <= 10.')
        self._approx = approx

        if approx is True:  # Fit cubic-spline interpolator.
            N_samples = 10 ** 3

            # Find interval LHS after which samples will be evaluated exactly.
            theta_max = np.pi
            while True:
                x = np.linspace(0, theta_max, N_samples)
                cx = np.cos(x)
                cy = self._exact_kernel(cx)
                zero_cross = np.diff(np.sign(cy))
                N_cross = np.abs(np.sign(zero_cross)).sum()

                if N_cross > 10:
                    theta_max /= 2
                else:
                    break

            window = func.Tukey(T=2 - 2 * np.cos(2 * theta_max),
                                beta=1,
                                alpha=0.5)

            x = np.r_[np.linspace(np.cos(theta_max * 2), np.cos(theta_max),
                                  N_samples, endpoint=False),
                      np.linspace(np.cos(theta_max), 1, N_samples)]
            y = self._exact_kernel(x) * window(x)
            self.__cs_interp = interpolate.interp1d(x, y,
                                                    kind='cubic',
                                                    bounds_error=False,
                                                    fill_value=0)

    @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def __call__(self, x):
        r"""
        Sample the order-N spherical Dirichlet kernel.

        Parameters
        ----------
        x : float or :py:class:`~numpy.ndarray`
            Values at which to compute :math:`K_{N}(x)`.

        Returns
        -------
        K_N(x) : :py:class:`~numpy.ndarray`
        """
        if chk.is_scalar(x):
            x = np.array([x], dtype=float)
        else:
            x = np.array(x, copy=False, dtype=float)

        if not np.all((-1 <= x) & (x <= 1)):
            raise ValueError('Parameter[x] must lie in [-1, 1].')

        if self._approx is True:
            f = self._approx_kernel
        else:
            f = self._exact_kernel

        amplitude = f(x)
        return amplitude

    # @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def _exact_kernel(self, x):
        amplitude = (sp.eval_legendre(self._N + 1, x) -
                     sp.eval_legendre(self._N, x))
        with warnings.catch_warnings():
            # The kernel is so condensed near 1 at high N that np.isclose()
            # does a terrible job at letting us manually treat values close to
            # the upper limit.
            # The best way to implement K_N(t) is to let the floating point
            # division fail and then replace NaNs.
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            amplitude /= x - 1
        amplitude[np.isnan(amplitude)] = self._N + 1

        return amplitude

    # @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def _approx_kernel(self, x):
        amplitude = self.__cs_interp(x)
        return amplitude
