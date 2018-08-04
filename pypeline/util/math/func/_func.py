# #############################################################################
# _func.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

import warnings

import astropy.units as u
import numpy as np
import scipy.interpolate as interpolate
import scipy.special as sp

import pypeline.core as core
import pypeline.util.argcheck as chk


class Tukey(core.Block):
    r"""
    Parameterized Tukey function.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.func import Tukey

    .. doctest::

       >>> tukey = Tukey(T=1, beta=0.5, alpha=0.25)

       >>> sample_points = np.linspace(0, 1, 25).reshape(5, 5)  # any shape
       >>> amplitudes = tukey(sample_points)
       >>> np.around(amplitudes, 2)
       array([[0.  , 0.25, 0.75, 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 0.75, 0.25, 0.  ]])

    Notes
    -----
    The Tukey function is defined as:

    .. math::

       \text{Tukey}(T, \beta, \alpha)(\varphi): \mathbb{R} & \to [0, 1] \\
       \varphi & \to
       \begin{cases}
           % LINE 1
           \sin^{2} \left( \frac{\pi}{T \alpha}
                    \left[ \frac{T}{2} - \beta + \varphi \right] \right) &
           0 \le \frac{T}{2} - \beta + \varphi < \frac{T \alpha}{2} \\
           % LINE 2
           1 &
           \frac{T \alpha}{2} \le \frac{T}{2} - \beta +
           \varphi \le T - \frac{T \alpha}{2} \\
           % LINE 3
           \sin^{2} \left( \frac{\pi}{T \alpha}
                    \left[ \frac{T}{2} + \beta - \varphi \right] \right) &
           T - \frac{T \alpha}{2} < \frac{T}{2} - \beta + \varphi \le T \\
           % LINE 4
           0 &
           \text{otherwise.}
       \end{cases}
    """

    @chk.check(dict(T=chk.is_real,
                    beta=chk.is_real,
                    alpha=chk.is_real))
    def __init__(self, T, beta, alpha):
        """
        Parameters
        ----------
        T : float
            Function support.
        beta : float
            Function mid-point.
        alpha : float
           Normalized decay-rate.
        """
        super().__init__()
        self._beta = beta

        if not (T > 0):
            raise ValueError('Parameter[T] must be positive.')
        self._T = T

        if not (0 <= alpha <= 1):
            raise ValueError('Parameter[alpha] must be in [0, 1].')
        self._alpha = alpha

    @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def __call__(self, x):
        """
        Sample the Tukey(T, beta, alpha) function.

        Parameters
        ----------
        x : float or array-like(float)
            Sample points.

        Returns
        -------
        Tukey(T, beta, alpha)(x) : :py:class:`~numpy.ndarray`
        """
        x = np.array(x, copy=False)

        y = x - self._beta + self._T / 2
        left_lim = float(self._T * self._alpha / 2)
        right_lim = float(self._T - (self._T * self._alpha / 2))

        ramp_up = (0 <= y) & (y < left_lim)
        body = (left_lim <= y) & (y <= right_lim)
        ramp_down = (right_lim < y) & (y <= self._T)

        amplitude = np.zeros_like(x)
        amplitude[body] = 1
        if not np.isclose(self._alpha, 0):
            amplitude[ramp_up] = np.sin(np.pi / (self._T * self._alpha) *
                                        y[ramp_up]) ** 2
            amplitude[ramp_down] = np.sin(np.pi / (self._T * self._alpha) *
                                          (self._T - y[ramp_down])) ** 2
        return amplitude


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
            theta_max = 180 * u.deg
            while True:
                x = np.linspace(0, theta_max, N_samples)
                cx = np.cos(x).value
                cy = self._exact_kernel(cx)
                zero_cross = np.diff(np.sign(cy))
                N_cross = np.abs(np.sign(zero_cross)).sum()

                if N_cross > 10:
                    theta_max /= 2
                else:
                    break

            window = Tukey(T=2 - 2 * np.cos(2 * theta_max.to_value(u.rad)),
                           beta=1,
                           alpha=0.5)

            x = np.r_[np.linspace(np.cos(theta_max * 2), np.cos(theta_max),
                                  N_samples, endpoint=False),
                      np.linspace(np.cos(theta_max), 1, N_samples)].value
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
        x : float or array-like(float)
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
