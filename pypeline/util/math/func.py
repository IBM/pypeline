# #############################################################################
# func.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
1d functions not available in `SciPy <https://www.scipy.org/>`_.
"""

import numpy as np
import scipy.interpolate as interpolate
import scipy.special as sp

import pypeline.util.argcheck as chk


@chk.check(dict(T=chk.is_real,
                beta=chk.is_real,
                alpha=chk.is_real))
def tukey(T, beta, alpha):
    r"""
    Parameterized Tukey function.

    Parameters
    ----------
    T : float
        Function support.
    beta : float
        Function mid-point.
    alpha : float
        Normalized decay-rate.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Function that outputs the amplitude of the parameterized Tukey function at specified locations.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.func import tukey

    .. doctest::

       >>> f = tukey(T=1, beta=0.5, alpha=0.25)
       >>> sample_points = np.linspace(0, 1, 25).reshape(5, 5)  # any shape

       >>> amplitudes = f(sample_points)
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
    if not (T > 0):
        raise ValueError('Parameter[T] must be positive.')
    if not (0 <= alpha <= 1):
        raise ValueError('Parameter[alpha] must be in [0, 1].')

    @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def tukey_func(x):
        x = np.array(x, copy=False)

        y = x - beta + T / 2
        if np.isclose(alpha, 0):
            left_lim = 0.
            right_lim = T
        else:
            left_lim = T * alpha / 2
            right_lim = T - (T * alpha / 2)

        ramp_up = (0 <= y) & (y < left_lim)
        body = (left_lim <= y) & (y <= right_lim)
        ramp_down = (right_lim < y) & (y <= T)

        amplitude = np.zeros_like(x)
        amplitude[body] = 1
        if not np.isclose(alpha, 0):
            amplitude[ramp_up] = np.sin(np.pi / (T * alpha) *
                                        y[ramp_up]) ** 2
            amplitude[ramp_down] = np.sin(np.pi / (T * alpha) *
                                          (T - y[ramp_down])) ** 2
        return amplitude

    tukey_func.__doc__ = (rf"""
        Tukey(T={T}, beta={beta}, alpha={alpha}) function.

        Parameters
        ----------
        x : float or array-like(float)
            Values at which to compute Tukey({T}, {beta}, {alpha})(x).

        Returns
        -------
        :py:class:`~numpy.ndarray`
        """)

    return tukey_func


@chk.check(dict(N=chk.is_integer,
                interp=chk.is_boolean))
def sph_dirichlet(N, interp=False):
    r"""
    Parameterized spherical Dirichlet kernel.

    Parameters
    ----------
    N : int
        Kernel order.
    interp : bool
        Use a cubic-spline to compute kernel values.
        This method can only be used if `N` is greater that 50, but is orders-of-magnitude faster than the exact computation while retaining high accuracy.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Function that outputs the amplitude of the parameterized spherical Dirichlet function at specified locations.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.func import sph_dirichlet

    .. doctest::

       >>> N = 4
       >>> f = sph_dirichlet(N)
       >>> sample_points = np.linspace(-1, 1, 25).reshape(5, 5)  # any shape

       >>> amplitudes = f(sample_points)
       >>> np.around(amplitudes, 2)
       array([[ 1.  ,  0.2 , -0.25, -0.44, -0.44],
              [-0.32, -0.13,  0.07,  0.26,  0.4 ],
              [ 0.47,  0.46,  0.38,  0.22,  0.  ],
              [-0.24, -0.48, -0.67, -0.76, -0.68],
              [-0.37,  0.27,  1.3 ,  2.84,  5.  ]])

    Notes
    -----
    The spherical Dirichlet function :math:`K_{N}(t): [-1, 1] \to \mathbb{R}` is defined as:

    .. math:: K_{N}(t) = \frac{P_{N+1}(t) - P_{N}(t)}{t - 1},

    where :math:`P_{N}(t)` is the `Legendre polynomial <https://en.wikipedia.org/wiki/Legendre_polynomials>`_ of order :math:`N`.
    """
    if N < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
    def exact_func(x):
        x = np.array(x, copy=False, dtype=float)

        if not np.all((-1 <= x) & (x <= 1)):
            raise ValueError('Parameter[x] must lie in [-1, 1].')

        amplitude = np.zeros_like(x)
        _1_mask = np.isclose(x, 1)

        amplitude[_1_mask] = N + 1
        amplitude[~_1_mask] = (sp.eval_legendre(N + 1, x[~_1_mask]) -
                               sp.eval_legendre(N, x[~_1_mask]))
        amplitude[~_1_mask] /= x[~_1_mask] - 1

        return amplitude

    sph_dirichlet_func = exact_func
    if interp:
        if N < 50:
            raise ValueError('Cannot use interpolation method if '
                             'Parameter[N] < 50.')

        x_boundary = 1 - 10 / N
        x_l = np.linspace(-1, -x_boundary, 10 * N, endpoint=False)
        x_m = np.linspace(-x_boundary, x_boundary, 10 * N, endpoint=False)
        x_r = np.linspace(x_boundary, 1, 50 * N)
        x_all = np.unique(np.r_[x_l, x_m, x_r])
        x_all = x_all[~np.isclose(x_all, 1)]
        y = exact_func(x_all)

        f = interpolate.interp1d(x_all, y,
                                 kind='cubic',
                                 bounds_error=False,
                                 fill_value=N + 1)

        @chk.check('x', chk.accept_any(chk.is_real, chk.has_reals))
        def interp_func(x):
            x = np.array(x, copy=False, dtype=float)

            if not np.all((-1 <= x) & (x <= 1)):
                raise ValueError('Parameter[x] must lie in [-1, 1].')

            amplitude = f(x)

            return amplitude

        sph_dirichlet_func = interp_func

    sph_dirichlet_func.__doc__ = (rf"""
        Order-{N} spherical Dirichlet kernel.

        Parameters
        ----------
        x : float or array-like(float)
            Values at which to compute :math:`K_{{{N}}}(x)`.

        Returns
        -------
        :py:class:`~numpy.ndarray`
        """)

    return sph_dirichlet_func
