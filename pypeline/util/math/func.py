# #############################################################################
# func.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
1d functions not available in :py:mod:`scipy`.
"""

import numpy as np

import pypeline.util.argcheck as chk


@chk.check(dict(T=chk.is_real,
                beta=chk.is_real,
                alpha=chk.is_real))
def tukey(T, beta, alpha):
    r"""
    Return function to output values of a parameterized Tukey window.

    The Tukey window is defined as:

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

    :param T: [:py:class:`~numbers.Real` > 0] window width.
    :param beta: [:py:class:`~numbers.Real`] window mid-point.
    :param alpha: [:py:class:`~numbers.Real` in [0, 1]] decay rate.
    :return: [:py:obj:`~typing.Callable`] parameterized Tukey function.

    .. testsetup::

       import numpy as np
       from pypeline.util.math.func import tukey

    .. doctest::

       >>> f = tukey(T=1, beta=0.5, alpha=0.25)
       >>> sample_points = np.linspace(0, 1, 25)
       >>> amplitudes = f(sample_points)
       >>> np.around(amplitudes, 2)
       array([0.  , 0.25, 0.75, 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
              1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
              0.75, 0.25, 0.  ])

       >>> amplitudes = f(sample_points.reshape(5, 5))  # multi-dim arrays
       >>> np.around(amplitudes, 2)
       array([[0.  , 0.25, 0.75, 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 1.  , 1.  , 1.  ],
              [1.  , 1.  , 0.75, 0.25, 0.  ]])
    """
    if not (T > 0):
        raise ValueError('Parameter[T] must be positive.')
    if not (0 <= alpha <= 1):
        raise ValueError('Parameter[alpha] must be in [0, 1].')

    @chk.check('x', lambda _: chk.has_reals(_) or chk.is_real(_))
    def window_func(x) -> np.ndarray:
        """
        Return the Tukey(T, beta, alpha) function.

        :param x: [:py:class:`~numpy.ndarray`, :py:class:`~numbers.Real`]
            sample points.
        :return: [:py:class:`~numpy.ndarray`] Tukey(T, beta, alpha)(x).
        """
        x = np.array(x, copy=False)

        y = x - beta + T / 2
        if np.isclose(alpha, 0):
            left_lim = 0
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

    return window_func
