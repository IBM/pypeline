# ##############################################################################
# _sphere.py
# ==========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# ##############################################################################

import numpy as np
import tqdm

import pypeline.core as core
import pypeline.util.argcheck as chk
import pypeline.util.math.func as func
import _pypeline_util_math_sphere_pybind11 as sph


@chk.check('N', chk.is_integer)
def ea_sample(N):
    r"""
    Open grid of Equal-Angle sample-points on the sphere.

    Parameters
    ----------
    N : int
        Order of the grid, i.e. there will be :math:`4 (N + 1)^{2}` points on the sphere.

    Returns
    -------
    colat : :py:class:`~numpy.ndarray`
        (2N + 2, 1) polar angles [rad].

    lon : :py:class:`~numpy.ndarray`
        (1, 2N + 2) azimuthal angles [rad].

    Examples
    --------
    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on the sphere:

    .. testsetup::

       import numpy as np
       from pypeline.util.math.sphere import ea_sample

    .. doctest::

       >>> N = 3
       >>> colat, lon = ea_sample(N)

       >>> np.around(colat, 2)
       array([[0.2 ],
              [0.59],
              [0.98],
              [1.37],
              [1.77],
              [2.16],
              [2.55],
              [2.95]])
       >>> np.around(lon, 2)
       array([[0.  , 0.79, 1.57, 2.36, 3.14, 3.93, 4.71, 5.5 ]])


    Notes
    -----
    The sample positions on the unit sphere are given (in radians) by [1]_:

    .. math::

       \theta_{q} & = \frac{\pi}{2 N + 2} \left( q + \frac{1}{2} \right), \qquad & q \in \{ 0, \ldots, 2 N + 1 \},

       \phi_{l} & = \frac{2 \pi}{2N + 2} l, \qquad & l \in \{ 0, \ldots, 2 N + 1 \}.


    .. [1] B. Rafaely, "Fundamentals of Spherical Array Processing", Springer 2015
    """
    if N <= 0:
        raise ValueError('Parameter[N] must be non-negative.')

    _2N2 = 2 * N + 2
    q, l = np.ogrid[:_2N2, :_2N2]

    colat = (np.pi / _2N2) * (0.5 + q)
    lon = (2 * np.pi / _2N2) * l

    return colat, lon


class EqualAngleInterpolator(core.Block):
    r"""
    Interpolate order-limited zonal function from Equal-Angle samples.

    Computes :math:`f(r) = \sum_{q, l} \alpha_{q} f(r_{q, l}) K_{N}(\langle r, r_{q, l} \rangle)`, where :math:`r_{q, l} \in \mathbb{S}^{2}` are points from an Equal-Angle sampling scheme, :math:`K_{N}(\cdot)` is the spherical Dirichlet kernel of order :math:`N`, and the :math:`\alpha_{q}` are scaling factors tailored to an Equal-Angle sampling scheme.

    Examples
    --------
    Let :math:`\gamma_{N}(r): \mathbb{S}^{2} \to \mathbb{R}` be the order-:math:`N` approximation of :math:`\gamma(r) = \delta(r - r_{0})`:

    .. math::

       \gamma_{N}(r) = \frac{N + 1}{4 \pi} \frac{P_{N + 1}(\langle r, r_{0} \rangle) - P_{N}(\langle r, r_{0} \rangle)}{\langle r, r_{0} \rangle -1}.

    As :math:`\gamma_{N}` is order-limited, it can be exactly reconstructed from it's samples on an order-:math:`N` Equal-Angle grid:

    .. testsetup::

       import numpy as np
       from pypeline.util.math.sphere import ea_sample, EqualAngleInterpolator, pol2cart
       from pypeline.util.math.func import SphericalDirichlet

       def gammaN(r, r0, N):
           similarity = np.tensordot(r0, r, axes=1)
           d_func = SphericalDirichlet(N)
           return d_func(similarity) * (N + 1) / (4 * np.pi)

    .. doctest::

       # \gammaN Parameters
       >>> N = 3
       >>> r0 = np.array([0, 0, 1])

       # Interpolate \gammaN from it's samples
       >>> colat, lon = ea_sample(N)
       >>> r = np.stack(pol2cart(1, colat, lon), axis=0)
       >>> g_samples = gammaN(r, r0, N)
       >>> q, l = np.meshgrid(np.arange(colat.size),
       ...                    np.arange(lon.size),
       ...                    indexing='ij')
       >>> ea_interp = EqualAngleInterpolator(q.reshape(-1),
       ...                                    l.reshape(-1),
       ...                                    g_samples.reshape(-1),
       ...                                    N)

       # Compare with exact solution at off-sample positions
       >>> colat, lon = ea_sample(2 * N)  # denser grid
       >>> g_interp = ea_interp(colat, lon)
       >>> g_exact = gammaN(pol2cart(1, colat, lon), r0, N)
       >>> np.allclose(g_interp, g_exact)
       True
    """

    @chk.check(dict(q=chk.has_integers,
                    l=chk.has_integers,
                    f=chk.accept_any(chk.has_reals, chk.has_complex),
                    N=chk.is_integer,
                    approximate_kernel=chk.is_boolean))
    def __init__(self, q, l, f, N, approximate_kernel=False):
        r"""
        Parameters
        ----------
        q : :py:class:`~numpy.ndarray`
            (N_s,) polar indices of an order-`N` Equal-Angle grid.
        l : :py:class:`~numpy.ndarray`
            (N_s,) azimuthal indices of an order-`N` Equal-Angle grid.
        f : :py:class:`~numpy.ndarray`
            (N_s,) samples of the zonal function at data-points. (float or complex)

            :math:`L`-dimensional zonal functions are also supported by supplying an (N_s, L) array instead.
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~pypeline.util.math.func.SphericalDirichlet`.

        Notes
        -----
        If :math:`f(r)` only takes non-negligeable values when :math:`r \in \mathcal{S} \subset \mathbb{S}^{2}`, then the runtime of :py:meth:`~pypeline.util.math.sphere.EqualAngleInterpolator.__call__` can be significantly reduced by only supplying the triplets (`q`, `l`, `f`) that belong to :math:`\mathcal{S}`.
        """
        super().__init__()

        colat_sph, _ = ea_sample(N)
        _2N2 = colat_sph.size
        q, l = np.array(q), np.array(l)
        if not ((q.shape == l.shape) and
                chk.has_shape((q.size,))(q)):
            raise ValueError("Parameter[q, l] must be 1D and of equal length.")
        if not all(np.all(0 <= _) and np.all(_ < _2N2) for _ in [q, l]):
            raise ValueError(f"Parameter[q, l] must contain entries in "
                             f"{{0, ..., 2N + 1}}.")
        self._N = N
        self._q = q
        self._l = l

        N_s = q.size
        f = np.array(f, copy=False)
        if (f.ndim == 1) and (len(f) == N_s):
            self._L = 1
        elif (f.ndim == 2) and (len(f) == N_s):
            self._L = f.shape[1]
        else:
            raise ValueError("Parameter[f] must have shape (N_s,) or (N_s, L).")
        f = f.reshape(N_s, self._L)

        _2m1 = np.reshape(2 * np.r_[:N + 1] + 1, (1, N + 1))
        alpha = (np.sin(colat_sph) / _2N2 *
                 np.sum(np.sin(_2m1 * colat_sph) / _2m1, axis=1, keepdims=True))
        self._weight = f * alpha[q]

        self._kernel_func = func.SphericalDirichlet(N, approximate_kernel)

    @chk.check(dict(colat=chk.accept_any(chk.is_real, chk.has_reals),
                    lon=chk.accept_any(chk.is_real, chk.has_reals)))
    def __call__(self, colat, lon):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        colat : float or :py:class:`~numpy.ndarray`
            Polar/Zenith angle [rad].
        lon : float or :py:class:`~numpy.ndarray`
            Longitude angle [rad].

        Returns
        -------
        :py:class:`~numpy.ndarray`
            (L, ...) function values at specified coordinates.
        """
        r = np.stack(sph.pol2cart(1, colat, lon), axis=0)
        sh_kern = (1,) + r.shape[1:]
        sh_weight = (self._L,) + (1,) * len(r.shape[1:])

        colat_sph, lon_sph = ea_sample(self._N)

        f_interp = np.zeros((self._L,) + r.shape[1:], dtype=self._weight.dtype)
        with tqdm.tqdm(total=len(self._weight)) as pbar:
            for w, t, p in zip(self._weight,
                               colat_sph[self._q, 0],
                               lon_sph[0, self._l]):
                similarity = np.tensordot(np.stack(sph.pol2cart(1, t, p), axis=-1),
                                          r, axes=1)
                kernel = self._kernel_func(similarity)

                f_interp += kernel.reshape(sh_kern) * w.reshape(sh_weight)

                pbar.update()

        return f_interp
