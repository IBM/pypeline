# ##############################################################################
# sphere.py
# =========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# ##############################################################################

"""
Spherical geometry tools.
"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import scipy.sparse as sp

import pypeline.core as core
import pypeline.util.argcheck as chk
import pypeline.util.math.func as func


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


@chk.check('N', chk.is_integer)
def fibonacci_sample(N):
    r"""
    Grid of Fibonacci sample-points on the sphere.

    Parameters
    ----------
    N : int
        Order of the grid, i.e. there will be :math:`4 (N + 1)^{2}` points on the sphere.

    Returns
    -------
    R : :py:class:`~numpy.ndarray`
        (3, N_px) support points.

    Examples
    --------
    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on the sphere:

    .. testsetup::

       import numpy as np
       from pypeline.util.math.sphere import fibonacci_sample

    .. doctest::

       >>> N = 3
       >>> R = fibonacci_sample(N)

       >>> np.around(R, 2)


    Notes
    -----
    The sample positions on the unit sphere are given (in radians) by [2]_:

    .. math::

       \cos(\theta_{q}) & = 1 - \frac{2 q + 1}{4 (N + 1)^{2}}, \qquad & q \in \{ 0, \ldots, 4 (N + 1)^{2} - 1 \},

       \phi_{q} & = \frac{4 \pi}{1 + \sqrt{5}} q, \qquad & q \in \{ 0, \ldots, 4 (N + 1)^{2} - 1 \}.


    .. [2] B. Rafaely, "Fundamentals of Spherical Array Processing", Springer 2015
    """
    if N <= 0:
        raise ValueError('Parameter[N] must be non-negative.')

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)
    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = ((4 * np.pi) / (1 + np.sqrt(5))) * n
    XYZ = np.stack(pol2cart(1, colat, lon), axis=0)

    return XYZ


class Interpolator(core.Block):
    r"""
    Interpolate order-limited zonal function from spatial samples.

    Computes :math:`f(r) = \sum_{q} \alpha_{q} f(r_{q}) K_{N}(\langle r, r_{q} \rangle)`, where :math:`r_{q} \in \mathbb{S}^{2}`
    are points from a spatial sampling scheme, :math:`K_{N}(\cdot)` is the spherical Dirichlet kernel of order :math:`N`,
    and the :math:`\alpha_{q}` are scaling factors tailored to the sampling scheme.
    """

    @chk.check(dict(N=chk.is_integer,
                    approximate_kernel=chk.is_boolean))
    def __init__(self, N, approximate_kernel=False):
        r"""
        Parameters
        ----------
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~pypeline.util.math.func.SphericalDirichlet`.
        """
        super().__init__()

        if not (N > 0):
            raise ValueError('Parameter[N] must be positive.')
        self._N = N
        self._kernel_func = func.SphericalDirichlet(N, approximate_kernel)

    @chk.check(dict(weight=chk.has_reals,
                    support=chk.has_reals,
                    f=chk.accept_any(chk.has_reals, chk.has_complex),
                    r=chk.has_reals,
                    sparsity_mask=chk.allow_None(chk.require_all(chk.is_instance(sp.spmatrix),
                                                                 lambda _: np.issubdtype(_.dtype, np.bool_)))))
    def __call__(self, weight, support, f, r, sparsity_mask=None):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        weight : :py:class:`~numpy.ndarray`
            (N_s,) weights to apply per support point.
        support : :py:class:`~numpy.ndarray`
            (3, N_s) critical support points.
        f : :py:class:`~numpy.ndarray`
            (L, N_s) zonal function values at support points. (float or complex)
        r : :py:class:`~numpy.ndarray`
            (3, N_px) evaluation points.
        sparsity_mask : :py:class:`~scipy.sparse.spmatrix`
            (N_s, N_px) sparsity mask to perform localized kernel evaluation.

        Returns
        -------
        f_interp : :py:class:`~numpy.ndarray`
            (L, N_px) function values at specified coordinates.
        """
        if not (weight.shape == (weight.size,)):
            raise ValueError('Parameter[weight] must have shape (N_s,).')
        N_s = weight.size

        if not (support.shape == (3, N_s)):
            raise ValueError('Parameter[support] must have shape (3, N_s).')

        L = len(f)
        if not (f.shape == (L, N_s)):
            raise ValueError('Parameter[f] must have shape (L, N_s).')

        if not ((r.ndim == 2) and (r.shape[0] == 3)):
            raise ValueError('Parameter[r] must have shape (3, N_px).')
        N_px = r.shape[1]

        if sparsity_mask is not None:
            if not (sparsity_mask.shape == (N_s, N_px)):
                raise ValueError('Parameter[sparsity_mask] must have shape (N_s, N_px).')

        if sparsity_mask is None:  # Dense evaluation
            kernel = self._kernel_func(support.T @ r)
            beta = f * weight
            f_interp = beta @ kernel
        else:  # Sparse evaluation
            raise NotImplementedError
            # kernel = sp.csc_matrix(sparsity_mask.tocsc(), dtype=np.float)  # (N_px, N_s) CSC
            # for i in range(N_s):
            #     kernel[:, i] = self._kernel_func()  # compute sparse kernel
            # beta = (f * weight).T
            # f_interp = kernel.dot(beta).T

        return f_interp


class EqualAngleInterpolator(Interpolator):
    r"""
    Interpolate order-limited zonal function from Equal-Angle samples.

    Computes :math:`f(r) = \sum_{q, l} \alpha_{q} f(r_{q, l}) K_{N}(\langle r, r_{q, l} \rangle)`, where
    :math:`r_{q, l} \in \mathbb{S}^{2}` are points from an Equal-Angle sampling scheme, :math:`K_{N}(\cdot)` is the
    spherical Dirichlet kernel of order :math:`N`, and the :math:`\alpha_{q}` are scaling factors tailored to an
    Equal-Angle sampling scheme.

    Examples
    --------
    Let :math:`\gamma_{N}(r): \mathbb{S}^{2} \to \mathbb{R}` be the order-:math:`N` approximation of :math:`\gamma(r) = \delta(r - r_{0})`:

    .. math::

       \gamma_{N}(r) = \frac{N + 1}{4 \pi} \frac{P_{N + 1}(\langle r, r_{0} \rangle) - P_{N}(\langle r, r_{0} \rangle)}{\langle r, r_{0} \rangle -1}.

    As :math:`\gamma_{N}` is order-limited, it can be exactly reconstructed from it's samples on an order-:math:`N` Equal-Angle grid:

    .. testsetup::

       import numpy as np
       from pypeline.util.math.func import SphericalDirichlet
       from pypeline.util.math.sphere import EqualAngleInterpolator, ea_sample, pol2cart

       def gammaN(r, r0, N):
           similarity = np.tensordot(r0, r, axes=1)
           d_func = SphericalDirichlet(N)
           return d_func(similarity) * (N + 1) / (4 * np.pi)

    .. doctest::

       # \gammaN Parameters
       >>> N = 3
       >>> r0 = np.array([1, 0, 0])

       # Solution at Nyquist resolution
       >>> colat_nyquist, lon_nyquist = ea_sample(N)
       >>> N_colat, N_lon = colat_nyquist.size, lon_nyquist.size
       >>> R_nyquist = pol2cart(1, colat_nyquist, lon_nyquist)
       >>> g_nyquist = gammaN(R_nyquist, r0, N)

       # Solution at high resolution
       >>> colat_dense, lon_dense = ea_sample(2 * N)
       >>> R_dense = pol2cart(1, colat_dense, lon_dense).reshape(3, -1)
       >>> g_exact = gammaN(R_dense, r0, N)

       >>> ea_interp = EqualAngleInterpolator(N)
       >>> g_interp = ea_interp(colat_idx=np.arange(2 * (N + 1)),
       ...                      lon_idx=np.arange(2 * (N + 1)),
       ...                      f=g_nyquist.reshape(1, N_colat, N_lon),
       ...                      r=R_dense)

       >>> np.allclose(g_exact, g_interp)
       True
    """

    @chk.check(dict(N=chk.is_integer,
                    approximate_kernel=chk.is_boolean))
    def __init__(self, N, approximate_kernel=False):
        r"""
        Parameters
        ----------
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~pypeline.util.math.func.SphericalDirichlet`.
        """
        super().__init__(N, approximate_kernel)

    @chk.check(dict(colat_idx=chk.has_integers,
                    lon_idx=chk.has_integers,
                    f=chk.accept_any(chk.has_reals, chk.has_complex),
                    r=chk.has_reals,
                    sparsity_mask=chk.allow_None(chk.require_all(chk.is_instance(sp.spmatrix),
                                                                 lambda _: np.issubdtype(_.dtype, np.bool_)))))
    def __call__(self, colat_idx, lon_idx, f, r, sparsity_mask=None):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        colat_idx : :py:class:`~numpy.ndarray`
            (N_colat,) polar indices from :py:func:`~pypeline.phased_array.util.grid.ea_harmonic_grid`.
        lon_idx : :py:class:`~numpy.ndarray`
            (N_lon,) azimuthal indices from :py:func:`~pypeline.phased_array.util.grid.ea_harmonic_grid`.
        f : :py:class:`~numpy.ndarray`
            (L, N_colat, N_lon) zonal function values at support points. (float or complex)
        r : :py:class:`~numpy.ndarray`
            (3, N_px) evaluation points.
        sparsity_mask : :py:class:`~scipy.sparse.spmatrix`
            (N_colat * N_lon, N_px) sparsity mask to perform localized kernel evaluation.

        Returns
        -------
        f_interp : :py:class:`~numpy.ndarray`
            (L, N_px) function values at specified coordinates.
        """
        N_colat = colat_idx.size
        if not (colat_idx.shape == (N_colat,)):
            raise ValueError('Parameter[colat_idx] must have shape (N_colat,).')

        N_lon = lon_idx.size
        if not (lon_idx.shape == (N_lon,)):
            raise ValueError('Parameter[lon_idx] must have shape (N_lon,).')

        L = len(f)
        if not (f.shape == (L, N_colat, N_lon)):
            raise ValueError('Parameter[f] must have shape (L, N_colat, N_lon).')

        if not ((r.ndim == 2) and (r.shape[0] == 3)):
            raise ValueError('Parameter[r] must have shape (3, N_px).')
        N_px = r.shape[1]

        if sparsity_mask is not None:
            if not (sparsity_mask.shape == (N_colat * N_lon, N_px)):
                raise ValueError('Parameter[sparsity_mask] must have shape (N_colat * N_lon, N_px).')

        # Apply weights directly onto `f` to avoid memory blow-up.
        colat, lon = ea_sample(self._N)
        a = np.arange(self._N + 1)
        weight = (np.sum(np.sin((2 * a + 1) * colat[colat_idx]) / (2 * a + 1), axis=1, keepdims=True) *
                  np.sin(colat[colat_idx]) /
                  (2 * self._N + 2))  # (N_colat,)
        fw = f * weight.reshape(1, N_colat, 1)  # (L, N_colat, N_lon)

        f_interp = super().__call__(weight=np.broadcast_to([1], (N_colat * N_lon,)),
                                    support=pol2cart(1, colat[colat_idx], lon[:, lon_idx]).reshape(3, -1),
                                    f=fw.reshape(L, -1),
                                    r=r,
                                    sparsity_mask=sparsity_mask)
        return f_interp


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                colat=chk.accept_any(chk.is_real, chk.has_reals),
                lon=chk.accept_any(chk.is_real, chk.has_reals)))
def pol2eq(r, colat, lon):
    """
    Polar coordinates to Equatorial coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].
    """
    lat = (np.pi / 2) - colat
    return r, lat, lon


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                lat=chk.accept_any(chk.is_real, chk.has_reals),
                lon=chk.accept_any(chk.is_real, chk.has_reals)))
def eq2pol(r, lat, lon):
    """
    Equatorial coordinates to Polar coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].
    """
    colat = (np.pi / 2) - lat
    return r, colat, lon


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                lat=chk.accept_any(chk.is_real, chk.has_reals),
                lon=chk.accept_any(chk.is_real, chk.has_reals)))
def eq2cart(r, lat, lon):
    """
    Equatorial coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.sphere import eq2cart

    .. doctest::

       >>> xyz = eq2cart(1, 0, 0)
       >>> np.around(xyz, 2)
       array([[1.],
              [0.],
              [0.]])
    """
    r = np.array([r]) if chk.is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
           .to_cartesian()
           .xyz
           .to_value(u.dimensionless_unscaled))
    return XYZ


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                colat=chk.accept_any(chk.is_real, chk.has_reals),
                lon=chk.accept_any(chk.is_real, chk.has_reals)))
def pol2cart(r, colat, lon):
    """
    Polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].
    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.sphere import pol2cart

    .. doctest::

       >>> xyz = pol2cart(1, 0, 0)
       >>> np.around(xyz, 2)
       array([[0.],
              [0.],
              [1.]])
    """
    lat = (np.pi / 2) - colat
    return eq2cart(r, lat, lon)


@chk.check(dict(x=chk.accept_any(chk.is_real, chk.has_reals),
                y=chk.accept_any(chk.is_real, chk.has_reals),
                z=chk.accept_any(chk.is_real, chk.has_reals)))
def cart2pol(x, y, z):
    """
    Cartesian coordinates to Polar coordinates.

    Parameters
    ----------
    x : float or array-like(float)
        X coordinate.
    y : float or array-like(float)
        Y coordinate.
    z : float or array-like(float)
        Z coordinate.

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    colat : :py:class:`~numpy.ndarray`
        Polar/Zenith angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.sphere import cart2pol

    .. doctest::

       >>> r, colat, lon = cart2pol(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

       >>> np.around(r, 2)
       1.0

       >>> np.around(np.rad2deg(colat), 2)
       45.0

       >>> np.around(np.rad2deg(lon), 2)
       90.0
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


@chk.check(dict(x=chk.accept_any(chk.is_real, chk.has_reals),
                y=chk.accept_any(chk.is_real, chk.has_reals),
                z=chk.accept_any(chk.is_real, chk.has_reals)))
def cart2eq(x, y, z):
    """
    Cartesian coordinates to Equatorial coordinates.

    Parameters
    ----------
    x : float or array-like(float)
        X coordinate.
    y : float or array-like(float)
        Y coordinate.
    z : float or array-like(float)
        Z coordinate.

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    lat : :py:class:`~numpy.ndarray`
        Elevation angle [rad].

    lon : :py:class:`~numpy.ndarray`
        Longitude angle [rad].

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.util.math.sphere import cart2eq

    .. doctest::

       >>> r, lat, lon = cart2eq(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

       >>> np.around(r, 2)
       1.0

       >>> np.around(np.rad2deg(lat), 2)
       45.0

       >>> np.around(np.rad2deg(lon), 2)
       90.0
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon
