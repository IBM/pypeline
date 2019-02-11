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
    colat : :py:class:`~astropy.units.Quantity`
        (2N + 2, 1) polar angles.

    lon : :py:class:`~astropy.units.Quantity`
        (1, 2N + 2) azimuthal angles.

    Examples
    --------
    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on the sphere:

    .. testsetup::

       import numpy as np
       import astropy.units as u
       from pypeline.util.math.sphere import ea_sample

    .. doctest::

       >>> N = 3
       >>> colat, lon = ea_sample(N)

       >>> np.around(colat.to_value(u.rad), 2)
       array([[0.2 ],
              [0.59],
              [0.98],
              [1.37],
              [1.77],
              [2.16],
              [2.55],
              [2.95]])
       >>> np.around(lon.to_value(u.rad), 2)
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

    colat = (np.pi / _2N2) * (0.5 + q) * u.rad
    lon = (2 * np.pi / _2N2) * l * u.rad

    return colat, lon


class Interpolator(core.Block):
    r"""
    Interpolate order-limited zonal function from spatial samples.

    Computes :math:`f(r) = \sum_{q} \alpha_{q} f(r_{q}) K_{N}(\langle r, r_{q} \rangle)`, where :math:`r_{q} \in \mathbb{S}^{2}`
    are points from a spatial sampling scheme, :math:`K_{N}(\cdot)` is the spherical Dirichlet kernel of order :math:`N`,
    and the :math:`\alpha_{q}` are scaling factors tailored to the sampling scheme.
    """

    @chk.check(dict(weight=chk.has_reals,
                    support=chk.has_reals,
                    N=chk.is_integer,
                    approximate_kernel=chk.is_boolean))
    def __init__(self, weight, support, N, approximate_kernel=False):
        r"""
        Parameters
        ----------
        weight : :py:class:`~numpy.ndarray`
            (N_s,) weighting scheme to apply to each support point.
        support : :py:class:`~numpy.ndarray`
            (3, N_s) critical support points for a given sampling scheme.
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~pypeline.util.math.func.SphericalDirichlet`.

        Notes
        -----
        * `weight` corresponds to :math:`\alpha_{ql}` in :ref:`ZOL_def`.

        * `support` corresponds to :math:`r_{ql}` in :ref:`ZOL_def`.
        """
        super().__init__()
        if not (weight.ndim == 1):
            raise ValueError('Parameter[weight] must be (N_s,).')
        N_s = len(weight)
        self._weight = weight

        if not (support.shape == (3, N_s)):
            raise ValueError('Parameter[support] must be (3, N_s).')
        self._support = support

        if not (N > 0):
            raise ValueError('Parameter[N] must be positive.')
        self._N = N

        self._kernel_func = func.SphericalDirichlet(N, approximate_kernel)

    @chk.check(dict(f=chk.accept_any(chk.has_reals, chk.has_complex),
                    r=chk.has_reals,
                    sparse_kernel=chk.is_boolean))
    def __call__(self, f, r, sparse_kernel=False):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        f : :py:class:`~numpy.ndarray`
            (N_s,) samples of the zonal function at data-points. (float or complex)
            :math:`L`-dimensional zonal functions are also supported by supplying an (L, N_s) array instead.
        r : :py:class:`~numpy.ndarray`
            (3, N_px) evaluation points.
        sparse_kernel : bool
            If :py:obj:`True`, take advantage of kernel sparsity to perform localised operations.

        Returns
        -------
        f_interp : :py:class:`~numpy.ndarray`
            (L, N_px) function values at specified coordinates.


        .. todo::

           As currently implemented, using `sparse_kernel=True` is slower than direct evaluation.
           Need to look into this when required.
           self._kernel_func._zero_threshold may be useful here.
        """
        N_s = len(self._weight)
        if f.shape == (N_s,):
            f = f.reshape(1, N_s)
        elif (f.ndim == 2) and (f.shape[1] == N_s):
            pass
        else:
            raise ValueError('Parameter[f] must have shape (N_s,) or (L, N_s).')

        if not ((r.ndim == 2) and (r.shape[0] == 3)):
            raise ValueError('Parameter[r] must have shape (3, N_px).')
        # N_px = r.shape[1]

        if sparse_kernel:
            kernel = self._kernel_func(np.tensordot(self._support.T, r, axes=1)) * ((self._N + 1) / (4 * np.pi))
            kernel = sp.csc_matrix(kernel)
            f_interp = kernel.T.dot((self._weight * f).T).T
        else:
            kernel = self._kernel_func(np.tensordot(self._support.T, r, axes=1)) * ((self._N + 1) / (4 * np.pi))
            f_interp = (self._weight * f) @ kernel
        return f_interp


class EqualAngleInterpolator(Interpolator):
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
       from pypeline.util.math.func import SphericalDirichlet
       from pypeline.util.math.sphere import ea_sample, pol2cart, EqualAngleInterpolator

       def gammaN(r, r0, N):
           similarity = np.tensordot(r0, r, axes=1)
           d_func = SphericalDirichlet(N)
           return d_func(similarity) * (N + 1) / (4 * np.pi)

    .. doctest::

       # \gammaN Parameters
       >>> N = 3
       >>> r0 = np.array([0, 0, 1])

       # Solution at Nyquist resolution
       >>> colat, lon = ea_sample(N)
       >>> r = pol2cart(1, colat, lon)
       >>> g_nyquist = gammaN(pol2cart(1, colat, lon), r0, N)

       # Solution at high resolution
       >>> colat, lon = ea_sample(2 * N)  # dense grid
       >>> g_exact = gammaN(pol2cart(1, colat, lon), r0, N)

       # Interpolate Nyquist solution to high resolution solution
       >>> ea_interp = EqualAngleInterpolator(N)
       >>> g_samples = g_nyquist.reshape(-1)
       >>> g_interp = ea_interp(g_samples, colat, lon)

       >>> np.allclose(g_interp, g_exact)
       True
    """

    @chk.check(dict(N=chk.is_integer,
                    approximate_kernel=chk.is_boolean))
    def __init__(self, N, approximate_kernel=False):
        """
        Parameters
        ----------
        N : int
            Order of the reconstructed zonal function.
        approximate_kernel : bool
            If :py:obj:`True`, pass the `approx` option to :py:class:`~pypeline.util.math.func.SphericalDirichlet`.
        """
        colat, lon = ea_sample(N)
        support = pol2cart(1, colat, lon).reshape(3, -1)
        N_colat, N_lon, N_s = colat.size, lon.size, support.shape[1]

        colat = colat.to_value(u.rad)
        a = np.arange(N + 1)
        weight = np.sum(np.sin((2 * a + 1) * colat) / (2 * a + 1), axis=1, keepdims=True) * np.sin(colat)
        weight = np.broadcast_to(weight, (N_colat, N_lon)).reshape(N_s)
        weight *= (2 * np.pi) / ((N + 1) ** 2)

        super().__init__(weight, support, N, approximate_kernel)

    @chk.check(dict(f=chk.accept_any(chk.has_reals, chk.has_complex),
                    colat=chk.accept_any(chk.is_angle, chk.has_angles),
                    lon=chk.accept_any(chk.is_angle, chk.has_angles),
                    sparse_kernel=chk.is_boolean))
    def __call__(self, f, colat, lon, sparse_kernel=False):
        """
        Interpolate function samples at order `N`.

        Parameters
        ----------
        f : :py:class:`~numpy.ndarray`
            (N_s,) samples of the zonal function at data-points. (float or complex)
            :math:`L`-dimensional zonal functions are also supported by supplying an (L, N_s) array instead.
        colat : :py:class:`~astropy.units.Quantity`
            (A, 1) Polar/Zenith angle.
        lon : :py:class:`~astropy.units.Quantity`
            (1, B) Longitude angle.
        sparse_kernel : bool
            If :py:obj:`True`, take advantage of kernel sparsity to perform localised operations.

        Returns
        -------
        f_interp : :py:class:`~numpy.ndarray`
            (L, A, B) function values at specified coordinates.
        """
        A, B = colat.size, lon.size
        if not (colat.shape == (A, 1) and (lon.shape == (1, B))):
            raise ValueError('Parameters[colat, lon] are ill-formed.')

        if f.ndim == 1:
            f = f.reshape(1, -1)
        L, N_s = len(f), (2 * self._N + 2) ** 2
        if f.shape != (L, N_s):
            raise ValueError('Parameter[f] inconsistent with interpolation order N.')

        r = pol2cart(1, colat, lon).reshape(3, A * B)
        f_interp = super().__call__(f, r, sparse_kernel)
        f_interp = f_interp.reshape(L, A, B)
        return f_interp


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                colat=chk.accept_any(chk.is_angle, chk.has_angles),
                lon=chk.accept_any(chk.is_angle, chk.has_angles)))
def pol2eq(r, colat, lon):
    """
    Polar coordinates to Equatorial coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    colat : :py:class:`~astropy.units.Quantity`
        Polar/Zenith angle.
    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    lat : :py:class:`~astropy.units.Quantity`
        Elevation angle.

    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.
    """
    lat = (90 * u.deg) - colat
    return r, lat, lon


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                lat=chk.accept_any(chk.is_angle, chk.has_angles),
                lon=chk.accept_any(chk.is_angle, chk.has_angles)))
def eq2pol(r, lat, lon):
    """
    Equatorial coordinates to Polar coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    lat : :py:class:`~astropy.units.Quantity`
        Elevation angle.
    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.

    Returns
    -------
    r : :py:class:`~numpy.ndarray`
        Radius.

    colat : :py:class:`~astropy.units.Quantity`
        Polar/Zenith angle.

    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.
    """
    colat = (90 * u.deg) - lat
    return r, colat, lon


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                lat=chk.accept_any(chk.is_angle, chk.has_angles),
                lon=chk.accept_any(chk.is_angle, chk.has_angles)))
def eq2cart(r, lat, lon):
    """
    Equatorial coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    lat : :py:class:`~astropy.units.Quantity`
        Elevation angle.
    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import astropy.units as u
       from pypeline.util.math.sphere import eq2cart

    .. doctest::

       >>> xyz = eq2cart(1, 0 * u.deg, 0 * u.deg)
       >>> np.around(xyz, 2)
       array([[1.],
              [0.],
              [0.]])
    """
    r = np.array([r]) if chk.is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (coord.SphericalRepresentation(lon, lat, r)
           .to_cartesian()
           .xyz
           .to_value(u.dimensionless_unscaled))

    return XYZ


@chk.check(dict(r=chk.accept_any(chk.is_real, chk.has_reals),
                colat=chk.accept_any(chk.is_angle, chk.has_angles),
                lon=chk.accept_any(chk.is_angle, chk.has_angles)))
def pol2cart(r, colat, lon):
    """
    Polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float or array-like(float)
        Radius.
    colat : :py:class:`~astropy.units.Quantity`
        Polar/Zenith angle.
    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, ...) Cartesian XYZ coordinates.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import astropy.units as u
       from pypeline.util.math.sphere import pol2cart

    .. doctest::

       >>> xyz = pol2cart(1, 0 * u.deg, 0 * u.deg)
       >>> np.around(xyz, 2)
       array([[0.],
              [0.],
              [1.]])
    """
    lat = (90 * u.deg) - colat
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

    colat : :py:class:`~astropy.units.Quantity`
        Polar/Zenith angle.

    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import astropy.units as u
       from pypeline.util.math.sphere import cart2pol

    .. doctest::

       >>> r, colat, lon = cart2pol(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

       >>> np.around(r, 2)
       1.0

       >>> np.around(colat.to_value(u.deg), 2)
       45.0

       >>> np.around(lon.to_value(u.deg), 2)
       90.0
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to(u.rad)
    lon = u.Quantity(sph.lon).to(u.rad)

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

    lat : :py:class:`~astropy.units.Quantity`
        Elevation angle.

    lon : :py:class:`~astropy.units.Quantity`
        Longitude angle.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import astropy.units as u
       from pypeline.util.math.sphere import cart2eq

    .. doctest::

       >>> r, lat, lon = cart2eq(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

       >>> np.around(r, 2)
       1.0

       >>> np.around(lat.to_value(u.deg), 2)
       45.0

       >>> np.around(lon.to_value(u.deg), 2)
       90.0
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (90 * u.deg) - colat
    return r, lat, lon
