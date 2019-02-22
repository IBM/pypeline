# #############################################################################
# grid.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Pixel-grid generation and utilities for spherical surfaces.
"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import scipy.linalg as linalg

import pypeline.util.argcheck as chk
import pypeline.util.math.linalg as pylinalg
import pypeline.util.math.sphere as sph


@chk.check(dict(direction=chk.require_all(chk.has_reals,
                                          chk.has_shape([3, ])),
                FoV=chk.is_real,
                size=chk.require_all(chk.has_integers,
                                     chk.has_shape([2, ]))))
def spherical_grid(direction, FoV, size):
    """
    Spherical pixel grid.

    Parameters
    ----------
    direction : array-like(float)
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid [rad] centered at `direction`.
    size : array-like(int)
        (N_height, N_width)

        The grid will consist of `N_height` concentric circles around `direction`, each containing `N_width` pixels.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (3, N_height, N_width) pixel grid.
    """
    direction = np.array(direction, dtype=float)
    direction /= linalg.norm(direction)

    if not (0 < np.rad2deg(FoV) <= 179):
        raise ValueError('Parameter[FoV] must be in (0, 179] degrees.')

    size = np.array(size, copy=False)
    if np.any(size <= 0):
        raise ValueError('Parameter[size] must contain positive entries.')

    N_height, N_width = size
    colat, lon = np.meshgrid(np.linspace(0, FoV / 2, N_height),
                             np.linspace(0, 2 * np.pi, N_width),
                             indexing='ij')
    XYZ = sph.pol2cart(1, colat, lon)

    # Center grid at 'direction'
    _, dir_colat, _ = sph.cart2pol(*direction)
    R_axis = np.cross([0, 0, 1], direction)
    if np.allclose(R_axis, 0):
        # R_axis is in span(E_z), so we must manually set R
        R = np.eye(3)
        if direction[2] < 0:
            R[2, 2] = -1
    else:
        R = pylinalg.rot(axis=R_axis, angle=dir_colat)

    XYZ = np.tensordot(R, XYZ, axes=1)
    return XYZ


@chk.check(dict(direction=chk.require_all(chk.has_reals,
                                          chk.has_shape([3, ])),
                FoV=chk.is_real,
                size=chk.require_all(chk.has_integers,
                                     chk.has_shape([2, ]))))
def uniform_grid(direction, FoV, size):
    """
    Uniform pixel grid.

    Parameters
    ----------
    direction : array-like(float)
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid [rad] centered at `direction`.
    size : array-like(int)
        (N_height, N_width)

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (3, N_height, N_width) pixel grid.
    """
    direction = np.array(direction, dtype=float)
    direction /= linalg.norm(direction)

    if not (0 < np.rad2deg(FoV) <= 179):
        raise ValueError('Parameter[FoV] must be in (0, 179] degrees.')

    size = np.array(size, copy=False)
    if np.any(size <= 0):
        raise ValueError('Parameter[size] must contain positive entries.')

    N_height, N_width = size
    lim = np.sin(FoV / 2)
    Y, X = np.meshgrid(np.linspace(-lim, lim, N_height),
                       np.linspace(-lim, lim, N_width),
                       indexing='ij')
    Z = 1 - X ** 2 - Y ** 2
    X[Z < 0], Y[Z < 0], Z[Z < 0] = 0, 0, 0
    Z = np.sqrt(Z)
    XYZ = np.stack([X, Y, Z], axis=0)

    # Center grid at 'direction'
    _, dir_colat, dir_lon = sph.cart2pol(*direction)
    R1 = pylinalg.rot(axis=[0, 0, 1], angle=dir_lon)
    R2_axis = np.cross([0, 0, 1], direction)
    if np.allclose(R2_axis, 0):
        # R2_axis is in span(E_z), so we must manually set R2.
        R2 = np.eye(3)
        if direction[2] < 0:
            R2[2, 2] = -1
    else:
        R2 = pylinalg.rot(axis=R2_axis, angle=dir_colat)
    R = R2 @ R1

    XYZ = np.tensordot(R, XYZ, axes=1)
    return XYZ


@chk.check(dict(direction=chk.require_all(chk.has_reals,
                                          chk.has_shape([3, ])),
                FoV=chk.is_real,
                size=chk.require_all(chk.has_integers,
                                     chk.has_shape([2, ]))))
def ea_grid(direction, FoV, size):
    """
    Equal-Angle pixel grid.

    Parameters
    ----------
    direction : array-like(float)
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid [rad] centered at `direction`.
    size : array-like(int)
        (N_height, N_width)

    Returns
    -------
    colat : :py:class:`~numpy.ndarray`
        (N_height, 1) polar angles [rad].

    lon : :py:class:`~numpy.ndarray`
        (1, N_width) azimuthal angles [rad].
    """
    direction = np.array(direction, dtype=float)
    direction /= linalg.norm(direction)

    if np.allclose(np.cross([0, 0, 1], direction), 0):
        raise ValueError('Generating Equal-Angle grids centered at poles currently not supported.')

    if not (0 < np.rad2deg(FoV) <= 179):
        raise ValueError('Parameter[FoV] must be in (0, 179] degrees.')

    size = np.array(size, copy=False)
    if np.any(size <= 0):
        raise ValueError('Parameter[size] must contain positive entries.')

    _, dir_colat, dir_lon = sph.cart2pol(*direction)
    lim_lon = dir_lon + (FoV / 2) * np.r_[-1, 1]
    lim_colat = dir_colat + (FoV / 2) * np.r_[-1, 1]
    lim_colat = (max(np.deg2rad(0.5), lim_colat[0]),
                 min(lim_colat[1], np.deg2rad(179.5)))

    N_height, N_width = size
    colat = np.linspace(*lim_colat, num=N_height).reshape(-1, 1)
    lon = np.linspace(*lim_lon, num=N_width).reshape(1, -1)
    return colat, lon


@chk.check(dict(direction=chk.require_all(chk.has_reals,
                                          chk.has_shape([3, ])),
                FoV=chk.is_real,
                N=chk.is_integer))
def ea_harmonic_grid(direction, FoV, N):
    """
    Region-limited Equal-Angle pixel grid of order `N`.

    Parameters
    ----------
    direction : array-like(float)
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid [rad] centered at `direction`.
    N : int
        Order of the grid, i.e. there will be :math:`4 (N + 1)^{2}` points on the sphere.

    Returns
    -------
    q : :py:class:`~numpy.ndarray`
        (N_height,) polar indices.

    l : :py:class:`~numpy.ndarray`
        (N_width,) azimuthal indices.

    colat : :py:class:`~numpy.ndarray`
        (N_height, 1) polar angles [rad].

    lon : :py:class:`~numpy.ndarray`
        (1, N_width) azimuthal angles [rad].

    See Also
    --------
    :py:class:`~pypeline.util.math.sphere.EqualAngleInterpolator`
    """
    direction = np.array(direction, dtype=float)
    direction /= linalg.norm(direction)

    if np.allclose(np.cross([0, 0, 1], direction), 0):
        raise ValueError('Generating Equal-Angle grids centered at poles currently not supported.')

    if not (0 < np.rad2deg(FoV) <= 179):
        raise ValueError('Parameter[FoV] must be in (0, 179] degrees.')

    if N <= 0:
        raise ValueError('Parameter[N] must be non-negative.')

    _, dir_colat, dir_lon = sph.cart2pol(*direction)
    lim_lon = dir_lon + (FoV / 2) * np.r_[-1, 1]
    lim_lon = coord.Angle(lim_lon * u.rad).wrap_at(360 * u.deg).to_value(u.rad)
    lim_colat = dir_colat + (FoV / 2) * np.r_[-1, 1]
    lim_colat = (max(np.deg2rad(0.5), lim_colat[0]),
                 min(lim_colat[1], np.deg2rad(179.5)))

    colat_full, lon_full = sph.ea_sample(N)
    q_full = np.arange(colat_full.size).reshape(-1, 1)
    l_full = np.arange(lon_full.size).reshape(1, -1)

    q_mask = ((lim_colat[0] <= colat_full) & (colat_full <= lim_colat[1]))
    if lim_lon[0] < lim_lon[1]:
        l_mask = ((lim_lon[0] <= lon_full) & (lon_full <= lim_lon[1]))
    else:
        l_mask = ((lim_lon[0] <= lon_full) | (lon_full <= lim_lon[1]))

    q = q_full[q_mask]
    l = l_full[l_mask]
    colat = colat_full[q_mask].reshape(-1, 1)
    lon = lon_full[l_mask].reshape(1, -1)
    return q, l, colat, lon


@chk.check(dict(direction=chk.require_all(chk.has_reals,
                                          chk.has_shape([3, ])),
                FoV=chk.is_real,
                N=chk.is_integer))
def fibonacci_harmonic_grid(direction, FoV, N):
    """
    Region-limited Fibonacci pixel grid of order `N`.

    Parameters
    ----------
    direction : array-like(float)
        (3,) vector around which the grid is centered.
    FoV : float
        Span of the grid [rad] centered at `direction`.
    N : int
        Order of the grid, i.e. there will be :math:`4 (N + 1)^{2}` points on the sphere.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        (3, N_px) pixel grid.

    See Also
    --------
    :py:class:`~pypeline.util.math.sphere.FibonacciInterpolator`
    """
    direction = np.array(direction, dtype=float)
    direction /= linalg.norm(direction)

    if not (0 < FoV < 2 * np.pi):
        raise ValueError('Parameter[FoV] must be in (0, 360) degrees.')

    if N <= 0:
        raise ValueError('Parameter[N] must be non-negative.')

    # TODO: Current grid generation is highly inefficient when interested in small FoVs
    XYZ = sph.fibonacci_sample(N)

    min_similarity = np.cos(FoV / 2)
    mask = (direction @ XYZ) >= min_similarity
    XYZ = XYZ[:, mask]

    return XYZ
