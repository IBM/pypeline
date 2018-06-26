# #############################################################################
# image.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Image containers, visualization and export facilities.
"""

import astropy.coordinates as coord
import astropy.io.fits as fits
import astropy.units as u
import matplotlib.axes as axes
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pyproj
import scipy.linalg as linalg
from astropy.wcs import WCS

import pypeline.phased_array.util.data_gen.sky as sky
import pypeline.util.argcheck as chk
import pypeline.util.math.sphere as sph
import pypeline.util.plot as plot


@chk.check('file_name', chk.is_instance(str))
def from_fits(file_name):
    """
    Load image from FITS file.

    Images must have been saved by calling :py:meth:`~pypeline.phased_array.util.io.image.SphericalImage.to_fits`.

    Parameters
    ----------
    file_name : str
        Name of file.

    Returns
    -------
    I : :py:class:`~pypeline.phased_array.util.io.image.SphericalImage`
    """
    with fits.open(file_name, mode='readonly',
                   memmap=True, lazy_load_hdus=True) as hdulist:
        # PrimaryHDU: grid / class info
        primary_hdu = hdulist[0]
        image_hdu = hdulist['IMAGE']
        klass = globals()[primary_hdu.header['IMG_TYPE']]

        I = klass._from_fits(primary_hdu, image_hdu)
        return I


class SphericalImage:
    """
    Container for storing real-valued images defined on :math:`\mathbb{S}^{2}`.

    Main features:

    * import images from FITS format;
    * export images to FITS format;
    * advanced 2D plotting based on `Matplotlib <https://matplotlib.org/>`_;
    * view exported images with `DS9 <http://ds9.si.edu/site/Home.html>`_.

    Examples
    --------
    .. doctest::

       import astropy.units as u
       import numpy as np
       import pypeline.phased_array.util.grid as grid
       import pypeline.phased_array.util.io.image as image
       import pypeline.util.math.sphere as sph
       import pypeline.util.math.stat as stat

       # grid settings =======================
       direction = sph.eq2cart(1, lat=30 * u.deg, lon=20 * u.deg).reshape(-1)
       FoV = 60 * u.deg
       N_height, N_width = 256, 384
       px_grid = grid.uniform_grid(direction, FoV, size=[N_height, N_width])

       # data settings =======================
       beta0, a0 = 0.7, [1, 1, 1]
       beta1, a1 = 0.9, [0, 0, 1]
       kent0 = stat.Kent(k=stat.Kent.min_scale(FoV, beta0) * 2,
                         beta=beta0,
                         g1=direction,
                         a=a0)
       kent1 = stat.Kent(k=stat.Kent.min_scale(FoV, beta1) * 2,
                         beta=beta1,
                         g1=direction,
                         a=a1)

       data0 = (kent0
                .pdf(px_grid.reshape(3, N_height * N_width).T)
                .reshape(N_height, N_width))
       data1 = (kent1
                .pdf(px_grid.reshape(3, N_height * N_width).T)
                .reshape(N_height, N_width))
       data = np.stack([data0, data1], axis=0)

       # Image creation ======================
       I = image.SphericalImage(data, px_grid)

    Data IO:

    .. doctest::

       I.to_fits('test.fits')  # save to FITS
       I2 = image.from_fits('test.fits')  # load from FITS

    Interactive plotting:

    .. doctest::

       I.draw()  # AEQD projection by default, all layers.

    .. image:: _img/sphericalimage_aeqd_example.png

    .. doctest::

       I.draw(index=0, projection='GNOM')  # Only show first data slice.

    .. image:: _img/sphericalimage_gnom_example.png

    .. doctest::

       I.draw(index=1, projection='LCC', data_kwargs=dict(cmap='jet'))

    .. image:: _img/sphericalimage_lcc_example.png
    """

    @chk.check(dict(data=chk.has_reals,
                    grid=chk.has_reals))
    def __init__(self, data, grid):
        """
        Parameters
        ----------
        data : array-like(float)
            multi-level data-cube.

            Possible shapes are:

            * (N_height, N_width);
            * (N_image, N_height, N_width);
            * (N_points,);
            * (N_image, N_points).
        grid : array-like(float)
            (3, ...) Cartesian coordinates of the sky on which the data points are defined.

            Possible shapes are:

            * (3, N_height, N_width);
            * (3, N_points).

        Notes
        -----
        For efficiency reasons, `data` and `grid` are not copied internally.
        """
        grid = np.array(grid, copy=False)
        grid_shape_error_msg = ('Parameter[grid] must have shape '
                                '(3, N_height, N_width) or (3, N_points).')
        if len(grid) != 3:
            raise ValueError(grid_shape_error_msg)
        if grid.ndim == 2:
            self._is_gridded = False
        elif grid.ndim == 3:
            self._is_gridded = True
        else:
            raise ValueError(grid_shape_error_msg)
        self._grid = grid / linalg.norm(grid, axis=0)

        data = np.array(data, copy=False)
        if self._is_gridded is True:
            N_height, N_width = self._grid.shape[1:]
            if ((data.ndim == 2) and
                    chk.has_shape([N_height, N_width])(data)):
                self._data = data[np.newaxis]
            elif ((data.ndim == 3) and
                  chk.has_shape([N_height, N_width])(data[0])):
                self._data = data
            else:
                raise ValueError('Parameters[grid, data] are inconsistent.')
        else:
            N_points = self._grid.shape[1]
            if ((data.ndim == 1) and
                    chk.has_shape([N_points, ])(data)):
                self._data = data[np.newaxis]
            elif ((data.ndim == 2) and
                  chk.has_shape([N_points, ])(data[0])):
                self._data = data
            else:
                raise ValueError('Parameters[grid, data] are inconsistent.')

    @property
    def data(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (N_image, ...) data cube.
        """
        return self._data

    @property
    def grid(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (3, ...) Cartesian coordinates of the sky on which the data points are defined.
        """
        return self._grid

    @chk.check('file_name', chk.is_instance(str))
    def to_fits(self, file_name):
        """
        Save image to FITS file.

        Parameters
        ----------
        file_name : str
            Name of file.

        Notes
        -----
        * :py:class:`~pypeline.phased_array.util.io.image.SphericalImage` subclasses that write WCS information assume the grid is specified in ICRS.
          If this is not the case, rotate the grid accordingly before calling :py:meth:`~pypeline.phased_array.util.io.image.SphericalImage.to_fits`.

        * Data cubes are stored in a secondary IMAGE frame and can be viewed with DS9 using::

              $ ds9 <FITS_file>.fits[IMAGE]

          WCS information is only available in external FITS viewers if using :py:class:`~pypeline.phased_array.util.io.image.EqualAngleImage` or :py:class:`~pypeline.phased_array.util.io.image.HEALPixImage`.
        """
        primary_hdu = self._PrimaryHDU()
        image_hdu = self._ImageHDU()

        hdulist = fits.HDUList([primary_hdu, image_hdu])
        hdulist.writeto(file_name, overwrite=True)

    def _PrimaryHDU(self):
        """
        Generate primary Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        """
        metadata = dict(IMG_TYPE=(self.__class__.__name__,
                                  'SphericalImage subclass'), )

        # grid: stored as angles to reduce file size.
        _, colat, lon = sph.cart2pol(*self._grid)
        coordinates = np.stack([colat.to_value(u.deg),
                                lon.to_value(u.deg)], axis=0)

        hdu = fits.PrimaryHDU(data=coordinates)
        for k, v in metadata.items():
            hdu.header[k] = v
        return hdu

    def _ImageHDU(self):
        """
        Generate image Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.ImageHDU`
        """
        hdu = fits.ImageHDU(data=self._data, name='IMAGE')
        return hdu

    @classmethod
    @chk.check(dict(primary_hdu=chk.is_instance(fits.PrimaryHDU),
                    image_hdu=chk.is_instance(fits.ImageHDU)))
    def _from_fits(cls, primary_hdu, image_hdu):
        """
        Load image from Header Descriptor Units.

        Parameters
        ----------
        primary_hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        image_hdu : :py:class:`~astropy.io.fits.ImageHDU`

        Returns
        -------
        I : :py:class:`~pypeline.phased_array.util.io.image.SphericalImage`
        """
        # PrimaryHDU: grid specification.
        colat, lon = primary_hdu.data * u.deg
        grid = sph.pol2cart(1, colat, lon)

        # ImageHDU: extract data cube.
        data = image_hdu.data

        I = cls(data=data, grid=grid)
        return I

    @property
    def shape(self):
        """
        Returns
        -------
        tuple
            Shape of data cube.
        """
        return self._data.shape

    @chk.check(dict(index=chk.accept_any(chk.is_integer, chk.has_integers,
                                         chk.is_instance(slice)),
                    projection=chk.is_instance(str),
                    catalog=chk.allow_None(chk.is_instance(sky.SkyEmission)),
                    show_gridlines=chk.is_boolean,
                    show_colorbar=chk.is_boolean,
                    ax=chk.allow_None(chk.is_instance(axes.Axes)),
                    data_kwargs=chk.allow_None(chk.is_instance(dict)),
                    grid_kwargs=chk.allow_None(chk.is_instance(dict)),
                    catalog_kwargs=chk.allow_None(chk.is_instance(dict))))
    def draw(self,
             index=slice(None),
             projection='AEQD',
             catalog=None,
             show_gridlines=True,
             show_colorbar=True,
             ax=None,
             data_kwargs=None,
             grid_kwargs=None,
             catalog_kwargs=None):
        """
        Plot spherical image using a 2D projection.

        Parameters
        ----------
        index : int, array-like(int), slice
            Slices of the data-cube to show.

            If multiple layers are provided, they are summed together.
        projection : str
            Plot projection.

            Must be one of (case-insensitive):

            * AEQD: `Azimuthal Equi-Distant <https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection>`_; (default)
            * LAEA: `Lambert Equal-Area <https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection>`_;
            * LCC: `Lambert Conformal Conic <https://en.wikipedia.org/wiki/Lambert_conformal_conic_projection>`_;
            * ROBIN: `Robinson <https://en.wikipedia.org/wiki/Robinson_projection>`_;
            * GNOM: `Gnomonic <https://en.wikipedia.org/wiki/Gnomonic_projection>`_;
            * HEALPIX: `Hierarchical Equal-Area Pixelisation <https://en.wikipedia.org/wiki/HEALPix>`_.

            Notes
            -----
            * (AEQD, LAEA, LCC, GNOM) are recommended for mapping portions of the sphere.

                * LCC breaks down when mapping polar regions.

            * (ROBIN, HEALPIX) are recommended for mapping the entire sphere.
        catalog : :py:class:`~pypeline.phased_array.util.data_gen.sky.SkyEmission`
            Source catalog to overlay on top of images. (Default: no overlay)
        show_gridlines : bool
            Show RA/DEC gridlines. (Default: True)
        show_colorbar : bool
            Show colorbar. (Default: True)
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to draw on.

            If :py:obj:`None`, a new axes is used.
        data_kwargs : dict
            Keyword arguments related to data-cube visualization.

            Accepted keys are:

            * :py:meth:`~matplotlib.axes.Axes.contourf` options.
            * :py:meth:`~matplotlib.axes.Axes.tricontourf` options.
        grid_kwargs : dict
            Keyword arguments related to grid visualization.

            Accepted keys are:

            * N_parallel : int
                Number declination lines to show in viewable region. (Default: 3)
            * N_meridian : int
                Number of right-ascension lines to show in viewable region. (Default: 3)
            * polar_plot : bool
                Correct RA/DEC gridlines when mapping polar regions. (Default: False)

                When mapping polar regions, meridian lines may be doubled at 180W/E, making it seem like a meridian line is missing.
                Setting `polar_plot` to :py:obj:`True` redistributes the meridians differently to correct the issue.

                This option only makes sense when mapping polar regions, and will produce incorrect gridlines otherwise.
            * ticks : bool
                Add RA/DEC labels next to gridlines. (Default: False)
                TODO: change to True once implemented
        catalog_kwargs : dict
            Keyword arguments related to catalog visualization.

            Accepted keys are:

            * :py:meth:`~matplotlib.axes.Axes.scatter` options.

        Returns
        -------
        ax : :py:class:`~matplotlib.axes.Axes`
        """
        if ax is None:
            fig, ax = plt.subplots()

        proj = self._draw_projection(projection)
        scm = self._draw_data(index, data_kwargs, proj, ax)
        cbar = self._draw_colorbar(show_colorbar, scm, ax)  # noqa: F841
        self._draw_gridlines(show_gridlines, grid_kwargs, proj, ax)
        self._draw_catalog(catalog, catalog_kwargs, proj, ax)
        self._draw_beautify(proj, ax)

        return ax

    @chk.check('projection', chk.is_instance(str))
    def _draw_projection(self, projection):
        """
        Setup :py:class:`pyproj.Proj` object to do (lon,lat) <-> (x,y) transforms.

        Parameters
        ----------
        projection : str
            `projection` parameter given to :py:meth:`draw`.

        Returns
        -------
        proj : :py:class:`pyproj.Proj`
        """
        # Most projections can be provided a point in space around which distortions are minimized.
        # We choose this point to approximately map to the center of the grid when appropriate.
        # (approximate since it is not always a spherical cap.)
        if self._is_gridded:  # (3, N_height, N_width) grid
            grid_dir = np.mean(self._grid, axis=(1, 2))
        else:  # (3, N_points) grid
            grid_dir = np.mean(self._grid, axis=1)
        _, grid_lat, grid_lon = sph.cart2eq(*grid_dir)
        grid_lat = (coord.Angle(grid_lat)
                    .to_value(u.deg))
        grid_lon = (coord.Angle(grid_lon)
                    .wrap_at(180 * u.deg)
                    .to_value(u.deg))

        p_name = projection.lower()
        if p_name == 'lcc':
            # Lambert Conformal Conic
            proj = pyproj.Proj(proj='lcc',
                               lon_0=grid_lon,
                               lat_0=grid_lat,
                               R=1)
        elif p_name == 'aeqd':
            # Azimuthal Equi-Distant
            proj = pyproj.Proj(proj='aeqd',
                               lon_0=grid_lon,
                               lat_0=grid_lat,
                               R=1)
        elif p_name == 'laea':
            # Lambert Equal-Area
            proj = pyproj.Proj(proj='laea',
                               lon_0=grid_lon,
                               lat_0=grid_lat,
                               R=1)
        elif p_name == 'robin':
            # Robinson
            proj = pyproj.Proj(proj='robin',
                               lon_0=grid_lon,
                               R=1)
        elif p_name == 'gnom':
            # Gnomonic
            proj = pyproj.Proj(proj='gnom',
                               lon_0=grid_lon,
                               lat_0=grid_lat,
                               R=1)
        elif p_name == 'healpix':
            # Hierarchical Equal-Area Pixelisation
            proj = pyproj.Proj(proj='healpix',
                               lon_0=grid_lon,
                               lat_0=grid_lat,
                               R=1)
        else:
            raise ValueError('Parameter[projection] is not a valid projection '
                             'specifier.')

        return proj

    @chk.check(dict(index=chk.accept_any(chk.is_integer, chk.has_integers,
                                         chk.is_instance(slice)),
                    data_kwargs=chk.allow_None(chk.is_instance(dict)),
                    projection=chk.is_instance(pyproj.Proj),
                    ax=chk.is_instance(axes.Axes)))
    def _draw_data(self, index, data_kwargs, projection, ax):
        """
        Contour plot of data.

        Parameters
        ----------
        index : int, array-like(int), slice
            `index` parameter given to :py:meth:`draw`.
        data_kwargs : dict
            `data_kwargs` parameter given to :py:meth:`draw`.
        projection : :py:class:`~pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.

        Returns
        -------
        scm : :py:class:`~matplotlib.cm.ScalarMappable`
        """
        if data_kwargs is None:
            data_kwargs = dict()

        N_image = self.shape[0]
        if chk.is_integer(index):
            index = np.array([index], dtype=int)
        elif chk.has_integers(index):
            index = np.array(index, dtype=int)
        else:  # slice()
            index = np.arange(N_image, dtype=int)[index]
            if index.size == 0:
                raise ValueError('No data-cube slice chosen.')
        if not np.all((0 <= index) & (index < N_image)):
            raise ValueError('Parameter[index] is out of bounds.')
        data = np.sum(self._data[index], axis=0)

        # Transform (lon,lat) to (x,y).
        # Some projections have unmappable regions or exhibit singularities at certain points.
        # These regions are colored white in contour plots by replacing their incorrect value (1e30) with NaN.
        _, grid_lat, grid_lon = sph.cart2eq(*self._grid)
        grid_lat = (coord.Angle(grid_lat)
                    .to_value(u.deg))
        grid_lon = (coord.Angle(grid_lon)
                    .wrap_at(180 * u.deg)
                    .to_value(u.deg))

        grid_x, grid_y = projection(grid_lon, grid_lat, errcheck=False)
        grid_x[np.isclose(grid_x, 1e30)] = np.nan
        grid_y[np.isclose(grid_y, 1e30)] = np.nan

        # Colormap choice
        if 'cmap' in data_kwargs:
            obj = data_kwargs.pop('cmap')
            if chk.is_instance(str)(obj):
                cmap = cm.get_cmap(obj)
            else:
                cmap = obj
        else:
            cmap = plot.cmap('matthieu-custom-sky', N=38)

        if self._is_gridded:
            scm = ax.contourf(grid_x, grid_y, data,
                              cmap.N, cmap=cmap,
                              **data_kwargs)
        else:
            triangulation = tri.Triangulation(grid_x, grid_y)
            scm = ax.tricontourf(triangulation, data,
                                 cmap.N, cmap=cmap,
                                 **data_kwargs)

        # Show coordinates in status bar
        def sexagesimal_coords(x, y):
            lon, lat = projection(x, y, errcheck=False, inverse=True)
            lon = (coord.Angle(lon * u.deg)
                   .wrap_at(180 * u.deg)
                   .to_string(unit=u.hourangle, sep='hms'))
            lat = (coord.Angle(lat * u.deg)
                   .to_string(unit=u.degree, sep='dms'))

            msg = f'RA: {lon}, DEC: {lat}'
            return msg

        ax.format_coord = sexagesimal_coords

        return scm

    @chk.check(dict(show_colorbar=chk.is_boolean,
                    scm=chk.is_instance(cm.ScalarMappable),
                    ax=chk.is_instance(axes.Axes)))
    def _draw_colorbar(self, show_colorbar, scm, ax):
        """
        Attach colorbar.

        Parameters
        ----------
        show_colorbar : bool
            `show_colorbar` parameter given to :py:meth:`draw`.
        scm : :py:class:`~matplotlib.cm.ScalarMappable`
            Intensity scale.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.

        Returns
        -------
        cbar : :py:class:`~matplotlib.colorbar.Colorbar`
        """
        if show_colorbar:
            cbar = plot.colorbar(scm, ax)
        else:
            cbar = None

        return cbar

    @chk.check(dict(show_gridlines=chk.is_boolean,
                    grid_kwargs=chk.allow_None(chk.is_instance(dict)),
                    projection=chk.is_instance(pyproj.Proj),
                    ax=chk.is_instance(axes.Axes)))
    def _draw_gridlines(self, show_gridlines, grid_kwargs, projection, ax):
        """
        Plot Right-Ascension / Declination lines.

        Parameters
        ----------
        show_gridlines : bool
            `show_gridlines` parameter given to :py:meth:`draw`.
        grid_kwargs : dict
            `grid_kwargs` parameter given to :py:meth:`draw`.
        projection : :py:class:`pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.
        """
        if grid_kwargs is None:
            grid_kwargs = dict()

        if 'N_parallel' in grid_kwargs:
            N_parallel = grid_kwargs.pop('N_parallel')
            if not (chk.is_integer(N_parallel) and
                    (N_parallel >= 3)):
                raise ValueError('Value[N_parallel] must be at least 3.')
        else:
            N_parallel = 3

        if 'N_meridian' in grid_kwargs:
            N_meridian = grid_kwargs.pop('N_meridian')
            if not (chk.is_integer(N_meridian) and
                    (N_meridian >= 3)):
                raise ValueError('Value[N_meridian] must be at least 3.')
        else:
            N_meridian = 3

        if 'polar_plot' in grid_kwargs:
            polar_plot = grid_kwargs.pop('polar_plot')
            if not chk.is_boolean(polar_plot):
                raise ValueError('Value[polar_plot] must be boolean.')
        else:
            polar_plot = False

        if 'ticks' in grid_kwargs:
            show_ticks = grid_kwargs.pop('ticks')
            if not chk.is_boolean(show_ticks):
                raise ValueError('Value[ticks] must be boolean.')
        else:
            # TODO: change to True once implemented.
            show_ticks = False

        plot_style = dict(alpha=0.5,
                          color='k',
                          linewidth=1,
                          linestyle='solid')
        plot_style.update(grid_kwargs)

        _, grid_lat, grid_lon = sph.cart2eq(*self._grid)
        grid_lat = (coord.Angle(grid_lat)
                    .to_value(u.deg))
        grid_lon = (coord.Angle(grid_lon)
                    .wrap_at(180 * u.deg)
                    .to_value(u.deg))

        # RA curves
        meridian = dict()
        dec_span = np.linspace(grid_lat.min(), grid_lat.max(), 200)
        if polar_plot:
            ra = np.linspace(-180, 180, N_meridian, endpoint=False)
        else:
            ra = np.linspace(grid_lon.min(), grid_lon.max(), N_meridian)
        for _ in ra:
            ra_span = _ * np.ones_like(dec_span)

            # Transform (lon,lat) to (x,y).
            # Some projections have unmappable regions or exhibit singularities at certain points.
            # These regions are colored white in contour plots by replacing their incorrect value (1e30) with NaN.
            grid_x, grid_y = projection(ra_span, dec_span, errcheck=False)
            grid_x[np.isclose(grid_x, 1e30)] = np.nan
            grid_y[np.isclose(grid_y, 1e30)] = np.nan

            if show_gridlines:
                mer = ax.plot(grid_x, grid_y, **plot_style)[0]
                meridian[_] = mer

        # DEC curves
        parallel = dict()
        ra_span = np.linspace(grid_lon.min(), grid_lon.max(), 200)
        if polar_plot:
            dec = np.linspace(grid_lat.min(), grid_lat.max(), N_parallel + 1)
        else:
            dec = np.linspace(grid_lat.min(), grid_lat.max(), N_parallel)
        for _ in dec:
            dec_span = _ * np.ones_like(ra_span)

            # Transform (lon,lat) to (x,y).
            # Some projections have unmappable regions or exhibit singularities at certain points.
            # These regions are colored white in contour plots by replacing their incorrect value (1e30) with NaN.
            grid_x, grid_y = projection(ra_span, dec_span, errcheck=False)
            grid_x[np.isclose(grid_x, 1e30)] = np.nan
            grid_y[np.isclose(grid_y, 1e30)] = np.nan

            if show_gridlines:
                par = ax.plot(grid_x, grid_y, **plot_style)[0]
                parallel[_] = par

        # LAT/LON ticks
        if show_gridlines and show_ticks:
            raise NotImplementedError('Not yet implemented.')

    @chk.check(dict(catalog=chk.allow_None(chk.is_instance(sky.SkyEmission)),
                    projection=chk.is_instance(pyproj.Proj),
                    ax=chk.is_instance(axes.Axes)))
    def _draw_catalog(self, catalog, catalog_kwargs, projection, ax):
        """
        Overlay catalog on top of map.

        Parameters
        ----------
        catalog : :py:class:`~pypeline.phased_array.util.data_gen.sky.SkyEmission`
            `catalog` parameter given to :py:meth:`draw`.
        catalog_kwargs : dict
            `catalog_kwargs` parameter given to :py:meth:`draw`.
        projection : :py:class:`pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.
        """
        if catalog is not None:
            _, c_lat, c_lon = sph.cart2eq(*catalog.xyz.T)
            c_lat = (coord.Angle(c_lat)
                     .to_value(u.deg))
            c_lon = (coord.Angle(c_lon)
                     .wrap_at(180 * u.deg)
                     .to_value(u.deg))

            c_x, c_y = projection(c_lon, c_lat, errcheck=False)
            c_x[np.isclose(c_x, 1e30)] = np.nan
            c_y[np.isclose(c_y, 1e30)] = np.nan

            if catalog_kwargs is None:
                catalog_kwargs = dict()

            plot_style = dict(s=400,
                              facecolors='none',
                              edgecolors='w')
            plot_style.update(catalog_kwargs)

            ax.scatter(c_x, c_y, **plot_style)

    @chk.check(dict(projection=chk.is_instance(pyproj.Proj),
                    ax=chk.is_instance(axes.Axes)))
    def _draw_beautify(self, projection, ax):
        """
        Format plot.

        Parameters
        ----------
        projection : :py:class:`pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to draw on.
        """
        ax.axis('off')
        ax.axis('equal')


class EqualAngleImage(SphericalImage):
    """
    Specialized container for Equal-Angle sampled images on :math:`\mathbb{S}^{2}.`
    """

    @chk.check(dict(colat=chk.has_angles,
                    lon=chk.has_angles))
    def __init__(self, data, colat, lon):
        """
        Parameters
        ----------
        data : array-like(float)
            multi-level data-cube.

            Possible shapes are:

            * (N_height, N_width);
            * (N_image, N_height, N_width).
        colat : :py:class:`~astropy.units.Quantity`
            (N_height, 1) equi-spaced polar angles.
        lon : :py:class:`~astropy.units.Quantity`
            (1, N_width) equi-spaced azimuthal angles.
        """
        N_height = colat.size
        if not chk.has_shape([N_height, 1])(colat):
            raise ValueError('Parameter[colat] must have shape (N_height, 1).')
        N_width = lon.size
        if not chk.has_shape([1, N_width])(lon):
            raise ValueError('Parameter[lon] must have shape (1, N_width).')

        grid = sph.pol2cart(1, colat, lon)
        super().__init__(data, grid)

        # Reorder grid/data for longitude/colatitude to be in increasing order.
        colat_idx = np.argsort(colat, axis=0)
        lon_idx = np.argsort(lon, axis=1)
        self._grid = self._grid[:, colat_idx, lon_idx]
        self._data = self._data[:, colat_idx, lon_idx]

        # Make sure colat/lon were actually equi-spaced.
        lon = lon[0, lon_idx][0]
        lon_step = lon[1] - lon[0]
        if not u.allclose(np.diff(lon), lon_step):
            raise ValueError('Parameter[lon] must be equi-spaced.')
        self._lon = lon.reshape(1, N_width)

        colat = colat[colat_idx, 0][:, 0]
        colat_step = colat[1] - colat[0]
        if not u.allclose(np.diff(colat), colat_step):
            raise ValueError('Parameter[colat] must be equi-spaced.')
        self._colat = colat.reshape(N_height, 1)

    def _PrimaryHDU(self):
        """
        Generate primary Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        """
        N_height = self._colat.size
        N_width = self._lon.size

        metadata = {'IMG_TYPE': (self.__class__.__name__,
                                 'SphericalImage subclass'),
                    'N_HEIGHT': (N_height, 'N_rows'),
                    'N_WIDTH': (N_width, 'N_columns')}

        # grid: store ogrid-style mesh in 1D form for compactness.
        coordinates = np.r_[self._colat.to_value(u.deg).reshape(N_height),
                            self._lon.to_value(u.deg).reshape(N_width)]

        hdu = fits.PrimaryHDU(data=coordinates)
        for k, v in metadata.items():
            hdu.header[k] = v
        return hdu

    def _ImageHDU(self):
        """
        Generate image Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.ImageHDU`
        """
        # Coordinate information for external FITS viewers:
        #
        # * arrays are Fortran-ordered with indices starting at 1;
        # * angles (lon/lat) must be specified in degrees.
        _, lat, lon = sph.pol2eq(1, self._colat, self._lon)
        RA = lon.to_value(u.deg).reshape(-1)
        DEC = lat.to_value(u.deg).reshape(-1)

        # Embed WCS header
        hdu = super()._ImageHDU()
        wcs = WCS(hdu.header)
        wcs.wcs.cd = np.diag([RA[1] - RA[0], DEC[1] - DEC[0], 1])  # CD_ija
        wcs.wcs.cname = ['RA', 'DEC', 'ENERGY_LEVEL']  # CNAMEa
        wcs.wcs.crpix = [1, 1, 1]  # CRPIX_ja
        wcs.wcs.crval = [RA[0], DEC[0], 0]  # CRVAL_ia
        wcs.wcs.ctype = ['RA', 'DEC', '']  # CTYPE_ia
        wcs.wcs.cunit = ['deg', 'deg', '']  # CUNIT_ia
        wcs.wcs.name = 'ICRS'  # WCSNAMEa

        hdu.header.extend(wcs.to_header().cards, update=True)
        return hdu

    @classmethod
    @chk.check(dict(primary_hdu=chk.is_instance(fits.PrimaryHDU),
                    image_hdu=chk.is_instance(fits.ImageHDU)))
    def _from_fits(cls, primary_hdu, image_hdu):
        """
        Load image from Header Descriptor Units.

        Parameters
        ----------
        primary_hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        image_hdu : :py:class:`~astropy.io.fits.ImageHDU`

        Returns
        -------
        I : :py:class:`~pypeline.phased_array.util.io.image.SphericalImage`
        """
        # PrimaryHDU: grid specification.
        N_height = primary_hdu.header['N_HEIGHT']
        colat = primary_hdu.data[:N_height].reshape(N_height, 1) * u.deg

        N_width = primary_hdu.header['N_WIDTH']
        lon = primary_hdu.data[-N_width:].reshape(1, N_width) * u.deg

        # ImageHDU: extract data cube.
        data = image_hdu.data

        I = cls(data=data, colat=colat, lon=lon)
        return I


class HEALPixImage(SphericalImage):
    """
    Specialized container for HEALPix-sampled images on :math:`\mathbb{S}^{2}`.
    """

    def __init__(self):
        """
        """
        raise NotImplementedError
