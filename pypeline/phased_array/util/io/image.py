# #############################################################################
# image.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Image containers, visualization and export facilities.
"""

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import scipy.linalg as linalg
from astropy.wcs import WCS

import pypeline.util.argcheck as chk
import pypeline.util.math.sphere as sph


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
    * advanced 2D plotting based on `Matplotlib <https://matplotlib.org/>`_  and `Cartopy <https://scitools.org.uk/cartopy/docs/latest/#>`_;
    * view exported images with `DS9 <http://ds9.si.edu/site/Home.html>`_.

    Examples
    --------
    .. todo:: put something here
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

            # Buffer draw()'s triangulation object; useful for large images.
            self._triangulation = None
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
