# #############################################################################
# io.py
# =====
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
IO-related operations and tools.
"""

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import scipy.linalg as linalg

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.util.data_gen as dgen
import pypeline.util.argcheck as chk
import pypeline.util.math.sphere as sph


@chk.check(dict(S=chk.is_instance(dgen.VisibilityMatrix),
                W=chk.is_instance(beamforming.BeamWeights)))
def filter_data(S, W):
    """
    Fix mis-matches to make data streams compatible.

    Visibility matrices from MS files typically include broken beams and/or may not match beams specified in beamforming matrices.
    This mis-match causes computations further down the imaging pypeline to be less efficient or completely wrong.
    This function applies 2 corrections to visibility and beamforming matrices to make them compliant:

    * Drop beams in `S` that do not appear in `W`;
    * Insert 0s in `W` where `S` has broken beams.

    Parameters
    ----------
    S : :py:class:`~pypeline.phased_array.util.data_gen.VisibilityMatrix`
        (N_beam1, N_beam1) visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) beamforming matrix.

    Returns
    -------
    S : :py:class:`~pypeline.phased_array.util.data_gen.VisibilityMatrix`
        (N_beam2, N_beam2) filtered visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) filtered beamforming matrix.
    """
    # Stage 1: Drop beams in S that do not appear in W
    beam_idx1 = S.index[0]
    beam_idx2 = W.index[1]
    beams_to_drop = beam_idx1.difference(beam_idx2)
    beams_to_keep = beam_idx1.drop(beams_to_drop)

    mask = np.any(beam_idx1.values.reshape(-1, 1) ==
                  beams_to_keep.values.reshape(1, -1), axis=1)
    S_f = dgen.VisibilityMatrix(data=S.data[np.ix_(mask, mask)],
                                beam_idx=beam_idx1[mask])

    # Stage 2: Insert 0s in W where S had broken beams
    broken_beam_idx = beam_idx2[np.isclose(np.sum(S_f.data, axis=1), 0)]
    mask = np.any(beam_idx2.values.reshape(-1, 1) ==
                  broken_beam_idx.values.reshape(1, -1), axis=1)

    w_f = W.data.copy()
    w_f[:, mask] = 0
    W_f = beamforming.BeamWeights(data=w_f,
                                  ant_idx=W.index[0],
                                  beam_idx=beam_idx2)

    return S_f, W_f


class SphericalImage:
    """
    Container for storing curved images defined on :math:`\mathbb{S}^{2}`.
    """

    @chk.check(dict(data=chk.has_reals,
                    grid=chk.has_reals))
    def __init__(self, data, grid):
        """
        Parameters
        ----------
        data : array-like(float)
            (N_height, N_width) or (N_image, N_height, N_width) data cube.
        grid : array-like(float)
            (3, N_height, N_width) Cartesian coordinates of the sky on which the data points are defined.
        """
        grid = np.array(grid, copy=False)
        if len(grid.shape) != 3:
            raise ValueError('Parameter[grid] must have shape '
                             '(3, N_height, N_width).')
        N_height, N_width = grid.shape[1:]
        if not chk.has_shape([3, N_height, N_width])(grid):
            raise ValueError('Parameter[grid] must have shape '
                             '(3, N_height, N_width).')
        self._grid = grid / linalg.norm(grid, axis=0)

        data = np.array(data, copy=False)
        N_image = len(data)
        if not (chk.has_shape([N_height, N_width])(data) or
                chk.has_shape([N_image, N_height, N_width])(data)):
            raise ValueError('Parameter[data] must have shape '
                             '(N_height, N_width) or '
                             '(N_image, N_height, N_width).')
        self._data = data[np.newaxis] if (data.ndim == 2) else data

    @property
    def data(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (N_image, N_height, N_width) data cube.
        """
        return self._data

    @property
    def grid(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (3, N_height, N_width) Cartesian coordinates of the sky on which the data points are defined.
        """
        return self._grid

    def draw(self):
        """

        Returns
        -------

        """

    @chk.check('file_name', chk.is_instance(str))
    def to_fits(self, file_name):
        """
        Save image to FITS file.

        Parameters
        ----------
        file_name : str
            Name of file.
        """
        # PrimaryHDU containing grid information.
        # (Stored as angles to reduce file size.)
        _, colat, lon = sph.cart2pol(*self._grid)
        coordinates = np.stack([colat.to_value(u.deg),
                                lon.to_value(u.deg)], axis=0)
        primary_hdu = fits.PrimaryHDU(data=coordinates)

        # ImageHDU containing data cube.
        image_hdu = fits.ImageHDU(data=self._data, name='IMAGE')

        hdulist = fits.HDUList([primary_hdu, image_hdu])
        hdulist.writeto(file_name, overwrite=True)

    @classmethod
    @chk.check('file_name', chk.is_instance(str))
    def from_fits(cls, file_name):
        """
        Load image from FITS file.

        Parameters
        ----------
        file_name : str
            Name of file.

        Returns
        -------
        :py:class:`~pypeline.phased_array.util.io.SphericalImage`
        """
        with fits.open(file_name, mode='readonly',
                       memmap=True, lazy_load_hdus=True) as hdulist:
            # PrimaryHDU: extract grid
            primary_hdu = hdulist[0]
            colat, lon = primary_hdu.data * u.deg
            grid = sph.pol2cart(1, colat, lon)

            # ImageHDU: extract data
            image_hdu = hdulist['IMAGE']
            data = image_hdu.data

            I = cls(data=data, grid=grid)
            return I
