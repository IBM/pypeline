# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
IO-related operations and tools.
"""

import astropy.io.fits as fits
import astropy.units as u
import matplotlib.axes as axes
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.util.data_gen as dgen
import pypeline.util.argcheck as chk
import pypeline.util.math.sphere as sph
import pypeline.util.plot as plot


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

    if (np.any(mask) and sparse.isspmatrix(W.data)):
        w_lil = W.data.tolil()  # for efficiency
        w_lil[:, mask] = 0
        w_f = w_lil.tocsr()
    else:
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

    @chk.check(dict(index=chk.accept_any(chk.has_integers,
                                         chk.is_instance(slice)),
                    mode=chk.is_instance(str),
                    ax=chk.allow_None(chk.is_instance(axes.Axes)),
                    catalog=chk.allow_None(chk.is_instance(dgen.SkyEmission))))
    def draw(self,
             index=slice(None),
             mode='CAE',
             ax=None,
             catalog=None,
             **kwargs):
        """
        2D contour plot of the data.

        Parameters
        ----------
        index : array-like(int) or slice
            Choice of images to show. (Default = all)
            If multiple layers are selected, they are summed together before display.
        mode : str
            Plot mode. Two available options:

            * CAE: Contour plot in Azimuth-Elevation. (Default)
            * CUV: Contour plot on the projected tangent plane.
              This option is useful if plotting a region which lies at a longitudinal discontinuity.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to draw on. If none is provided, a new figure is generated.
        catalog : :py:class:`~pypeline.phased_array.util.data_gen.SkyEmission`
            Optional ICRS source overlay.
            If the imaging grid was not provided in ICRS, then the overlays will be incorrectly placed.
        **kwargs
            Keyword arguments. These are forwarded to :py:func:`~matplotlib.axes.Axes.contourf`.

        Returns
        -------
        :py:class:`~matplotlib.axes.Axes`
            Axes object.
        """
        I = np.sum(self._data[index], axis=0)

        if ax is None:
            fig, ax = plt.subplots()

        if 'cmap' in kwargs:
            cmap = cm.get_cmap(kwargs.pop('cmap'))
        else:
            cmap = plot.cmap('matthieu-custom-sky', N=38)

        if mode == 'CAE':
            _, lat, lon = sph.cart2eq(*self._grid)
            lat = lat.to_value(u.deg)
            lon = lon.to_value(u.deg)

            im = ax.contourf(lon, lat, I, cmap.N, cmap=cmap, **kwargs)
            plot.colorbar(im, ax)

            ax.set_xlabel('AZ [deg]')
            ax.set_xlim(lon.min(), lon.max())
            ax.set_ylabel('EL [deg]')
            ax.set_ylim(lat.min(), lat.max())
            ax.set_aspect(aspect='equal', adjustable='box')

            if catalog is not None:
                _, c_lat, c_lon = sph.cart2eq(*catalog.xyz.T)
                c_lat = c_lat.to_value(u.deg)
                c_lon = c_lon.to_value(u.deg)
                ax.scatter(c_lon, c_lat, s=400,
                           facecolors='none', edgecolors='w')
        elif mode == 'CUV':
            # UVW coordinates are defined by 3 axes (U,V,W) such that:
            # - U: points East (Northern Hemisphere) or West (Southern Hemisphere)
            # - V: points North (Northern Hemisphere) or South (Southern Hemisphere)
            # - W: points away from the origin.
            #
            # To plot images in UV mode, we need to project the grid on the U, V axes.
            f_dir = np.mean(self._grid, axis=(1, 2))
            f_dir /= linalg.norm(f_dir)
            _, f_lat, _ = sph.cart2eq(*f_dir)
            pole_dir = 1 if (f_lat >= 0 * u.deg) else -1
            pole_similarity = np.stack([[0, 0, 1], [0, 0, -1]], axis=0) @ f_dir

            if np.any(pole_similarity >= np.cos(0.05 * u.deg)):
                # When `direction` is facing towards a pole, U and V can
                # be chosen arbitrarily. As a convention, we will always choose
                # (U,V) = (X,Y) in these circumstances.
                u_axis = np.r_[1, 0, 0]
                v_axis = np.r_[0, 1, 0]
                w_axis = np.r_[0, 0, pole_dir]
            else:
                w_axis = f_dir
                u_axis = np.cross([0, 0, pole_dir], w_axis)
                u_axis /= linalg.norm(u_axis)
                v_axis = np.cross(w_axis, u_axis)

            UV = np.stack((u_axis, v_axis), axis=0)
            g_u, g_v = np.tensordot(UV, self._grid, axes=1)

            im = ax.contourf(g_u, g_v, I, cmap.N, cmap=cmap, **kwargs)
            plot.colorbar(im, ax)

            ax.set_xlabel('U')
            ax.set_xlim(g_u.min(), g_u.max())
            ax.set_ylabel('V')
            ax.set_ylim(g_v.min(), g_v.max())
            ax.set_aspect(aspect='equal', adjustable='box')

            if catalog is not None:
                c_u, c_v = np.tensordot(UV, catalog.xyz.T, axes=1)
                ax.scatter(c_u, c_v, s=400,
                           facecolors='none', edgecolors='w')
        else:
            raise ValueError('Parameter[mode] only accepts {CAE, CUV}.')

        return ax

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
