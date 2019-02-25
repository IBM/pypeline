# #############################################################################
# spatial_domain.py
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Field synthesizers that work in the spatial domain.
"""

import numexpr as ne
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

import pypeline.phased_array.bluebild.field_synthesizer as synth
import pypeline.util.argcheck as chk


def _have_matching_shapes(V, XYZ, W):
    if (V.ndim == 2) and (XYZ.ndim == 2) and (W.ndim == 2):
        if V.shape[0] != W.shape[1]:  # N_beam
            return False
        if W.shape[0] != XYZ.shape[0]:  # N_antenna
            return False
        return True

    return False


class SpatialFieldSynthesizerBlock(synth.FieldSynthesizerBlock):
    """
    Field synthesizer based on StandardSynthesis.

    Examples
    --------
    Assume we are imaging a portion of the Bootes field with LOFAR's 24 core stations.

    The short script below shows how to use :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.spatial_domain.SpatialFieldSynthesizerBlock` to form continuous energy level estimates.

    .. testsetup::

       import numpy as np
       import astropy.units as u
       import astropy.time as atime
       import astropy.coordinates as coord
       import scipy.constants as constants
       from tqdm import tqdm as ProgressBar
       from pypeline.phased_array.bluebild.data_processor import IntensityFieldDataProcessorBlock
       from pypeline.phased_array.bluebild.field_synthesizer.spatial_domain import SpatialFieldSynthesizerBlock
       from pypeline.phased_array.instrument import LofarBlock
       from pypeline.phased_array.beamforming import MatchedBeamformerBlock
       from pypeline.phased_array.util.gram import GramBlock
       from pypeline.phased_array.util.data_gen.sky import from_tgss_catalog
       from pypeline.phased_array.util.data_gen.visibility import VisibilityGeneratorBlock
       from pypeline.phased_array.util.grid import spherical_grid

       np.random.seed(0)

    .. doctest::

       ### Experiment setup ================================================
       # Observation
       >>> obs_start = atime.Time(56879.54171302732, scale='utc', format='mjd')
       >>> field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
       >>> field_of_view = np.deg2rad(5)
       >>> frequency = 145e6
       >>> wl = constants.speed_of_light / frequency

       # instrument
       >>> N_station = 24
       >>> dev = LofarBlock(N_station)
       >>> mb = MatchedBeamformerBlock([(_, _, field_center) for _ in range(N_station)])
       >>> gram = GramBlock()

       # Visibility generation
       >>> sky_model=from_tgss_catalog(field_center, field_of_view, N_src=10)
       >>> vis = VisibilityGeneratorBlock(sky_model,
       ...                                T=8,
       ...                                fs=196000,
       ...                                SNR=np.inf)

       ### Energy-level imaging ============================================
       # Pixel grid
       >>> px_grid = spherical_grid(field_center.transform_to('icrs').cartesian.xyz.value,
       ...                          FoV=field_of_view,
       ...                          size=[256, 386])

       >>> I_dp = IntensityFieldDataProcessorBlock(N_eig=7,  # assumed obtained from IntensityFieldParameterEstimator.infer_parameters()
       ...                                         cluster_centroids=[124.927,  65.09 ,  38.589,  23.256])
       >>> I_fs = SpatialFieldSynthesizerBlock(wl, px_grid)
       >>> t_img = obs_start + np.arange(20) * 400 * u.s  # well-spaced snapshots
       >>> for t in ProgressBar(t_img):
       ...     XYZ = dev(t)
       ...     W = mb(XYZ, wl)
       ...     S = vis(XYZ, W, wl)
       ...     G = gram(XYZ, W, wl)
       ...
       ...     D, V, c_idx = I_dp(S, G)
       ...
       ...     # (N_eig, N_height, N_width) energy levels (compact descriptor, not the same thing as [D, V]).
       ...     field_stat = I_fs(V, XYZ.data, W.data)
       ...
       ...     # (N_eig, N_height, N_width) energy levels
       ...     # These are the actual field values. Depending on the implementation of FieldSynthesizerBlock, `field_stat` and `field` may differ.
       ...     field = I_fs.synthesize(field_stat)

       # For SpatialFieldSynthesizerBlock(), `field` and `field_stat` are actually identical.
       >>> np.allclose(field_stat, field)
       True

    In the example above, individual snapshots were not added together, hence the final image is just the last field snapshot and can be quite noisy:

    .. doctest::

       from pypeline.phased_array.util.io.image import SphericalImage
       I_snapshot = SphericalImage(data=field, grid=px_grid)

       ax = I_snapshot.draw(index=slice(None),  # Collapse all energy levels
                            catalog=sky_model,
                            data_kwargs=dict(cmap='cubehelix'),
                            catalog_kwargs=dict(s=600))
       ax.get_figure().show()

    .. image:: _img/bluebild_SpatialFieldSynthesizer_snapshot_example.png
    """

    @chk.check(dict(wl=chk.is_real,
                    pix_grid=chk.has_reals,
                    precision=chk.is_integer))
    def __init__(self, wl, pix_grid, precision=64):
        """
        Parameters
        ----------
        wl : float
            Wavelength [m] of observations.
        pix_grid : :py:class:`~numpy.ndarray`
            (3, N_height, N_width) pixel vectors.
        precision : int
            Numerical accuracy of floating-point operations.

            Must be 32 or 64.
        """
        super().__init__()

        if precision == 32:
            self._fp = np.float32
            self._cp = np.complex64
        elif precision == 64:
            self._fp = np.float64
            self._cp = np.complex128
        else:
            raise ValueError('Parameter[precision] must be 32 or 64.')

        self._wl = wl

        if not ((pix_grid.ndim == 3) and (len(pix_grid) == 3)):
            raise ValueError('Parameter[pix_grid] must have dimensions (3, N_height, N_width).')
        self._grid = pix_grid / linalg.norm(pix_grid, axis=0)

    @chk.check(dict(V=chk.has_complex,
                    XYZ=chk.has_reals,
                    W=chk.is_instance(np.ndarray,
                                      sparse.csr_matrix,
                                      sparse.csc_matrix)))
    def __call__(self, V, XYZ, W):
        """
        Compute instantaneous field statistics.

        Parameters
        ----------
        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.
        XYZ : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be defined in the same reference frame as `pix_grid` from :py:meth:`~pypeline.phased_array.bluebild.field_synthesizer.spatial_domain.SpatialFieldSynthesizerBlock.__init__`.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (N_eig, N_height, N_width) field statistics.

            (Note: StandardSynthesis statistics correspond to the actual field values.)
        """
        if not _have_matching_shapes(V, XYZ, W):
            raise ValueError('Parameters[V, XYZ, W] are inconsistent.')
        V = V.astype(self._cp, copy=False)
        XYZ = XYZ.astype(self._fp, copy=False)
        W = W.astype(self._cp, copy=False)

        N_antenna, N_beam = W.shape
        N_height, N_width = self._grid.shape[1:]

        XYZ = XYZ - XYZ.mean(axis=0)
        P = np.zeros((N_antenna, N_height, N_width), dtype=self._cp)
        ne.evaluate('exp(A * B)',
                    dict(A=1j * 2 * np.pi / self._wl,
                         B=np.tensordot(XYZ, self._grid, axes=1)),
                    out=P,
                    casting='same_kind')  # Due to limitations of NumExpr2

        PW = W.T @ P.reshape(N_antenna, N_height * N_width)
        PW = PW.reshape(N_beam, N_height, N_width)

        E = np.tensordot(V.T, PW, axes=1)
        I = E.real ** 2 + E.imag ** 2
        return I

    @chk.check('stat', chk.has_reals)
    def synthesize(self, stat):
        """
        Compute field values from statistics.

        Parameters
        ----------
        stat : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_width) field statistics.

        Returns
        -------
        field : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_width) field values.
        """
        stat = np.array(stat, copy=False)

        if stat.ndim != 3:
            raise ValueError('Parameter[stat] is incorrectly shaped.')

        N_level = len(stat)
        N_height, N_width = self._grid.shape[1:]

        if not chk.has_shape([N_level, N_height, N_width])(stat):
            raise ValueError("Parameter[stat] does not match the grid's dimensions.")

        field = stat
        return field
