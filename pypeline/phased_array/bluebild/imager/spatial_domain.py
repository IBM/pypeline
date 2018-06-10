# #############################################################################
# spatial_domain.py
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
High-level Bluebild interfaces that work in the spatial domain.
"""

import numpy as np
import scipy.sparse as sparse

import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as ssd
import pypeline.phased_array.bluebild.imager as bim
import pypeline.phased_array.util.io as io
import pypeline.util.argcheck as chk
import pypeline.util.array as array


class Spatial_IMFS_Block(bim.IntegratingMultiFieldSynthesizerBlock):
    """
    Multi-field synthesizer based on StandardSynthesis.
    """

    @chk.check(dict(freq=chk.is_frequency,
                    pix_grid=chk.has_reals,
                    N_level=chk.is_integer,
                    precision=chk.is_integer))
    def __init__(self, freq, pix_grid, N_level, precision=64):
        """
        Parameters
        ----------
        freq : :py:class:`~astropy.units.Quantity`
            Frequency of observations.
        pix_grid : :py:class:`~numpy.ndarray`
            (3, N_height, N_width) pixel vectors.
        N_level : int
            Number of clustered energy-levels to output.
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

        if N_level <= 0:
            raise ValueError('Parameter[N_level] must be positive.')
        self._N_level = N_level

        self._synthesizer = ssd.SpatialFieldSynthesizerBlock(
            freq, pix_grid, precision)

    @chk.check(dict(D=chk.has_reals,
                    V=chk.has_complex,
                    XYZ=chk.has_reals,
                    W=chk.is_instance(np.ndarray,
                                      sparse.csr_matrix,
                                      sparse.csc_matrix),
                    cluster_idx=chk.has_integers))
    def __call__(self, D, V, XYZ, W, cluster_idx):
        """
        Compute (clustered) integrated field statistics for least-squares and standardized estimates.

        Parameters
        ----------
        D : :py:class:`~numpy.ndarray`
            (N_eig,) positive eigenvalues.
        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.
        XYZ : :py:class:`~numpy.ndarary`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be defined in the same reference frame as `pix_grid` from :py:meth:`~pypeline.phased_array.bluebild.imager.Spatial_IMFS_Block.__init__`.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.
        cluster_idx : :py:class:`~numpy.ndarray`
            (N_eig,) cluster indices of each eigenpair.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (2, N_level, N_height, N_width) field statistics.
        """
        D = D.astype(self._fp, copy=False)

        stat_std = self._synthesizer(V, XYZ, W)
        stat_lsq = stat_std * D.reshape(-1, 1, 1)

        stat = np.stack([stat_std, stat_lsq], axis=0)
        stat = array._cluster_layers(stat, cluster_idx,
                                     N=self._N_level, axis=1)

        self._update(stat)
        return stat

    def as_image(self):
        """
        Transform integrated statistics to viewable image.

        Returns
        -------
        std : :py:class:`~pypeline.phased_array.util.io.SphericalImage`
            (N_level, N_height, N_width) standardized energy-levels.

        lsq : :py:class:`~pypeline.phased_array.util.io.SphericalImage`
            (N_level, N_height, N_width) least-squares energy-levels.
        """
        grid = self._synthesizer._grid

        stat_std = self._statistics[0]
        std = io.SphericalImage(stat_std, grid)

        stat_lsq = self._statistics[1]
        lsq = io.SphericalImage(stat_lsq, grid)

        return std, lsq
