# #############################################################################
# spatial_domain.py
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
High-level Bluebild interfaces that work in Fourier Series domain.
"""

import numpy as np
import scipy.sparse as sparse

import pypeline.phased_array.bluebild.field_synthesizer.fourier_domain as psd
import pypeline.phased_array.bluebild.imager as bim
import pypeline.phased_array.util.io as io
import pypeline.util.argcheck as chk
import pypeline.util.array as array
import pypeline.util.math.sphere as sph


class Fourier_IMFS_Block(bim.IntegratingMultiFieldSynthesizerBlock):
    """
    Multi-field synthesizer based on PeriodicSynthesis.
    """

    @chk.check(dict(freq=chk.is_frequency,
                    grid_colat=chk.has_angles,
                    grid_lon=chk.has_angles,
                    N_FS=chk.is_odd,
                    T=chk.is_angle,
                    R=chk.require_all(chk.has_shape([3, 3]),
                                      chk.has_reals),
                    N_level=chk.is_integer,
                    precision=chk.is_integer))
    def __init__(self, freq, grid_colat, grid_lon, N_FS, T, R, N_level,
                 precision=64):
        """
        Parameters
        ----------
        freq : :py:class:`~astropy.units.Quantity`
            Frequency of observations.
        grid_colat : :py:class:`~astropy.units.Quantity`
            (N_height, 1) BFSF polar angles.
        grid_lon : :py:class:`~astropy.units.Quantity`
            (1, N_width) equi-spaced BFSF azimuthal angles.
        N_FS : int
            :math:`2\pi`-periodic kernel bandwidth. (odd-valued)
        T : :py:class:`~astropy.units.Quantity`
            Kernel periodicity to use for imaging.
        R : array-like(float)
            (3, 3) ICRS -> BFSF rotation matrix.
        N_level : int
            Number of clustered energy-levels to output.
        precision : int
            Numerical accuracy of floating-point operations.

            Must be 32 or 64.

        Notes
        -----
        * `grid_colat` and `grid_lon` should be generated using :py:func:`~pypeline.phased_array.util.grid.ea_grid` or :py:func:`~pypeline.phased_array.util.grid.ea_harmonic_grid`.
        * `N_FS` can be optimally chosen by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.kernel_bandwidth`.
        * `R` can be obtained by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.icrs2bfsf_rot`.
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

        self._synthesizer = psd.FourierFieldSynthesizerBlock(
            freq, grid_colat, grid_lon, N_FS, T, R, precision)

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

            `XYZ` must be given in ICRS.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.
        cluster_idx : :py:class:`~numpy.ndarray`
            (N_eig,) cluster indices of each eigenpair.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (2, N_level, N_height, N_FS + Q) field statistics.
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
        Transform integrated statistics to viewable ICRS image.

        Returns
        -------
        std : :py:class:`~pypeline.phased_array.util.io.SphericalImage`
            (N_level, N_height, N_width) standardized energy-levels.

        lsq : :py:class:`~pypeline.phased_array.util.io.SphericalImage`
            (N_level, N_height, N_width) least-squares energy-levels.
        """
        bfsf_grid = sph.pol2cart(1,
                                 self._synthesizer._grid_colat,
                                 self._synthesizer._grid_lon)
        icrs_grid = np.tensordot(self._synthesizer._R.T,
                                 bfsf_grid,
                                 axes=1)

        stat_std = self._statistics[0]
        field_std = self._synthesizer.synthesize(stat_std)
        std = io.SphericalImage(field_std, icrs_grid)

        stat_lsq = self._statistics[1]
        field_lsq = self._synthesizer.synthesize(stat_lsq)
        lsq = io.SphericalImage(field_lsq, icrs_grid)

        return std, lsq
