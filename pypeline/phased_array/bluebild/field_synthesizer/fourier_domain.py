# #############################################################################
# fourier_domain.py
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Field synthesizers that work in Fourier Series domain.
"""

import astropy.units as u
import numexpr as ne
import numpy as np
import scipy.fftpack as fftpack
import scipy.linalg as linalg
import scipy.sparse as sparse

import pypeline
import pypeline.phased_array.bluebild.field_synthesizer as synth
import pypeline.phased_array.bluebild.field_synthesizer.spatial_domain as fsd
import pypeline.util.argcheck as chk
import pypeline.util.math.fourier as fourier
import pypeline.util.math.func as func
import pypeline.util.math.linalg as pylinalg
import pypeline.util.math.sphere as sph


class FourierFieldSynthesizerBlock(synth.FieldSynthesizerBlock):
    """
    Field synthesizer based on PeriodicSynthesis.

    Examples
    --------
    Assume we are imaging a portion of the Bootes field with LOFAR's 24 core stations.

    The short script below shows how to use :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.fourier_domain.FourierFieldSynthesizerBlock` to form continuous energy level estimates.

    .. testsetup::

       import numpy as np
       import astropy.units as u
       import astropy.time as atime
       import astropy.coordinates as coord
       from tqdm import tqdm as ProgressBar
       from pypeline.phased_array.bluebild.data_processor import IntensityFieldDataProcessorBlock
       from pypeline.phased_array.bluebild.field_synthesizer.fourier_domain import FourierFieldSynthesizerBlock
       from pypeline.phased_array.instrument import LofarBlock
       from pypeline.phased_array.beamforming import MatchedBeamformerBlock
       from pypeline.phased_array.util.gram import GramBlock
       from pypeline.phased_array.util.data_gen.sky import from_tgss_catalog
       from pypeline.phased_array.util.data_gen.visibility import VisibilityGeneratorBlock
       from pypeline.phased_array.util.grid import ea_grid
       from pypeline.util.math.sphere import pol2cart

       np.random.seed(0)

    .. doctest::

       ### Experiment setup ================================================
       # Observation
       >>> obs_start = atime.Time(56879.54171302732, scale='utc', format='mjd')
       >>> field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
       >>> field_of_view = 5 * u.deg
       >>> frequency = 145 * u.MHz

       # instrument
       >>> N_station = 24
       >>> dev = LofarBlock(N_station)
       >>> mb = MatchedBeamformerBlock([(_, _, field_center) for _ in range(N_station)])
       >>> gram = GramBlock()

       # Visibility generation
       >>> sky_model=from_tgss_catalog(field_center, field_of_view, N_src=10)
       >>> vis = VisibilityGeneratorBlock(sky_model,
       ...                                T=8 * u.s,
       ...                                fs=196 * u.kHz,
       ...                                SNR=np.inf)

       ### Energy-level imaging ============================================
       # Kernel parameters
       >>> t_img = obs_start + np.arange(200) * 8 * u.s  # fine-grained snapshots
       >>> obs_start, obs_end = t_img[0], t_img[-1]
       >>> R = dev.icrs2bfsf_rot(obs_start, obs_end)
       >>> N_FS = dev.bfsf_kernel_bandwidth(frequency, obs_start, obs_end)
       >>> T_kernel = 10 * u.deg

       # Pixel grid: make sure to generate it in BFSF coordinates by applying R.
       >>> px_colat, px_lon = ea_grid(direction=np.dot(R, field_center.transform_to('icrs').cartesian.xyz.value),
       ...                            FoV=field_of_view,
       ...                            size=[256, 386])

       >>> I_dp = IntensityFieldDataProcessorBlock(N_eig=7,  # assumed obtained from IntensityFieldParameterEstimator.infer_parameters()
       ...                                         cluster_centroids=[124.927,  65.09 ,  38.589,  23.256])
       >>> I_fs = FourierFieldSynthesizerBlock(frequency, px_colat, px_lon,
       ...                                     N_FS, T_kernel, R)
       >>> for t in ProgressBar(t_img):
       ...     XYZ = dev(t)
       ...     W = mb(XYZ, frequency)
       ...     S = vis(XYZ, W, frequency)
       ...     G = gram(XYZ, W, frequency)
       ...
       ...     D, V, c_idx = I_dp(S, G)
       ...
       ...     # (N_eig, N_height, N_FS+Q) energy levels (compact descriptor, not the same thing as [D, V]).
       ...     field_stat = I_fs(V, XYZ.data, W.data)

       # (N_eig, N_height, N_width) energy levels
       # These are the actual field values. Depending on the implementation of FieldSynthesizerBlock, `field_stat` and `field` may differ.
       >>> field = I_fs.synthesize(field_stat)

    In the example above, individual snapshots were not added together, hence the final image is just the last field snapshot and can be quite noisy:

    .. doctest::

       from pypeline.phased_array.util.io.image import SphericalImage
       # Transform grid to ICRS coordinates before plotting.
       px_grid = np.tensordot(R.T, pol2cart(1, px_colat, px_lon), axes=1)
       I_snapshot = SphericalImage(data=field, grid=px_grid)

       ax = I_snapshot.draw(index=slice(None),  # Collapse all energy levels
                            catalog=sky_model,
                            data_kwargs=dict(cmap='cubehelix'),
                            catalog_kwargs=dict(s=600))
       ax.get_figure().show()

    .. image:: _img/bluebild_FourierFieldSynthesizer_snapshot_example.png
    """

    @chk.check(dict(freq=chk.is_frequency,
                    grid_colat=chk.has_angles,
                    grid_lon=chk.has_angles,
                    N_FS=chk.is_odd,
                    T=chk.is_angle,
                    R=chk.require_all(chk.has_shape([3, 3]),
                                      chk.has_reals),
                    precision=chk.is_integer))
    def __init__(self, freq, grid_colat, grid_lon, N_FS, T, R, precision=64):
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
        precision : int
            Numerical accuracy of floating-point operations.

            Must be 32 or 64.

        Notes
        -----
        * `grid_colat` and `grid_lon` should be generated using :py:func:`~pypeline.phased_array.util.grid.ea_grid` or :py:func:`~pypeline.phased_array.util.grid.ea_harmonic_grid`.
        * `N_FS` can be optimally chosen by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.bfsf_kernel_bandwidth`.
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

        wps = pypeline.config.getfloat('phased_array', 'wps') * (u.m / u.s)
        self._wl = (wps / freq).to_value(u.m)

        if N_FS <= 0:
            raise ValueError('Parameter[N_FS] must be positive.')

        if not (0 < T <= 360 * u.deg):
            raise ValueError(f'Parameter[T] is out of bounds.')

        if not u.isclose(T, 360 * u.deg):  # PeriodicSynthesis
            self._alpha_window = 0.1
            T_min = (1 + self._alpha_window) * grid_lon.ptp()
            if T < T_min:
                raise ValueError(f'Parameter[T] must be greater that {T_min}.')
            self._T = T

            aw = self._alpha_window
            lon_start, lon_end = grid_lon[0, [0, -1]]
            T_start, T_end = lon_end + T * np.r_[0.5 * aw - 1, 0.5 * aw]
            self._Tc = (T_start + T_end) / 2
            self._mps = lon_start - (T_start + 0.5 * T * aw)  # max_phase_shift

            N_FS_trunc = N_FS / (2 * np.pi) * T.to_value(u.rad)
            N_FS_trunc = int(np.ceil(N_FS_trunc))
            N_FS_trunc += 1 if chk.is_even(N_FS_trunc) else 0
            self._NFS = N_FS_trunc
        else:  # No PeriodicSynthesis, but set params to still work.
            self._alpha_window = 0
            self._T = 360 * u.deg
            self._Tc = 180 * u.deg
            self._mps = 360 * u.deg  # max_phase_shift
            self._NFS = N_FS

        self._grid_colat = u.Quantity(grid_colat)
        self._grid_lon = u.Quantity(grid_lon)
        self._R = np.array(R)

        # Transform angles to radians to simplify later steps.
        self._T = self._T.to_value(u.rad)
        self._Tc = self._Tc.to_value(u.rad)
        self._mps = self._mps.to_value(u.rad)

        # Buffered state
        self._FSk = None  # (N_antenna, N_height, N_FS+Q) FS coefficients
        self._XYZk = None  # (N_antenna, 3) BFSF coordinates

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

            `XYZ` must be given in ICRS.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (N_eig, N_height, N_FS + Q) field statistics.
        """
        if not fsd._have_matching_shapes(V, XYZ, W):
            raise ValueError('Parameters[V, XYZ, W] are inconsistent.')
        V = V.astype(self._cp, copy=False)
        XYZ = XYZ.astype(self._fp, copy=False)
        W = W.astype(self._cp, copy=False)

        bfsf_XYZ = XYZ @ self._R.T
        if self._XYZk is None:
            phase_shift = np.inf
        else:
            phase_shift = self._phase_shift(bfsf_XYZ)

        if self._regen_required(phase_shift):
            self._regen_kernel(bfsf_XYZ)
            phase_shift = 0

        N_antenna, N_height, _2N1Q = self._FSk.shape
        N = (self._NFS - 1) // 2
        Q = _2N1Q - self._NFS
        N_beam = W.shape[1]

        PW_FS = W.T @ self._FSk.reshape(N_antenna, N_height * _2N1Q)
        E_FS = np.tensordot(V.T, PW_FS.reshape(N_beam, N_height, _2N1Q), axes=1)

        mod_phase = (-1j * 2 * np.pi * phase_shift / self._T)
        E_FS *= np.exp(mod_phase) ** np.r_[-N:N + 1, np.zeros(Q)]

        E_Ny = fourier.iffs(E_FS, self._T, self._Tc, self._NFS, axis=2)
        I_Ny = E_Ny.real ** 2 + E_Ny.imag ** 2
        return I_Ny

    @chk.check('stat', chk.has_reals)
    def synthesize(self, stat):
        """
        Compute field values from statistics.

        Parameters
        ----------
        stat : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_FS + Q) field statistics.

        Returns
        -------
        field : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_width) field values.
        """
        stat = np.array(stat, copy=False)

        if stat.ndim != 3:
            raise ValueError('Parameter[stat] is incorrectly shaped.')

        N_level = len(stat)
        N_height, _2N1Q = self._FSk.shape[1:]

        if not chk.has_shape([N_level, N_height, _2N1Q])(stat):
            raise ValueError("Parameter[stat] does not match "
                             "the kernel's dimensions.")

        field_FS = fourier.ffs(stat, self._T, self._Tc, self._NFS, axis=2)
        field = fourier.fs_interp(field_FS[:, :, :self._NFS],
                                  T=self._T,
                                  a=self._grid_lon[0, 0].to_value(u.rad),
                                  b=self._grid_lon[0, -1].to_value(u.rad),
                                  M=self._grid_lon.size,
                                  axis=2,
                                  real_x=True)
        return field

    def _phase_shift(self, XYZ):
        """
        Angular shift w.r.t kernel antenna coordinates.

        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be given in BFSF.

        Returns
        -------
        theta : float
            Angular shift (radians) such that ``dot(_XYZk, R(theta).T) == XYZ``.
        """
        R_T, *_ = linalg.lstsq(self._XYZk[:, :2], XYZ[:, :2])

        R = np.eye(3)
        R[:2, :2] = R_T.T
        theta = pylinalg.z_rot2angle(R)

        return theta.to_value(u.rad)

    def _regen_required(self, shift):
        lhs = np.radians(-0.1)  # Slightly below 0 due to numerical rounding
        if lhs <= shift <= self._mps:
            return False
        else:
            return True

    def _regen_kernel(self, XYZ):
        """
        Compute kernel.

        Parameters
        ----------
        XYZ : :py:class:`~numpy.ndarray`
            (N_antenna, 3) Cartesian instrument geometry.

            `XYZ` must be given in BFSF.
        """
        N_samples = fftpack.next_fast_len(self._NFS)
        lon_smpl = fourier.ffs_sample(self._T, self._NFS, self._Tc, N_samples)
        pix_smpl = sph.pol2cart(1,
                                self._grid_colat,
                                lon_smpl.reshape(1, -1) * u.rad)

        N_antenna = len(XYZ)
        N_height = len(self._grid_colat)

        # `self._NFS` assumes imaging is performed with `XYZ` centered at the origin.
        XYZ_c = XYZ - XYZ.mean(axis=0)
        window = func.Tukey(self._T, self._Tc, self._alpha_window)
        k_smpl = np.zeros((N_antenna, N_height, N_samples), dtype=self._cp)
        ne.evaluate('exp(A * B) * C',
                    dict(A=1j * 2 * np.pi / self._wl,
                         B=np.tensordot(XYZ_c, pix_smpl, axes=1),
                         C=window(lon_smpl)),
                    out=k_smpl,
                    casting='same_kind')  # Due to limitations of NumExpr2

        self._FSk = fourier.ffs(k_smpl, self._T, self._Tc, self._NFS, axis=2)
        self._XYZk = XYZ
