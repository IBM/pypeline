# #############################################################################
# visibility.py
# =============
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Visibility generation utilities.

Due to the high data-rates emanating from antennas, raw antenna time-series are rarely archived.
Instead, signals from different antennas are correlated together to form *visibility* matrices.
"""

import numpy as np
import scipy.fftpack as fftpack
import skimage.util as sku

import pypeline.core as core
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.util.data_gen.sky as sky
import pypeline.util.argcheck as chk
import pypeline.util.array as array
import pypeline.util.math.stat as stat
import pypeline.util.math.func as func


class VisibilityMatrix(array.LabeledMatrix):
    """
    Visibility coefficients.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import pandas as pd
       from pypeline.phased_array.util.data_gen.visibility import VisibilityMatrix

    .. doctest::

       >>> N_beam = 5
       >>> beam_idx = pd.Index(range(N_beam), name='BEAM_ID')
       >>> S = VisibilityMatrix(np.eye(N_beam), beam_idx)

       >>> S.data
       array([[1., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0.],
              [0., 0., 1., 0., 0.],
              [0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1.]])
    """

    @chk.check(dict(data=chk.accept_any(chk.has_reals, chk.has_complex),
                    beam_idx=beamforming.is_beam_index))
    def __init__(self, data, beam_idx):
        """
        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            (N_beam, N_beam) visibility coefficients.
        beam_idx
            (N_beam,) index.
        """
        data = np.array(data, copy=False)
        N_beam = len(beam_idx)

        if not chk.has_shape((N_beam, N_beam))(data):
            raise ValueError('Parameters[data, beam_idx] are not consistent.')

        if not np.allclose(data, data.conj().T):
            raise ValueError('Parameter[data] must be hermitian symmetric.')

        super().__init__(data, beam_idx, beam_idx)


class VisibilityGeneratorBlock(core.Block):
    """
    Generate synthetic visibility matrices.
    """

    @chk.check(dict(sky_model=chk.is_instance(sky.SkyEmission),
                    T=chk.is_real,
                    fs=chk.is_real,
                    SNR=chk.is_real))
    def __init__(self, sky_model, T, fs, SNR):
        """
        Parameters
        ----------
        sky_model : :py:class:`~pypeline.phased_array.util.data_gen.sky.SkyEmission`
            Source model from which to generate data.
        T : float
            Integration time [s].
        fs : float
            Sampling rate [Hz].
        SNR : float
            Signal-to-Noise-Ratio (dB).
        """
        super().__init__()

        if T <= 0:
            raise ValueError('Parameter[T] must be positive.')
        if fs <= 0:
            raise ValueError('Parameter[fs] must be positive.')

        self._N_sample = int(T * fs) + 1
        self._SNR = 10 ** (SNR / 10)
        self._sky_model = sky_model

    @chk.check(dict(XYZ=chk.is_instance(instrument.InstrumentGeometry),
                    W=chk.is_instance(beamforming.BeamWeights),
                    wl=chk.is_real))
    def __call__(self, XYZ, W, wl):
        """
        Compute visibility matrix.

        Parameters
        ----------
        XYZ : :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            (N_antenna, 3) ICRS instrument geometry.
        W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            (N_antenna, N_beam) synthesis beamweights.
        wl : float
            Wave-length [m] at which to generate visibilities.

        Returns
        -------
        :py:class:`~pypeline.phased_array.util.data_gen.visibility.VisibilityMatrix`
            (N_beam, N_beam) visibility matrix.

        Examples
        --------
        .. testsetup::

           from pypeline.phased_array.instrument import LofarBlock
           from pypeline.phased_array.beamforming import MatchedBeamformerBlock
           import astropy.units as u
           import astropy.time as atime
           import astropy.coordinates as coord
           from pypeline.phased_array.util.data_gen.visibility import VisibilityGeneratorBlock
           from pypeline.phased_array.util.data_gen.sky import from_tgss_catalog
           from scipy.constants import speed_of_light

        .. doctest::

           # Configure instrument and beamformer
           >>> instr = LofarBlock(N_station=24)
           >>> station_id = instr._layout.index.get_level_values('STATION_ID')

           >>> mb_cfg = [(_, _, coord.SkyCoord(0 * u.deg, 90 * u.deg))
           ...           for _ in station_id.drop_duplicates()]
           >>> mb = MatchedBeamformerBlock(mb_cfg)

           # Configure visibility generator
           >>> sky_model = from_tgss_catalog(coord.SkyCoord(0 * u.deg, 90 * u.deg),
           ...                               FoV=np.radians(5),
           ...                               N_src=10)
           >>> S_gen = VisibilityGeneratorBlock(sky_model,
           ...                                  T=8,
           ...                                  fs=196e3,
           ...                                  SNR=np.inf)

           # Generate data
           >>> XYZ = instr(atime.Time('J2000'))
           >>> freq = 145e6
           >>> wl = speed_of_light / freq
           >>> W = mb(XYZ, wl)
           >>> S = S_gen(XYZ, W, wl)

           # Only 10 sources & no noise, so rank(S) <= 10
           >>> np.linalg.matrix_rank(S.data) <= 10
           True
        """
        if wl <= 0:
            raise ValueError('Parameter[wl] must be positive.')

        if not XYZ.is_consistent_with(W, axes=[0, 0]):
            raise ValueError('Parameters[XYZ, W] are inconsistent.')

        A = np.exp((1j * 2 * np.pi / wl) *
                   (self._sky_model.xyz @ XYZ.data.T))
        S_sky = ((W.data.conj().T @ (A.conj().T * self._sky_model.intensity)) @
                 (A @ W.data))

        noise_var = np.sum(self._sky_model.intensity) / (2 * self._SNR)
        S_noise = W.data.conj().T @ (noise_var * W.data)

        wishart = stat.Wishart(V=S_sky + S_noise, n=self._N_sample)
        S = wishart()[0] / self._N_sample
        return VisibilityMatrix(data=S, beam_idx=W.index[1])


@chk.check(dict(data=chk.accept_any(chk.has_reals, chk.has_complex),
                T=chk.is_real,
                fs=chk.is_integer,
                channel_boundaries=chk.has_reals,
                stft_window_alpha=chk.is_real))
def ts2vs(data, T, fs, channel_boundaries, stft_window_alpha):
    """
    Transform time-series to a sequence of visibility matrices.

    Parameters
    ----------
    data : :py:class:`~numpy.ndarray`
        (N_samples, N_stream) antenna samples.
    T : float
        Integration time [s].
    fs : int
        Sampling rate [Hz].
    channel_boundaries : :py:class:`~numpy.ndarray`
        (N_band, 2) frequency band borders [Hz].
    stft_window_alpha : float
        Normalized decay-rate in (0, 1].

        Small values denote sharp windows.

    Returns
    -------
    N_samples : int
        Number of time-samples used to form each visibility matrix.
    S : :py:class:`~numpy.ndarray`
        (N_time_slot, N_band, N_stream, N_stream) visibility matrices.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pypeline.phased_array.util.data_gen.visibility import ts2vs

    .. doctest::

       # Generate some pure-tone time-series for 6 microphones.
       >>> N_stream = 6
       >>> fs = 48000
       >>> f = np.array([1500, 4000, 8000])  # tone frequencies
       >>> t = np.arange(2 * fs) / fs        # 2[s] recording
       >>> phase = 2 * np.pi * np.random.rand(N_stream)
       >>> time_series = np.cos(2 * np.pi * f.reshape(1, -1, 1) *
       ...                                  t.reshape(-1, 1, 1) +
       ...                                  phase.reshape(1, 1, -1)).sum(axis=1)

       >>> T = 50e-3
       >>> channel_boundaries = np.array([[ 500, 1400],  # no signal
       ...                                [1400, 1600],  # signal present
       ...                                [3700, 4300],  # signal present
       ...                                [7800, 8100]]) # signal present
       >>> stft_window_alpha = 0.1
       >>> N_samples, S = ts2vs(time_series, T, fs, channel_boundaries, stft_window_alpha)

       # Since the 0-th frequency band contains no signal, it should contain
       # less total energy than bands 1,2,3:
       >>> D = np.linalg.eigvalsh(S)        # (N_time, N_channel, N_stream)
       >>> energy = np.sum(D, axis=(0, 2))  # (N_channel,) energy per band.
       >>> np.all(energy[0] < energy[1:])
       True
    """
    if T <= 0:
        raise ValueError('Parameter[T] must be positive.')
    if fs <= 0:
        raise ValueError('Parameter[fs] must be positive.')
    if not ((channel_boundaries.ndim == 2) and
            (channel_boundaries.shape[1] == 2)):
        raise ValueError('Parameter[channel_boundaries] must have shape (N_band, 2).')
    if not np.all(channel_boundaries > 0):
        raise ValueError('Parameter[channel_boundaries] must contain positive values.')
    if not (0 < stft_window_alpha <= 1):
        raise ValueError("Parameter[stft_window_alpha] must lie in (0, 1].")

    data = np.array(data, copy=False)
    channel_boundaries = np.array(channel_boundaries, dtype=np.float64)

    # Partition data into time-slots.
    N_samples = int(fs * T)
    N_stream = data.shape[1]
    block_data = (sku.view_as_blocks(data[:((len(data) // N_samples) * N_samples)],
                                     (N_samples, N_stream))
                  .squeeze(axis=1))
    N_time_slot = block_data.shape[0]

    # Apply windowing
    tukey = func.Tukey(N_samples / fs,
                       N_samples / (fs * 2),
                       stft_window_alpha)
    window = tukey(np.arange(N_samples) / fs)
    block_data *= window.reshape(1, N_samples, 1)

    # Block-level STFT
    stft_data = fftpack.fft(block_data, axis=1)
    frequency = np.linspace(0, fs, N_samples)

    # Visibility formation for frequency bands of interest only.
    N_band = len(channel_boundaries)
    S = np.zeros((N_time_slot, N_band, N_stream, N_stream), dtype=np.complex64)

    idx = np.digitize(frequency, np.sort(channel_boundaries, axis=-1).flatten())
    for _idx in idx:
        if (0 < _idx < 2 * N_band) and chk.is_odd(_idx):
            mask = (idx == _idx)
            freq_data = stft_data[:, mask]
            S[:, (_idx - 1) // 2] += np.sum(freq_data[:, :, :, np.newaxis] *
                                            freq_data[:, :, np.newaxis, :].conj(),
                                            axis=1) / np.sum(mask)

    return N_samples, S
