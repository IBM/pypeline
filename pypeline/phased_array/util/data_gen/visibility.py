# #############################################################################
# data_gen.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Visibility generation utilities.

Due to the high data-rates emanating from antennas, raw antenna time-series are rarely archived.
Instead, signals from different antennas are correlated together to form *visibility* matrices.
"""

import astropy.units as u
import numpy as np

import pypeline
import pypeline.core as core
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.phased_array.util.data_gen.sky as sky
import pypeline.util.argcheck as chk
import pypeline.util.array as array
import pypeline.util.math.stat as stat


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
        data : array-like(real, complex)
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
                    T=chk.is_duration,
                    fs=chk.is_frequency,
                    SNR=chk.is_real))
    def __init__(self, sky_model, T, fs, SNR):
        """
        Parameters
        ----------
        sky_model : :py:class:`~pypeline.phased_array.util.data_gen.sky.SkyEmission`
            Source model from which to generate data.
        T : :py:class:`~astropy.units.Quantity`
            Integration time.
        fs : :py:class:`~astropy.units.Quantity`
            Sampling rate.
        SNR : float
            Signal-to-Noise-Ratio (dB).
        """
        super().__init__()
        self._N_sample = int((T * fs).to_value(u.dimensionless_unscaled)) + 1
        self._SNR = 10 ** (SNR / 10)
        self._sky_model = sky_model

    @chk.check(dict(XYZ=chk.is_instance(instrument.InstrumentGeometry),
                    W=chk.is_instance(beamforming.BeamWeights),
                    freq=chk.is_frequency))
    def __call__(self, XYZ, W, freq):
        """
        Compute visibility matrix.

        Parameters
        ----------
        XYZ : :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            (N_antenna, 3) ICRS instrument geometry.
        W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            (N_antenna, N_beam) synthesis beamweights.
        freq : :py:class:`~astropy.units.Quantity`
            Frequency at which to generate visibilities.

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

        .. doctest::

           # Configure instrument and beamformer
           >>> instr = LofarBlock(N_station=24)
           >>> station_id = instr._layout.index.get_level_values('STATION_ID')

           >>> mb_cfg = [(_, _, coord.SkyCoord(0 * u.deg, 90 * u.deg))
           ...           for _ in station_id.drop_duplicates()]
           >>> mb = MatchedBeamformerBlock(mb_cfg)

           # Configure visibility generator
           >>> sky_model = from_tgss_catalog(coord.SkyCoord(0 * u.deg, 90 * u.deg),
           ...                               FoV=5 * u.deg,
           ...                               N_src=10)
           >>> S_gen = VisibilityGeneratorBlock(sky_model,
           ...                                  T=8 * u.s,
           ...                                  fs=196 * u.kHz,
           ...                                  SNR=np.inf)

           # Generate data
           >>> XYZ = instr(atime.Time('J2000'))
           >>> W = mb(XYZ, freq=145 * u.MHz)
           >>> S = S_gen(XYZ, W, freq=145 * u.MHz)

           # Only 10 sources & no noise, so rank(S) <= 10
           >>> np.linalg.matrix_rank(S.data) <= 10
           True
        """
        if not XYZ.is_consistent_with(W, axes=[0, 0]):
            raise ValueError('Parameters[XYZ, W] are inconsistent.')

        wps = pypeline.config.getfloat('phased_array', 'wps') * (u.m / u.s)
        wl = (wps / freq).to_value(u.m)

        A = np.exp((1j * 2 * np.pi / wl) *
                   (self._sky_model.xyz @ XYZ.data.T))
        S_sky = ((W.data.conj().T @ (A.conj().T * self._sky_model.intensity)) @
                 (A @ W.data))

        noise_var = np.sum(self._sky_model.intensity) / (2 * self._SNR)
        S_noise = W.data.conj().T @ (noise_var * W.data)

        wishart = stat.Wishart(V=S_sky + S_noise, n=self._N_sample)
        S = wishart()[0] / self._N_sample
        return VisibilityMatrix(data=S, beam_idx=W.index[1])
