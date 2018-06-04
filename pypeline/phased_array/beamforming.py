# #############################################################################
# beamforming.py
# ==============
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Beamforming-related operations and tools.
"""

import collections.abc as abc

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import scipy.sparse as sparse

import pypeline
import pypeline.core as core
import pypeline.phased_array.instrument as instrument
import pypeline.util.argcheck as chk
import pypeline.util.array as array


def is_beam_index(x):
    """
    Return :py:obj:`True` if `x` is a valid beam index.

    A valid beam index is a :py:class:`~pandas.Index` with column

    * BEAM_ID : int
    """
    if (chk.is_instance(pd.Index)(x) and
            not chk.is_instance(pd.MultiIndex)(x)):
        col_name = 'BEAM_ID'
        if x.name == col_name:
            ids = x.values
            if chk.has_integers(ids):
                return True

    return False


def _as_BeamWeights(df):
    df = df.sort_values(by=['STATION_ID', 'ANTENNA_ID', 'BEAM_ID'])

    row_map = (df
               .loc[:, ['STATION_ID', 'ANTENNA_ID']]
               .drop_duplicates()
               .sort_values(['STATION_ID', 'ANTENNA_ID'])
               .assign(ROW_ID=lambda df: np.arange(len(df))))
    N_antenna = len(row_map)

    col_map = (df
               .loc[:, ['BEAM_ID']]
               .drop_duplicates()
               .sort_values('BEAM_ID')
               .assign(COL_ID=lambda s: np.arange(len(s))))
    N_beam = len(col_map)

    data = (df
            .merge(row_map, on=['STATION_ID', 'ANTENNA_ID'])
            .merge(col_map, on='BEAM_ID')
            .loc[:, ['ROW_ID', 'COL_ID', 'W']])

    sparsity_ratio = len(data) / (N_antenna * N_beam)
    max_sparsity_ratio = pypeline.config.getfloat('phased_array.beamforming',
                                                  'bw_max_sparsity_ratio')
    if sparsity_ratio <= max_sparsity_ratio:  # Use sparse matrix
        W = sparse.csr_matrix(
            (data.W.values, (data.ROW_ID.values, data.COL_ID.values)),
            shape=(N_antenna, N_beam),
            dtype=complex)
    else:  # Use dense matrix
        W = np.zeros(shape=(N_antenna, N_beam), dtype=complex)
        W[data.ROW_ID.values, data.COL_ID.values] = data.W.values

    ant_idx = (pd.MultiIndex
               .from_arrays([row_map.STATION_ID, row_map.ANTENNA_ID],
                            names=['STATION_ID', 'ANTENNA_ID']))
    beam_idx = pd.Index(col_map.BEAM_ID, name='BEAM_ID')

    bW = BeamWeights(W, ant_idx, beam_idx)
    return bW


class BeamWeights(array.LabeledMatrix):
    """
    Beamforming coefficients.

    Examples
    --------
    .. testsetup::

       from pypeline.phased_array.beamforming import BeamWeights
       import numpy as np
       import pandas as pd

    .. doctest::

       >>> N_antenna, N_beam = 5, 3
       >>> data = 1j * np.arange(N_antenna * N_beam).reshape(N_antenna, N_beam)
       >>> ant_idx = pd.MultiIndex.from_product([range(N_antenna), [0]],
       ...                                      names=['STATION_ID', 'ANTENNA_ID'])
       >>> beam_idx = pd.Index(range(3), name='BEAM_ID')
       >>> W = BeamWeights(data, ant_idx, beam_idx)

       >>> W.data
       array([[0. +0.j, 0. +1.j, 0. +2.j],
              [0. +3.j, 0. +4.j, 0. +5.j],
              [0. +6.j, 0. +7.j, 0. +8.j],
              [0. +9.j, 0.+10.j, 0.+11.j],
              [0.+12.j, 0.+13.j, 0.+14.j]])
    """

    @chk.check(dict(data=chk.accept_any(chk.has_complex, sparse.isspmatrix),
                    ant_idx=instrument.is_antenna_index,
                    beam_idx=is_beam_index))
    def __init__(self, data, ant_idx, beam_idx):
        """
        Parameters
        ----------
        data : array-like(complex)
            (N_antenna, N_beam) beamforming weights.
        ant_idx
            (N_antenna,) index.
        beam_idx
            (N_beam,) index.
        """
        if sparse.isspmatrix(data):
            if not np.issubdtype(data.dtype, np.complexfloating):
                raise ValueError('Parameter[data] must be complex-valued.')

        N_antenna, N_beam = len(ant_idx), len(beam_idx)
        if not chk.has_shape((N_antenna, N_beam))(data):
            raise ValueError('Parameters[data, ant_idx, beam_idx] are not '
                             'consistent.')

        super().__init__(data=data, row_idx=ant_idx, col_idx=beam_idx)


class BeamformerBlock(core.Block):
    """
    Compute beamweights for synthesis operators.
    """

    def __init__(self):
        """

        """
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        Determine beamweights to apply to each (antenna, beam) pair.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            Synthesis beamweights.
        """
        raise NotImplementedError


def is_mb_beam_config(x):
    """
    Return :py:obj:`True` if `x` is a valid matched-beamforming config specification.

    A beam config spec is considered valid if it is a collection of triplets (a, b, c), with

    * a : int
        station identifier;
    * b : int
        beam identifier;
    * c : :py:class:`~astropy.coordinates.SkyCoord`
        focal point of beam `b`.
    """
    if chk.is_instance(abc.Collection)(x):
        for entry in x:
            if not chk.is_instance(abc.Sequence)(entry):
                return False

            N_field = len(entry)
            if N_field != 3:
                return False

            station_id, beam_id = entry[:2]
            if not (chk.is_integer(station_id) and
                    chk.is_integer(beam_id)):
                return False

            focus_dir = entry[2]
            if not chk.is_instance(coord.SkyCoord)(focus_dir):
                return False
        return True

    return False


class MatchedBeamformerBlock(BeamformerBlock):
    """
    Compute matched-beamforming (MB) weights.
    """

    @chk.check('beam_config', is_mb_beam_config)
    def __init__(self, beam_config):
        """
        Parameters
        ----------
        beam_config
            Matched Beamforming configuration.
            Must satisfy :py:func:`~pypeline.phased_array.beamforming.is_mb_beam_config`.
        """
        super().__init__()

        N_info = len(beam_config)
        station_id = [None] * N_info
        beam_id = [None] * N_info
        focus_dir = [None] * N_info
        for i, (s_id, b_id, f_dir) in enumerate(beam_config):
            station_id[i] = s_id
            beam_id[i] = b_id
            focus_dir[i] = (f_dir
                            .transform_to('icrs')
                            .cartesian
                            .xyz
                            .value)

        station_id = np.array(station_id)
        beam_id = np.array(beam_id)
        focus_dir = np.stack(focus_dir, axis=1)

        self._config = pd.DataFrame(dict(STATION_ID=station_id,
                                         BEAM_ID=beam_id,
                                         F_X=focus_dir[0],
                                         F_Y=focus_dir[1],
                                         F_Z=focus_dir[2]))

    @chk.check(dict(XYZ=chk.is_instance(instrument.InstrumentGeometry),
                    freq=chk.is_frequency))
    def __call__(self, XYZ, freq):
        """
        Determine beamweights to apply to each (antenna, beam) pair.

        Parameters
        ----------
        XYZ : :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            Instrument geometry.
        freq : :py:class:`~astropy.units.Quantity`
            Frequency at which to generate beamweights.

        Returns
        -------
        :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            Synthesis beamweights.
        """
        wps = pypeline.config.getfloat('phased_array', 'wps') * (u.m / u.s)
        wl = (wps / freq).to_value(u.m)

        xyz = XYZ.as_frame()
        xyz = (xyz - xyz.mean()) / wl

        data = pd.merge(xyz.reset_index(), self._config, on='STATION_ID')
        XYZ = data.loc[:, ['X', 'Y', 'Z']].values
        F_XYZ = data.loc[:, ['F_X', 'F_Y', 'F_Z']].values
        similarity = np.squeeze(XYZ.reshape(-1, 1, 3) @
                                F_XYZ.reshape(-1, 3, 1), axis=(1, 2))
        W = np.exp((-1j * 2 * np.pi) * similarity)

        df = pd.DataFrame(dict(STATION_ID=data.STATION_ID,
                               ANTENNA_ID=data.ANTENNA_ID,
                               BEAM_ID=data.BEAM_ID,
                               W=W))
        return _as_BeamWeights(df)
