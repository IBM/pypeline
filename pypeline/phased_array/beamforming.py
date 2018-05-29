# #############################################################################
# beamforming.py
# ==============
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Beamforming-related operations and tools.

.. todo::

   * Implement draw()
"""

import pandas as pd
import scipy.sparse as sparse

import pypeline.core as core
import pypeline.phased_array.instrument as instrument
import pypeline.util.argcheck as chk
import pypeline.util.array as array


class BeamformingMatrix(array.LabeledArray):
    """
    Attach antenna/beam identifiers to a beamforming matrix.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import scipy.sparse as sparse
       from pypeline.phased_array.beamforming import BeamformingMatrix
       import pandas as pd

    .. doctest::

       >>> N_station, N_antenna, N_beam = 3, 5, 4
       >>> antenna_id = (pd.MultiIndex
       ...               .from_product([range(N_station), range(N_antenna)],
       ...                             names=['STATION_NAME', 'ANTENNA_ID']))
       >>> beam_id = pd.Index(range(N_beam), name='BEAM_ID')

       # Dense beamforming matrix
       >>> data = np.ones((N_station * N_antenna, N_beam))
       >>> A = BeamformingMatrix(data, antenna_id, beam_id)
       >>> A.data[:3, :3]
       array([[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]])

       # Sparse beamforming matrix
       >>> data = sparse.csr_matrix(np.ones((N_station * N_antenna, N_beam)))
       >>> A = BeamformingMatrix(data, antenna_id, beam_id)
       >>> A.data[:3, :3].todense()
       matrix([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])
    """

    @chk.check(dict(data=chk.accept_any(chk.has_complex,
                                        chk.has_reals,
                                        chk.is_instance(sparse.spmatrix)),
                    antenna_id=chk.is_instance(pd.MultiIndex),
                    beam_id=chk.is_instance(pd.Index)))
    def __init__(self, data, antenna_id, beam_id):
        if not (antenna_id.names == ('STATION_NAME', 'ANTENNA_ID')):
            raise ValueError('Parameter[antenna_id] is not a valid index.')

        if beam_id.name != 'BEAM_ID':
            raise ValueError('Parameter[beam_id] is not a valid index.')

        super().__init__(data, antenna_id, beam_id)

    def is_sparse(self):
        """
        Returns
        -------
        bool
            :py:obj:`True` if the underlying data format is sparse.
        """
        return sparse.issparse(self.data)


class BeamformerBlock(core.Block):
    """
    Compute beamweights for synthesis operators.
    """

    def __init__(self):
        super().__init__()

    @chk.check(dict(inst_cfg=chk.is_instance(instrument.InstrumentConfig),
                    freq=chk.is_frequency))
    def __call__(self, inst_cfg, freq):
        """
        Compute beamforming matrix.

        Parameters
        ----------
        inst_cfg : :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
            Instrument configuration.
        freq : :py:class:`~astropy.units.Quantity`
            Frequency at which to compute the beamweights.

        Returns
        -------
        :py:class:`BeamformingMatrix`
            Synthesis beamweights.
        """
        raise NotImplementedError


class MatchedBeamformerBlock(BeamformerBlock):
    """
    Compute matched-beamforming weights.
    """
