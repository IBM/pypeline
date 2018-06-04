# #############################################################################
# gram.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Gram-related operations and tools.
"""

import astropy.units as u
import numpy as np
import scipy.linalg as linalg

import pypeline
import pypeline.core as core
import pypeline.phased_array.beamforming as beamforming
import pypeline.util.argcheck as chk
import pypeline.util.array as array


class GramMatrix(array.LabeledMatrix):
    """
    Gram coefficients.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import pandas as pd
       from pypeline.phased_array.gram import GramMatrix

    .. doctest::

       >>> N_beam = 5
       >>> beam_idx = pd.Index(range(N_beam), name='BEAM_ID')
       >>> G = GramMatrix(np.eye(N_beam), beam_idx)

       >>> G.data
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
        data : array-like(complex)
            (N_beam, N_beam) Gram coefficients.
        beam_idx
            (N_beam,) index.
        """
        N_beam = len(beam_idx)
        if not chk.has_shape((N_beam, N_beam))(data):
            raise ValueError('Parameters[data, beam_idx] are not consistent.')

        super().__init__(data, beam_idx, beam_idx)


class GramBlock(core.Block):
    """
    Compute Gram matrices.
    """

    def __init__(self):
        """

        """
        super().__init__()

    def __call__(self, XYZ, W, freq):
        """
        Compute Gram matrix.

        Parameters
        ----------
        XYZ : :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            (N_antenna, 3) Cartesian antenna coordinates in any reference frame.
        W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            (N_antenna, N_beam) synthesis beamweights.
        freq : :py:class:`~astropy.units.Quantity`
            Frequency at which to compute the Gram.

        Returns
        -------
        :py:class:`~pypeline.phased_array.gram.GramMatrix`
            (N_beam, N_beam) Gram matrix.
        """
        wps = pypeline.config.getfloat('phased_array', 'wps') * (u.m / u.s)
        wl = (wps / freq).to_value(u.m)

        N_antenna = XYZ.shape[0]
        baseline = linalg.norm(XYZ.data.reshape(N_antenna, 1, 3) -
                               XYZ.data.reshape(1, N_antenna, 3), axis=-1)

        G_1 = (4 * np.pi) * np.sinc((2 / wl) * baseline)
        G_2 = W.data.conj().T @ G_1 @ W.data

        return GramMatrix(data=G_2, beam_idx=W.index[1])
