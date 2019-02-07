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

import pypeline.core as core
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
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
       from pypeline.phased_array.util.gram import GramMatrix

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
        data = np.array(data, copy=False)
        N_beam = len(beam_idx)

        if not chk.has_shape((N_beam, N_beam))(data):
            raise ValueError('Parameters[data, beam_idx] are not consistent.')

        if not np.allclose(data, data.conj().T):
            raise ValueError('Parameter[data] must be hermitian symmetric.')

        super().__init__(data, beam_idx, beam_idx)


class GramBlock(core.Block):
    """
    Compute Gram matrices.
    """

    def __init__(self):
        """

        """
        super().__init__()

    @chk.check(dict(XYZ=chk.is_instance(instrument.InstrumentGeometry),
                    W=chk.is_instance(beamforming.BeamWeights),
                    wl=chk.is_wavelength))
    def __call__(self, XYZ, W, wl):
        """
        Compute Gram matrix.

        Parameters
        ----------
        XYZ : :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            (N_antenna, 3) Cartesian antenna coordinates in any reference frame.
        W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
            (N_antenna, N_beam) synthesis beamweights.
        wl : :py:class:`~astropy.units.Quantity`
            Wavelength at which to compute the Gram.

        Returns
        -------
        :py:class:`~pypeline.phased_array.gram.GramMatrix`
            (N_beam, N_beam) Gram matrix.

        Examples
        --------
        .. testsetup::

           import astropy.constants as constants
           import astropy.units as u
           import astropy.time as atime
           import astropy.coordinates as coord
           from pypeline.phased_array.instrument import LofarBlock
           from pypeline.phased_array.beamforming import MatchedBeamformerBlock
           from pypeline.phased_array.util.gram import GramBlock

        .. doctest::

           >>> instr = LofarBlock()
           >>> station_id = instr._layout.index.get_level_values('STATION_ID')
           >>> freq = 145 * u.MHz
           >>> wl = constants.c / freq

           >>> mb_cfg = [(_, _, coord.SkyCoord(0 * u.deg, 90 * u.deg))
           ...           for _ in station_id.drop_duplicates()]
           >>> mb = MatchedBeamformerBlock(mb_cfg)

           >>> XYZ = instr(atime.Time('J2000'))
           >>> W = mb(XYZ, wl)

           >>> gr = GramBlock()
           >>> G = gr(XYZ, W, wl)

           >>> np.around(np.abs(G.data[:4, :4]), 2)
           array([[3.0774e+02, 3.3000e-01, 1.0000e-02, 1.8000e-01],
                  [3.3000e-01, 3.0774e+02, 4.0000e-02, 9.0000e-02],
                  [1.0000e-02, 4.0000e-02, 3.0654e+02, 8.2000e-01],
                  [1.8000e-01, 9.0000e-02, 8.2000e-01, 2.6708e+02]])
        """
        if not XYZ.is_consistent_with(W, axes=[0, 0]):
            raise ValueError('Parameters[XYZ, W] are inconsistent.')

        wl = wl.to_value(u.m)

        N_antenna = XYZ.shape[0]
        baseline = linalg.norm(XYZ.data.reshape(N_antenna, 1, 3) -
                               XYZ.data.reshape(1, N_antenna, 3), axis=-1)

        G_1 = (4 * np.pi) * np.sinc((2 / wl) * baseline)
        G_2 = W.data.conj().T @ G_1 @ W.data

        return GramMatrix(data=G_2, beam_idx=W.index[1])
