# #############################################################################
# io.py
# =====
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
IO-related operations and tools.
"""

import numpy as np

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.util.data_gen as dgen
import pypeline.util.argcheck as chk


@chk.check(dict(S=chk.is_instance(dgen.VisibilityMatrix),
                W=chk.is_instance(beamforming.BeamWeights)))
def filter_data(S, W):
    """
    Fix mis-matches to make data streams compatible.

    Visibility matrices from MS files typically include broken beams and/or may not match beams specified in beamforming matrices.
    This mis-match causes computations further down the imaging pypeline to be less efficient or completely wrong.
    This function applies 2 corrections to visibility and beamforming matrices to make them compliant:

    * Drop beams in `S` that do not appear in `W`;
    * Insert 0s in `W` where `S` has broken beams.

    Parameters
    ----------
    S : :py:class:`~pypeline.phased_array.util.data_gen.VisibilityMatrix`
        (N_beam1, N_beam1) visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) beamforming matrix.

    Returns
    -------
    S : :py:class:`~pypeline.phased_array.util.data_gen.VisibilityMatrix`
        (N_beam2, N_beam2) filtered visibility matrix.
    W : :py:class:`~pypeline.phased_array.beamforming.BeamWeights`
        (N_antenna, N_beam2) filtered beamforming matrix.
    """
    # Stage 1: Drop beams in S that do not appear in W
    beam_idx1 = S.index[0]
    beam_idx2 = W.index[1]
    beams_to_drop = beam_idx1.difference(beam_idx2)
    beams_to_keep = beam_idx1.drop(beams_to_drop)

    mask = np.any(beam_idx1.values.reshape(-1, 1) ==
                  beams_to_keep.values.reshape(1, -1), axis=1)
    S_f = dgen.VisibilityMatrix(data=S.data[np.ix_(mask, mask)],
                                beam_idx=beam_idx1[mask])

    # Stage 2: Insert 0s in W where S had broken beams
    broken_beam_idx = beam_idx2[np.isclose(np.sum(S_f.data, axis=1), 0)]
    mask = np.any(beam_idx2.values.reshape(-1, 1) ==
                  broken_beam_idx.values.reshape(1, -1), axis=1)

    w_f = W.data.copy()
    w_f[:, mask] = 0
    W_f = beamforming.BeamWeights(data=w_f,
                                  ant_idx=W.index[0],
                                  beam_idx=beam_idx2)

    return S_f, W_f
