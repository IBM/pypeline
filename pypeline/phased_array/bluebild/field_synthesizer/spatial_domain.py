# #############################################################################
# spatial_domain.py
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Field synthesizers that work in the spatial domain.
"""

import astropy.units as u
import numexpr as ne
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

import pypeline
import pypeline.phased_array.bluebild.field_synthesizer as synth
import pypeline.util.argcheck as chk


def _have_matching_shapes(V, XYZ, W):
    if (V.ndim == 2) and (XYZ.ndim == 2) and (W.ndim == 2):
        if V.shape[0] != W.shape[1]:  # N_beam
            return False
        if W.shape[0] != XYZ.shape[0]:  # N_antenna
            return False
        return True

    return False


class SpatialFieldSynthesizerBlock(synth.FieldSynthesizerBlock):
    """
    Field synthesizer based on StandardSynthesis.
    """

    @chk.check(dict(freq=chk.is_frequency,
                    pix_grid=chk.has_reals,
                    precision=chk.is_integer))
    def __init__(self, freq, pix_grid, precision=64):
        """
        Parameters
        ----------
        freq : :py:class:`~astropy.units.Quantity`
            Frequency of observations.
        pix_grid : :py:class:`~numpy.ndarray`
            (3, N_height, N_width) pixel vectors.
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

        wps = pypeline.config.getfloat('phased_array', 'wps') * (u.m / u.s)
        self._wl = (wps / freq).to_value(u.m)

        if not ((pix_grid.ndim == 3) and (len(pix_grid) == 3)):
            raise ValueError('Parameter[pix_grid] must have '
                             'dimensions (3, N_height, N_width).')
        self._grid = pix_grid / linalg.norm(pix_grid, axis=0)

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

            `XYZ` must be defined in the same reference frame as `pix_grid` from :py:meth:`~pypeline.phased_array.bluebild.field_synthesizer.spatial_domain.SpatialFieldSynthesizerBlock.__init__`.
        W : :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.csr_matrix` or :py:class:`~scipy.sparse.csc_matrix`
            (N_antenna, N_beam) synthesis beamweights.

        Returns
        -------
        stat : :py:class:`~numpy.ndarray`
            (N_eig, N_height, N_width) field statistics.

            (Note: StandardSynthesis statistics correspond to the actual field values.)
        """
        if not _have_matching_shapes(V, XYZ, W):
            raise ValueError('Parameters[V, XYZ, W] are inconsistent.')
        V = V.astype(self._cp, copy=False)
        XYZ = XYZ.astype(self._fp, copy=False)
        W = W.astype(self._cp, copy=False)

        N_antenna, N_beam = W.shape
        N_height, N_width = self._grid.shape[1:]

        XYZ = XYZ - XYZ.mean(axis=0)
        P = np.zeros((N_antenna, N_height, N_width), dtype=self._cp)
        ne.evaluate('exp(A * B)',
                    dict(A=1j * 2 * np.pi / self._wl,
                         B=np.tensordot(XYZ, self._grid, axes=1)),
                    out=P,
                    casting='same_kind')  # Due to limitations of NumExpr2

        PW = W.T @ P.reshape(N_antenna, N_height * N_width)
        PW = PW.reshape(N_beam, N_height, N_width)

        E = np.tensordot(V.T, PW, axes=1)
        I = E.real ** 2 + E.imag ** 2
        return I

    @chk.check('stat', chk.has_reals)
    def synthesize(self, stat):
        """
        Compute field values from statistics.

        Parameters
        ----------
        stat : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_width) field statistics.

        Returns
        -------
        field : :py:class:`~numpy.ndarray`
            (N_level, N_height, N_width) field values.
        """
        stat = np.array(stat, copy=False)

        if stat.ndim != 3:
            raise ValueError('Parameter[stat] is incorrectly shaped.')

        N_level = len(stat)
        N_height, N_width = self._grid.shape[1:]

        if not chk.has_shape([N_level, N_height, N_width])(stat):
            raise ValueError("Parameter[stat] does not match "
                             "the grid's dimensions.")

        field = stat
        return field
