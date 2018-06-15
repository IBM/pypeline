# #############################################################################
# data_processor.py
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Data processors.
"""

import numpy as np

import pypeline.core as core
import pypeline.phased_array.util.data_gen.visibility as vis
import pypeline.phased_array.util.gram as gram
import pypeline.util.argcheck as chk
import pypeline.util.math.linalg as pylinalg


class DataProcessorBlock(core.Block):
    """
    Top-level public interface of Bluebild data processors.
    """

    def __init__(self):
        """

        """
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        fPCA decomposition and data formatting for :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock` objects.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.
        """
        raise NotImplementedError


class IntensityFieldDataProcessorBlock(DataProcessorBlock):
    """
    Data processor for computing intensity fields.
    """

    @chk.check(dict(N_eig=chk.is_integer,
                    cluster_centroids=chk.has_reals))
    def __init__(self, N_eig, cluster_centroids):
        """
        Parameters
        ----------
        N_eig : int
            Number of eigenpairs to output after PCA decomposition.
        cluster_centroids : array-like(float)
            Intensity centroids for energy-level clustering.

        Notes
        -----
        Both parameters should preferably be set by calling the :py:meth:`~pypeline.phased_array.bluebild.parameter_estimator.IntensityFieldParameterEstimator.infer_parameters` method from :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.IntensityFieldParameterEstimator`.
        """
        if N_eig <= 0:
            raise ValueError('Parameter[N_eig] must be positive.')

        super().__init__()
        self._N_eig = N_eig
        self._cluster_centroids = np.array(cluster_centroids, copy=False)

    @chk.check(dict(S=chk.is_instance(vis.VisibilityMatrix),
                    G=chk.is_instance(gram.GramMatrix)))
    def __call__(self, S, G):
        """
        fPCA decomposition and data formatting for :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock` objects.

        .. todo:: How to deal with ill-conditioned G?

        Parameters
        ----------
        S : :py:class:`~pypeline.phased_array.util.data_gen.visibility.VisibilityMatrix`
            (N_beam, N_beam) visibility matrix.
        G : :py:class:`~pypeline.phased_array.util.gram.GramMatrix`
            (N_beam, N_beam) gram matrix.

        Returns
        -------
        D : :py:class:`~numpy.ndarray`
            (N_eig,) positive eigenvalues.

        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.

        cluster_idx : :py:class:`~numpy.ndarray`
            (N_eig,) cluster indices of each eigenpair.

        Examples
        --------
        .. testsetup::

           from pypeline.phased_array.util.data_gen.visibility import VisibilityMatrix
           from pypeline.phased_array.util.gram import GramMatrix
           from pypeline.phased_array.bluebild.data_processor import IntensityFieldDataProcessorBlock
           import numpy as np
           import pandas as pd
           import scipy.linalg as linalg

           def hermitian_array(N: int) -> np.ndarray:
               '''
               Construct a (N, N) Hermitian matrix.
               '''
               D = np.arange(N)
               Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
               Q, _ = linalg.qr(Rmtx)

               A = (Q * D) @ Q.conj().T
               return A

           np.random.seed(0)

        .. doctest::

           >>> N_beam = 5
           >>> beam_idx = pd.Index(range(N_beam), name='BEAM_ID')

           # Some random visibility matrix
           >>> S = VisibilityMatrix(hermitian_array(N_beam), beam_idx)

           # Some random positive-definite Gram matrix
           >>> G = GramMatrix(hermitian_array(N_beam) + 100*np.eye(N_beam), beam_idx)

           # Get compact energy level descriptors.
           >>> I_dp = IntensityFieldDataProcessorBlock(N_eig=2,
           ...                                         cluster_centroids=[0., 20.])
           >>> D, V, cluster_idx = I_dp(S, G)

           >>> np.around(D, 2)
           array([0.04, 0.03])

           >>> cluster_idx  # useful for aggregation stage.
           array([0, 0])
        """
        if not S.is_consistent_with(G, axes=[0, 0]):
            raise ValueError('Parameters[S, G] are inconsistent.')

        # Remove broken BEAM_IDs
        N_beam = len(G.data)
        broken_row_id = np.flatnonzero(np.isclose(np.sum(S.data, axis=0),
                                                  np.sum(S.data, axis=1)))
        working_row_id = list(set(np.arange(N_beam)) - set(broken_row_id))
        idx = np.ix_(working_row_id, working_row_id)
        S, G = S.data[idx], G.data[idx]

        # Functional PCA
        if not np.allclose(S, 0):
            D, V = pylinalg.eigh(S, G, tau=1, N=self._N_eig)
        else:  # S is broken beyond use
            D, V = np.zeros(self._N_eig), 0

        # Add broken BEAM_IDs
        V_aligned = np.zeros((N_beam, self._N_eig), dtype=np.complex)
        V_aligned[working_row_id] = V

        # Determine energy-level clustering
        cluster_dist = np.absolute(D.reshape(-1, 1) -
                                   self._cluster_centroids.reshape(1, -1))
        cluster_idx = np.argmin(cluster_dist, axis=1)

        return D, V_aligned, cluster_idx


class SensitivityFieldDataProcessorBlock(DataProcessorBlock):
    """
    Data processor for computing sensitivity fields.
    """

    @chk.check('N_eig', chk.is_integer)
    def __init__(self, N_eig):
        """
        Parameters
        ----------
        N_eig : int
            Number of eigenpairs to output after PCA decomposition.

        Notes
        -----
        `N_eig` should preferably be set by calling the :py:meth:`~pypeline.phased_array.bluebild.parameter_estimator.SensitivityFieldParameterEstimator.infer_parameters` method from :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.SensitivityFieldParameterEstimator`.
        """
        if N_eig <= 0:
            raise ValueError('Parameter[N_eig] must be positive.')

        super().__init__()
        self._N_eig = N_eig

    @chk.check('G', chk.is_instance(gram.GramMatrix))
    def __call__(self, G):
        """
        fPCA decomposition and data formatting for :py:class:`~pypeline.phased_array.bluebild.field_synthesizer.FieldSynthesizerBlock` objects.

        Parameters
        ----------
        G : :py:class:`~pypeline.phased_array.util.gram.GramMatrix`
            (N_beam, N_beam) gram matrix.

        Returns
        -------
        D : :py:class:`~numpy.ndarray`
            (N_eig,) positive eigenvalues.

        V : :py:class:`~numpy.ndarray`
            (N_beam, N_eig) complex-valued eigenvectors.

        Examples
        --------
        .. testsetup::

           from pypeline.phased_array.util.gram import GramMatrix
           from pypeline.phased_array.bluebild.data_processor import SensitivityFieldDataProcessorBlock
           import numpy as np
           import pandas as pd
           import scipy.linalg as linalg

           def hermitian_array(N: int) -> np.ndarray:
               '''
               Construct a (N, N) Hermitian matrix.
               '''
               D = np.arange(N)
               Rmtx = np.random.randn(N,N) + 1j * np.random.randn(N, N)
               Q, _ = linalg.qr(Rmtx)

               A = (Q * D) @ Q.conj().T
               return A

           np.random.seed(0)

        .. doctest::

           # Some random positive-definite Gram matrix
           >>> N_beam = 5
           >>> beam_idx = pd.Index(range(N_beam), name='BEAM_ID')
           >>> G = GramMatrix(hermitian_array(N_beam) + 100*np.eye(N_beam), beam_idx)

           # Get compact energy level descriptors.
           >>> S_dp = SensitivityFieldDataProcessorBlock(N_eig=2)
           >>> D, V = S_dp(G)

           >>> np.around(D, 6)
           array([9.2e-05, 9.4e-05])
        """
        N_beam = len(G.data)
        D, V = pylinalg.eigh(G.data, np.eye(N_beam), tau=1, N=self._N_eig)
        Dg = 1 / (D ** 2)

        return Dg, V
