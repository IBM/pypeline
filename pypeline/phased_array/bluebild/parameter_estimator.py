# #############################################################################
# parameter_estimator.py
# ======================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Parameter estimators.
"""

import numpy as np
import sklearn.cluster as skcl

import pypeline.phased_array.util.data_gen as dgen
import pypeline.phased_array.util.gram as gr
import pypeline.util.argcheck as chk
import pypeline.util.math.linalg as pylinalg


class ParameterEstimator:
    """
    Top-level public interface of Bluebild parameter estimators.
    """

    def __init__(self):
        """

        """
        super().__init__()

    def collect(self, *args, **kwargs):
        """
        Ingest data to internal queue for inference.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.
        """
        raise NotImplementedError

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        Returns
        -------
        tuple
            Parameters as defined by subclasses.
        """
        raise NotImplementedError


class IntensityFieldParameterEstimator(ParameterEstimator):
    """
    Parameter estimator for computing intensity fields.
    """

    @chk.check(dict(N_level=chk.is_integer,
                    sigma=chk.is_real))
    def __init__(self, N_level, sigma):
        """
        Parameters
        ----------
        N_level : int
            Number of clustered energy levels to output.
        sigma : float
            Normalized energy ratio for fPCA decomposition.
        """
        super().__init__()

        if N_level <= 0:
            raise ValueError('Parameter[N_level] must be positive.')
        self._N_level = N_level

        if not (0 < sigma <= 1):
            raise ValueError('Parameter[sigma] must lie in (0,1].')
        self._sigma = sigma

        # Collected data.
        self._visibilities = []
        self._grams = []

    @chk.check(dict(S=chk.is_instance(dgen.VisibilityMatrix),
                    G=chk.is_instance(gr.GramMatrix)))
    def collect(self, S, G):
        """
        Ingest data to internal queue for inference.

        Parameters
        ----------
        S : :py:class:`~pypeline.phased_array.util.data_gen.VisibilityMatrix`
            (N_beam, N_beam) visibility matrix.
        G : :py:class:`~pypeline.phased_array.util.gram.GramMatrix`
            (N_beam, N_beam) gram matrix.
        """
        if not S.is_consistent_with(G, axes=[0, 0]):
            raise ValueError('Parameters[S, G] are inconsistent.')

        self._visibilities.append(S)
        self._grams.append(G)

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        Returns
        -------
        N_eig : int
            Number of eigenpairs to use.

        cluster_centroid : :py:class:`~numpy.ndarray`
            (N_level,) intensity field cluster centroids.
        """
        N_data = len(self._visibilities)
        N_beam = N_eig_max = self._visibilities[0].shape[0]

        D_all = np.zeros((N_data, N_eig_max))
        for i, (S, G) in enumerate(zip(self._visibilities, self._grams)):
            # Remove broken BEAM_IDs
            broken_row_id = np.flatnonzero(np.isclose(np.sum(S.data, axis=0),
                                                      np.sum(S.data, axis=1)))
            working_row_id = list(set(np.arange(N_beam)) - set(broken_row_id))
            idx = np.ix_(working_row_id, working_row_id)
            S, G = S.data[idx], G.data[idx]

            # Functional PCA
            if not np.allclose(S, 0):
                D, _ = pylinalg.eigh(S, G, tau=self._sigma)
                D_all[i, :len(D)] = D

        D_all = D_all[D_all.nonzero()]
        kmeans = (skcl.KMeans(n_clusters=self._N_level)
                  .fit(np.log(D_all).reshape(-1, 1)))

        # For extremely small telescopes or datasets that are mostly 'broken', we can have (N_eig < N_level).
        # In this case we have two options: (N_level = N_eig) or (N_eig = N_level).
        # In the former case, the user's choice of N_level is not respected and subsequent code written by the user
        # could break due to a false assumption. In the latter case, we modify N_eig to match the user's choice.
        # This has the disadvantage of increasing the computational load of Bluebild, but as the N_eig energy levels
        # are clustered together anyway, the trailing energy levels will be (close to) all-0 and can be discarded
        # on inspection.
        N_eig = max(int(np.ceil(len(D_all) / N_data)), self._N_level)
        cluster_centroid = np.sort(np.exp(kmeans.cluster_centers_)[:, 0])[::-1]

        return N_eig, cluster_centroid


class SensitivityFieldParameterEstimator(ParameterEstimator):
    """
    Parameter estimator for computing sensitivity fields.
    """

    @chk.check('sigma', chk.is_real)
    def __init__(self, sigma):
        """
        Parameters
        ----------
        sigma : float
            Normalized energy ratio for fPCA decomposition.
        """
        super().__init__()

        if not (0 < sigma <= 1):
            raise ValueError('Parameter[sigma] must lie in (0,1].')
        self._sigma = sigma

        # Collected data.
        self._grams = []

    @chk.check('G', chk.is_instance(gr.GramMatrix))
    def collect(self, G):
        """
        Ingest data to internal queue for inference.

        Parameters
        ----------
        G : :py:class:`~pypeline.phased_array.util.gram.GramMatrix`
            (N_beam, N_beam) gram matrix.
        """
        self._grams.append(G)

    def infer_parameters(self):
        """
        Estimate parameters given ingested data.

        Returns
        -------
        N_eig : int
            Number of eigenpairs to use.
        """
        N_data = len(self._grams)
        N_beam = N_eig_max = self._grams[0].shape[0]

        D_all = np.zeros((N_data, N_eig_max))
        for i, G in enumerate(self._grams):
            # Functional PCA
            D, _ = pylinalg.eigh(G.data, np.eye(N_beam), tau=self._sigma)
            D_all[i, :len(D)] = D

        D_all = D_all[D_all.nonzero()]

        N_eig = int(np.ceil(len(D_all) / N_data))
        return N_eig
