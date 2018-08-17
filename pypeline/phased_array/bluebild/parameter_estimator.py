# #############################################################################
# parameter_estimator.py
# ======================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

r"""
Parameter estimators.

Bluebild field synthesizers output :math:`N_{\text{beam}}` energy levels, with :math:`N_{\text{beam}}` being the height of the visibility/Gram matrices :math:`\Sigma, G`.
We are often not interested in such fined-grained energy decompositions but would rather have 4-5 well-separated energy levels as output.
This is accomplished by clustering energy levels together during the aggregation stage.

As the energy scale depends on the visibilities, it is preferable to infer the cluster centroids (and any other parameters of interest) by scanning a portion of the data stream.
Subclasses of :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.ParameterEstimator` are specifically tailored for such tasks.
"""

import numpy as np
import sklearn.cluster as skcl

import pypeline.phased_array.util.data_gen.visibility as vis
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

    Examples
    --------
    Assume we are imaging a portion of the Bootes field with LOFAR's 24 core stations.
    As 24 energy levels exhibit a smooth spectrum, we decide to aggregate them into 4 well-separated energy levels through clustering.

    The short script below shows how to use :py:class:`~pypeline.phased_array.bluebild.parameter_estimator.IntensityFieldParameterEstimator` to optimally choose cluster centroids.

    .. testsetup::

       import numpy as np
       import astropy.units as u
       import astropy.time as atime
       import astropy.coordinates as coord
       from pypeline.phased_array.bluebild.parameter_estimator import IntensityFieldParameterEstimator
       from pypeline.phased_array.instrument import LofarBlock
       from pypeline.phased_array.beamforming import MatchedBeamformerBlock
       from pypeline.phased_array.util.gram import GramBlock
       from pypeline.phased_array.util.data_gen.sky import from_tgss_catalog
       from pypeline.phased_array.util.data_gen.visibility import VisibilityGeneratorBlock
       from scipy.constants import speed_of_light

       np.random.seed(0)

    .. doctest::

       ### Experiment setup ================================================
       # Observation
       >>> obs_start = atime.Time(56879.54171302732, scale='utc', format='mjd')
       >>> field_center = coord.SkyCoord(218 * u.deg, 34.5 * u.deg)
       >>> field_of_view = np.radians(5)
       >>> frequency = 145e6
       >>> wl = speed_of_light / frequency

       # instrument
       >>> N_station = 24
       >>> dev = LofarBlock(N_station)
       >>> mb = MatchedBeamformerBlock([(_, _, field_center) for _ in range(N_station)])
       >>> gram = GramBlock()

       # Visibility generation
       >>> vis = VisibilityGeneratorBlock(sky_model=from_tgss_catalog(field_center, field_of_view, N_src=10),
       ...                                T=8,
       ...                                fs=196e3,
       ...                                SNR=np.inf)

       ### Parameter estimation ============================================
       >>> I_est = IntensityFieldParameterEstimator(N_level=4, sigma=0.95)
       >>> t_est = obs_start + np.arange(20) * 400 * u.s  # sample visibilities throughout the 8h observation.
       >>> for t in t_est:
       ...    XYZ = dev(t)
       ...    W = mb(XYZ, wl)
       ...    S = vis(XYZ, W, wl)
       ...    G = gram(XYZ, W, wl)
       ...
       ...    I_est.collect(S, G)  # Store S, G internally until full 8h interval has been sampled.
       ...
       >>> N_eig, c_centroid = I_est.infer_parameters()  # optimal estimate

       # notice that it is less than N_src=10 because sigma=0.95 made
       # it throw away the trailing eigenpairs that account for 5% of
       # the sky's energy.
       >>> N_eig
       7

       >>> np.around(c_centroid, 3)
       array([124.927,  65.09 ,  38.589,  23.256])
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

    @chk.check(dict(S=chk.is_instance(vis.VisibilityMatrix),
                    G=chk.is_instance(gr.GramMatrix)))
    def collect(self, S, G):
        """
        Ingest data to internal queue for inference.

        Parameters
        ----------
        S : :py:class:`~pypeline.phased_array.util.data_gen.visibility.VisibilityMatrix`
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
