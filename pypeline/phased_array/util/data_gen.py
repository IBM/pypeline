# #############################################################################
# data_gen.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Tools and helper functions to generate synthetic data.
"""

import collections.abc as abc
import pathlib
import shutil
import urllib.request

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import pypeline
import pypeline.core as core
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.instrument as instrument
import pypeline.util.argcheck as chk
import pypeline.util.array as array
import pypeline.util.math.sphere as sph
import pypeline.util.math.stat as stat


def is_source_config(x):
    """
    Return :py:obj:`True` if `x` is a valid :py:class:`~pypeline.phased_array.util.data_gen.SkyEmission` config specification.

    A source config spec is considered valid if it is a collection of pairs (a, b), with

    * a : :py:class:`~astropy.coordinates.SkyCoord`
        source position in the sky;
    * b : float
        source intensity.
    """
    if chk.is_instance(abc.Collection)(x):
        for entry in x:
            if not chk.is_instance(abc.Sequence)(entry):
                return False

            N_field = len(entry)
            if N_field != 2:
                return False

            intensity = entry[1]
            if not (chk.is_real(intensity) and
                    (intensity > 0)):
                return False

            direction = entry[0]
            if not chk.is_instance(coord.SkyCoord)(direction):
                return False
        return True

    return False


class SkyEmission:
    """
    Data structure holding far-field source positions and intensities.

    Examples
    --------
    .. testsetup::

       from astropy.coordinates import SkyCoord
       import astropy.units as u
       import numpy as np
       from pypeline.phased_array.util.data_gen import SkyEmission

    .. doctest::

       >>> src_config = [(SkyCoord(_ * u.deg, _ * u.deg, frame='gcrs'), _)
       ...               for _ in range(1, 5, 1)]
       >>> sky_model = SkyEmission(src_config)

       >>> np.around(sky_model.xyz, 2)
       array([[1.  , 0.02, 0.02],
              [1.  , 0.03, 0.03],
              [1.  , 0.05, 0.05],
              [1.  , 0.07, 0.07]])

       >>> sky_model.intensity
       array([1, 2, 3, 4])
    """

    @chk.check('source_config', is_source_config)
    def __init__(self, source_config):
        """
        Parameters
        ----------
        source_config
            Source configuration.
            Must satisfy :py:func:`~pypeline.phased_array.util.data_gen.is_source_config`.
        """
        N_src = len(source_config)
        intensity = [None] * N_src
        direction = [None] * N_src
        for i, (s_dir, s_int) in enumerate(source_config):
            intensity[i] = s_int
            direction[i] = (s_dir
                            .transform_to('icrs')
                            .cartesian
                            .xyz
                            .value)

        self.__xyz = np.stack(direction, axis=0)
        self.__intensity = np.array(intensity)

    @property
    def xyz(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (N_src, 3) Cartesian ICRS source locations.
        """
        return self.__xyz

    @property
    def intensity(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (N_src,) source intensities.
        """
        return self.__intensity

    def draw(self):
        """
        Plot source positions in ICRS coordinates.

        Returns
        -------
        :py:class:`plotly.graph_objs.Figure`
            3D Figure.

        Examples
        --------
        .. doctest::

           import plotly.offline as py
           import pypeline.phased_array.util.data_gen as dgen

           src_config = {(SkyCoord(_ * u.deg, _ * u.deg, frame='gcrs'), _) for _ in range(1, 45, 1)}
           sky_model = dgen.SkyEmission(src_config)

           fig = sky_model.draw()
           py.plot(fig)
        """
        # Hollow Sphere
        N_height = N_width = 128
        colat, lon = np.meshgrid(np.linspace(0, 180, N_height) * u.deg,
                                 np.linspace(0, 360, N_width) * u.deg,
                                 indexing='ij')
        sph_X, sph_Y, sph_Z = sph.pol2cart(1, colat, lon)
        globe_trace = go.Surface(
            contours={'x': {'show': False, 'highlight': False},
                      'y': {'show': False, 'highlight': False},
                      'z': {'show': False, 'highlight': False}},
            hoverinfo='none',
            lighting={'diffuse': 0},
            opacity=0.5,
            showscale=False,
            surfacecolor=np.ones((N_height, N_width)),
            x=sph_X,
            y=sph_Y,
            z=sph_Z)

        # Cartesian Axes
        frame_trace = go.Scatter3d(hoverinfo='text+name',
                                   line={'width': 6},
                                   marker={'color': 'rgb(0,0.5,0)'},
                                   mode='lines+markers+text',
                                   name='ICRS frame',
                                   showlegend=False,
                                   text=['X', 'O', 'Y', 'O', 'Z'],
                                   textfont={'color': 'black'},
                                   textposition='middle center',
                                   x=[1, 0, 0, 0, 0],
                                   y=[0, 0, 1, 0, 0],
                                   z=[0, 0, 0, 0, 1])

        # Source Locations
        sX, sY, sZ = self.__xyz.T
        sI = self.__intensity
        src_trace = go.Scatter3d(hoverinfo='text+name',
                                 mode='markers',
                                 marker={'cmin': 0,
                                         'cmax': sI.max(),
                                         'color': sI,
                                         'colorbar': {'title': 'Intensity'},
                                         'colorscale': 'Viridis',
                                         'showscale': True,
                                         'size': 2},
                                 name='source',
                                 showlegend=False,
                                 x=sX,
                                 y=sY,
                                 z=sZ)

        data = go.Data([globe_trace, frame_trace, src_trace])
        layout = go.Layout(scene={'aspectmode': 'data',
                                  'camera': {'eye': {
                                      'x': 1.25,
                                      'y': 1.25,
                                      'z': 1.25}},
                                  'xaxis': {'title': 'X', 'range': [-1, 1]},
                                  'yaxis': {'title': 'Y', 'range': [-1, 1]},
                                  'zaxis': {'title': 'Z', 'range': [-1, 1]}},
                           title='ICRS Source Location')

        fig = go.Figure(data=data, layout=layout)
        return fig


@chk.check(dict(direction=chk.is_instance(coord.SkyCoord),
                FoV=chk.is_angle,
                N_src=chk.is_integer))
def from_tgss_catalog(direction, FoV, N_src):
    """
    Generate :py:class:`~pypeline.phased_array.util.data_gen.SkyEmission` from the `TGSS <http://tgssadr.strw.leidenuniv.nl/doku.php>`_ catalog.

    This function will automatically download and cache the catalog on disk for subsequent calls.

    Parameters
    ----------
    direction : :py:class:`~astropy.coordinates.SkyCoord`
        Direction in the sky.
    FoV : :py:class:`~astropy.units.Quantity`
        Spherical angle of the sky centered at `direction` from which sources are extracted.
    N_src : int
        Number of dominant sources to extract.

    Returns
    -------
    :py:class:`~pypeline.phased_array.util.data_gen.SkyEmission`
        Sky model.
    """
    if FoV.to_value(u.deg) <= 0:
        raise ValueError('Parameter[FoV] must be positive.')

    if N_src <= 0:
        raise ValueError('Parameter[N_src] must be positive.')

    disk_path = (pathlib.Path.home() /
                 '.pypeline' /
                 'catalog' /
                 'TGSSADR1_7sigma_catalog.tsv')

    if not disk_path.exists():
        # Download catalog from web.
        catalog_dir = disk_path.parent
        if not catalog_dir.exists():
            catalog_dir.mkdir(parents=True)

        web_path = ('http://tgssadr.strw.leidenuniv.nl/'
                    'catalogs/TGSSADR1_7sigma_catalog.tsv')
        print(f'Downloading catalog from {web_path}')
        with urllib.request.urlopen(web_path) as response:
            with disk_path.open(mode='wb') as f:
                shutil.copyfileobj(response, f)

    # Read catalog from disk path
    catalog_full = pd.read_table(disk_path)

    lat = catalog_full.loc[:, 'DEC'].values * u.deg
    lon = catalog_full.loc[:, 'RA'].values * u.deg
    xyz = sph.eq2cart(1, lat, lon)
    I = catalog_full.loc[:, 'Total_flux'].values * 1e-3  # mJy in catalog.

    # Reduce catalog to area of interest
    f_dir = (direction
             .transform_to('icrs')
             .cartesian
             .xyz
             .value)
    mask = (f_dir @ xyz) >= np.cos(FoV / 2)

    if mask.sum() < N_src:
        raise ValueError('There are less than Parameter[N_src] '
                         'sources in the field.')

    I_region, xyz_region = I[mask], xyz[:, mask]
    idx = np.argsort(I_region)[-N_src:]
    I_region, xyz_region = I_region[idx], xyz_region[:, idx]
    _, lat_region, lon_region = sph.cart2eq(*xyz_region)

    source_config = [(coord.SkyCoord(az, el, frame='icrs'), intensity)
                     for el, az, intensity in
                     zip(lat_region, lon_region, I_region)]
    sky_model = SkyEmission(source_config)
    return sky_model


class VisibilityMatrix(array.LabeledMatrix):
    """
    Visibility coefficients.

    Examples
    --------
    .. testsetup::

       import numpy as np
       import pandas as pd
       from pypeline.phased_array.util.data_gen import VisibilityMatrix

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
        data : array-like(complex)
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

    @chk.check(dict(T=chk.is_duration,
                    fs=chk.is_frequency,
                    SNR=chk.is_real))
    def __init__(self, T, fs, SNR):
        """
        Parameters
        ----------
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

    @chk.check(dict(XYZ=chk.is_instance(instrument.InstrumentGeometry),
                    W=chk.is_instance(beamforming.BeamWeights),
                    freq=chk.is_frequency,
                    sky_model=chk.is_instance(SkyEmission)))
    def __call__(self, XYZ, W, freq, sky_model):
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
        sky_model : :py:class:`~pypeline.phased_array.util.data_gen.SkyEmission`
            (N_src,) source description.

        Returns
        -------
        :py:class:`~pypeline.phased_array.util.data_gen.VisibilityMatrix`
            (N_beam, N_beam) visibility matrix.
        """
        wps = pypeline.config.getfloat('phased_array', 'wps') * (u.m / u.s)
        wl = (wps / freq).to_value(u.m)

        A = np.exp((1j * 2 * np.pi / wl) * (sky_model.xyz @ XYZ.data.T))
        S_sky = ((W.data.conj().T @ (A.conj().T * sky_model.intensity)) @
                 (A @ W.data))

        noise_var = np.sum(sky_model.intensity) / (2 * self._SNR)
        S_noise = W.data.conj().T @ (noise_var * W.data)

        S = stat.wishrnd(S=S_sky + S_noise, df=self._N_sample) / self._N_sample
        return VisibilityMatrix(data=S, beam_idx=W.index[1])
