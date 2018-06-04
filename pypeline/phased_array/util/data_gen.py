# #############################################################################
# data_gen.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Tools and helper functions to generate synthetic data.
"""

import collections.abc as abc

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import plotly.graph_objs as go

import pypeline.util.argcheck as chk
import pypeline.util.math.sphere as sph


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
        theta, phi = np.meshgrid(np.linspace(0, 180, N_height) * u.deg,
                                 np.linspace(0, 360, N_width) * u.deg,
                                 indexing='ij')
        sph_X, sph_Y, sph_Z = sph.pol2cart(1, theta, phi)
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
