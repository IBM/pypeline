# #############################################################################
# instrument.py
# =============
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Instrument-related operations and configuration.

.. todo::

   * Add examples to illustrate usage of __init__() and __call__() methods;
   * Implement MwaBlock(), AskapBlock().
"""

import collections.abc as abc
import pathlib

import astropy.coordinates as coord
import astropy.time as time
import numpy as np
import pandas as pd
import pkg_resources as pkg
import plotly.graph_objs as go
import scipy.linalg as linalg

import pypeline.core as core
import pypeline.util.argcheck as chk


class StationConfig:
    """
    Configuration data of a single phased-array station.

    A phased-array station consists of a collection of antennas and provides
    the following information:

    * Station name;
    * Antenna identifiers;
    * Antenna coordinates: specified w.r.t an absolute or relative reference
      frame. (Typically specified by the object's creator.);
    * (More stuff in the future.)

    Examples
    --------
    .. testsetup::

       from pypeline.phased_array.instrument import StationConfig
       import numpy as np

    .. doctest::

       >>> cfg = StationConfig(name=0,
       ...                     antenna_ids=range(5),
       ...                     antenna_xyz=np.arange(3 * 5).reshape(3, 5))

       >>> cfg.name
       0

       >>> cfg.antenna_ids
      array([0, 1, 2, 3, 4])

       >>> cfg.xyz
       array([[ 0.,  1.,  2.,  3.,  4.],
              [ 5.,  6.,  7.,  8.,  9.],
              [10., 11., 12., 13., 14.]])
    """

    @chk.check(dict(name=chk.is_integer,
                    antenna_ids=chk.has_integers,
                    antenna_xyz=chk.has_reals))
    def __init__(self, name, antenna_ids, antenna_xyz):
        """
        Parameters
        ----------
        name : int
            Name of the station.
        antenna_ids : array-like(int)
            (N_antenna,) names of the individual antennas.
        antenna_xyz : array-like(float)
            (3, N_antenna) Cartesian coordinates of antennas.
        """
        self.__name = name

        antenna_ids = np.array(antenna_ids, dtype=int)
        N_antenna = antenna_ids.size
        if not chk.has_shape((N_antenna,))(antenna_ids):
            raise ValueError('Parameter[antenna_ids] must be 1-dimensional.')
        if np.unique(antenna_ids).size != N_antenna:
            raise ValueError('Parameter[antenna_ids] does not contain unique '
                             'names.')
        self.__antenna_ids = antenna_ids
        self.__antenna_ids.setflags(write=False)

        if not chk.has_shape((3, N_antenna))(antenna_xyz):
            raise ValueError('Lengths of Parameter[antenna_xyz] and '
                             'Parameter[antenna_ids] are not compatible.')
        self.__antenna_xyz = np.array(antenna_xyz, dtype=float)
        self.__antenna_xyz.setflags(write=False)

    @property
    def name(self):
        """
        Returns
        -------
        str
            Name of the station.
        """
        return self.__name

    @property
    def antenna_ids(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (N_antenna,) names of the individual antennas.
        """
        return self.__antenna_ids

    @property
    def xyz(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian coordinates of antennas.
        """
        return self.__antenna_xyz

    @chk.check('title', chk.is_instance(str))
    def draw(self, title=''):
        """
        Plot antenna coordinates in 3D.

        Parameters
        ----------
        title : str
            Title of the graph. (Default = '')

        Returns
        -------
        :py:class:`plotly.graph_objs.Figure`
            3D Figure.

        Examples
        --------
        .. doctest::

           import plotly.offline as py

           cfg = StationConfig(name=0,
                               antenna_ids=range(5),
                               antenna_xyz=np.arange(3 * 5).reshape(3, 5))

           fig = cfg.draw(title='Linear Station Geometry')
           py.plot(fig)

        Notes
        -----
        For visualization purposes, the station's center of gravity is mappped
        to the origin.
        """
        # Cartesian Axes
        arm_length = np.std(self.xyz, axis=1).max()
        frame_trace = go.Scatter3d(hoverinfo='text+name',
                                   line={'width': 6},
                                   marker={'color': 'rgb(0,0.5,0)'},
                                   mode='lines+markers+text',
                                   name='Reference Frame',
                                   text=['X', 'O', 'Y', 'O', 'Z'],
                                   textfont={'color': 'black'},
                                   textposition='middle center',
                                   x=[arm_length, 0, 0, 0, 0],
                                   y=[0, 0, arm_length, 0, 0],
                                   z=[0, 0, 0, 0, arm_length])

        # Station Layout
        X, Y, Z = self.xyz - self.xyz.mean(axis=1, keepdims=True)
        labels = [f'antenna {_}' for _ in self.antenna_ids]
        station_trace = go.Scatter3d(hoverinfo='text+name',
                                     marker={'size': 2},
                                     mode='markers',
                                     name=f'station {self.name}',
                                     text=labels,
                                     x=X,
                                     y=Y,
                                     z=Z)

        # Merge traces
        data = go.Data([frame_trace, station_trace])
        layout = go.Layout(scene={'aspectmode': 'data',
                                  'camera': {'eye': {'x': 1.25,
                                                     'y': 1.25,
                                                     'z': 1.25}},
                                  'xaxis': {'title': 'X [m]'},
                                  'yaxis': {'title': 'Y [m]'},
                                  'zaxis': {'title': 'Z [m]'}},
                           title=title)

        fig = go.Figure(data=data, layout=layout)
        return fig


class InstrumentConfig:
    """
    Configuration data for an entire instrument.

    A phased-array instrument consists of a collection of
    :py:class:`~pypeline.phased_array.instrument.StationConfig` objects and
    provides the following information:

    * Instrument name;
    * Access to individual station data.

    Examples
    --------
    .. testsetup::

       from pypeline.phased_array.instrument import (StationConfig,
                                                     InstrumentConfig)
       import numpy as np

    .. doctest::

       >>> st_cfgs = []
       >>> for i in range(5):
       ...     name = i
       ...     antenna_ids = np.arange(i, i + 4)
       ...     antenna_xyz = i + np.arange(3 * 4).reshape(3, 4)
       ...
       ...     st_cfgs.append(StationConfig(name, antenna_ids, antenna_xyz))

       >>> inst_cfg = InstrumentConfig(name='instrument',
       ...                             station_configs=st_cfgs)

       >>> inst_cfg.name
       'instrument'

       # Get the antenna identifiers of station '3'
       >>> inst_cfg.station[3].antenna_ids
       array([3, 4, 5, 6])

       # Get the antenna coordinataes of station '3'
       >>> inst_cfg.station[3].xyz
       array([[ 3.,  4.,  5.,  6.],
              [ 7.,  8.,  9., 10.],
              [11., 12., 13., 14.]])

       # Get the positions of all antennas.
       >>> inst_cfg.xyz.T
       array([[ 0.,  4.,  8.],
              [ 1.,  5.,  9.],
              [ 2.,  6., 10.],
              [ 3.,  7., 11.],
              [ 1.,  5.,  9.],
              [ 2.,  6., 10.],
              [ 3.,  7., 11.],
              [ 4.,  8., 12.],
              [ 2.,  6., 10.],
              [ 3.,  7., 11.],
              [ 4.,  8., 12.],
              [ 5.,  9., 13.],
              [ 3.,  7., 11.],
              [ 4.,  8., 12.],
              [ 5.,  9., 13.],
              [ 6., 10., 14.],
              [ 4.,  8., 12.],
              [ 5.,  9., 13.],
              [ 6., 10., 14.],
              [ 7., 11., 15.]])
    """

    @chk.check(dict(name=chk.is_instance(str),
                    station_configs=chk.is_instance(abc.Collection)))
    def __init__(self, name, station_configs):
        """
        Parameters
        ----------
        name : str
            Name of the instrument.
        station_configs : list( :py:class:`~pypeline.phased_array.instrument.StationConfig`)
            (N_station,) configs of the instrument's stations.
        """
        self.__name = name

        for station in station_configs:
            if not chk.is_instance(StationConfig)(station):
                raise ValueError('Parameter[station_configs] must only '
                                 'contain StationConfigs.')
        self.__station_configs = {_.name: _ for _ in station_configs}

        if len(self.__station_configs) != len(station_configs):
            raise ValueError('All stations in Parameter[station_configs] must '
                             'have unique names.')

        # Buffered attributes
        self.__xyz = None

    @property
    def name(self):
        """
        Returns
        -------
        str
            Name of instrument.
        """
        return self.__name

    @property
    def station(self):
        """
        Returns
        -------
        dict(int, :py:class:`~pypeline.phased_array.instrument.StationConfig`)
            Station name -> station data mapping.
        """
        return self.__station_configs

    @property
    def xyz(self):
        """
        Returns
        -------
        :py:class:`~numpy.ndarray`
            (3, N_antenna) Cartesian coordinates of all antennas in the
            instrument, sorted by ascending lexicographic station name.
        """
        if self.__xyz is None:
            station_names = sorted(self.station.keys())
            antenna_xyz = [self.station[_].xyz for _ in station_names]
            self.__xyz = np.concatenate(antenna_xyz, axis=1)
            self.__xyz.setflags(write=False)

        return self.__xyz

    @chk.check('title', chk.is_instance(str))
    def draw(self, title=''):
        """
        Plot antenna coordinates from all stations in 3D.

        Parameters
        ----------
        title : str
            Title of the graph. (Default = '')

        Returns
        -------
        :py:class:`plotly.graph_objs.Figure`
            3D Figure.

        Examples
        --------
        .. doctest::

           import plotly.offline as py

           st_cfgs = []
           for i in range(5):
               name = i
               antenna_ids = np.arange(i, i + 4)
               antenna_xyz = i + np.random.rand(3, 4)

               st_cfgs.append(StationConfig(name, antenna_ids, antenna_xyz))

           inst_cfg = InstrumentConfig(name='instrument',
                                       station_configs=st_cfgs)

           fig = inst_cfg.draw(title='Microphone Array')
           py.plot(fig)

        Notes
        -----
        For visualization purposes, the instrument's center of gravity is
        mapped to the origin.
        """
        # Cartesian Axes
        arm_length = np.std(self.xyz, axis=1).max()
        frame_trace = go.Scatter3d(hoverinfo='text+name',
                                   line={'width': 6},
                                   marker={'color': 'rgb(0,0.5,0)'},
                                   mode='lines+markers+text',
                                   name='Reference Frame',
                                   text=['X', 'O', 'Y', 'O', 'Z'],
                                   textfont={'color': 'black'},
                                   textposition='middle center',
                                   x=[arm_length, 0, 0, 0, 0],
                                   y=[0, 0, arm_length, 0, 0],
                                   z=[0, 0, 0, 0, arm_length])

        # Instrument Layout
        instrument_trace = []
        cog = np.mean(self.xyz, axis=1, keepdims=True)
        for station in self.station.values():
            X, Y, Z = station.xyz - cog
            labels = [f'antenna {_}' for _ in station.antenna_ids]
            station_trace = go.Scatter3d(hoverinfo='text+name',
                                         marker={'size': 2},
                                         mode='markers',
                                         name=f'station {station.name}',
                                         text=labels,
                                         x=X,
                                         y=Y,
                                         z=Z)
            instrument_trace += [station_trace]

        data = go.Data([frame_trace] + instrument_trace)
        layout = go.Layout(scene={'aspectmode': 'data',
                                  'camera': {'eye': {'x': 1.25,
                                                     'y': 1.25,
                                                     'z': 1.25}},
                                  'xaxis': {'title': 'X [m]'},
                                  'yaxis': {'title': 'Y [m]'},
                                  'zaxis': {'title': 'Z [m]'}},
                           title=title)

        fig = go.Figure(data=data, layout=layout)
        return fig


@chk.check(dict(name=chk.is_instance(str),
                layout=chk.is_instance(pd.DataFrame)))
def _as_InstrumentConfig(name, layout):
    """
    Transform :py:class:`~pandas.DataFrame` to
    :py:class:`~pypeline.phased_array.instrument.InstrumentConfig` format.

    Parameters
    ----------
    name : str
        Name of the instrument.
    layout : :py:class:`~pandas.DataFrame`
        Must have same format as the `__layout` attribute from
        :py:class:`~pypeline.phased_array.instrument.CoordinateComputerBlock`.

    Returns
    -------
    :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
    """
    station_cfg = []
    for station_name, station_layout in layout.groupby(level='STATION_NAME'):
        antenna_ids = (station_layout
                       .index
                       .get_level_values('ANTENNA_ID')
                       .tolist())
        antenna_xyz = station_layout.values.T
        cfg = StationConfig(station_name, antenna_ids, antenna_xyz)

        station_cfg.append(cfg)

    inst_cfg = InstrumentConfig(name, station_cfg)
    return inst_cfg


class CoordinateComputerBlock(core.Block):
    """
    Compute antenna positions.
    """

    @chk.check(dict(inst_config=chk.is_instance(InstrumentConfig),
                    N_station=chk.allow_None(chk.is_integer)))
    def __init__(self, inst_config, N_station=None):
        """
        Parameters
        ----------
        inst_config : :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
            Instrument geometry.
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument's stations are desired.
            Setting `N_station` limits the number of stations to those that
            lexicographically appear first in `inst_config`.
        """
        super().__init__()

        if N_station is not None:
            if N_station < 1:
                raise ValueError('Parameter[N_station] must be positive.')

        if N_station is None:
            valid_station = sorted(inst_config.station.keys())
        else:
            valid_station = sorted(inst_config.station.keys())[:N_station]

        station_xyz = {}
        for station in inst_config.station.values():
            if station.name in valid_station:
                df = pd.DataFrame(data=station.xyz.T,
                                  columns=('X', 'Y', 'Z'),
                                  index=pd.Index(station.antenna_ids,
                                                 name='ANTENNA_ID'))
                station_xyz[station.name] = df

        self._layout = pd.concat(station_xyz, names=('STATION_NAME',
                                                     'ANTENNA_ID'))

    def __call__(self, *args, **kwargs):
        """
        Determine instrument antenna positions.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
            Instrument configuration.
        """
        raise NotImplementedError


class StationaryArrayCoordinateComputerBlock(CoordinateComputerBlock):
    """
    Sub-class specialized in instruments that are stationary, i.e. that do not
    move with time.
    """

    @chk.check(dict(inst_config=chk.is_instance(InstrumentConfig),
                    N_station=chk.allow_None(chk.is_integer)))
    def __init__(self, inst_config, N_station=None):
        """
        Parameters
        ----------
        inst_config : :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
            Instrument geometry.
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument's stations are desired.
            Setting `N_station` limits the number of stations to those that
            lexicographically appear first in `inst_config`.
        """
        super().__init__(inst_config, N_station)

    def __call__(self):
        """
        Determine instrument antenna positions.

        As the instrument's configuration does not change through time,
        :py:func:`~pypeline.phased_array.instrument.StationaryArrayCoordinateComputerBlock.__call__`
        always returns the same object.

        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
            Instrument configuration.
        """
        return _as_InstrumentConfig('', self._layout)


class PyramicBlock(StationaryArrayCoordinateComputerBlock):
    """
    Pyramic 48-element 3D microphone array.
    """

    def __init__(self):
        """

        """
        inst_config = self._get_config()
        super().__init__(inst_config)

    def _get_config(self):
        """
        Generate instrument configuration.

        Returns
        -------
        :py:class:`pypeline.phased_array.instrument.InstrumentConfig`
        """
        x = 0.27 + 2 * 0.015  # length of one side
        c1, c2, c3, c4 = 1 / np.sqrt(3), np.sqrt(2 / 3), np.sqrt(3) / 6, 0.5
        corners = np.array([[0, x * c1, -x * c3, -x * c3],
                            [0, 0, x * c4, -x * c4],
                            [0, x * c2, x * c2, x * c2]])

        # Relative placement of microphones on one PCB.
        pcb = np.r_[-0.100, -0.060, -0.020, -0.004, 0.004, 0.020, 0.060, 0.100]

        def line(p1, p2, dist):
            center = (p1 + p2) / 2.
            unit_vec = (p2 - p1) / linalg.norm(p2 - p1)

            pts = [center + d * unit_vec for d in dist]
            return pts

        coordinates = np.array(line(corners[:, 0], corners[:, 3], pcb) +
                               line(corners[:, 3], corners[:, 2], pcb) +
                               line(corners[:, 0], corners[:, 1], pcb) +
                               line(corners[:, 1], corners[:, 3], pcb) +
                               line(corners[:, 0], corners[:, 2], pcb) +
                               line(corners[:, 2], corners[:, 1], pcb))

        # Reference point is 1cm below zero-th microphone
        coordinates[:, 2] += 0.01 - coordinates[0, 2]

        station_cfgs = []
        for i, xyz in enumerate(coordinates):
            station_name = i
            antenna_ids = [0]
            antenna_xyz = xyz.reshape(3, 1)

            cfg = StationConfig(station_name, antenna_ids, antenna_xyz)
            station_cfgs.append(cfg)

        inst_cfg = InstrumentConfig('Pyramic', station_cfgs)
        return inst_cfg


class CompactSixBlock(StationaryArrayCoordinateComputerBlock):
    """
    Compact-Six 6-element ring microphone array.
    """

    def __init__(self):
        """

        """
        inst_config = self._get_config()
        super().__init__(inst_config)

    def _get_config(self):
        """
        Generate instrument configuration.

        Returns
        -------
        :py:class:`pypeline.phased_array.instrument.InstrumentConfig`
        """
        N_mic = 6
        radius = 0.0675

        angle = np.linspace(0, 2 * np.pi, num=N_mic, endpoint=False)
        coordinates = radius * np.stack((np.cos(angle),
                                         np.sin(angle),
                                         np.zeros(N_mic)), axis=1)

        station_cfgs = []
        for i, xyz in enumerate(coordinates):
            station_name = i
            antenna_ids = [0]
            antenna_xyz = xyz.reshape(3, 1)

            cfg = StationConfig(station_name, antenna_ids, antenna_xyz)
            station_cfgs.append(cfg)

        inst_cfg = InstrumentConfig('CompactSix', station_cfgs)
        return inst_cfg


class EarthBoundArrayCoordinateComputerBlock(CoordinateComputerBlock):
    """
    Sub-class specialized in instruments that move with the Earth, such as
    radio telescopes.
    """

    @chk.check(dict(inst_config=chk.is_instance(InstrumentConfig),
                    N_station=chk.allow_None(chk.is_integer)))
    def __init__(self, inst_config, N_station=None):
        """
        Parameters
        ----------
        inst_config : :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
            ITRS instrument geometry.
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument's stations are desired.
            Setting `N_station` limits the number of stations to those that
            lexicographically appear first in `inst_config`.
        """
        super().__init__(inst_config, N_station)

    @chk.check('time', chk.is_instance(time.Time))
    def __call__(self, time):
        """
        Determine instrument antenna positions in ICRS.

        Parameters
        ----------
        time : :py:class:`~astropy.time.Time`
            Moment at which the coordinates are wanted.

        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.InstrumentConfig`
            ICRS instrument configuration.
        """
        layout = self._layout.loc[:, ['X', 'Y', 'Z']].values.T
        r = linalg.norm(layout, axis=0)

        itrs_layout = coord.CartesianRepresentation(layout)
        itrs_position = coord.SkyCoord(itrs_layout, obstime=time, frame='itrs')
        icrs_position = r * (itrs_position
                             .transform_to('icrs')
                             .cartesian
                             .xyz)
        icrs_layout = pd.DataFrame(data=icrs_position.T,
                                   index=self._layout.index,
                                   columns=('X', 'Y', 'Z'))
        return _as_InstrumentConfig('', icrs_layout)


class LofarBlock(EarthBoundArrayCoordinateComputerBlock):
    """
    LOw-Frequency ARray (LOFAR) located in Europe: lofar_.

    This LOFAR model consists of 62 stations, each containing between 17 to 24
    HBA dipole antennas.

    .. _lofar: http://www.lofar.org/
    """

    @chk.check('N_station', chk.allow_None(chk.is_integer))
    def __init__(self, N_station=None):
        """
        Parameters
        ----------
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument's stations are desired.
            Setting `N_station` limits the number of stations to those that
            lexicographically appear first in `inst_config`.
        """
        inst_cfg = self._get_config()
        super().__init__(inst_cfg, N_station)

    def _get_config(self):
        """
        Load instrument configuration.

        Returns
        -------
        :py:class:`pypeline.phased_array.instrument.InstrumentConfig`
            ITRS instrument geometry.
        """
        rel_path = pathlib.Path('data', 'phased_array',
                                'instrument', 'LOFAR.csv')
        abs_path = pkg.resource_filename('pypeline', str(rel_path))

        itrs_cfg = (pd.read_csv(abs_path)
                    .set_index(['STATION_NAME', 'ANTENNA_ID']))

        inst_cfg = _as_InstrumentConfig('LOFAR', itrs_cfg)
        return inst_cfg
