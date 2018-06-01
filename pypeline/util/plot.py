# #############################################################################
# plot.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
`Matplotlib <https://matplotlib.org/>`_ helpers.
"""

import pathlib

import matplotlib.axes as axes
import matplotlib.cm as cm
import matplotlib.colors as col
import mpl_toolkits.axes_grid1 as ax_grid
import pandas as pd
import pkg_resources as pkg

import pypeline.util.argcheck as chk


@chk.check(dict(scm=chk.is_instance(cm.ScalarMappable),
                ax=chk.is_instance(axes.Axes)))
def colorbar(scm, ax):
    """
    Attach colorbar to side of a plot.

    Parameters
    ----------
    scm : :py:class:`~matplotlib.cm.ScalarMappable`
        Intensity scale.
    ax : :py:class:`~matplotlib.axes.Axes`
        Plot next to which the colorbar is placed.

    Returns
    -------
    :py:class:`~matplotlib.colorbar.Colorbar`
        Attached colorbar.

    Examples
    --------
    .. doctest::

       import numpy as np
       import matplotlib.pyplot as plt
       from pypeline.util.plot import colorbar

       x, y = np.ogrid[-1:1:100j, -1:1:100j]

       fig, ax = plt.subplots()
       im = ax.imshow(x + y, cmap='jet')
       cb = colorbar(im, ax)

       fig.show()

    .. image:: _img/colorbar_example.png
    """
    fig = ax.get_figure()
    divider = ax_grid.make_axes_locatable(ax)
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05,
                                      axes_class=axes.Axes)

    return fig.colorbar(scm, cax=ax_colorbar)


@chk.check(dict(name=chk.is_instance(str),
                N=chk.allow_None(chk.is_integer)))
def cmap(name, N=None):
    """
    Load one of Pypeline's custom colormaps.

    All maps are defined under ``<pypeline_dir>/data/colormap/``.

    Parameters
    ----------
    name : str
        colormap name.
    N : int, optional
        Number of color levels. (Default: all).

        If `N` is smaller than the number of levels available in the colormap, then the last `N` colors will be used.

    Returns
    -------
    :py:class:`~matplotlib.colors.ListedColormap`
        `Matplotlib <https://matplotlib.org/>`_ colormap.

    Examples
    --------
    .. doctest::

       import numpy as np
       import matplotlib.pyplot as plt
       from pypeline.util.plot import cmap

       x, y = np.ogrid[-1:1:100j, -1:1:100j]

       fig, ax = plt.subplots(ncols=2)
       ax[0].imshow(x + y, cmap='jet')
       ax[0].set_title('Jet')

       ax[1].imshow(x + y, cmap=cmap('matthieu-custom-sky'))
       ax[1].set_title('Matthieu-Custom-Sky')

       fig.show()

    .. image:: _img/cmap_example.png
    """
    if (N is not None) and (N <= 0):
        raise ValueError('Parameter[N] must be a positive integer.')

    cmap_rel_dir = pathlib.Path('data', 'colormap')
    cmap_rel_path = cmap_rel_dir / f'{name}.csv'

    if pkg.resource_exists('pypeline', str(cmap_rel_path)):
        cmap_abs_path = pkg.resource_filename('pypeline', str(cmap_rel_path))
        colors = (pd.read_csv(cmap_abs_path)
                  .loc[:, ['R', 'G', 'B']]
                  .values)

        N = len(colors) if (N is None) else N
        colormap = col.ListedColormap(colors[-N:])
        return colormap

    else:  # no cmap under that name.
        # List available cmaps.
        cmap_names = [pathlib.Path(_).stem for _ in
                      pkg.resource_listdir('pypeline', str(cmap_rel_dir))
                      if _.endswith('csv')]
        raise ValueError(f'{name} is not a pypeline-defined colormap. '
                         f'Available options: {cmap_names}')
