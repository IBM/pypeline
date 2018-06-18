# ##############################################################################
# special.py
# ==========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# ##############################################################################

"""
Special mathematical functions.
"""

import pathlib

import numpy as np
import pandas as pd
import pkg_resources as pkg

import pypeline.util.argcheck as chk


@chk.check('x', chk.is_real)
def jv_threshold(x):
    r"""
    Decay threshold of Bessel function :math:`J_{n}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    int
        Value of `n` in :math:`J_{n}(x)` past which :math:`J_{n}(x) \approx 0`.
    """
    rel_path = pathlib.Path('data', 'util', 'math',
                            'special', 'jv_threshold.csv')
    abs_path = pkg.resource_filename('pypeline', str(rel_path))

    data = pd.read_csv(abs_path).sort_values(by='x')
    x = np.abs(x)
    idx = int(np.digitize(x, bins=data['x'].values))
    if idx == 0:  # Below smallest known x.
        n = data['n_threshold'].iloc[0]
    else:
        if idx == len(data):  # Above largest known x.
            ratio = data['n_threshold'].iloc[-1] / data['x'].iloc[-1]
        else:
            ratio = data['n_threshold'].iloc[idx - 1] / data['x'].iloc[idx - 1]
        n = int(np.ceil(ratio * x))

    return n


@chk.check('x', chk.is_real)
def spherical_jn_threshold(x):
    r"""
    Decay threshold of spherical Bessel function :math:`j_{n}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    int
        Value of `n` in :math:`j_{n}(x)` past which :math:`j_{n}(x) \approx 0`.
    """
    rel_path = pathlib.Path('data', 'util', 'math',
                            'special', 'spherical_jn_threshold.csv')
    abs_path = pkg.resource_filename('pypeline', str(rel_path))

    data = pd.read_csv(abs_path).sort_values(by='x')
    x = np.abs(x)
    idx = int(np.digitize(x, bins=data['x'].values))
    if idx == 0:  # Below smallest known x.
        n = data['n_threshold'].iloc[0]
    else:
        if idx == len(data):  # Above largest known x.
            ratio = data['n_threshold'].iloc[-1] / data['x'].iloc[-1]
        else:
            ratio = data['n_threshold'].iloc[idx - 1] / data['x'].iloc[idx - 1]
        n = int(np.ceil(ratio * x))

    return n


@chk.check('x', chk.is_real)
def iv_threshold(x):
    r"""
    Decay threshold of Bessel function :math:`I_{n}(x)`.

    Parameters
    ----------
    x : float

    Returns
    -------
    int
        Value of `n` in :math:`I_{n}(x)` past which :math:`I_{n}(x) \approx 0`.
    """
    rel_path = pathlib.Path('data', 'util', 'math',
                            'special', 'iv_threshold.csv')
    abs_path = pkg.resource_filename('pypeline', str(rel_path))

    data = pd.read_csv(abs_path).sort_values(by='x')
    x = np.abs(x)
    idx = int(np.digitize(x, bins=data['x'].values))
    if idx == 0:  # Below smallest known x.
        n = data['n_threshold'].iloc[0]
    else:
        if idx == len(data):  # Above largest known x.
            ratio = data['n_threshold'].iloc[-1] / data['x'].iloc[-1]
        else:
            ratio = data['n_threshold'].iloc[idx - 1] / data['x'].iloc[idx - 1]
        n = int(np.ceil(ratio * x))

    return n
