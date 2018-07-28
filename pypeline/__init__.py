# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Top-level Pypeline module.

Attributes
----------
config : :py:class:`~configparser.ConfigParser`
    Pypeline configuration data.
"""

import configparser
import os
import pathlib

import pkg_resources as pkg


def ___load_config():
    cfg = configparser.ConfigParser()

    # Load default configuration
    cfg_path = pkg.resource_filename('pypeline',
                                     str(pathlib.Path('data', 'pypeline.cfg')))
    with open(cfg_path, mode='r') as f:
        cfg.read_file(f)

    running_tests = bool(os.environ.get('PYPELINE_RUNNING_TESTS', False))
    if not running_tests:
        # Let user override defaults with his config file.
        u_cfg_path = pathlib.Path.home() / '.pypeline' / 'pypeline.cfg'
        if u_cfg_path.exists():
            cfg.read(u_cfg_path)
            print(f'Loaded user config from {u_cfg_path}.')

    return cfg


config = ___load_config()


def reload_config():
    """
    Reload Pypeline's configuration file(s).
    """

    global config

    config = ___load_config()
