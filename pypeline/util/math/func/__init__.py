# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Special functions.
"""

import _pypeline_util_math_func_pybind11 as __cpp

from . import _func as __py

SphericalDirichlet = __py.SphericalDirichlet

Tukey = __cpp.Tukey
