# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Linear algebra routines.
"""

import _pypeline_util_math_linalg_pybind11 as __cpp

from . import _linalg as __py

eigh = __py.eigh

rot = __cpp.rot
z_rot2angle = __cpp.z_rot2angle
