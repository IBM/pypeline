# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Spherical geometry tools.
"""

import _pypeline_util_math_sphere_pybind11 as __cpp

from . import _sphere as __py

ea_sample = __py.ea_sample
EqualAngleInterpolator = __py.EqualAngleInterpolator

colat2lat = __cpp.colat2lat
lat2colat = __cpp.lat2colat
eq2cart = __cpp.eq2cart
pol2cart = __cpp.pol2cart
cart2pol = __cpp.cart2pol
cart2eq = __cpp.cart2eq
