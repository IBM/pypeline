# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Image containers, visualization and export facilities.
"""

import _pypeline_phased_array_util_io_image_pybind11 as __cpp

from . import _image as __py

from_fits = __py.from_fits
SphericalImage = __py.SphericalImage
# EqualAngleImage = __py.EqualAngleImage

SphericalImageContainer_float32 = __cpp.SphericalImageContainer_float32
SphericalImageContainer_float64 = __cpp.SphericalImageContainer_float64
