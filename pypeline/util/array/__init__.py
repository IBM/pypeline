# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Tools and utilities for manipulating arrays.
"""

import _pypeline_util_array_pybind11 as __cpp

from . import _array as __py

LabeledMatrix = __py.LabeledMatrix
index = __py.index

cluster_layers = __cpp.cluster_layers
