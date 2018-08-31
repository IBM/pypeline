# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Field synthesizers that work in Fourier Series domain.
"""

import _pypeline_phased_array_bluebild_field_synthesizer_fourier_domain_pybind11 as __cpp

from . import _fourier_domain as __py

ReferenceFourierFieldSynthesizerBlock = __py.ReferenceFourierFieldSynthesizerBlock

FourierFieldSynthesizerBlock = __cpp.FourierFieldSynthesizerBlock_c128
