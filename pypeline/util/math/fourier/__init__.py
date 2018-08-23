# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
FFT-based tools.

The :py:class:`FFTW_xx` objects are more efficient implementations of their
lower-case counterparts written in C++.
"""

import _pypeline_util_math_fourier_pybind11 as __cpp

from . import _fourier as __py

ffs = __py.ffs
iffs = __py.iffs
czt = __py.czt
fs_interp = __py.fs_interp

ffs_sample = __cpp.ffs_sample
planning_effort = __cpp.planning_effort
FFTW_FFT = __cpp.FFTW_FFT
FFTW_FFS = __cpp.FFTW_FFS
FFTW_CZT = __cpp.FFTW_CZT
FFTW_FS_INTERP = __cpp.FFTW_FS_INTERP
