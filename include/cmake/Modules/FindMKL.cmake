# #############################################################################
# FindMKL.cmake
# =============
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

# Find Intel MKL Libraries
#
# This module defines the following variables:
#
#    MKL_FOUND          - True if the library was found
#    MKL_INCLUDE_DIRS   - MKL header file directory.
#    MKL_LIBRARIES      - MKL libraries to link against.

include(FindPackageHandleStandardArgs)
set(MKL_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "Conda environment root folder.")

### Header Directory ==========================================================
find_path(MKL_INCLUDE_DIR mkl.h
          PATHS           "${MKL_ROOT}/include/")

### Libraries =================================================================
# MKL is composed by four layers: Interface, Threading, Computational and RTL.
# We need to select one of each.
set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
## Interface layer ------------------------------------------------------------
find_library(MKL_INTERFACE_LIBRARY mkl_intel_lp64
             PATHS                 "${MKL_ROOT}/lib/")
## Threading layer ------------------------------------------------------------
find_library(MKL_THREADING_LIBRARY mkl_gnu_thread
             PATHS                 "${MKL_ROOT}/lib/")
## Computational layer --------------------------------------------------------
find_library(MKL_CORE_LIBRARY mkl_core
             PATHS            "${MKL_ROOT}/lib/")
## RTL layer ------------------------------------------------------------------
find_library(MKL_RTL_LIBRARY gomp)

set(MKL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_RTL_LIBRARY})

find_package_handle_standard_args(MKL DEFAULT_MSG
                                      MKL_INCLUDE_DIR
                                      MKL_LIBRARY)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS "${MKL_INCLUDE_DIR}")
    set(MKL_LIBRARIES "${MKL_LIBRARY}")
endif()
