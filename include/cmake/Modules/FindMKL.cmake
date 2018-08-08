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
if(${PYPELINE_USE_OPENMP})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_library(MKL_THREADING_LIBRARY mkl_intel_thread
                     PATHS                 "${MKL_ROOT}/lib/")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        find_library(MKL_THREADING_LIBRARY mkl_gnu_thread
                     PATHS                 "${MKL_ROOT}/lib/")
    else(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "Unknown compiler.")
    endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
else(${PYPELINE_USE_OPENMP})
    find_library(MKL_THREADING_LIBRARY mkl_sequential
                 PATHS                 "${MKL_ROOT}/lib/")
endif(${PYPELINE_USE_OPENMP})

## Computational layer --------------------------------------------------------
find_library(MKL_CORE_LIBRARY mkl_core
             PATHS            "${MKL_ROOT}/lib/")
## RTL layer ------------------------------------------------------------------
if(${PYPELINE_USE_OPENMP})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_library(MKL_RTL_LIBRARY iomp5)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        find_library(MKL_RTL_LIBRARY gomp)
    else(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "Unknown compiler.")
    endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
else(${PYPELINE_USE_OPENMP})
    set(MKL_RTL_LIBRARY "")
endif(${PYPELINE_USE_OPENMP})

message(STATUS "*************************************************************")
message(STATUS "MKL_INTERFACE_LIBRARY : ${MKL_INTERFACE_LIBRARY}")
message(STATUS "MKL_THREADING_LIBRARY : ${MKL_THREADING_LIBRARY}")
message(STATUS "MKL_CORE_LIBRARY : ${MKL_CORE_LIBRARY}")
message(STATUS "MKL_RTL_LIBRARY : ${MKL_RTL_LIBRARY}")
message(STATUS "*************************************************************")

set(MKL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_RTL_LIBRARY})

find_package_handle_standard_args(MKL DEFAULT_MSG
                                      MKL_INCLUDE_DIR
                                      MKL_LIBRARY)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS "${MKL_INCLUDE_DIR}")
    set(MKL_LIBRARIES "${MKL_LIBRARY}")
endif()
