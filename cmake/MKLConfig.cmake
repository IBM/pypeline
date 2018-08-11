# #############################################################################
# MKLConfig.cmake
# ===============
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
set (MKL_INCLUDE_DIR "${MKL_ROOT}/include/")

### Libraries =================================================================
# MKL is composed by four layers: Interface, Threading, Computational and RTL.
# We need to select one of each.
## Interface layer ------------------------------------------------------------
set (MKL_INTERFACE_LIBRARY "${MKL_ROOT}/lib/libmkl_intel_lp64.so")
## Threading layer ------------------------------------------------------------
if(${PYPELINE_USE_OPENMP})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set (MKL_THREADING_LIBRARY "${MKL_ROOT}/lib/libmkl_intel_thread.so")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set (MKL_THREADING_LIBRARY "${MKL_ROOT}/lib/libmkl_gnu_thread.so")
    else(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "Unknown compiler.")
    endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
else(${PYPELINE_USE_OPENMP})
    set (MKL_THREADING_LIBRARY "${MKL_ROOT}/lib/libmkl_sequential.so")
endif(${PYPELINE_USE_OPENMP})

## Computational layer --------------------------------------------------------
set (MKL_CORE_LIBRARY "${MKL_ROOT}/lib/libmkl_core.so")
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
