# #############################################################################
# FFTW3Config.cmake
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

# Find custom-compiled FFTW3 Libraries
#
# This module defines the following variables:
#
#    FFTW3_FOUND          - True if the library was found
#    FFTW3_INCLUDE_DIRS   - FFTW3 header file directory.
#    FFTW3_LIBRARIES      - FFTW3 libraries to link against.

include(FindPackageHandleStandardArgs)

### Header Directory ==========================================================
set (FFTW3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/")

### Libraries =================================================================
set (FFTW3_LIBRARY_SINGLE  "${PROJECT_SOURCE_DIR}/lib64/libfftw3.so")
if(${PYPELINE_USE_OPENMP})
    set (FFTW3_LIBRARY_OPENMP  "${PROJECT_SOURCE_DIR}/lib64/libfftw3_omp.so")
else(${PYPELINE_USE_OPENMP})
    set (FFTW3_LIBRARY_OPENMP  "")
endif(${PYPELINE_USE_OPENMP})

set(FFTW3_LIBRARY ${FFTW3_LIBRARY_SINGLE} ${FFTW3_LIBRARY_OPENMP})

find_package_handle_standard_args(FFTW3 DEFAULT_MSG
                                        FFTW3_INCLUDE_DIR
                                        FFTW3_LIBRARY)

if(FFTW3_FOUND)
    set(FFTW3_INCLUDE_DIRS "${FFTW3_INCLUDE_DIR}")
    set(FFTW3_LIBRARIES "${FFTW3_LIBRARY}")
endif()
