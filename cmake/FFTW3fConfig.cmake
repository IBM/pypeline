# #############################################################################
# FFTW3fConfig.cmake
# ==================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

# Find custom-compiled FFTW3f Libraries
#
# This module defines the following variables:
#
#    FFTW3f_FOUND          - True if the library was found
#    FFTW3f_INCLUDE_DIRS   - FFTW3f header file directory.
#    FFTW3f_LIBRARIES      - FFTW3f libraries to link against.

include(FindPackageHandleStandardArgs)

### Header Directory ==========================================================
set (FFTW3f_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/")

### Libraries =================================================================
set (FFTW3f_LIBRARY_SINGLE  "${PROJECT_SOURCE_DIR}/lib64/libfftw3f.so")
if(${PYPELINE_USE_OPENMP})
    set (FFTW3f_LIBRARY_OPENMP  "${PROJECT_SOURCE_DIR}/lib64/libfftw3f_omp.so")
else(${PYPELINE_USE_OPENMP})
    set (FFTW3f_LIBRARY_OPENMP  "")
endif(${PYPELINE_USE_OPENMP})

set(FFTW3f_LIBRARY ${FFTW3f_LIBRARY_SINGLE} ${FFTW3f_LIBRARY_OPENMP})

find_package_handle_standard_args(FFTW3f DEFAULT_MSG
                                         FFTW3f_INCLUDE_DIR
                                         FFTW3f_LIBRARY)

if(FFTW3f_FOUND)
    set(FFTW3f_INCLUDE_DIRS "${FFTW3f_INCLUDE_DIR}")
    set(FFTW3f_LIBRARIES "${FFTW3f_LIBRARY}")
endif()
