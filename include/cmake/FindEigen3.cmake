# #############################################################################
# FindEigen3.cmake
# ================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

# Find Eigen3 Libraries
#
# This module defines the following variables:
#
#    EIGEN3_FOUND          - True if the library was found
#    EIGEN3_INCLUDE_DIRS   - Eigen3 header file directory.

include(FindPackageHandleStandardArgs)

### Header Directory ==========================================================
set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

find_package_handle_standard_args(EIGEN3 DEFAULT_MSG
                                         EIGEN3_INCLUDE_DIR)

if(EIGEN3_FOUND)
    set(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
endif()
