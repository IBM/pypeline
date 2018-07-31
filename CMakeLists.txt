# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

cmake_minimum_required(VERSION 3.11)
project(pypeline
        DESCRIPTION "`GPLv3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_ signal processing toolkit to design and deploy holistic imaging pipelines."
        LANGUAGES C CXX)

set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_MODULE_PATH      ${CMAKE_MODULE_PATH}
                            "${PROJECT_SOURCE_DIR}/include/cmake/Modules/")

### C++/Python Standards ======================================================
set (CMAKE_CXX_STANDARD          17)
set (CMAKE_CXX_EXTENSIONS        OFF)  # Force c++1[7z] instead of gnu++1[7z]
set (CMAKE_CXX_STANDARD_REQUIRED ON)   # Fail if no c++17 compiler available
set (PYBIND11_CPP_STANDARD       -std=c++17)
set (PYTHON_EXECUTABLE           "$ENV{CONDA_PREFIX}/bin/python3")

### Automatic Header/Library Discovery ========================================
find_package(Eigen3       REQUIRED NO_MODULE)
find_package(MKL          REQUIRED)
find_package(OpenMP       REQUIRED)
find_package(pybind11     REQUIRED)
find_package(xsimd        REQUIRED)
find_package(xtensor      REQUIRED)

message(STATUS "")
message(STATUS "MKL Discovery ===============================================")
message(STATUS "MKL_INCLUDE_DIRS:      ${MKL_INCLUDE_DIRS}")
message(STATUS "MKL_LIBRARIES:         ${MKL_LIBRARIES}")
message(STATUS "PyBind11 Discovery ==========================================")
message(STATUS "PYBIND11_CPP_STANDARD: ${PYBIND11_CPP_STANDARD}")
message(STATUS "PYTHON_EXECUTABLE:     ${PYTHON_EXECUTABLE}")
message(STATUS "=============================================================")
message(STATUS "")

### Custom Header Search Paths ================================================
set (PYPELINE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include/)
include_directories(BEFORE ${MKL_INCLUDE_DIRS}
                           ${pybind11_INCLUDE_DIRS}
                           ${xsimd_INCLUDE_DIRS}
                           ${xtensor_INCLUDE_DIRS}
                           ${PYPELINE_INCLUDE_DIRS})

### Debug/Release Macro Definitions ===========================================
add_definitions(-DEIGEN_USE_MKL_ALL)  # Eigen MKL acceleration
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    add_definitions(-DXTENSOR_ENABLE_ASSERT)
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
    add_definitions(-DNDEBUG)   # disable (most) safety checks
    add_definitions(-DXTENSOR_USE_XSIMD)
endif(${CMAKE_BUILD_TYPE} STREQUAL Debug)

### Debug/Release Compilation Flags ===========================================
set (CMAKE_CXX_FLAGS         "-Wall -Wextra -m64")
set (CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -g -Og")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -mfma -mavx -O3 -march=native")

### BUILD Targets =============================================================
file(GLOB_RECURSE pypeline_cpp_files
     LIST_DIRECTORIES false
     "pypeline/*.cpp")
list(FILTER pypeline_cpp_files EXCLUDE REGEX ".*_pybind11.cpp")

## Pypeline C++ Linkable Library ----------------------------------------------
add_library(pypeline SHARED ${pypeline_cpp_files})
target_link_libraries(pypeline Eigen3::Eigen
                               ${MKL_LIBRARIES}
                               OpenMP::OpenMP_CXX)

## C++ Executables ------------------------------------------------------------
add_executable(test test.cpp)
target_link_libraries(test pypeline)

## Python Extension Modules ---------------------------------------------------
pybind11_add_module  (_pypeline_util_array_pybind11 pypeline/util/array/_array_pybind11.cpp)
target_link_libraries(_pypeline_util_array_pybind11 PRIVATE pypeline)
