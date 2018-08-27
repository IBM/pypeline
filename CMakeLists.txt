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
set (CMAKE_PREFIX_PATH      "${PROJECT_SOURCE_DIR}/cmake"
                            "${PROJECT_SOURCE_DIR}/lib/cmake"
                            "${PROJECT_SOURCE_DIR}/lib64/cmake")

### C++/Python Standards ======================================================
set (CMAKE_CXX_STANDARD          14)
set (CMAKE_CXX_EXTENSIONS        OFF)  # Force c++1[4y] instead of gnu++1[4y]
set (CMAKE_CXX_STANDARD_REQUIRED ON)   # Fail if no c++14 compiler available
set (PYBIND11_CPP_STANDARD       -std=c++14)
set (PYTHON_EXECUTABLE           "$ENV{CONDA_PREFIX}/bin/python3")

### Automatic Header/Library Discovery ========================================
find_package(OpenMP           REQUIRED)                                       # Provides target OpenMP::OpenMP_CXX
find_package(Eigen3    3.3.5  REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)  # Provides target Eigen3::Eigen
find_package(pybind11  2.2.3  REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)
find_package(xsimd     6.1.5  REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)
find_package(xtensor   0.17.1 REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)

# We provide custom module files for the following packages.
find_package(MKL              REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)
find_package(FFTW3            REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)
find_package(FFTW3f           REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)

### Custom Header Search Paths ================================================
# All Pypeline C++ code have their dependency headers (except MKL) under include/, so no need to specify each of them individually.
include_directories(BEFORE "${PROJECT_SOURCE_DIR}/include/"
                           "${MKL_INCLUDE_DIRS}")

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
file(GLOB_RECURSE libpypeline_files
     LIST_DIRECTORIES false
     "${PROJECT_SOURCE_DIR}/include/pypeline/*.hpp")

## Pypeline C++ Linkable Library ----------------------------------------------
add_library(pypeline SHARED ${libpypeline_files})
set_target_properties(pypeline PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(pypeline Eigen3::Eigen
                               ${FFTW3_LIBRARIES}
                               ${FFTW3f_LIBRARIES}
                               ${MKL_LIBRARIES})
if(${PYPELINE_USE_OPENMP})
    target_link_libraries(pypeline OpenMP::OpenMP_CXX)
endif(${PYPELINE_USE_OPENMP})

## C++ Executables ------------------------------------------------------------
add_executable(test test.cpp)
target_link_libraries(test pypeline)

## Python Extension Modules ---------------------------------------------------
pybind11_add_module  (_pypeline_util_array_pybind11 ${PROJECT_SOURCE_DIR}/pypeline/util/array/_array_pybind11.cpp)
target_link_libraries(_pypeline_util_array_pybind11 PRIVATE pypeline)

pybind11_add_module  (_pypeline_util_math_linalg_pybind11 ${PROJECT_SOURCE_DIR}/pypeline/util/math/linalg/_linalg_pybind11.cpp)
target_link_libraries(_pypeline_util_math_linalg_pybind11 PRIVATE pypeline)

pybind11_add_module  (_pypeline_util_math_func_pybind11 ${PROJECT_SOURCE_DIR}/pypeline/util/math/func/_func_pybind11.cpp)
target_link_libraries(_pypeline_util_math_func_pybind11 PRIVATE pypeline)

pybind11_add_module  (_pypeline_util_math_sphere_pybind11 ${PROJECT_SOURCE_DIR}/pypeline/util/math/sphere/_sphere_pybind11.cpp)
target_link_libraries(_pypeline_util_math_sphere_pybind11 PRIVATE pypeline)

pybind11_add_module  (_pypeline_util_math_fourier_pybind11 ${PROJECT_SOURCE_DIR}/pypeline/util/math/fourier/_fourier_pybind11.cpp)
target_link_libraries(_pypeline_util_math_fourier_pybind11 PRIVATE pypeline)

pybind11_add_module  (_pypeline_phased_array_util_io_image_pybind11 ${PROJECT_SOURCE_DIR}/pypeline/phased_array/util/io/image/_image_pybind11.cpp)
target_link_libraries(_pypeline_phased_array_util_io_image_pybind11 PRIVATE pypeline)

pybind11_add_module  (_pypeline_phased_array_bluebild_field_synthesizer_fourier_domain_pybind11 ${PROJECT_SOURCE_DIR}/pypeline/phased_array/bluebild/field_synthesizer/fourier_domain/_fourier_domain_pybind11.cpp)
target_link_libraries(_pypeline_phased_array_bluebild_field_synthesizer_fourier_domain_pybind11 PRIVATE pypeline)

### Install Build Targets to lib64/ ===========================================
install(TARGETS  pypeline
                _pypeline_util_array_pybind11
                _pypeline_util_math_linalg_pybind11
                _pypeline_util_math_func_pybind11
                _pypeline_util_math_sphere_pybind11
                _pypeline_util_math_fourier_pybind11
                _pypeline_phased_array_util_io_image_pybind11
                _pypeline_phased_array_bluebild_field_synthesizer_fourier_domain_pybind11
        LIBRARY
        DESTINATION "${PROJECT_SOURCE_DIR}/lib64/")
