// ############################################################################
// _array_pybind11.cpp
// ===================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "pypeline/types.hpp"
#include "pypeline/util/array.hpp"
#include "pypeline/util/cpp_py3_interop.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace array = pypeline::util::array;

template <typename T>
pybind11::array_t<T> py_cluster_layers(pybind11::array_t<T> x,
                                       std::vector<int> idx,
                                       const int N,
                                       const int axis) {
    auto xview = cpp_py3_interop::numpy_to_xview<T>(x);
    if (N <= 0) {
        std::string msg = "Parameter[N] must be positive.";
        throw std::runtime_error(msg);
    }
    size_t cpp_N = N;
    auto cpp_idx = cpp_py3_interop::cpp_index_convention(cpp_N, idx);
    auto cpp_axis = cpp_py3_interop::cpp_index_convention(xview.dimension(), axis);

    auto y = array::cluster_layers(xview, cpp_idx, cpp_N, cpp_axis);
    return cpp_py3_interop::xtensor_to_numpy(y);
}

PYBIND11_MODULE(_pypeline_util_array_pybind11, m) {
    m.def("cluster_layers",
          &py_cluster_layers<int32_t>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"));
    m.def("cluster_layers",
          &py_cluster_layers<int64_t>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"));
    m.def("cluster_layers",
          &py_cluster_layers<uint32_t>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"));
    m.def("cluster_layers",
          &py_cluster_layers<uint64_t>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"));
    m.def("cluster_layers",
          &py_cluster_layers<float>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"));
    m.def("cluster_layers",
          &py_cluster_layers<double>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"));
    m.def("cluster_layers",
          &py_cluster_layers<cfloat_t>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"));
    m.def("cluster_layers",
          &py_cluster_layers<cdouble_t>,
          pybind11::arg("x").noconvert(),
          pybind11::arg("idx"),
          pybind11::arg("N"),
          pybind11::arg("axis"),
          pybind11::doc(R"EOF(
Additive tensor compression along an axis.

Parameters
----------
x : :py:class:`~numpy.ndarray`
    (..., K, ...) array.
idx : array-like(int)
    (K,) cluster indices.
N : int
    Total number of levels along compression axis.
axis : int
    Dimension along which to compress.

Returns
-------
:py:class:`~numpy.ndarray`
    (..., N, ...) array

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.array import cluster_layers

.. doctest::

   >>> A = np.arange(5*3).reshape(5, 3)
   >>> B = cluster_layers(A, [0, 0, 1, 3, 5], N=10, axis=0)

   >>> B
   array([[ 3,  5,  7],
          [ 6,  7,  8],
          [ 0,  0,  0],
          [ 9, 10, 11],
          [ 0,  0,  0],
          [12, 13, 14],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]], dtype=int64)

)EOF"));
}
