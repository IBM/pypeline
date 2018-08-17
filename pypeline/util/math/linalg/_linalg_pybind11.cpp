// ############################################################################
// _linalg_pybind11.cpp
// ====================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <utility>

#include "pybind11/pybind11.h"

#include "pypeline/util/cpp_py3_interop.hpp"
#include "pypeline/util/math/linalg.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace linalg = pypeline::util::math::linalg;

double py_z_rot2angle(pybind11::array_t<double> R) {
    const auto& R_view = cpp_py3_interop ::numpy_to_xview<double>(R);

    double angle = linalg::z_rot2angle(R_view);
    return angle;
}

pybind11::array_t<double> py_rot(pybind11::array_t<double> axis,
                                 const double angle) {
    const auto& axis_view = cpp_py3_interop::numpy_to_xview<double>(axis);

    auto R = linalg::rot(axis_view, angle);
    return cpp_py3_interop::xtensor_to_numpy(std::move(R));
}

PYBIND11_MODULE(_pypeline_util_math_linalg_pybind11, m) {
    pybind11::options options;
    options.disable_function_signatures();

    m.def("z_rot2angle",
          &py_z_rot2angle,
          pybind11::arg("R").none(false),
          pybind11::doc(R"EOF(
z_rot2angle(R)

Determine rotation angle from Z-axis rotation matrix.

Parameters
----------
R : :py:class:`~numpy.ndarray`
    (3, 3) rotation matrix around the Z-axis.

Returns
-------
angle : float
    Signed rotation angle [rad].

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.linalg import z_rot2angle

.. doctest::

   >>> R = np.eye(3)
   >>> angle = z_rot2angle(R)
   >>> np.around(angle, 2)
   0.0

   >>> R = [[0, -1, 0],
   ...      [1,  0, 0],
   ...      [0,  0, 1]]
   >>> angle = z_rot2angle(R)
   >>> np.around(angle, 2)
   1.57
)EOF"));

    m.def("rot",
          &py_rot,
          pybind11::arg("axis").none(false),
          pybind11::arg("angle").none(false),
          pybind11::doc(R"EOF(
rot(axis, angle)

3D rotation matrix.

Parameters
----------
axis : :py:class:`~numpy.ndarray`
    (3,) rotation axis.
angle : float
    Signed rotation angle [rad].

Returns
-------
R : :py:class:`~numpy.ndarray`
    (3, 3) rotation matrix.

Examples
--------
 .. testsetup::

    from pypeline.util.math.linalg import rot

 .. doctest::

    >>> R = rot(np.r_[0, 0, 1], np.pi / 2)
    >>> np.around(R, 2)
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])

    >>> R = rot([1, 0, 0], -1)
    >>> np.around(R, 2)
    array([[ 1.  ,  0.  ,  0.  ],
           [ 0.  ,  0.54,  0.84],
           [ 0.  , -0.84,  0.54]])
)EOF"));
}
