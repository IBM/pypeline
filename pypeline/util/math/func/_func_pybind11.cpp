// ############################################################################
// _func_pybind11.cpp
// ==================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <string>

#include "pybind11/pybind11.h"
#include "xtensor/xtensor.hpp"

#include "pypeline/util/cpp_py3_interop.hpp"
#include "pypeline/util/math/func.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace func = pypeline::util::math::func;

template <typename T>
pybind11::array_t<double> py_Tukey___call__(const func::Tukey& tukey,
                                            pybind11::array_t<T> x) {
    const auto& xview = cpp_py3_interop::numpy_to_xview<T>(x);

    auto amplitude = const_cast<func::Tukey&>(tukey)(xview);
    return cpp_py3_interop::xtensor_to_numpy(std::move(amplitude));
}

template <typename T>
double py_Tukey___call__(const func::Tukey& tukey, double x) {
    xt::xtensor<double, 1> _x {x};

    double amplitude = const_cast<func::Tukey&>(tukey)(_x)[0];
    return amplitude;
}

PYBIND11_MODULE(_pypeline_util_math_func_pybind11, m) {
    pybind11::options options;
    options.disable_function_signatures();

    pybind11::class_<func::Tukey>(m, "Tukey", R"EOF(
Tukey(T, beta, alpha)

Parameterized Tukey function.

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.func import Tukey

.. doctest::

   >>> tukey = Tukey(T=1, beta=0.5, alpha=0.25)

   >>> sample_points = np.linspace(0, 1, 25).reshape(5, 5)  # any shape
   >>> amplitudes = tukey(sample_points)
   >>> np.around(amplitudes, 2)
   array([[0.  , 0.25, 0.75, 1.  , 1.  ],
          [1.  , 1.  , 1.  , 1.  , 1.  ],
          [1.  , 1.  , 1.  , 1.  , 1.  ],
          [1.  , 1.  , 1.  , 1.  , 1.  ],
          [1.  , 1.  , 0.75, 0.25, 0.  ]])

Notes
-----
The Tukey function is defined as:

.. math::

   \text{Tukey}(T, \beta, \alpha)(\varphi): \mathbb{R} & \to [0, 1] \\
   \varphi & \to
   \begin{cases}
       % LINE 1
       \sin^{2} \left( \frac{\pi}{T \alpha}
                \left[ \frac{T}{2} - \beta + \varphi \right] \right) &
       0 \le \frac{T}{2} - \beta + \varphi < \frac{T \alpha}{2} \\
       % LINE 2
       1 &
       \frac{T \alpha}{2} \le \frac{T}{2} - \beta +
       \varphi \le T - \frac{T \alpha}{2} \\
       % LINE 3
       \sin^{2} \left( \frac{\pi}{T \alpha}
                \left[ \frac{T}{2} + \beta - \varphi \right] \right) &
       T - \frac{T \alpha}{2} < \frac{T}{2} - \beta + \varphi \le T \\
       % LINE 4
       0 &
       \text{otherwise.}
   \end{cases}
)EOF")
        .def(pybind11::init<const double, const double, const double>(),
             pybind11::arg("T").none(false),
             pybind11::arg("beta").none(false),
             pybind11::arg("alpha").none(false),
             pybind11::doc(R"EOF(
__init__(T, beta, alpha)

Parameters
----------
T : float
    Function support.
beta : float
    Function mid-point.
alpha : float
   Decay-rate in [0, 1].
)EOF"))
        .def("__repr__",
             &func::Tukey::__repr__)
        .def("__call__",
             pybind11::overload_cast<const func::Tukey&,
                                     double>(&py_Tukey___call__<double>),
             pybind11::arg("x"))
        .def("__call__",
             pybind11::overload_cast<const func::Tukey&,
                                     pybind11::array_t<float>>(&py_Tukey___call__<float>),
             pybind11::arg("x").noconvert().none(false))
        .def("__call__",
             pybind11::overload_cast<const func::Tukey&,
                                     pybind11::array_t<double>>(&py_Tukey___call__<double>),
             pybind11::arg("x").noconvert().none(false),
             pybind11::doc(R"EOF(
__call__(x)

Sample the Tukey(T, beta, alpha) function.

Parameters
----------
x : float or :py:class:`~numpy.ndarray`
    Sample points.

Returns
-------
amplitude : float or :py:class:`~numpy.ndarray`
)EOF"));
}
