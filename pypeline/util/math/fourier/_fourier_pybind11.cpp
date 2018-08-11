// ############################################################################
// _fourier_pybind11.cpp
// =====================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <stdexcept>
#include <string>
#include <utility>

#include "pybind11/pybind11.h"

#include "pypeline/util/cpp_py3_interop.hpp"
#include "pypeline/util/math/fourier.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace fourier = pypeline::util::math::fourier;

pybind11::array_t<double> py_ffs_sample(const double T,
                                        const int N_FS,
                                        const double T_c,
                                        const int N_s) {
    if (N_FS < 0) {
        std::string msg = "Parameter[N_FS] must be positive.";
        throw std::runtime_error(msg);
    }
    size_t cpp_N_FS = N_FS;

    if (N_s < 0) {
        std::string msg = "Parameter[N_s] must be positive.";
        throw std::runtime_error(msg);
    }
    size_t cpp_N_s = N_s;

    const auto& sample_points = fourier::ffs_sample(T, cpp_N_FS, T_c, cpp_N_s);
    return cpp_py3_interop::xtensor_to_numpy(std::move(sample_points));
}

PYBIND11_MODULE(_pypeline_util_math_fourier_pybind11, m) {
    m.def("ffs_sample",
          &py_ffs_sample,
          pybind11::arg("T"),
          pybind11::arg("N_FS"),
          pybind11::arg("T_c"),
          pybind11::arg("N_s"),
          pybind11::doc(R"EOF(
Signal sample positions for :py:func:`~pypeline.util.math.fourier.ffs` and :py:class:`~pypeline.util.math.fourier.FFS`.

Return the coordinates at which a signal must be sampled to use :py:func:`~pypeline.util.math.fourier.ffs` and :py:class:`~pypeline.util.math.fourier.FFS`.

Parameters
----------
T : float
    Function period.
N_FS : int
    Function bandwidth.
T_c : float
    Period mid-point.
N_s : int
    Number of samples.

Returns
-------
:py:class:`~numpy.ndarray`
    (N_s,) coordinates at which to sample a signal (in the right order).

Examples
--------
Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a bandlimited periodic function of period :math:`T = 1`, bandwidth :math:`N_{FS} = 5`, and with one period centered at :math:`T_{c} = \pi`.
The sampling points :math:`t[n] \in \mathbb{R}` at which :math:`\phi` must be evaluated to compute the Fourier Series coefficients :math:`\left\{ \phi_{k}^{FS}, k = -2, \ldots, 2 \right\}` with :py:func:`~pypeline.util.math.fourier.ffs` and :py:class:`~pypeline.util.math.fourier.FFS` are obtained as follows:

.. testsetup::

   import numpy as np
   from pypeline.util.math.fourier import ffs_sample

.. doctest::

   # Ideally choose N_s to be highly-composite for ffs().
   >>> smpl_pts = ffs_sample(T=1, N_FS=5, T_c=np.pi, N_s=8)
   >>> np.around(smpl_pts, 2)  # Notice points are not sorted.
   array([3.2 , 3.33, 3.45, 3.58, 2.7 , 2.83, 2.95, 3.08])

See Also
--------
:py:func:`~pypeline.util.math.fourier.ffs`, :py:class:`~pypeline.util.math.fourier.FFS`.
)EOF"));
}
