// ############################################################################
// _fourier_pybind11.cpp
// =====================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "pypeline/util/cpp_py3_interop.hpp"
#include "pypeline/util/math/fourier.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace fourier = pypeline::util::math::fourier;

void ffs_sample_bindings(pybind11::module &m) {
    m.def("ffs_sample",
          [](const double T,
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
          },
          pybind11::arg("T").none(false),
          pybind11::arg("N_FS").none(false),
          pybind11::arg("T_c").none(false),
          pybind11::arg("N_s").none(false),
          pybind11::doc(R"EOF(
ffs_sample(T, N_FS, T_c, N_s)

Signal sample positions for :py:func:`~pypeline.util.math.fourier.ffs`.

Return the coordinates at which a signal must be sampled to use :py:func:`~pypeline.util.math.fourier.ffs`.

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
The sampling points :math:`t[n] \in \mathbb{R}` at which :math:`\phi` must be evaluated to compute the Fourier Series coefficients :math:`\left\{ \phi_{k}^{FS}, k = -2, \ldots, 2 \right\}` with :py:func:`~pypeline.util.math.fourier.ffs` are obtained as follows:

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
:py:func:`~pypeline.util.math.fourier.ffs`
)EOF"));
}

void planning_effort_bindings(pybind11::module &m) {
    auto obj = pybind11::enum_<fourier::planning_effort>(m,
                                                         "planning_effort",
                                                         R"EOF(
FFTW planning flags that can be used in FFTW_xx objects.

Available options are:

* :py:attr:`~pypeline.util.math.fourier.planning_effort.NONE`;
* :py:attr:`~pypeline.util.math.fourier.planning_effort.MEASURE`.
)EOF");

    obj.value("NONE", fourier::planning_effort::NONE);
    obj.value("MEASURE", fourier::planning_effort::MEASURE);
}

template <typename T>
void FFTW_FFT_bindings(pybind11::module &m,
                       const std::string &class_name) {
    auto obj = pybind11::class_<fourier::FFTW_FFT<T>>(m,
                                                      class_name.data(),
                                                      R"EOF(
FFTW_FFT(shape, axis, inplace, N_threads, effort)

FFTW wrapper to plan 1D complex->complex (i)FFTs on multi-dimensional tensors.

This object automatically allocates input/output buffers and provides a
tensor interface to the underlying memory using NumPy arrays.

Examples
--------
.. testsetup::

   import numpy as np
   import scipy.fftpack as fftpack
   from pypeline.util.math.fourier import FFTW_FFT, planning_effort

.. doctest::

   >>> shape, axis = (3, 4), 0
   >>> inplace, N_threads = False, 1
   >>> effort = planning_effort.NONE

   >>> transform = FFTW_FFT(shape, axis, inplace, N_threads, effort)

   >>> transform.input[:] = 1
   >>> transform.fft()
   >>> np.allclose(transform.output, fftpack.fft(transform.input, axis=axis))
   True
)EOF");

    obj.def(pybind11::init([](std::vector<int> shape,
                              const int axis,
                              const bool inplace,
                              const int N_threads,
                              fourier::planning_effort effort) {
        /*
         * PYBIND11 BUG
         * ------------
         * Tensors of rank \ge 2 suffer from numerous problems:
         * * In Debug Mode, FFTW plans fail allocation assertions.
         * * In Release Mode, FFTW plans pass allocation assertions, but transform
         *   output are garbage.
         */
        if (shape.size() > 2) {
            std::string msg = "PYBIND11 BUG: Transforms on tensors of rank >= 3 currently disabled.";
            throw std::runtime_error(msg);
        }

        if (xt::any(xt::adapt(shape) <= 0)) {
            std::string msg = "Parameter[shape] must contain positive integers.";
            throw std::runtime_error(msg);
        }
        std::vector<size_t> cpp_shape(shape.size());
        std::copy(shape.begin(), shape.end(), cpp_shape.begin());

        const auto& cpp_axis = cpp_py3_interop::cpp_index_convention(shape.size(), axis);

        if (N_threads < 0) {
            std::string msg = "Parameter[N_threads] must be positive.";
            throw std::runtime_error(msg);
        }

        return std::make_unique<fourier::FFTW_FFT<T>>(cpp_shape, cpp_axis,
                                                      inplace, N_threads, effort);
    }), pybind11::arg("shape").none(false),
        pybind11::arg("axis").none(false),
        pybind11::arg("inplace").none(false),
        pybind11::arg("N_threads").none(false),
        pybind11::arg("effort").none(false),
        pybind11::doc(R"EOF(
__init__(shape, axis, inplace, N_threads, effort)

Parameters
----------
shape : tuple(int)
    Dimensions of input/output arrays.
axis : int
    Dimension along which to apply transform.
inplace : bool
    Perform in-place transforms.
    If enabled, only one array will be internally allocated.
N_threads : int
    Number of threads to use.
effort : :py:class:`~pypeline.util.math.fourier.planning_effort`
    Amount of time spent finding the best transform.

Notes
-----
* Input and output buffers are initialized to 0 by default.
)EOF"));

    obj.def_property_readonly("shape", [](fourier::FFTW_FFT<T> &fftw_fft) {
        return fftw_fft.shape();
    }, pybind11::doc(R"EOF(
Returns
-------
shape : tuple(int)
    Dimensions of input/output buffers.
)EOF"));

    obj.def_property_readonly("input", [](fourier::FFTW_FFT<T> &fftw_fft) {
        return cpp_py3_interop::xtensor_to_numpy(fftw_fft.view_in(), false);
    }, pybind11::doc(R"EOF(
Returns
-------
input : :py:class:`~numpy.ndarray`
    T-valued complex array.
)EOF"));

    obj.def_property_readonly("output", [](fourier::FFTW_FFT<T> &fftw_fft) {
        return cpp_py3_interop::xtensor_to_numpy(fftw_fft.view_out(), false);
    }, pybind11::doc(R"EOF(
Returns
-------
output : :py:class:`~numpy.ndarray`
    T-valued complex array.
)EOF"));

    obj.def("fft", [](fourier::FFTW_FFT<T> &fftw_fft) {
        fftw_fft.fft();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.input` using 1D-FFT, result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.output`.
)EOF"));

    obj.def("fft_r", [](fourier::FFTW_FFT<T> &fftw_fft) {
        fftw_fft.fft_r();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.output` using 1D-FFT, result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.input`.
)EOF"));

    obj.def("ifft", [](fourier::FFTW_FFT<T> &fftw_fft) {
        fftw_fft.ifft();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.input` using 1D-iFFT, result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.output`.
)EOF"));

    obj.def("ifft_r", [](fourier::FFTW_FFT<T> &fftw_fft) {
        fftw_fft.ifft_r();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.output` using 1D-iFFT, result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFT.input`.
)EOF"));

    obj.def("__repr__", [](fourier::FFTW_FFT<T> &fftw_fft) {
        return fftw_fft.__repr__();
    });
}

template <typename TT>
void FFTW_FFS_bindings(pybind11::module &m,
                       const std::string &class_name) {
    auto obj = pybind11::class_<fourier::FFTW_FFS<TT>>(m,
                                                       class_name.data(),
                                                       R"EOF(
FFTW_FFS(shape, axis, T, T_c, N_FS, inplace, N_threads, effort)

FFTW wrapper to compute Fourier Series coefficients from signal samples.

This object automatically allocates input/output buffers and provides a tensor
interface to the underlying memory using NumPy arrays.

Examples
--------
Let :math:`\phi(t)` be a shifted Dirichlet kernel of period :math:`T` and bandwidth :math:`N_{FS} = 2 N + 1`:

.. math::

   \phi(t) = \sum_{k = -N}^{N} \exp\left( j \frac{2 \pi}{T} k (t - T_{c}) \right)
           = \frac{\sin\left( N_{FS} \pi [t - T_{c}] / T \right)}{\sin\left( \pi [t - T_{c}] / T \right)}.

It's Fourier Series (FS) coefficients :math:`\phi_{k}^{FS}` can be analytically evaluated using the shift-modulation theorem:

.. math::

   \phi_{k}^{FS} =
   \begin{cases}
       \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
       0 & \text{otherwise}.
   \end{cases}

Being bandlimited, we can use :py:func:`~pypeline.util.math.fourier.FFTW_FFS` to numerically evaluate :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}`:

.. testsetup::

   import numpy as np
   import math
   from pypeline.util.math.fourier import ffs_sample, FFTW_FFS, planning_effort

   def dirichlet(x, T, T_c, N_FS):
       y = x - T_c

       n, d = np.zeros((2, len(x)))
       nan_mask = np.isclose(np.fmod(y, np.pi), 0)
       n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
       d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
       n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
       d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)

       return n / d

.. doctest::

   >>> T, T_c, N_FS = math.pi, math.e, 15
   >>> N_samples = 16  # Any >= N_FS will do, but highly-composite best.

   # Sample the kernel and do the transform.
   >>> sample_points = ffs_sample(T, N_FS, T_c, N_samples)
   >>> diric_samples = dirichlet(sample_points, T, T_c, N_FS)

   >>> transform = FFTW_FFS((N_samples,), 0, T, T_c, N_FS,
   ...                      inplace=False, N_threads=1,
   ...                      effort=planning_effort.NONE)
   >>> transform.input[:] = diric_samples
   >>> transform.ffs()

   # Compare with theoretical result.
   >>> N = (N_FS - 1) // 2
   >>> diric_FS_exact = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])

   >>> np.allclose(transform.output[:N_FS], diric_FS_exact)
   True

Notes
-----
Theory: :ref:`FFS_def`.

See Also
--------
:py:func:`~pypeline.util.math.fourier.ffs_sample`,
:py:func:`~pypeline.util.math.fourier.ffs`,
:py:func:`~pypeline.util.math.fourier.iffs`
)EOF");

    obj.def(pybind11::init([](std::vector<int> shape,
                              const int axis,
                              const double T,
                              const double T_c,
                              const int N_FS,
                              const bool inplace,
                              const int N_threads,
                              fourier::planning_effort effort) {
        /*
         * PYBIND11 BUG
         * ------------
         * Tensors of rank \ge 2 suffer from numerous problems:
         * * In Debug Mode, FFTW plans fail allocation assertions.
         * * In Release Mode, FFTW plans pass allocation assertions, but transform
         *   output are garbage.
         */
        if (shape.size() > 2) {
            std::string msg = "PYBIND11 BUG: Transforms on tensors of rank >= 3 currently disabled.";
            throw std::runtime_error(msg);
        }

        if (xt::any(xt::adapt(shape) <= 0)) {
            std::string msg = "Parameter[shape] must contain positive integers.";
            throw std::runtime_error(msg);
        }
        std::vector<size_t> cpp_shape(shape.size());
        std::copy(shape.begin(), shape.end(), cpp_shape.begin());

        const auto& cpp_axis = cpp_py3_interop::cpp_index_convention(shape.size(), axis);

        if (N_FS < 0) {
            std::string msg = "Parameter[N_FS] must be positive.";
        }

        if (N_threads < 0) {
            std::string msg = "Parameter[N_threads] must be positive.";
            throw std::runtime_error(msg);
        }

        return std::make_unique<fourier::FFTW_FFS<TT>>(cpp_shape, cpp_axis,
                                                       T, T_c, N_FS,
                                                       inplace, N_threads, effort);
    }), pybind11::arg("shape").none(false),
        pybind11::arg("axis").none(false),
        pybind11::arg("T").none(false),
        pybind11::arg("T_c").none(false),
        pybind11::arg("N_FS").none(false),
        pybind11::arg("inplace").none(false),
        pybind11::arg("N_threads").none(false),
        pybind11::arg("effort").none(false),
        pybind11::doc(R"EOF(
__init__(shape, axis, T, T_c, N_FS, inplace, N_threads, effort)

Parameters
----------
shape : tuple(int)
    Dimensions of input/output arrays.
axis : int
    Dimension along which function samples are stored.
T : float
    Function period.
T_c : float
    period mid-point
N_FS : int
    Function bandwidth.
inplace : bool
    Perform in-place transforms.
    If enabled, only one array will be allocated internally.
N_threads : int
    Number of threads to use.
effort : :py:class:`~pypeline.util.math.fourier.planning_effort`
    Amount of time spent finding best transform.

Notes
-----
* Input and output buffers are initialized to 0 by default.
)EOF"));

    obj.def_property_readonly("shape", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        return fftw_ffs.shape();
    }, pybind11::doc(R"EOF(
Returns
-------
shape : tuple(int)
    Dimensions of input/output buffers.
)EOF"));

    obj.def_property_readonly("input", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        return cpp_py3_interop::xtensor_to_numpy(fftw_ffs.view_in(), false);
    }, pybind11::doc(R"EOF(
Returns
-------
input : :py:class:`~numpy.ndarray`
    T-valued complex array.
)EOF"));

    obj.def_property_readonly("output", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        return cpp_py3_interop::xtensor_to_numpy(fftw_ffs.view_out(), false);
    }, pybind11::doc(R"EOF(
Returns
-------
output : :py:class:`~numpy.ndarray`
    T-valued complex array.
)EOF"));

    obj.def("ffs", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        fftw_ffs.ffs();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input` using 1D-FFS,
result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output`.

It is assumed :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input` contains
function values at sampling points specified by :py:func:`~pypeline.util.math.fourier.ffs_sample`.
After this function call, :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output`
will contain FS coefficients :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}}` along dimension `axis`.
)EOF"));

    obj.def("ffs_r", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        fftw_ffs.ffs_r();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output` using 1D-FFS,
result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input`.

It is assumed :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output` contains
function values at sampling points specified by :py:func:`~pypeline.util.math.fourier.ffs_sample`.
After this function call, :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input`
will contain FS coefficients :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{s}}` along dimension `axis`.
)EOF"));

    obj.def("iffs", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        fftw_ffs.iffs();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input` using 1D-iFFS,
result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output`.

It is assumed :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input` contains
FS coefficients ordered as :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{samples}}`.
along dimension `axis`.
After this function call, :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output`
will contain the original function samples in the same order specified by
:py:func:`~pypeline.util.math.fourier.ffs_sample`.
)EOF"));

    obj.def("iffs_r", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        fftw_ffs.iffs_r();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output` using 1D-iFFS,
result available in :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input`.

It is assumed :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.output` contains
FS coefficients ordered as :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{samples}}`.
along dimension `axis`.
After this function call, :py:attr:`~pypeline.util.math.fourier.FFTW_FFS.input`
will contain the original function samples in the same order specified by
:py:func:`~pypeline.util.math.fourier.ffs_sample`.
)EOF"));

    obj.def("__repr__", [](fourier::FFTW_FFS<TT> &fftw_ffs) {
        return fftw_ffs.__repr__();
    });
}

template <typename T>
void FFTW_CZT_bindings(pybind11::module &m,
                       const std::string &class_name) {
    auto obj = pybind11::class_<fourier::FFTW_CZT<T>>(m,
                                                      class_name.data(),
                                                      R"EOF(
FFTW_CZT(shape, axis, A, W, M, N_threads, effort)

FFTW wrapper to compute the 1D *Chirp Z-Transform* on multi-dimensional tensors.

This implementation follows the semantics defined in :ref:`CZT_def`.

This object automatically allocates an optimally-sized buffer and provides a
tensor interface to the underlying memory using NumPy arrays.

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.fourier import FFTW_CZT, planning_effort

Implementation of the DFT:

.. doctest::

   >>> N = M = 10
   >>> x = np.random.randn(N, 3) + 1j * np.random.randn(N, 3)  # multi-dim

   >>> dft_x = np.fft.fft(x, axis=0)
   >>> transform = FFTW_CZT(x.shape, axis=0,
   ...                      A=1, W=np.exp(-1j * 2 * np.pi / N), M=M,
   ...                      N_threads=1, effort=planning_effort.NONE)
   >>> transform.input[:] = x
   >>> transform.czt()

   >>> np.allclose(dft_x, transform.output)
   True
)EOF");

    obj.def(pybind11::init([](std::vector<int> shape,
                              const int axis,
                              const std::complex<double> A,
                              const std::complex<double> W,
                              const int M,
                              const int N_threads,
                              fourier::planning_effort effort) {
        /*
         * PYBIND11 BUG
         * ------------
         * Tensors of rank \ge 2 suffer from numerous problems:
         * * In Debug Mode, FFTW plans fail allocation assertions.
         * * In Release Mode, FFTW plans pass allocation assertions, but transform
         *   output are garbage.
         */
        if (shape.size() > 2) {
            std::string msg = "PYBIND11 BUG: Transforms on tensors of rank >= 3 currently disabled.";
            throw std::runtime_error(msg);
        }

        if (xt::any(xt::adapt(shape) <= 0)) {
            std::string msg = "Parameter[shape] must contain positive integers.";
            throw std::runtime_error(msg);
        }
        std::vector<size_t> cpp_shape(shape.size());
        std::copy(shape.begin(), shape.end(), cpp_shape.begin());

        const auto& cpp_axis = cpp_py3_interop::cpp_index_convention(shape.size(), axis);

        if (M <= 0) {
            std::string msg = "Parameter[M] must be positive.";
            throw std::runtime_error(msg);
        }

        if (N_threads < 0) {
            std::string msg = "Parameter[N_threads] must be positive.";
            throw std::runtime_error(msg);
        }

        return std::make_unique<fourier::FFTW_CZT<T>>(cpp_shape, cpp_axis,
                                                      A, W, M,
                                                      N_threads, effort);
    }), pybind11::arg("shape").none(false),
        pybind11::arg("axis").none(false),
        pybind11::arg("A").none(false),
        pybind11::arg("W").none(false),
        pybind11::arg("M").none(false),
        pybind11::arg("N_threads").none(false),
        pybind11::arg("effort").none(false),
        pybind11::doc(R"EOF(
__init__(shape, axis, A, W, M, N_threads, effort)

Parameters
----------
shape : tuple(int)
    Dimensions of the input array.
axis : int
    Dimension along which to apply the transform.
A : complex
    Circular offset from the positive real-axis.
W : complex
    Circular spacing between transform points.
M : int
    Length of the transform.
N_threads : int
    Number of threads to use.
effort : planning_effort

Notes
-----
Due to numerical instability when using large `M`, this implementation only
supports transforms where `A` and `W` have unit norm.
)EOF"));

    obj.def_property_readonly("shape_input", [](fourier::FFTW_CZT<T> &fftw_czt) {
        return fftw_czt.shape_in();
    }, pybind11::doc(R"EOF(
Returns
-------
shape_input : tuple(int)
    Dimensions of the input.
)EOF"));

    obj.def_property_readonly("shape_output", [](fourier::FFTW_CZT<T> &fftw_czt) {
        return fftw_czt.shape_out();
    }, pybind11::doc(R"EOF(
Returns
-------
shape_output : tuple(int)
    Dimensions of the output.
)EOF"));

    obj.def_property_readonly("input", [](fourier::FFTW_CZT<T> &fftw_czt) {
        return cpp_py3_interop::xtensor_to_numpy(fftw_czt.view_in(), false);
    }, pybind11::doc(R"EOF(
Returns
-------
input : :py:class:`~numpy.ndarray`
    T-valued complex array.
)EOF"));

    obj.def_property_readonly("output", [](fourier::FFTW_CZT<T> &fftw_czt) {
        return cpp_py3_interop::xtensor_to_numpy(fftw_czt.view_out(), false);
    }, pybind11::doc(R"EOF(
Returns
-------
output : :py:class:`~numpy.ndarray`
    T-valued complex array.
)EOF"));

    obj.def("czt", [](fourier::FFTW_CZT<T> &fftw_czt) {
        fftw_czt.czt();
    }, pybind11::doc(R"EOF(
Transform :py:attr:`~pypeline.util.math.fourier.FFTW_CZT.input` using 1D-CZT,
result available in :py:attr:`~pypeline.util.math.fourier.FFTW_CZT.output`.

Notes
-----
The contents of :py:attr:`~pypeline.util.math.fourier.FFTW_CZT.input` are not
preserved after calls to :py:meth:`~pypeline.util.math.fourier.FFTW_CZT.czt`.
)EOF"));

    obj.def("__repr__", [](fourier::FFTW_CZT<T> &fftw_czt) {
        return fftw_czt.__repr__();
    });
}

template <typename TT, typename T_array>
void _fs_interp_input(fourier::FFTW_FS_INTERP<TT> &fftw_fs_interp,
                      pybind11::array_t<T_array> x) {
    const auto& xview = cpp_py3_interop::numpy_to_xview<T_array>(x);

    fftw_fs_interp.in(xview);
}

template <typename TT>
void FFTW_FS_INTERP_bindings(pybind11::module &m,
                             const std::string &class_name) {
    auto obj = pybind11::class_<fourier::FFTW_FS_INTERP<TT>>(m,
                                                              class_name.data(),
                                                              R"EOF(
FFTW_FS_INTERP(shape, axis, T, a, b, M, real_valued_output, N_threads, effort)

Interpolate bandlimited periodic signals as described in :ref:`fp_interp_def`.

If given the Fourier Series coefficients of a bandlimited periodic function
:math:`x(t): \mathbb{R} \to \mathbb{C}`, then :py:meth:`FFTW_FS_INTERP.fs_interp`
computes the values of :math:`x(t)` at points :math:`t[k] = (a + \frac{b - a}{M - 1} k) 1_{[0,\ldots,M-1]}[k]`.

Examples
--------
.. testsetup::

   import numpy as np
   import math
   from pypeline.util.math.fourier import FFTW_FS_INTERP, planning_effort

   def dirichlet(x, T, T_c, N_FS):
       y = x - T_c
       n, d = np.zeros((2, len(x)))
       nan_mask = np.isclose(np.fmod(y, np.pi), 0)
       n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
       d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
       n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
       d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)
       return n / d

   # Parameters of the signal.
   T, T_c, N_FS = math.pi, math.e, 15
   N = (N_FS - 1) // 2
   # Generate interpolated signal
   a, b = T_c + (T / 2) *  np.r_[-1, 1]
   M = 100  # We want lots of points.
   diric_FS = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])


Let :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}` be the Fourier Series (FS) coefficients of a shifted Dirichlet kernel of period :math:`T`:

.. math::

   \phi_{k}^{FS} =
   \begin{cases}
       \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
       0 & \text{otherwise}.
   \end{cases}

.. doctest::

   # Parameters of the signal.
   >>> T, T_c, N_FS = math.pi, math.e, 15
   >>> N = (N_FS - 1) // 2

   # And the kernel's FS coefficients.
   >>> diric_FS = np.exp(-1j * (2 * np.pi / T) * T_c * np.r_[-N:N+1])

Being bandlimited, we can use :py:class:`~pypeline.util.math.fourier.FFTW_FS_INTERP` to numerically evaluate :math:`\phi(t)` on the interval :math:`\left[ T_{c} - \frac{T}{2}, T_{c} + \frac{T}{2} \right]`.

.. doctest::

   # Generate interpolated signal
   >>> a, b = T_c + (T / 2) *  np.r_[-1, 1]
   >>> M = 100  # We want lots of points.
   >>> transform = FFTW_FS_INTERP(diric_FS.shape, 0,
   ...                            T, a, b, M, real_valued_output=False,
   ...                            N_threads=1, effort=planning_effort.NONE)
   >>> transform.input(diric_FS)
   >>> transform.fs_interp()
   >>> diric_sig = transform.output

   # Compare with theoretical result.
   >>> t = a + (b - a) / (M - 1) * np.arange(M)
   >>> diric_sig_exact = dirichlet(t, T, T_c, N_FS)
   >>> np.allclose(diric_sig, diric_sig_exact)
   True


The Dirichlet kernel is real-valued, so we can set `real_valued_output` to use the accelerated algorithm instead:

.. doctest::

   # Generate interpolated signal
   >>> a, b = T_c + (T / 2) *  np.r_[-1, 1]
   >>> M = 100  # We want lots of points.
   >>> transform = FFTW_FS_INTERP(diric_FS.shape, 0,
   ...                            T, a, b, M, real_valued_output=True,
   ...                            N_threads=1, effort=planning_effort.NONE)
   >>> transform.input(diric_FS)
   >>> transform.fs_interp()
   >>> diric_sig = transform.output

   # Compare with theoretical result.
   >>> t = a + (b - a) / (M - 1) * np.arange(M)
   >>> diric_sig_exact = dirichlet(t, T, T_c, N_FS)
   >>> np.allclose(diric_sig, diric_sig_exact)
   True

Notes
-----
Theory: :ref:`fp_interp_def`.
)EOF");

    obj.def(pybind11::init([](std::vector<int> shape,
                              const int axis,
                              const double T,
                              const double a,
                              const double b,
                              const int M,
                              const bool real_valued_output,
                              const int N_threads,
                              fourier::planning_effort effort) {
        /*
         * PYBIND11 BUG
         * ------------
         * Tensors of rank \ge 2 suffer from numerous problems:
         * * In Debug Mode, FFTW plans fail allocation assertions.
         * * In Release Mode, FFTW plans pass allocation assertions, but transform
         *   output are garbage.
         */
        if (shape.size() > 2) {
            std::string msg = "PYBIND11 BUG: Transforms on tensors of rank >= 3 currently disabled.";
            throw std::runtime_error(msg);
        }

        if (xt::any(xt::adapt(shape) <= 0)) {
            std::string msg = "Parameter[shape] must contain positive integers.";
            throw std::runtime_error(msg);
        }
        std::vector<size_t> cpp_shape(shape.size());
        std::copy(shape.begin(), shape.end(), cpp_shape.begin());

        const auto& cpp_axis = cpp_py3_interop::cpp_index_convention(shape.size(), axis);

        if (M <= 0) {
            std::string msg = "Parameter[M] must be positive.";
            throw std::runtime_error(msg);
        }

        if (N_threads < 0) {
            std::string msg = "Parameter[N_threads] must be positive.";
            throw std::runtime_error(msg);
        }

        return std::make_unique<fourier::FFTW_FS_INTERP<TT>>(cpp_shape, cpp_axis,
                                                             T, a, b, M, real_valued_output,
                                                             N_threads, effort);
    }), pybind11::arg("shape").none(false),
        pybind11::arg("axis").none(false),
        pybind11::arg("T").none(false),
        pybind11::arg("a").none(false),
        pybind11::arg("b").none(false),
        pybind11::arg("M").none(false),
        pybind11::arg("real_valued_output").none(false),
        pybind11::arg("N_threads").none(false),
        pybind11::arg("effort").none(false),
        pybind11::doc(R"EOF(
__init__(shape, axis, T, a, b, M, real_valued_output, N_threads, effort)

Parameters
----------
shape : tuple(int)
    Dimensions of input array.
axis : int
    Dimension along which FS coefficients are stored.
T : float
    Function period.
a : float
    Interval LHS.
b : float
    Interval RHS.
M : int
    Number of points to interpolate.
real_valued_output : bool
    If :py:obj:`True`, it is assumed the interpolated signal is real-valued.
    In this context, only the FS coefficients corresponding to non-negative
    frequencies will be used, along with a more efficient interpolation algorithm.
N_threads : int
    Number of threads to use.
effort : planning_effort
)EOF"));

    obj.def_property_readonly("shape_input", [](fourier::FFTW_FS_INTERP<TT> &fftw_fs_interp) {
        return fftw_fs_interp.shape_in();
    }, pybind11::doc(R"EOF(
Returns
-------
shape_input : tuple(int)
    Dimensions of the input.
)EOF"));

    obj.def_property_readonly("shape_output", [](fourier::FFTW_FS_INTERP<TT> &fftw_fs_interp) {
        return fftw_fs_interp.shape_out();
    }, pybind11::doc(R"EOF(
Returns
-------
shape_output : tuple(int)
    Dimensions of the output.
)EOF"));

    obj.def("input", &_fs_interp_input<TT, float>,
            pybind11::arg("x").noconvert().none(false));
    obj.def("input", &_fs_interp_input<TT, double>,
            pybind11::arg("x").noconvert().none(false));
    obj.def("input", &_fs_interp_input<TT, std::complex<float>>,
            pybind11::arg("x").noconvert().none(false));
    obj.def("input", &_fs_interp_input<TT, std::complex<double>>,
            pybind11::arg("x").noconvert().none(false),
            pybind11::doc(R"EOF(
input(x)

Fill input buffer.

Parameters
----------
x : :py:class:`~numpy.ndarray`
    (..., N_FS, ...) FS coefficients in the order :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}\right]` along dimension `axis`.

Notes
-----
if `real_valued_output` was set to :py:obj:`True`, only the FS coefficients
corresponding to non-negative frequencies are stored.
)EOF"));

    obj.def_property_readonly("output", [](fourier::FFTW_FS_INTERP<TT> &fftw_fs_interp) {
        return cpp_py3_interop::xtensor_to_numpy(fftw_fs_interp.view_out(), false);
    }, pybind11::doc(R"EOF(
Returns
-------
output : :py:class:`~numpy.ndarray`
    (..., M, ...) interpolated values :math:`\left[ x(t[0]), \ldots, x(t[M-1]) \right]` along the axis indicated by `axis`.
    If `real_valued_output` was set to :py:obj:`True`, the output's imaginary part is guaranteed to be 0.
)EOF"));

    obj.def("fs_interp", [](fourier::FFTW_FS_INTERP<TT> &fftw_fs_interp) {
        fftw_fs_interp.fs_interp();
    }, pybind11::doc(R"EOF(
Interpolate bandlimited periodic signal.

This function is meant to be used as follows:

* Use :py:meth:`~pypeline.util.math.fourier.FFTW_FS_INTERP.input` to fill up the input buffer with FS coefficients.
* Call :py:meth:`~pypeline.util.math.fourier.FFTW_FS_INTERP.fs_interp` to perform the interpolation.
* Use :py:attr:`~pypeline.util.math.fourier.FFTW_FS_INTERP.output` to obtain the signal samples.
)EOF"));

    obj.def("__repr__", [](fourier::FFTW_FS_INTERP<TT> &fftw_fs_interp) {
        return fftw_fs_interp.__repr__();
    });
}

PYBIND11_MODULE(_pypeline_util_math_fourier_pybind11, m) {
    pybind11::options options;
    options.disable_function_signatures();

    ffs_sample_bindings(m);
    planning_effort_bindings(m);
    FFTW_FFT_bindings<double>(m, "FFTW_FFT");
    FFTW_FFS_bindings<double>(m, "FFTW_FFS");
    FFTW_CZT_bindings<double>(m, "FFTW_CZT");
    FFTW_FS_INTERP_bindings<double>(m, "FFTW_FS_INTERP");
}
