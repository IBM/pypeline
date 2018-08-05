// ############################################################################
// cpp_py3_interop.hpp
// ===================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Python3/C++ interfacing tools with PyBind11.
 */

#ifndef PYPELINE_UTIL_CPP_PY3_INTEROP_HPP
#define PYPELINE_UTIL_CPP_PY3_INTEROP_HPP

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "xtensor/xadapt.hpp"

namespace pypeline::util::cpp_py3_interop {
    /*
     * Reference C++ tensor from Python3 as NumPy array.
     *
     * The C++ tensor's memory is moved to a NumPy array whose lifetime is managed
     * by the Python interpreter.
     * In other words, the NumPy array's resources are automatically freed by the
     * Python interpreter when the array is no longer referenced from Python code.
     *
     * The C++ tensor should no longer be used on the C++ side after calling this
     * function.
     *
     * Parameters
     * ----------
     * _x : xt::xarray<T> or xt::xtensor<T, rank>
     *     C++ tensor.
     *
     * Returns
     * -------
     * x : pybind11::array_t<T>
     *     NumPy tensor.
     *
     * Examples
     * --------
     * Define Python module :py:mod:`example` with unique function :py:func:`example.test`
     * returning a statically-allocated tensor:
     *
     * .. literal_block:: cpp
     *
     *    #include <complex>
     *    #include "xtensor/xtensor.hpp"
     *    #include "pypeline/util/cpp_py3_interop.hpp"
     *
     *    PYBIND11_MODULE(example, m) {
     *        m.def("test", []() {
     *            using T = std::complex<float>;
     *            xt::xtensor<T, 2> x {{1, 2, 3}, {4, 5, 6}};
     *
     *            namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
     *            return cpp_py3_interop::xtensor_to_numpy(x);
     *        }, pybind11::return_value_policy::automatic);
     *    }
     *
     * After compilation and linking, :py:mod:`example` can be accessed from Python3:
     *
     * .. literal_block:: py
     *
     *    >>> import example
     *
     *    >>> A = example.test()
     *    >>> A
     *    array([[1.+0.j, 2.+0.j, 3.+0.j],
     *           [4.+0.j, 5.+0.j, 6.+0.j]], dtype=complex64)
     */
    template <typename E>
    auto xtensor_to_numpy(E &&_x) {
        /*
         * pybind11::array_t<T>() used below takes ownership of strides/shape
         * vectors, so we need to allocate copies.
         */
        using EE = typename std::decay_t<E>;
        using T = typename EE::value_type;
        EE *x = new EE(std::move(_x));

        std::vector<ssize_t> shape_x(x->dimension());
        std::copy(x->shape().begin(), x->shape().end(), shape_x.begin());

        /*
         * Stride information differs between Xtensor and NumPy:
         *     * Xtensor: dot(strides, index) gives T-offsets.
         *     * NumPy: dot(strides, index) gives char-offsets.
         * We therefore need to rescale Xtensor's strides to match NumPy's
         * conventions.
         */
        std::vector<ssize_t> strides_x(x->dimension());
        for (size_t i = 0; i < x->dimension(); ++i) {
            strides_x[i] = x->strides()[i] * sizeof(T);
        }

        /*
         * `x` took possession of `_x`'s memory, so we need to setup a mechanism
         * on Python-side to deallocate it's ressources.
         * :cpp:obj:`pybind11::capsule` is a thin wrapper around a pointer
         * (first argument) so that it can be referenced from Python.
         * The second argument is a deallocation function called by the Python
         * garbage collector when all references to `x` have expired on Python-side.
         */
        pybind11::capsule dealloc_handle(x, [](void *capsule) {
            EE *x = reinterpret_cast<EE *>(capsule);
            delete x;
        });

        return pybind11::array_t<T>(shape_x, strides_x, x->data(), dealloc_handle);
    }

    /*
     * Reference NumPy array from C++ as Xtensor container.
     *
     * The array's memory is still owned and managed from Python.
     *
     * Parameters
     * ----------
     * buf : pybind11::buffer
     *     Object that implements the `buffer-protocol <https://docs.python.org/3/c-api/buffer.html>`_.
     *     It is assumed `buf` was obtained by calling :cpp:func:`pybind11::array_t<T>::request()`.
     * T : type
     *     Type of individual elements in the buffer.
     *
     * Returns
     * -------
     * xview : xexpression
     *     View on the NumPy array.
     *
     * Notes
     * -----
     * * In-place operations do not work on non-contiguous NumPy views.
     *   No tests are done from C++ to make sure this is the case.
     *   The user is left responsible to enforce this condition.
     * * Strickly-speaking, specifying `T` is not required since buffers contain
     *   a format-string that fully describes the enclosed datatype.
     *   However we enforce the user to specify it so that the buffer can be
     *   checked to be consistent with what the user expects the function to
     *   output.
     *
     * Examples
     * --------
     * Define Python module :py:mod:`example` with unique function :py:func:`example.test`
     * that increments NumPy arrays in-place:
     *
     * .. literal_block:: cpp
     *
     *    #include <complex>
     *    #include <cstdint>
     *    #include "pybind11/pybind11.h"
     *    #include "pypeline/util/cpp_py3_interop.hpp"
     *
     *    template <typename T>
     *    void test(pybind11::array_t<T> x) {
     *        namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
     *        auto xview = cpp_py3_interop::numpy_to_xview<T>(x);
     *        xview += T(1);
     *    }
     *
     *    PYBIND11_MODULE(example, m) {
     *        m.def("test", &test<int32_t>,              pybind11::arg("test").noconvert());
     *        m.def("test", &test<int64_t>,              pybind11::arg("test").noconvert());
     *        m.def("test", &test<uint32_t>,             pybind11::arg("test").noconvert());
     *        m.def("test", &test<uint64_t>,             pybind11::arg("test").noconvert());
     *        m.def("test", &test<float>,                pybind11::arg("test").noconvert());
     *        m.def("test", &test<double>,               pybind11::arg("test").noconvert());
     *        m.def("test", &test<std::complex<float>>,  pybind11::arg("test").noconvert());
     *        m.def("test", &test<std::complex<double>>, pybind11::arg("test").noconvert());
     *    }
     *
     * It is important to specify `.noconvert()` above to forbid PyBind11 from
     * automatically casting non-conforming NumPy arrays without user knowledge.
     * Doing so would make some in-place operations fail.
     *
     * After compilation and linking, :py:mod:`example` can be accessed from Python3:
     *
     * .. literal_block:: py
     *
     *    >>> import example
     *    >>> A = np.arange(5, dtype=np.complex64)
     *    >>> B = np.array(A)
     *
     *    >>> example.test(A)
     *    >>> A
     *    array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j, 5.+0.j], dtype=complex64)
     *    >>> B
     *    array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j], dtype=complex64)
     *
     *    # In-place ops on non-contiguous views are not supported.
     *    >>> A = np.arange(15, dtype=np.uint64)
     *    >>> A_orig = np.array(A)
     *    >>> example.test(A[::2])
     *
     *    >>> A_orig
     *    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], dtype=uint64)
     *    >>> A  # incorrect
     *    array([ 1,  2,  3,  4,  5,  6,  6,  7,  8,  9, 10, 11], dtype=uint64)
     */
    template <typename T>
    auto numpy_to_xview(pybind11::buffer buf) {
        pybind11::buffer_info info = buf.request();

        if (info.itemsize != sizeof(T)) {
            std::string msg = ("Parameters[T, buf] have inconsistent item-sizes: " +
                               std::to_string(sizeof(T)) + " vs. " +
                               std::to_string(info.itemsize));
            throw std::runtime_error(msg);
        }

        /*
         * Different compilers can produce inconsistent format descriptors for
         * types such as (np.int64, int64_t).
         * So although testing format descriptors for equivalence should be
         * done, we have decided to drop the test and let the user be in control
         * here.
         *
         * if (info.format != pybind11::format_descriptor<T>::format()) {
         *     std::string msg = ("Parameters[T, buf] have inconsistent data types: " +
         *                        pybind11::format_descriptor<T>::format() +
         *                        " vs. " + info.format);
         *     throw std::runtime_error(msg);
         * }
         */

        /*
         * Stride information differs between Xtensor and NumPy:
         *     * Xtensor: dot(strides, index) gives T-offsets.
         *     * NumPy: dot(strides, index) gives char-offsets.
         * We therefore need to rescale NumPy's strides to match Xtensor's
         * conventions.
         */
        std::vector<ssize_t> strides_buf(info.ndim);
        std::copy(info.strides.begin(), info.strides.end(), strides_buf.begin());
        for (size_t i = 0; i < strides_buf.size(); ++i) {
            strides_buf[i] /= info.itemsize;
        }

        auto xview = xt::adapt(reinterpret_cast<T *>(info.ptr),
                               info.size,
                               xt::no_ownership(),
                               info.shape,
                               strides_buf);
        return xview;
    }

    /*
     * Transform Python signed index to C++ unsigned index.
     *
     * Parameters
     * ----------
     * N : size_t
     *     Length of the object to index.
     * index : int, vector<int>
     *     Signed Python index.
     *
     * Returns
     * -------
     * cpp_index : size_t, vector<size_t>
     *     Equivalent unsigned C++ index.
     *
     * Examples
     * --------
     * .. literal_block:: cpp
     *
     *    #include "pypeline/util/cpp_py3_interop.hpp"
     *    namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
     *
     *    size_t N = 10;
     *    cpp_py3_interop::cpp_index_convention(N, 0);   // 0
     *    cpp_py3_interop::cpp_index_convention(N, -1);  // 9
     */
    size_t cpp_index_convention(const size_t N, const int index) {
        bool index_in_bounds = (-((int) N) <= index) && (index < ((int) N));
        if (!index_in_bounds) {
            std::string msg = "Parameter[index] must lie in {-N, ..., N-1}.";
            throw std::runtime_error(msg);
        }

        size_t cpp_index = 0;
        if (index >= 0) {
            cpp_index = index;
        } else {
            cpp_index = index + N;
        }

        return cpp_index;
    }

    std::vector<size_t> cpp_index_convention(const size_t N,
                                             const std::vector<int>& index) {
        std::vector<size_t> cpp_index(index.size());
        for (size_t i = 0; i < index.size(); ++i) {
            cpp_index[i] = cpp_index_convention(N, index[i]);
        }

        return cpp_index;
    }
}

#endif //PYPELINE_UTIL_CPP_PY3_INTEROP_HPP
