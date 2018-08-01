// ############################################################################
// _cpp_py3_interop.cpp
// ====================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

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
    template <typename E>
    auto xtensor_to_numpy(E&& _x) {
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
