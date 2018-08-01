// ############################################################################
// cpp_py3_interop.hpp
// ===================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Python3/C++ interfacing tools with PyBind11.
 */

#ifndef PYPELINE_UTIL_CPP_PY3_INTEROP
#define PYPELINE_UTIL_CPP_PY3_INTEROP

#include <vector>

#include "pybind11/pybind11.h"

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
    auto xtensor_to_numpy(E&& _x);

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
     * * This function does not work on NumPy views, i.e. NumPy arrays where
     *   :py:attr:`ndarray.base` is not None.
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
     *    # Remember, views are not supported.
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
    auto numpy_to_xview(pybind11::buffer buf);

    /*
     * Transform Python signed index to C++ unsigned index.
     *
     * Parameters
     * ----------
     * N : size_t
     *     Length of the object to index.
     * index : int
     *     Signed Python index.
     *
     * Returns
     * -------
     * cpp_index : size_t
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
    size_t cpp_index_convention(const size_t N, const int index);
    std::vector<size_t> cpp_index_convention(const size_t N,
                                             const std::vector<int>& index);
}

#include "_cpp_py3_interop.tpp"

#endif //PYPELINE_UTIL_CPP_PY3_INTEROP
