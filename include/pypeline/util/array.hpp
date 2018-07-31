// ############################################################################
// array.hpp
// =========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Tools and utilities for manipulating arrays.
 */

#ifndef PYPELINE_UTIL_ARRAY_HPP
#define PYPELINE_UTIL_ARRAY_HPP

#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xstrided_view.hpp"

namespace pypeline::util::array {
    /*
     * Form indexing structure for Xtensor containers.
     *
     * Given an array `x`, generates the vector that has :cpp:`xt::all()` in
     * each axis except `axis`, where `idx_spec` is used instead.
     *
     * Parameters
     * ----------
     * x : xt::xexpression
     *     Array to index.
     * axis : size_t
     *     Dimension along which to apply `idx_spec`.
     * idx_spec : size_t or xt::xslice
     *     Accepted xt::xslice objects are:
     *         * xt::range(min, max, [step=1]);
     *         * xt::all().
     *
     * Returns
     * -------
     * idx_struct : xt::xtstrided_slice_vector
     *     Second argument to :cpp:`xt::strided_view(x, _)`.
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include "xtensor/xarray.hpp"
     *    #include "xtensor/xbuilder.hpp"
     *    #include "xtensor/xstrided_view.hpp"
     *    #include "pypeline/util/array.hpp"
     *
     *    xt::xarray<double> x = xt::reshape_view(xt::arange<double>(0, 3 * 4 * 5),
     *                                            {3, 4, 5});
     *
     *    namespace array = pypeline::util::array;
     *    auto idx = array::index(x, 2, xt::range(0, 5, 2));
     *    auto y = xt::strided_view(x, idx);
     */
    template <typename E, typename I>
    xt::xstrided_slice_vector index(E&& x, const size_t axis, I&& idx_spec);

    /*
     * Additive tensor compression along an axis.
     *
     * Parameters
     * ----------
     * x : xt::xexpression
     *     (..., K, ...) array.
     * idx : std::vector<size_t>
     *     (K,) cluster indices.
     * N : size_t
     *     Total number of levels along compression axis.
     * axis : size_t
     *     Dimension along which to compress.
     *
     * Returns
     * -------
     * clustered_x : xt::xarray
     *     (..., N, ...) array.
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include <vector>
     *    #include "xtensor/xarray.hpp"
     *    #include "xtensor/xbuilder.hpp"
     *    #include "xtensor/xstrided_view.hpp"
     *    #include "pypeline/util/array.hpp"
     *
     *    namespace array = pypeline::util::array;
     *    using T = double;
     *    using array_t = xt::xarray<T>;
     *
     *    array_t x = xt::reshape_view(xt::arange<T>(0, 3*4*5, 1),
     *                                 {3, 4, 5});
     *
     *    const size_t N{5}, axis{0};
     *    const std::vector<size_t> idx {0, 1, 1};
     *    auto y = array::cluster_layers(x, idx, N, axis);
     */
    template <typename E>
    auto cluster_layers(E&& x,
                        std::vector<size_t> idx,
                        const size_t N,
                        const size_t axis);
}

#include "_array.tpp"

#endif //PYPELINE_UTIL_ARRAY_HPP
