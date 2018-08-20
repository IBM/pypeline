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

#include <algorithm>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

namespace pypeline { namespace util { namespace array {
    /*
     * Form indexing structure for Xtensor containers.
     *
     * Given an array's rank, generate the vector that has :cpp:`xt::all()`
     * in each axis except `axis`, where `idx_spec` is used instead.
     *
     * Parameters
     * ----------
     * rank : size_t
     *     Rank of array to index.
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
     *     Second argument to :cpp:`xt::strided_view(xt::xcontainer, _)`.
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
     *    auto idx = array::index(x.dimension(), 2, xt::range(0, 5, 2));
     *    auto y = xt::strided_view(x, idx);
     */
    template <typename I>
    xt::xstrided_slice_vector index(const size_t rank, const size_t axis, I &&idx_spec) {
        if (axis >= rank) {
            std::string msg = "Parameter[axis] must be lie in {0, ..., rank-1}.";
            throw std::runtime_error(msg);
        }

        xt::xstrided_slice_vector idx_struct(rank);
        std::fill(idx_struct.begin(), idx_struct.end(), xt::all());
        idx_struct[axis] = idx_spec;

        return idx_struct;
    }

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
    auto cluster_layers(E &&x,
                        std::vector<size_t> idx,
                        const size_t N,
                        const size_t axis) {
        if (N == 0) {
            std::string msg = "Parameter[N] must be non-zero.";
            throw std::runtime_error(msg);
        }
        if (axis >= x.dimension()) {
            std::string msg = "Parameter[axis] must be lie in {0, ..., x.dimension()-1}.";
            throw std::runtime_error(msg);
        }
        if (idx.size() != static_cast<size_t>(x.shape()[axis])) {
            std::stringstream msg;
            msg << "Parameter[idx] contains " << std::to_string(idx.size())
                << " elements, but Parameter[x] contains " << std::to_string(x.shape()[axis])
                << " entries along dimension " << std::to_string(axis);
            throw std::runtime_error(msg.str());
        }
        if (!xt::all(xt::adapt(idx) < N)) {
            std::string msg = "Parameter[idx] contains out-of-bound entries w.r.t Parameter[N].";
            throw std::runtime_error(msg);
        }

        using T = typename std::decay_t<E>::value_type;
        auto shape_y = x.shape();
        shape_y[axis] = N;
        xt::xarray<T> y {xt::zeros<T>(shape_y)};

        for (size_t i = 0; i < idx.size(); ++i) {
            auto idx_x = index(x.dimension(), axis, i);
            auto view_x = xt::strided_view(x, idx_x);

            auto idx_y = index(y.dimension(), axis, idx[i]);
            auto view_y = xt::strided_view(y, idx_y);

            view_y.plus_assign(view_x);
        }

        return y;
    }
}}}

#endif //PYPELINE_UTIL_ARRAY_HPP
