// ############################################################################
// _array.cpp
// ==========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <algorithm>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

namespace pypeline::util::array {
    template <typename E, typename I>
    xt::xstrided_slice_vector index(E&& x, const size_t axis, I&& idx_spec) {
        if (axis >= x.dimension()) {
            std::string msg = "Parameter[axis] must be lie in {0, ..., x.dimension()-1}.";
            throw std::runtime_error(msg);
        }

        xt::xstrided_slice_vector idx_struct(x.dimension());
        std::fill(idx_struct.begin(), idx_struct.end(), xt::all());
        idx_struct[axis] = idx_spec;

        return idx_struct;
    }

    template <typename E>
    auto cluster_layers(E&& x,
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
        xt::xarray<T> y = xt::zeros<T>(shape_y);

        for (size_t i = 0; i < idx.size(); ++i) {
            auto idx_x = index(x, axis, i);
            auto view_x = xt::strided_view(x, idx_x);

            auto idx_y = index(y, axis, idx[i]);
            auto view_y = xt::strided_view(y, idx_y);

            view_y.plus_assign(view_x);
        }

        return y;
    }
}
