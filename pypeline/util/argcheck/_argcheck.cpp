// ############################################################################
// _argcheck.cpp
// =============
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <array>
#include <complex>
#include <type_traits>

namespace pypeline::util::argcheck {
    template<typename E>
    bool has_rank(E &&x, const size_t rank) {
        return x.dimension() == rank;
    }

    template<typename E, size_t rank>
    bool has_shape(E &&x, const std::array <size_t, rank> &shape) {
        auto shape_x = x.shape();
        size_t rank_x = x.dimension();

        if (shape.size() == rank_x) {
            for (size_t i = 0; i < rank_x; ++i) {
                if (((size_t) shape_x[i]) != shape[i]) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    template<typename E>
    bool has_floats(E &&) {
        using T = typename std::decay_t<E>::value_type;
        return std::is_floating_point<T>::value;
    }

    template<typename E>
    bool has_complex(E &&) {
        using T = typename std::decay_t<E>::value_type;
        using cfloat = std::complex<float>;
        using cdouble = std::complex<double>;
        using cldouble = std::complex<long double>;

        return (std::is_same<T, cfloat>::value ||
                std::is_same<T, cdouble>::value ||
                std::is_same<T, cldouble>::value);
    }
}
