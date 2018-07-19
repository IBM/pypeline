// ############################################################################
// linalg.hpp
// ==========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Linear algebra routines.
 */

#ifndef PYPELINE_UTIL_MATH_LINALG_HPP
#define PYPELINE_UTIL_MATH_LINALG_HPP

#include "xtensor/xtensor.hpp"

namespace pypeline::util::math::linalg {
    /*
     * Determine rotation angle from Z-axis rotation matrix.
     *
     * Parameters
     * ----------
     * R : xt::xexpression
     *     (3, 3) rotation matrix around the Z-axis;
     *
     * Returns
     * -------
     * angle : double
     *     Signed rotation angle [rad].
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include "xtensor/xarray.hpp"
     *    #include "pypeline/util/math/linalg.hpp"
     *
     *    auto R = xt::xarray<double> {{0, -1, 0},
     *                                 {1,  0, 0},
     *                                 {0,  0, 1}};
     *
     *    namespace linalg = pypeline::util::math::linalg;
     *    double angle = linalg::z_rot2angle(R);
     */
    template<typename E>
    double z_rot2angle(E &&R);

    /*
     * 3D rotation matrix.
     *
     * Parameters
     * ----------
     * axis : xt::xexpression
     *     (3,) rotation axis.
     * angle : double
     *     Signed rotation angle [rad].
     *
     * Returns
     * -------
     * R : xt::xtensor<double, 2>
     *     (3, 3) rotation matrix.
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include <cmath>
     *    #include "xtensor/xtensor.hpp"
     *    #include "pypeline/util/math/linalg.hpp"
     *
     *    xt::xtensor<double, 1> axis {1, 1, 1};
     *    double angle = M_PI / 2.0;
     *
     *    namespace linalg = pypeline::util::math::linalg;
     *    auto R = linalg::rot(axis, angle);
     */
    template<typename E>
    xt::xtensor<double, 2> rot(E &&axis, const double angle);
}

#include "_linalg.tpp"

#endif //PYPELINE_UTIL_MATH_LINALG_HPP
