// ############################################################################
// func.hpp
// ========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Special functions.
 */

#ifndef PYPELINE_UTIL_MATH_FUNC_HPP
#define PYPELINE_UTIL_MATH_FUNC_HPP

#include <cmath>
#include <stdexcept>
#include <string>

#include "xtensor/xarray.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xeval.hpp"

#include "pypeline/util/argcheck.hpp"

namespace pypeline { namespace util { namespace math { namespace func {
    /*
     * Parameterized Tukey function.
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include "pypeline/util/math/func.hpp"
     *
     *    namespace func = pypeline::util::math::func;
     *    auto tukey = func::Tukey(1, 0.5, 0.25);
     *
     *    auto sample_points = xt::linspace<double>(0, 1, 25);
     *    auto amplitude = tukey(sample_points);
     *
     * Notes
     * -----
     * The Tukey function is defined as:
     *
     * .. math::
     *
     *    \text{Tukey}(T, \beta, \alpha)(\varphi): \mathbb{R} & \to [0, 1] \\
     *    \varphi & \to
     *    \begin{cases}
     *        % LINE 1
     *        \sin^{2} \left( \frac{\pi}{T \alpha}
     *                 \left[ \frac{T}{2} - \beta + \varphi \right] \right) &
     *        0 \le \frac{T}{2} - \beta + \varphi < \frac{T \alpha}{2} \\
     *        % LINE 2
     *        1 &
     *        \frac{T \alpha}{2} \le \frac{T}{2} - \beta +
     *        \varphi \le T - \frac{T \alpha}{2} \\
     *        % LINE 3
     *        \sin^{2} \left( \frac{\pi}{T \alpha}
     *                 \left[ \frac{T}{2} + \beta - \varphi \right] \right) &
     *        T - \frac{T \alpha}{2} < \frac{T}{2} - \beta + \varphi \le T \\
     *        % LINE 4
     *        0 &
     *        \text{otherwise.}
     *    \end{cases}
     */
    class Tukey {
        public:
            /*
             * Parameters
             * ----------
             * T : double
             *     Function support.
             * beta : double
             *     Function mid-point.
             * alpha : double
             *     Decay-rate in [0, 1].
             */
            Tukey(const double T, const double beta, const double alpha):
                m_T(T), m_beta(beta), m_alpha(alpha) {
                if (T <= 0) {
                    std::string msg = "Parameter[T] must be positive.";
                    throw std::runtime_error(msg);
                }
                if (!((0 <= alpha) && (alpha <= 1))) {
                    std::string msg = "Parameter[alpha] must lie in [0, 1].";
                    throw std::runtime_error(msg);
                }
            }

            /*
             * Sample the Tukey(T, beta, alpha) function.
             *
             * Parameters
             * ----------
             * x : xt::xexpression
             *     Sample points.
             *
             * Returns
             * -------
             * amplitude : xt::xarray<double>
             */
            template <typename E>
            xt::xarray<double> operator()(E &&x) {
                namespace argcheck = pypeline::util::argcheck;
                if (!argcheck::has_floats(x)) {
                    std::string msg = "Parameter[x] must contain real values.";
                    throw std::runtime_error(msg);
                }

                xt::xarray<double> y {x - m_beta + (0.5 * m_T)};
                xt::xarray<double> amplitude {xt::zeros<double>(y.shape())};

                double lim_left = 0.5 * m_T * m_alpha;
                xt::xarray<bool> mask_left {(0 <= y) && (y < lim_left)};
                xt::xarray<double> sqrt_amp_left {xt::sin(M_PI / (m_T * m_alpha) * xt::filter(y, mask_left))};
                xt::filter(amplitude, mask_left) = xt::square(sqrt_amp_left);

                double lim_right = m_T - (0.5 * m_T * m_alpha);
                xt::xarray<bool> mask_right {(lim_right < y) & (y <= m_T)};
                xt::xarray<double> sqrt_amp_right {xt::sin(M_PI / (m_T * m_alpha) * (m_T - xt::filter(y, mask_right)))};
                xt::filter(amplitude, mask_right) = xt::square(sqrt_amp_right);

                xt::xarray<bool> mask_middle {(lim_left <= y) && (y <= lim_right)};
                xt::filter(amplitude, mask_middle) = 1;

                return amplitude;
            }

            std::string __repr__() {
                std::string repr = ("<Tukey(T=" + std::to_string(m_T) +
                                    ", beta=" + std::to_string(m_beta) +
                                    ", alpha=" + std::to_string(m_alpha) + ">");
                return repr;
            }

        private:
            const double m_T = 0;
            const double m_beta = 0;
            const double m_alpha = 0;
    };
}}}}

#endif //PYPELINE_UTIL_MATH_FUNC_HPP
