// ############################################################################
// fourier.hpp
// ===========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * FFT-based tools.
 */

#ifndef PYPELINE_UTIL_MATH_FOURIER_HPP
#define PYPELINE_UTIL_MATH_FOURIER_HPP

#include <stdexcept>
#include <string>
#include <tuple>

#include "xtensor/xtensor.hpp"
#include "xtensor/xbuilder.hpp"

#include "pypeline/util/argcheck.hpp"

namespace pypeline { namespace util { namespace math { namespace fourier {
    /*
     * Signal sample positions for :cpp:func:`ffs` and :cpp:class:`FFS`.
     *
     * Return the coordinates at which a signal must be sampled to use
     * :cpp:func:`ffs` and :cpp:class:`FFS`.
     *
     * Parameters
     * ----------
     * T : double
     *     Function period.
     * N_FS : size_t
     *     Function bandwidth, odd-valued.
     * T_c : double
     *     Period mid-point.
     * N_s : size_t
     *     Number of samples.
     *
     * Returns
     * -------
     * sample_points : xt::xtensor<double, 1>
     *     (N_s,) coordinates at which to sample a signal (in the right order).
     *
     * Examples
     * --------
     * Let :math:`\phi: \mathbb{R} \to \mathbb{C}` be a bandlimited periodic
     * function of period :math:`T = 1`, bandwidth :math:`N_{FS} = 5`, and with
     * one period centered at :math:`T_{c} = \pi`.
     * The sampling points :math:`t[n] \in \mathbb{R}` at which :math:`\phi`
     * must be evaluated to compute the Fourier Series coefficients
     * :math:`\left\{ \phi_{k}^{FS}, k = -2, \ldots, 2 \right\}` with
     * :cpp:func:`ffs` or :cpp:class:`FFS` are obtained as follows:
     *
     * .. literal_block::
     *
     *    #include <cmath>
     *    #include "pypeline/util/math/fourier.hpp"
     *    namespace fourier = pypeline::util::math::fourier;
     *
     *    double T {1}, T_c {M_PI};
     *    size_t N_FS {5}, N_samples {8};  // Ideally choose N_s to be
     *                                     // highly-composite for ffs()/FFS().
     *
     *    auto sample_points = fourier::ffs_sample(T, N_FS, T_c, N_samples);
     *
     * See Also
     * --------
     * :cpp:func:`ffs`, :cpp:class:`FFS`
     */
    xt::xtensor<double, 1> ffs_sample(const double T,
                                      const size_t N_FS,
                                      const double T_c,
                                      const size_t N_s) {
        namespace argcheck = pypeline::util::argcheck;
        if (T <= 0) {
            std::string msg = "Parameter[T] must be positive.";
            throw std::runtime_error(msg);
        }
        if (!(argcheck::is_odd(N_FS) && (N_FS >= 3))) {
            std::string msg = "Parameter[N_FS] must be odd-valued and at least 3.";
            throw std::runtime_error(msg);
        }
        if (N_s < N_FS) {
            std::string msg = "Parameter[N_s] must be greater or equal to the signal bandwidth.";
            throw std::runtime_error(msg);
        }

        xt::xtensor<double, 1> sample_points;
        if (argcheck::is_odd(N_s)) {
            size_t M = (N_s - 1) / 2;
            auto idx = xt::concatenate(std::make_tuple(
                           xt::arange<int>(0, M + 1),
                           xt::arange<int>(-M, 0)));
            sample_points = T_c + (T / (2 * M + 1)) * idx;
        } else {
            size_t M = N_s / 2;
            auto idx = xt::concatenate(std::make_tuple(
                           xt::arange<int>(0, M),
                           xt::arange<int>(-M, 0)));
            sample_points = T_c + (T / (2 * M)) * (0.5 + idx);
        }

        return sample_points;
    }
}}}}

#endif //PYPELINE_UTIL_MATH_FOURIER_HPP
