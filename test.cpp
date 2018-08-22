// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <complex>
#include <cmath>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xindex_view.hpp"
#include "pypeline/util/math/fourier.hpp"
namespace fourier = pypeline::util::math::fourier;

// Compute samples from a shifted Dirichlet kernel.
xt::xarray<double> dirichlet(xt::xarray<double> x,
                             const double T,
                             const double T_c,
                             const size_t N_FS) {
    xt::xarray<double> y {x - T_c};

    xt::xarray<double> numerator {xt::zeros<double>({x.size()})};
    xt::xarray<double> denominator {xt::zeros<double>({x.size()})};

    xt::xarray<bool> nan_mask {xt::isclose(xt::fmod(y, M_PI), 0)};
    xt::filter(numerator, ~nan_mask) = xt::sin((N_FS * M_PI / T) * xt::filter(y, ~nan_mask));
    xt::filter(denominator, ~nan_mask) = xt::sin((M_PI / T) * xt::filter(y, ~nan_mask));
    xt::filter(numerator, nan_mask) = N_FS * xt::cos((N_FS * M_PI / T) * xt::filter(y, nan_mask));
    xt::filter(denominator, nan_mask) = xt::cos((M_PI / T) * xt::filter(y, nan_mask));

    xt::xarray<double> vals {numerator / denominator};
    return vals;
}

// Analytical FS coefficients of a shifted Dirichlet kernel.
xt::xarray<std::complex<double>> dirichlet_FS_theory(const double T,
                                                     const double T_c,
                                                     const size_t N_FS) {
    const size_t N = (N_FS - 1) / 2;
    std::complex<double> _1j(0, 1);

    std::complex<double> base = exp(-_1j * (2 * M_PI * T_c) / T);
    xt::xarray<double> exponent {xt::arange<int>(-N, N+1)};

    xt::xarray<std::complex<double>> kernel {xt::pow(base, exponent)};
    return kernel;
}

int main() {
    // Signal Parameters
    const double T = M_PI;
    const double T_c = M_E;
    const size_t N_FS = 15;
    xt::xarray<std::complex<double>> diric_FS {dirichlet_FS_theory(T, T_c, N_FS)};

    // Ground-truth: exact interpolated result.
    const double a = T_c - 0.5 * T;
    const double b = T_c + 0.5 * T;
    const size_t M = 10;  // We want a lot of interpolated points.
    xt::xarray<double> sample_positions {a + ((b - a) / (M - 1)) * xt::arange<int>(M)};
    xt::xarray<double> diric_sig_exact {dirichlet(sample_positions, T, T_c, N_FS)};

    const std::vector<size_t> shape_transform {N_FS};
    const size_t axis_transform = 0;
    const size_t N_threads = 1;
    auto effort = fourier::planning_effort::NONE;

    // Option 1
    // --------
    // No assumptions on FS spectra: use generic algorithm.
    fourier::FFTW_FS_INTERP<double> interpolant(shape_transform, axis_transform,
                                                T, a, b, M, false,
                                                N_threads, effort);
    interpolant.in(diric_FS);  // fill input
    interpolant.fs_interp();
    xt::xarray<std::complex<double>> diric_sig {interpolant.view_out()};

    // Option 2
    // --------
    // You know that the output is real-valued: use accelerated algorithm.
    fourier::FFTW_FS_INTERP<double> interpolant_real(shape_transform, axis_transform,
                                                     T, a, b, M, true,
                                                     N_threads, effort);
    interpolant_real.in(diric_FS);  // fill input
    interpolant_real.fs_interp();
    xt::xarray<std::complex<double>> diric_sig_real {interpolant_real.view_out()};
    // .view_out() is always complex-valued, but its imaginary part will be 0.

    xt::allclose(diric_sig_exact, diric_sig);       // true
    xt::allclose(diric_sig_exact, diric_sig_real);  // true

    return 0;
}
