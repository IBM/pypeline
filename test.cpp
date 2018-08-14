// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <complex>
#include <cmath>
#include <vector>
#include <iostream>

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xio.hpp"

#include "pypeline/util/array.hpp"
#include "pypeline/util/math/fourier.hpp"

namespace array = pypeline::util::array;
namespace fourier = pypeline::util::math::fourier;

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
    const double T = M_PI;
    const double T_c = M_E;
    const size_t N_FS = 15;
    const size_t N_samples = 16;

    const std::vector<size_t> shape_transform {N_samples};
    const size_t axis_transform = 0;
    const bool inplace = false;
    const size_t N_threads = 1;
    const fourier::planning_effort effort = fourier::planning_effort::NONE;

    xt::xarray<double> sample_points {fourier::ffs_sample(T, N_FS, T_c, N_samples)};
    xt::xarray<double> diric_samples {dirichlet(sample_points, T, T_c, N_FS)};
    xt::xarray<std::complex<double>> diric_FS_exact {dirichlet_FS_theory(T, T_c, N_FS)};

    fourier::FFTW_FFS<double> transform(shape_transform,
                                        axis_transform,
                                        T, T_c, N_FS,
                                        inplace, N_threads, effort);

    transform.view_in() = diric_samples;
    transform.view_out() = 0;
    transform.ffs();
    xt::xarray<std::complex<double>> diric_FS {transform.view_out()};

    auto idx = array::index(diric_FS.dimension(), axis_transform, xt::range(0, N_FS));
    std::cout << xt::allclose(xt::strided_view(diric_FS, idx), diric_FS_exact) << std::endl;


    transform.view_out() = diric_samples;
    transform.view_in() = 0;
    transform.ffs_r();
    diric_FS = transform.view_in();

    idx = array::index(diric_FS.dimension(), axis_transform, xt::range(0, N_FS));
    std::cout << xt::allclose(xt::strided_view(diric_FS, idx), diric_FS_exact) << std::endl;

    return 0;
}
