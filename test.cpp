// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <cmath>
#include <iostream>
#include "pypeline/util/math/fourier.hpp"
#include "pypeline/phased_array/bluebild/field_synthesizer/fourier_domain.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xview.hpp"

namespace fourier = pypeline::util::math::fourier;
namespace f_synth = pypeline::phased_array::bluebild::field_synthesizer::fourier_domain;

int main() {
    const double wl = 2.0;
    const size_t N_height = 50;
    const size_t N_width = 51;
    const xt::xtensor<double, 2> grid_colat {xt::reshape_view(xt::linspace<double>(0, M_PI, N_height),
                                                              std::vector<size_t> {N_height, 1})};
    const xt::xtensor<double, 2> grid_lon {xt::reshape_view(xt::linspace<double>(0, 0.5 * M_PI, N_width),
                                                            std::vector<size_t>{1, N_width})};
    const size_t N_FS = 15;
    const double T = M_PI;
    const xt::xtensor<double, 2> R {{1, 0, 0},
                                    {0, 1, 0},
                                    {0, 0, 1}};
    const size_t N_eig = 4;
    const size_t N_antenna = 48;
    const size_t N_threads = 1;
    auto effort = fourier::planning_effort::NONE;

    auto bb = f_synth::FourierFieldSynthesizerBlock<double>(wl, grid_colat, grid_lon,
                                                            N_FS, T, R, N_eig, N_antenna,
                                                            N_threads, effort);
    std::cout << bb.__repr__() << std::endl;

    return 0;
}
