// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <cmath>
#include <complex>
#include <iostream>
#include "pypeline/util/math/fourier.hpp"
#include "pypeline/phased_array/bluebild/field_synthesizer/fourier_domain.hpp"
#include "pypeline/types.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include "eigen3/Eigen/Eigen"

namespace fourier = pypeline::util::math::fourier;
namespace f_synth = pypeline::phased_array::bluebild::field_synthesizer::fourier_domain;

int main() {
    using TT = double;
    using cTT = std::complex<TT>;

    const TT wl = 2.0;
    const size_t N_height = 50;
    const size_t N_width = 51;
    const xt::xtensor<TT, 2> grid_colat {xt::reshape_view(xt::linspace<TT>(0, M_PI, N_height, false),
                                                          std::vector<size_t> {N_height, 1})};
    const xt::xtensor<TT, 2> grid_lon {xt::reshape_view(xt::linspace<TT>(0, 0.5 * M_PI, N_width),
                                                        std::vector<size_t>{1, N_width})};
    const size_t N_FS = 17;
    const double T = M_PI;
    const xt::xtensor<TT, 2> R {{1, 0, 0},
                                {0, 1, 0},
                                {0, 0, 1}};
    const size_t N_eig = 4;
    const size_t N_antenna = 48;
    const size_t N_beam = 33;
    const size_t N_threads = 1;
    auto effort = fourier::planning_effort::NONE;

    auto bb = f_synth::FourierFieldSynthesizerBlock<TT>(wl, grid_colat, grid_lon,
                                                        N_FS, T, R, N_eig, N_antenna,
                                                        N_threads, effort);
    std::cout << bb.__repr__() << std::endl;

    xt::xtensor<cTT, 2> V {xt::reshape_view(xt::arange(N_beam * N_eig),
                                            std::vector<size_t> {N_beam, N_eig})};
    xt::xtensor<TT, 2> XYZ {xt::reshape_view(xt::arange(3 * N_antenna),
                                             std::vector<size_t> {N_antenna, 3})};
    xt::xtensor<cTT, 2> W {xt::reshape_view(xt::arange(N_antenna * N_beam),
                                            std::vector<size_t> {N_antenna, N_beam})};
    xt::xtensor<TT, 3> stat {bb(V, XYZ, W)};
    auto field = bb.synthesize(stat);

    std::cout << field << std::endl;

    return 0;
}
