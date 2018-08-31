// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <cmath>
#include <chrono>
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
#include "eigen3/Eigen/Sparse"

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
    const size_t N_eig = 12;
    const size_t N_antenna = 48;
    const size_t N_beam = 12;
    const size_t N_antenna_per_beam = N_antenna / N_beam;
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
    // xt::xtensor<cTT, 2> W {xt::zeros<TT>({N_antenna, N_beam})};
    // TT elem = 0;
    // for (size_t i = 0; i < N_beam; ++i) {
    //     for (size_t j = 0; j < N_antenna_per_beam; ++j) {
    //         size_t k = i * N_antenna_per_beam + j;
    //         W(k, i) = elem;
    //         elem += 1.0;
    //     }
    // }
    std::vector<Eigen::Triplet<cTT>> triplets(N_antenna);
    TT elem = 0;
    size_t l = 0;
    for (size_t i = 0; i < N_beam; ++i) {
        for (size_t j = 0; j < N_antenna_per_beam; ++j) {
            size_t k = i * N_antenna_per_beam + j;
            triplets[l] = Eigen::Triplet<cTT>(k, i, elem);
            elem += 1.0;
            l += 1;
        }
    }
    SpMatrixXX_t<cTT> W(N_antenna, N_beam);
    W.setFromTriplets(triplets.begin(), triplets.end());

    auto start_stat = std::chrono::high_resolution_clock::now();
    xt::xtensor<TT, 3> stat {bb(V, XYZ, W)};
    auto stop_stat = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_stat = stop_stat - start_stat;
    std::cout << "Elapsed time: " << elapsed_stat.count() << " [s]" << std::endl;

    auto start_stat2 = std::chrono::high_resolution_clock::now();
    xt::xtensor<TT, 3> stat2 {bb(V, XYZ, W)};
    auto stop_stat2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_stat2 = stop_stat2 - start_stat2;
    std::cout << "Elapsed time: " << elapsed_stat2.count() << " [s]" << std::endl;

    auto field = bb.synthesize(stat);

    return 0;
}
