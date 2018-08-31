// ############################################################################
// fourier_domain.hpp
// ==================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Field synthesizers that work in Fourier Series domain.
 */

#ifndef PYPELINE_PHASED_ARRAY_BLUEBILD_FIELD_SYNTHESIZER_FOURIER_DOMAIN
#define PYPELINE_PHASED_ARRAY_BLUEBILD_FIELD_SYNTHESIZER_FOURIER_DOMAIN

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "eigen3/Eigen/Eigen"
#include "xtensor/xtensor.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstrided_view.hpp"

#include "pypeline/types.hpp"
#include "pypeline/util/argcheck.hpp"
#include "pypeline/util/math/fourier.hpp"
#include "pypeline/util/math/func.hpp"
#include "pypeline/util/math/linalg.hpp"
#include "pypeline/util/math/sphere.hpp"

#include <iostream>
#include "xtensor/xio.hpp"

namespace argcheck = pypeline::util::argcheck;
namespace array = pypeline::util::array;
namespace fourier = pypeline::util::math::fourier;
namespace func = pypeline::util::math::func;
namespace linalg = pypeline::util::math::linalg;
namespace sphere = pypeline::util::math::sphere;

namespace pypeline { namespace phased_array { namespace bluebild { namespace field_synthesizer { namespace fourier_domain {
    /*
     * TODO: docstring + documentation.
     */
    template <typename TT>
    class FourierFieldSynthesizerBlock {
        private:
            static_assert(std::is_floating_point<TT>::value, "Only {float, double} are allowed for Type[T].");
            static constexpr bool is_float = std::is_same<TT, float>::value;
            using cTT = std::complex<TT>;

            // block_0_parameters
            size_t m_wl = 0;
            size_t m_N_antenna = 0;
            size_t m_N_eig = 0;
            xt::xtensor<TT, 2> m_grid_colat;
            xt::xtensor<TT, 2> m_grid_lon;
            xt::xtensor<TT, 2> m_R;
            std::unique_ptr<xt::xtensor<TT, 2>> m_XYZk = nullptr;  // Antenna positions at kernel eval time.

            // block_1_parameters
            TT m_alpha_window = 0;
            TT m_T = 0;
            TT m_Tc = 0;
            TT m_mps = 0;  // max_phase_shift
            size_t m_N_FS = 0;
            size_t m_N_samples = 0;

            // Resources
            std::unique_ptr<fourier::FFTW_FFS<TT>> m_FSK; // Fourier Series Kernel compute/storage.
            std::unique_ptr<fourier::FFTW_FFS<TT>> m_FST; // Field STatistics compute/storage.

            template <typename E_colat, typename E_lon, typename E_R>
            void set_block_0_parameters(const double wl,
                                        const size_t N_antenna,
                                        const size_t N_eig,
                                        E_colat &&grid_colat,
                                        E_lon &&grid_lon,
                                        E_R &&R) {
                if (wl <= 0) {
                    std::string msg = "Parameter[wl] must be positive.";
                    throw std::runtime_error(msg);
                }
                m_wl = wl;

                if(!(argcheck::has_shape(grid_colat, std::array<size_t, 2> {grid_colat.size(), 1}) &&
                     argcheck::has_floats(grid_colat))) {
                    std::string msg = "Parameter[grid_colat] must be (N_height, 1) real-valued.";
                    throw std::runtime_error(msg);
                }
                m_grid_colat = grid_colat;

                if(!(argcheck::has_shape(grid_lon, std::array<size_t, 2> {1, grid_lon.size()}) &&
                     argcheck::has_floats(grid_lon))) {
                    std::string msg = "Parameter[grid_lon] must be (1, N_width) real-valued.";
                    throw std::runtime_error(msg);
                }
                m_grid_lon = grid_lon;

                if(!(argcheck::has_shape(R, std::array<size_t, 2> {3, 3}) &&
                     argcheck::has_floats(R))) {
                    std::string msg = "Parameter[R] must be a (3, 3) rotation matrix.";
                    throw std::runtime_error(msg);
                }
                m_R = R;

                if (N_antenna == 0) {
                    std::string msg = "Parameter[N_antenna] must be positive.";
                    throw std::runtime_error(msg);
                }
                m_N_antenna = N_antenna;

                if (N_eig == 0) {
                    std::string msg = "Parameter[N_eig] must be positive.";
                    throw std::runtime_error(msg);
                }
                m_N_eig = N_eig;
            }

            void set_block_1_parameters(const size_t N_FS,
                                        const double T) {
                if (argcheck::is_even(N_FS)) {
                    std::string msg = "Parameter[N_FS] must be odd-valued.";
                    throw std::runtime_error(msg);
                }
                if (!((0 < T) && (T <= 2 * M_PI))) {
                    std::string msg = "Parameter[T] must lie in (0, 2pi].";
                    throw std::runtime_error(msg);
                }

                if (!xt::allclose(T, 2 * M_PI)) {  // PeriodicSynthesis
                    m_alpha_window = 0.1;
                    const double aw = m_alpha_window; // shorthand

                    const double T_min = (1 + aw) * (xt::amax(m_grid_lon)[0] - xt::amin(m_grid_lon)[0]);
                    if (T < T_min) {
                        std::stringstream msg;
                        msg << "Given Parameter[grid_lon], "
                            << "Parameter[T] must be at least "
                            << std::to_string(T_min);
                        throw std::runtime_error(msg.str());
                    }
                    m_T = T;

                    const double lon_start = m_grid_lon(0, 0);
                    const double lon_end = m_grid_lon(0, m_grid_lon.size() - 1);
                    const double T_start = lon_end + T * (0.5 * aw - 1);
                    const double T_end = lon_end + T * 0.5 * aw;
                    m_Tc = (T_start + T_end) / 2.0;
                    m_mps = lon_start - (T_start + 0.5 * T * aw);

                    size_t N_FS_trunc = std::ceil((N_FS * T) / (2 * M_PI));
                    N_FS_trunc += ((argcheck::is_even(N_FS_trunc)) ? 1 : 0);
                    m_N_FS = N_FS_trunc;
                } else {  // No PeriodicSynthesis, but set params to still work.
                    m_alpha_window = 0;
                    m_T = 2 * M_PI;
                    m_Tc = M_PI;
                    m_mps = 2 * M_PI;
                    m_N_FS = N_FS;
                }

                m_N_samples = fourier::FFTW_size_finder(m_N_FS).next_fast_len();
            }

            void allocate_resources(const size_t N_threads,
                                    fourier::planning_effort effort) {
                std::vector<size_t> shape_FSK {m_N_antenna * m_grid_colat.size(), m_N_samples};
                m_FSK = std::make_unique<fourier::FFTW_FFS<TT>>(shape_FSK, 1,
                                                                m_T, m_Tc, m_N_FS,
                                                                true, N_threads, effort);

                std::vector<size_t> shape_FST {m_N_eig * m_grid_colat.size(), m_N_samples};
                m_FST = std::make_unique<fourier::FFTW_FFS<TT>>(shape_FST, 1,
                                                                m_T, m_Tc, m_N_FS,
                                                                true, N_threads, effort);
            }

            double phase_shift(xt::xtensor<TT, 2> &XYZ) {
                Eigen::Map<MatrixXX_t<TT>> _XYZ(XYZ.data(), m_N_antenna, 3);
                Eigen::Map<MatrixXX_t<TT>> _mXYZ(m_XYZk->data(), m_N_antenna, 3);

                MatrixXX_t<TT> R_T = (_mXYZ.leftCols(2)
                                      .fullPivHouseholderQr()
                                      .solve(_XYZ.leftCols(2)));
                xt::xtensor<TT, 2> R {{R_T(0, 0), R_T(1, 0), 0},
                                      {R_T(0, 1), R_T(1, 1), 0},
                                      {        0,         0, 1}};

                const double theta = linalg::z_rot2angle(R);
                return theta;
            }

            bool regen_required(const double shift) {
                const double lhs = -0.1 * (M_PI / 180); // Slightly below 0 [rad] due to numerical rounding.
                if ((lhs <= shift) && (shift <= m_mps)) {
                    return false;
                } else {
                    return true;
                }
            }

            void regen_kernel(xt::xtensor<TT, 2> &XYZ) {
                xt::xtensor<TT, 1> lon_smpl {fourier::ffs_sample(m_T, m_N_FS, m_Tc, m_N_samples)};

                auto px_xyz = sphere::pol2cart(
                                xt::xtensor<TT, 1> {1},
                                m_grid_colat,
                                xt::reshape_view(lon_smpl,
                                                 std::vector<size_t> {1, m_N_samples}));
                xt::xtensor<TT, 3> pix_smpl {xt::stack(std::move(px_xyz), 0)};
                // m_N_samples assumes imaging is performed with XYZ centered at the origin.
                xt::xtensor<TT, 2> XYZ_c {XYZ - xt::mean(XYZ, {0})};

                const size_t N_height = m_grid_colat.size();
                Eigen::Map<MatrixXX_t<TT>> _pix_smpl(pix_smpl.data(), 3, N_height * m_N_samples);
                Eigen::Map<MatrixXX_t<TT>> _XYZ_c(XYZ_c.data(), m_N_antenna, 3);
                Eigen::Map<ArrayXX_t<cTT>> _FSK((m_FSK->view_in()).data(), m_N_antenna, N_height * m_N_samples);
                std::complex<TT> _1j(0, 1);
                _FSK = ((_1j * static_cast<TT>(2 * M_PI / m_wl) * _XYZ_c) *
                         _pix_smpl).array().exp();
                // TODO: exp() might be faster with explicit MKL?

                func::Tukey tukey(m_T, m_Tc, m_alpha_window);
                xt::xtensor<TT, 1> window {tukey(lon_smpl)};
                m_FSK->view_in().multiplies_assign(window);
                m_FSK->ffs();
                m_XYZk = std::make_unique<xt::xtensor<TT, 2>>(XYZ);
            }

            void validate_shapes(xt::xtensor<cTT, 2> &V,
                                 xt::xtensor<TT, 2> &XYZ,
                                 xt::xtensor<cTT, 2> &W) {
                const size_t N_beam = V.shape()[0];

                std::vector<size_t> shape_V(V.dimension());
                std::copy(V.shape().begin(), V.shape().end(), shape_V.begin());
                if (shape_V != std::vector<size_t> {N_beam, m_N_eig}) {
                    std::string msg = "Parameter[V] does not have shape (N_beam, N_eig).";
                    throw std::runtime_error(msg);
                }

                std::vector<size_t> shape_XYZ(XYZ.dimension());
                std::copy(XYZ.shape().begin(), XYZ.shape().end(), shape_XYZ.begin());
                if (shape_XYZ != std::vector<size_t> {m_N_antenna, 3}) {
                    std::string msg = "Parameter[XYZ] does not have shape (N_antenna, 3).";
                    throw std::runtime_error(msg);
                }

                std::vector<size_t> shape_W(W.dimension());
                std::copy(W.shape().begin(), W.shape().end(), shape_W.begin());
                if (shape_W != std::vector<size_t> {m_N_antenna, N_beam}) {
                    std::string msg = "Parameters[V, W] have inconsistent dimensions.";
                    throw std::runtime_error(msg);
                }
            }

        public:
            template <typename E_colat, typename E_lon, typename E_R>
            FourierFieldSynthesizerBlock(const double wl,
                                         E_colat &&grid_colat,
                                         E_lon &&grid_lon,
                                         const size_t N_FS,
                                         const double T,
                                         E_R && R,
                                         const size_t N_eig,
                                         const size_t N_antenna,
                                         const size_t N_threads,
                                         fourier::planning_effort effort) {
                set_block_0_parameters(wl, N_antenna, N_eig,
                                       grid_colat, grid_lon, R);
                set_block_1_parameters(N_FS, T);
                allocate_resources(N_threads, effort);
            }

            auto operator()(xt::xtensor<cTT, 2> &V,
                            xt::xtensor<TT, 2> &XYZ,
                            xt::xtensor<cTT, 2> &W) {
                validate_shapes(V, XYZ, W);

                // icrs_XYZ -> bfsf_XYZ
                Eigen::Map<MatrixXX_t<TT>> _XYZ(XYZ.data(), m_N_antenna, 3);
                Eigen::Map<MatrixXX_t<TT>> _R(m_R.data(), 3, 3);

                xt::xtensor<TT, 2> bfsf_XYZ {xt::zeros<TT>(std::vector<size_t>{m_N_antenna, 3})};
                Eigen::Map<MatrixXX_t<TT>> _bfsf_XYZ(bfsf_XYZ.data(), m_N_antenna, 3);
                _bfsf_XYZ = _XYZ * _R.transpose();

                // Phase shift + kernel evaluation
                TT shift = std::numeric_limits<TT>::infinity();
                if (m_XYZk != nullptr) {
                    shift = phase_shift(bfsf_XYZ);
                }
                if (regen_required(shift)) {
                    regen_kernel(bfsf_XYZ);
                    shift = 0;
                }

                cTT _1j(0, 1);
                const int N = (static_cast<int>(m_N_FS) - 1) / 2;
                const int Q = m_N_samples - m_N_FS;
                const size_t N_beam = W.shape()[1];
                const size_t N_height = m_grid_colat.size();
                Eigen::Map<MatrixXX_t<cTT>> _V(V.data(), N_beam, m_N_eig);
                Eigen::Map<MatrixXX_t<cTT>> _W(W.data(), m_N_antenna, N_beam);
                Eigen::Map<MatrixXX_t<cTT>> _FSK((m_FSK->view_in()).data(), m_N_antenna, N_height * m_N_samples);

                // Eigenfunctions (Fourier domain)
                auto E_FS = m_FST->view_in();  // (N_eig * N_height, N_samples)
                Eigen::Map<MatrixXX_t<cTT>> _E_FS(E_FS.data(), m_N_eig, N_height * m_N_samples);
                _E_FS = _V.transpose() * (_W.transpose() * _FSK);

                cTT base = std::exp(-_1j * static_cast<TT>((2 * M_PI * shift) / m_T));
                xt::xtensor<TT, 1> exponent = xt::concatenate(
                                                std::make_tuple(
                                                  xt::arange<int>(-N, N + 1),
                                                  xt::zeros<int>({Q})));
                xt::xtensor<cTT, 1> mod {xt::pow(base, exponent)};
                E_FS.multiplies_assign(mod);

                // Field Statistics
                m_FST->iffs();
                auto E_Ny = m_FST->view_out();
                auto _I_Ny = m_FST->view_out();
                _I_Ny.assign(xt::square(xt::abs(E_Ny)));

                // _I_Ny is a complex-valued container: extract its real part only.
                const size_t N_cells = m_N_eig * N_height * m_N_samples;
                auto I_Ny = xt::adapt(reinterpret_cast<TT*>(m_FST->data_out()),
                                      N_cells, xt::no_ownership(),
                                      std::vector<size_t> {m_N_eig, N_height, m_N_samples},
                                      std::vector<size_t> {2 * m_N_samples * N_height,
                                                           2 * m_N_samples,
                                                           2});
                return I_Ny;
            }

            template <typename E_stat>
            xt::xtensor<TT, 3> synthesize(E_stat &&stat) {
                const size_t N_level = stat.shape()[0];
                const size_t N_height = m_grid_colat.size();
                const size_t N_width = m_grid_lon.size();

                { // Verify arguments
                    namespace argcheck = pypeline::util::argcheck;
                    if (!argcheck::has_floats(stat)) {
                        std::string msg = "Parameter[stat] must have real-valued entries.";
                        throw std::runtime_error(msg);
                    }
                    std::array<size_t, 3> shape_stat {N_level, N_height, m_N_samples};
                    if (!argcheck::has_shape(stat, shape_stat)) {
                        std::string msg = "Parameter[stat] must have shape (N_level, N_height, N_samples).";
                        throw std::runtime_error(msg);
                    }
                }

                { // Fill m_FST->view_in() with statistics + go to FS domain.
                    m_FST->view_in() = 0;
                    const size_t N_cells = N_level * N_height * m_N_samples;
                    xt::adapt(m_FST->data_in(), N_cells, xt::no_ownership(),
                              std::vector<size_t> {N_level, N_height, m_N_samples}) = stat;
                    m_FST->ffs();
                }

                std::vector<size_t> shape_transform {N_level * N_height, m_N_FS};
                fourier::FFTW_FS_INTERP<TT> transform(shape_transform,
                                                      1, m_T,
                                                      m_grid_lon(0, 0),
                                                      m_grid_lon(0, N_width - 1),
                                                      N_width,
                                                      true, 1, fourier::planning_effort::NONE);

                { // Fill transform.in() with m_FST->view_out() + fs_interp()
                    const size_t N_cells = N_level * N_height * m_N_samples;
                    const auto& idx = array::index(shape_transform.size(), 1, xt::range(0, m_N_FS));
                    const auto& transform_in = xt::strided_view(
                        xt::adapt(m_FST->data_out(), N_cells, xt::no_ownership(),
                                  std::vector<size_t> {N_level * N_height, m_N_samples}), idx);
                    transform.in(transform_in);
                    transform.fs_interp();
                }

                // allocate output buffer and copy FFTW_FS_INTERP->view_out()
                xt::xtensor<TT, 3> field {xt::zeros<TT>({N_level, N_height, N_width})};
                {
                    const size_t N_cells = N_level * N_height * N_width;
                    auto _field = xt::adapt(field.data(), N_cells, xt::no_ownership(),
                                            std::vector<size_t> {N_level * N_height, N_width});
                    auto _strides = transform.view_out().strides();
                    std::vector<size_t> strides(2);
                    std::copy(_strides.begin(), _strides.end(), strides.begin());
                    for (size_t i = 0; i < strides.size(); ++i) {
                        strides[i] *= 2;
                    }
                    auto _transform_out = xt::adapt(reinterpret_cast<TT*>(transform.view_out().data()),
                                                    N_cells, xt::no_ownership(),
                                                    std::vector<size_t> {N_level * N_height, N_width},
                                                    std::vector<size_t> {strides});
                    _field = _transform_out;
                }
                return field;
            }

            std::string __repr__() {
                std::stringstream msg;
                msg << "FourierFieldSynthesizerBlock<" << ((is_float) ? "float" : "double") << ">("
                    << "wl=" << std::to_string(m_wl) << ", "
                    << "N_antenna=" << std::to_string(m_N_antenna) << ", "
                    << "N_eig=" << std::to_string(m_N_eig) << ", "
                    << "alpha_window=" << std::to_string(m_alpha_window) << ", "
                    << "T=" << std::to_string(m_T) << ", "
                    << "Tc=" << std::to_string(m_Tc) << ", "
                    << "mps=" << std::to_string(m_mps) << ", "
                    << "N_FS=" << std::to_string(m_N_FS) << ")";

                return msg.str();
            }
    };
}}}}}

#endif //PYPELINE_PHASED_ARRAY_BLUEBILD_FIELD_SYNTHESIZER_FOURIER_DOMAIN
