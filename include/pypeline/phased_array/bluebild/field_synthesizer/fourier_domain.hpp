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

#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "eigen3/Eigen/Eigen"
#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xbuilder.hpp"

#include "pypeline/types.hpp"
#include "pypeline/util/argcheck.hpp"
#include "pypeline/util/math/fourier.hpp"
#include "pypeline/util/math/func.hpp"
#include "pypeline/util/math/linalg.hpp"
#include "pypeline/util/math/sphere.hpp"

namespace argcheck = pypeline::util::argcheck;
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
            xt::xtensor<TT, 2> m_XYZk;  // Antenna positions at kernel eval time.

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
                Eigen::Map<MatrixXX_t<TT>> _mXYZ(m_XYZk.data(), m_N_antenna, 3);

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
                m_XYZk = XYZ;
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
