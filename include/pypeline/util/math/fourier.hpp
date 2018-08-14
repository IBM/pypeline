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

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fftw3.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstrided_view.hpp"

#include "pypeline/util/argcheck.hpp"
#include "pypeline/util/array.hpp"

namespace pypeline { namespace util { namespace math { namespace fourier {
    enum class planning_effort: unsigned int {
        NONE = FFTW_ESTIMATE,
        MEASURE = FFTW_MEASURE
    };

    /*
     * Signal sample positions for :cpp:class:`FFTW_FFS`.
     *
     * Return the coordinates at which a signal must be sampled to use :cpp:class:`FFTW_FFS`.
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
     * :cpp:class:`FFTW_FFS` are obtained as follows:
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
     * :cpp:class:`FFTW_FFS`
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
            int M = (static_cast<int>(N_s) - 1) / 2;
            auto idx = xt::concatenate(std::make_tuple(
                           xt::arange<int>(0, M + 1),
                           xt::arange<int>(-M, 0)));
            sample_points = T_c + (T / (2 * M + 1)) * idx;
        } else {
            int M = static_cast<int>(N_s) / 2;
            auto idx = xt::concatenate(std::make_tuple(
                           xt::arange<int>(0, M),
                           xt::arange<int>(-M, 0)));
            sample_points = T_c + (T / (2 * M)) * (0.5 + idx);
        }

        return sample_points;
    }

    /*
     * FFTW wrapper to plan 1D complex->complex (i)FFTs on multi-dimensional tensors.
     *
     * This object automatically allocates input/output buffers and provides a
     * tensor interface to the underlying memory using xtensor views.
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include <vector>
     *    #include "pypeline/util/math/fourier.hpp"
     *    namespace fourier = pypeline::util::math::fourier;
     *
     *    auto transform = fourier::FFTW_FFT<float>( // T = {float, double}.
     *        std::vector<size_t> {512, 200, 3},     // shape of input/output tensors.
     *        0,                                     // axis along which to do the transform.
     *        true,                                  // in-place transform.
     *        1,                                     // number of threads.
     *        fourier::planning_effort::MEASURE);    // effort to put into planning transforms.
     *
     *    transform.view_in() = 1;          // fill input array with constant. Broadcasting rules apply.
     *    transform.fft();                  // execute (one of the) planned transforms.
     *    auto out = transform.view_out();  // tensor-view of output.
     *
     *    auto *ptr = transform.data_out();  // raw pointer and shape access
     *    auto shape = transform.shape();    // for low-level manipulations.
     */
    template <typename T>
    class FFTW_FFT {
        private:
            static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                          "T only accepts {float, double}.");
            static constexpr bool is_float = std::is_same<T, float>::value;
            using fftw_data_t = std::conditional_t<is_float, fftwf_complex, fftw_complex>;
            using fftw_plan_t = std::conditional_t<is_float, fftwf_plan, fftw_plan>;

            fftw_plan_t m_plan_fft;
            fftw_plan_t m_plan_fft_r;
            fftw_plan_t m_plan_ifft;
            fftw_plan_t m_plan_ifft_r;
            std::complex<T> *m_data_in = nullptr;
            std::complex<T> *m_data_out = nullptr;
            size_t m_axis = 0;
            std::vector<size_t> m_shape {};

            void setup_threads(const size_t N_threads) {
                if (is_float) {
                    fftwf_init_threads();
                    fftwf_plan_with_nthreads(N_threads);
                } else {
                    fftw_init_threads();
                    fftw_plan_with_nthreads(N_threads);
                }
            }

            void allocate_buffers(const bool inplace) {
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}

                m_data_in = reinterpret_cast<std::complex<T>*>(fftw_malloc(sizeof(std::complex<T>) * N_cells));
                if (inplace) {
                    m_data_out = m_data_in;
                } else {
                    m_data_out = reinterpret_cast<std::complex<T>*>(fftw_malloc(sizeof(std::complex<T>) * N_cells));
                }

                view_in() = 0;
                view_out() = 0;
            }

            void allocate_plans(const planning_effort effort) {
                // Determine right planning function to use based on T.
                using fftw_plan_func_t = fftw_plan_t (*)(int, const fftw_iodim *,
                                                         int, const fftw_iodim *,
                                                         fftw_data_t *, fftw_data_t *,
                                                         int, unsigned int);
                fftw_plan_func_t plan_func;
                if (is_float) {
                    plan_func = (fftw_plan_func_t) &fftwf_plan_guru_dft;
                } else {
                    plan_func = (fftw_plan_func_t) &fftw_plan_guru_dft;
                }

                // Fill in Guru interface's parameters. =======================
                auto strides = view_in().strides();
                auto shape = view_in().shape();

                const int rank = 1;
                fftw_iodim dims_info {static_cast<int>(shape[m_axis]),
                                      static_cast<int>(strides[m_axis]),
                                      static_cast<int>(strides[m_axis])};
                std::vector<fftw_iodim> dims{dims_info};

                const int howmany_rank = shape.size() - 1;
                std::vector<fftw_iodim> howmany_dims(howmany_rank);
                for (size_t i = 0, j = 0; i < static_cast<size_t>(howmany_rank); ++i, ++j) {
                    if (i == m_axis) {j += 1;}

                    fftw_iodim info {static_cast<int>(shape[j]),
                                     static_cast<int>(strides[j]),
                                     static_cast<int>(strides[j])};
                    howmany_dims[i] = info;
                }

                fftw_data_t *data_in = reinterpret_cast<fftw_data_t*>(m_data_in);
                fftw_data_t *data_out = reinterpret_cast<fftw_data_t*>(m_data_out);
                // ============================================================

                m_plan_fft = plan_func(rank, dims.data(),
                                       howmany_rank, howmany_dims.data(),
                                       data_in, data_out, FFTW_FORWARD,
                                       static_cast<unsigned int>(effort));
                m_plan_fft_r = plan_func(rank, dims.data(),
                                         howmany_rank, howmany_dims.data(),
                                         data_out, data_in, FFTW_FORWARD,
                                         static_cast<unsigned int>(effort));
                m_plan_ifft = plan_func(rank, dims.data(),
                                        howmany_rank, howmany_dims.data(),
                                        data_in, data_out, FFTW_BACKWARD,
                                        static_cast<unsigned int>(effort));
                m_plan_ifft_r = plan_func(rank, dims.data(),
                                          howmany_rank, howmany_dims.data(),
                                          data_out, data_in, FFTW_BACKWARD,
                                          static_cast<unsigned int>(effort));
            }

        public:
            /*
             * Parameters
             * ----------
             * shape : std::vector<size_t>
             *     Dimensions of input/output arrays.
             * axis : size_t
             *     Dimension along which to apply transform.
             * inplace : bool
             *     Perform in-place transforms.
             *     If enabled, only one array will be internally allocated.
             * N_threads : size_t
             *     Number of threads to use.
             * effort : planning_effort
             *
             * Notes
             * -----
             * Input and output buffers are initialized to 0 by default.
             */
            FFTW_FFT(const std::vector<size_t> &shape,
                     const size_t axis,
                     const bool inplace,
                     const size_t N_threads,
                     const planning_effort effort):
                m_axis(axis), m_shape(shape) {
                if (shape.size() < 1) {
                    std::string msg = "Parameter[shape] cannot be empty.";
                    throw std::runtime_error(msg);
                }

                if (axis >= shape.size()) {
                    std::string msg = "Parameter[axis] must be lie in {0, ..., shape.size()-1}.";
                    throw std::runtime_error(msg);
                }

                if (N_threads < 1) {
                    std::string msg = "Parameter[N_threads] must be positive.";
                    throw std::runtime_error(msg);
                }

                setup_threads(N_threads);
                allocate_buffers(inplace);
                allocate_plans(effort);
            }

            ~FFTW_FFT() {
                // Determine right destroy function to use based on T.
                using fftw_destroy_plan_func_t = void (*)(fftw_plan_t);
                fftw_destroy_plan_func_t destroy_plan_func;
                if (is_float) {
                    destroy_plan_func = (fftw_destroy_plan_func_t) &fftwf_destroy_plan;
                } else {
                    destroy_plan_func = (fftw_destroy_plan_func_t) &fftw_destroy_plan;
                }

                destroy_plan_func(m_plan_fft);
                destroy_plan_func(m_plan_fft_r);
                destroy_plan_func(m_plan_ifft);
                destroy_plan_func(m_plan_ifft_r);

                // Determine right free function to use based on T.
                using fftw_free_func_t = void (*)(fftw_data_t*);
                fftw_free_func_t free_func;
                if (is_float) {
                    free_func = (fftw_free_func_t) &fftwf_free;
                } else {
                    free_func = (fftw_free_func_t) &fftw_free;
                }

                bool out_of_place = (m_data_in != m_data_out);
                free_func(reinterpret_cast<fftw_data_t*>(m_data_in));
                if (out_of_place) {
                    free_func(reinterpret_cast<fftw_data_t*>(m_data_out));
                }
            }

            /*
             * Returns
             * -------
             * data_in : std::complex<T>*
             *     Pointer to input array.
             */
            std::complex<T>* data_in() {
                return m_data_in;
            }

            /*
             * Returns
             * -------
             * data_out : std::complex<T>*
             *     Pointer to output array.
             *     If `inplace` was set to true, then ``data_in() == data_out()``.
             */
            std::complex<T>* data_out() {
                return m_data_out;
            }

            /*
             * Returns
             * -------
             * shape : std::vector<size_t>
             *     Dimensions of the input buffers.
             */
            std::vector<size_t> shape() {
                return m_shape;
            }

            /*
             * Returns
             * -------
             * view_in : xt::xstrided_view
             *     View on the input array.
             */
            auto view_in() {
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}

                auto view_in = xt::reshape_view(xt::adapt(m_data_in, N_cells, xt::no_ownership()),
                                                std::vector<size_t>{m_shape});
                return view_in;
            }

            /*
             * Returns
             * -------
             * view_out : xt::xstrided_view
             *     View on the output array.
             */
            auto view_out() {
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}

                auto view_out = xt::reshape_view(xt::adapt(m_data_out, N_cells, xt::no_ownership()),
                                                 std::vector<size_t>{m_shape});
                return view_out;
            }

            /*
             * Transform input buffer using 1D-FFT, result available in output buffer.
             */
            void fft() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_fft);
            }

            /*
             * Transform output buffer using 1D-FFT, result available in input buffer.
             */
            void fft_r() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_fft_r);
            }

            /*
             * Transform input buffer using 1D-iFFT, result available in output buffer.
             */
            void ifft() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_ifft);

                // Correct FFTW's lack of scaling during iFFTs.
                const xt::xtensor<T, 1> N {static_cast<T>(m_shape[m_axis])};
                view_out().multiplies_assign(1.0 / N);
            }

            /*
             * Transform output buffer using 1D-iFFT, result available in input buffer.
             */
            void ifft_r() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_ifft_r);

                // Correct FFTW's lack of scaling during iFFTs.
                const xt::xtensor<T, 1> N {static_cast<T>(m_shape[m_axis])};
                view_in().multiplies_assign(1.0 / N);
            }
    };

    /*
     * FFTW wrapper to compute Fourier Series coefficients from signal samples.
     *
     * This object automatically allocates input/output buffers and provides a
     * tensor interface to the underlying memory using xtensor views.
     *
     * Examples
     * --------
     * Let :math:`\phi(t)` be a shifted Dirichlet kernel of period :math:`T` and
     * bandwidth :math:`N_{FS} = 2 N + 1`:
     *
     * .. math::
     *
     *    \phi(t) = \sum_{k = -N}^{N} \exp\left( j \frac{2 \pi}{T} k (t - T_{c}) \right)
     *            = \frac{\sin\left( N_{FS} \pi [t - T_{c}] / T \right)}{\sin\left( \pi [t - T_{c}] / T \right)}.
     *
     * It's Fourier Series (FS) coefficients :math:`\phi_{k}^{FS}` can be analytically
     * evaluated using the shift-modulation theorem:
     *
     * .. math::
     *
     *    \phi_{k}^{FS} =
     *    \begin{cases}
     *        \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
     *        0 & \text{otherwise}.
     *    \end{cases}
     *
     * Being bandlimited, we can use :cpp:class:`FFTW_FFS` and :cpp:func:`ffs_sample`
     * to numerically evaluate :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}`:
     *
     * .. literal_block::
     *
     *    #include <complex>
     *    #include <cmath>
     *    #include <vector>
     *    #include "xtensor/xarray.hpp"
     *    #include "xtensor/xbuilder.hpp"
     *    #include "xtensor/xmath.hpp"
     *    #include "xtensor/xindex_view.hpp"
     *    #include "pypeline/util/math/fourier.hpp"
     *    namespace fourier = pypeline::util::math::fourier;
     *
     *    // Compute samples from a shifted Dirichlet kernel.
     *    xt::xarray<double> dirichlet(xt::xarray<double> x,
     *                                 const double T,
     *                                 const double T_c,
     *                                 const size_t N_FS) {
     *        xt::xarray<double> y {x - T_c};
     *
     *        xt::xarray<double> numerator {xt::zeros<double>({x.size()})};
     *        xt::xarray<double> denominator {xt::zeros<double>({x.size()})};
     *
     *        xt::xarray<bool> nan_mask {xt::isclose(xt::fmod(y, M_PI), 0)};
     *        xt::filter(numerator, ~nan_mask) = xt::sin((N_FS * M_PI / T) * xt::filter(y, ~nan_mask));
     *        xt::filter(denominator, ~nan_mask) = xt::sin((M_PI / T) * xt::filter(y, ~nan_mask));
     *        xt::filter(numerator, nan_mask) = N_FS * xt::cos((N_FS * M_PI / T) * xt::filter(y, nan_mask));
     *        xt::filter(denominator, nan_mask) = xt::cos((M_PI / T) * xt::filter(y, nan_mask));
     *
     *        xt::xarray<double> vals {numerator / denominator};
     *        return vals;
     *    }
     *
     *    // Analytical FS coefficients of a shifted Dirichlet kernel.
     *    xt::xarray<std::complex<double>> dirichlet_FS_theory(const double T,
     *                                                         const double T_c,
     *                                                         const size_t N_FS) {
     *        const size_t N = (N_FS - 1) / 2;
     *        std::complex<double> _1j(0, 1);
     *
     *        std::complex<double> base = exp(-_1j * (2 * M_PI * T_c) / T);
     *        xt::xarray<double> exponent {xt::arange<int>(-N, N+1)};
     *
     *        xt::xarray<std::complex<double>> kernel {xt::pow(base, exponent)};
     *        return kernel;
     *    }
     *
     *    const double T = M_PI;
     *    const double T_c = M_E;
     *    const size_t N_FS = 15;
     *    const size_t N_samples = 16;  // N_samples >= N_FS, but choose highly-
     *                                  // composite for best performance.
     *
     *    const std::vector<size_t> shape_transform {N_samples};
     *    const size_t axis_transform = 0;
     *    const bool inplace = false;
     *    const size_t N_threads = 1;
     *    const fourier::planning_effort effort = fourier::planning_effort::NONE;
     *
     *    xt::xarray<double> sample_points {fourier::ffs_sample(T, N_FS, T_c, N_samples)};  // sample signal in the right order.
     *    xt::xarray<double> diric_samples {dirichlet(sample_points, T, T_c, N_FS)};
     *    xt::xarray<std::complex<double>> diric_FS_exact {dirichlet_FS_theory(T, T_c, N_FS)};
     *
     *    fourier::FFTW_FFS<double> transform(shape_transform,
     *                                        axis_transform,
     *                                        T, T_c, N_FS,
     *                                        inplace, N_threads, effort);
     *    transform.view_in() = diric_samples;  // Copy Dirichlet samples to input buffer.
     *    transform.ffs();                      // Perform Fast-Fourier-Series transform.
     *    xt::xarray<std::complex<double>> diric_FS {transform.view_out()};  // Array indices {0, ..., N_FS-1} along
     *                                                                       // transform axis hold the FS coefficients.
     *
     *    // If you compare `diric_FS_exact` and `diric_FS`, they match perfectly.
     *
     * See Also
     * --------
     * :cpp:func:`ffs_sample`
     */
    template <typename TT>
    class FFTW_FFS {
        private:
            size_t m_axis = 0;
            std::vector<size_t> m_shape {};
            FFTW_FFT<TT> m_transform;
            xt::xarray<std::complex<TT>> m_mod_1;
            xt::xarray<std::complex<TT>> m_mod_2;

            void compute_modulation_vectors(const double T,
                                            const double T_c,
                                            const size_t N_FS) {
                const int N_samples = static_cast<int>(m_shape[m_axis]);
                const int M = N_samples / 2;
                const int N = static_cast<int>(N_FS) / 2;
                std::complex<TT> _1j(0, 1);

                std::complex<TT> B_2 = exp(-_1j * TT(2 * M_PI / N_samples));
                xt::xtensor<TT, 1> E_1 {xt::concatenate(std::make_tuple(
                                             xt::arange<int>(-N, N + 1),
                                             xt::zeros<int>({N_samples - N_FS})))};

                namespace argcheck = pypeline::util::argcheck;
                std::complex<TT> B_1;
                xt::xtensor<TT, 1> E_2;
                if (argcheck::is_odd(N_samples)) {
                    B_1 = exp(_1j * TT((2 * M_PI / T) * T_c));
                    E_2 = xt::concatenate(std::make_tuple(
                              xt::arange<int>(0, M + 1),
                              xt::arange<int>(-M, 0)));
                } else {
                    B_1 = exp(_1j * TT((2 * M_PI / T) *
                                       (T_c + (T / (2 * N_samples)))));
                    E_2 = xt::concatenate(std::make_tuple(
                              xt::arange<int>(0, M),
                              xt::arange<int>(-M, 0)));
                }

                std::vector<size_t> shape_view(m_shape.size());
                std::fill(shape_view.begin(), shape_view.end(), 1);
                shape_view[m_axis] = N_samples;

                m_mod_1 = xt::reshape_view(xt::pow(B_1, -E_1), std::vector<size_t> {shape_view});
                m_mod_2 = xt::reshape_view(xt::pow(B_2, -N * E_2), std::vector<size_t> {shape_view});
            }

        public:
            /*
             * Parameters
             * ----------
             * shape : std::vector<size_t>
             *     Dimensions of the input/output arrays.
             * axis : size_t
             *     Dimension along which function samples are stored.
             * T : double
             *     Function period.
             * T_c : double
             *     Period mid-point.
             * N_FS : int
             *     Function bandwidth.
             * inplace : bool
             *     Perform in-place transforms.
             *     If enabled, only one array will be internally allocated.
             * N_threads : size_t
             *     Number of threads to use.
             * effort : planning_effort
             *
             * Notes
             * -----
             * Input and output buffers are initialized to 0 by default.
             */
            FFTW_FFS(const std::vector<size_t> &shape,
                     const size_t axis,
                     const double T,
                     const double T_c,
                     const size_t N_FS,
                     const bool inplace,
                     const size_t N_threads,
                     const planning_effort effort):
                m_axis(axis), m_shape(shape),
                m_transform(shape, axis, inplace, N_threads, effort) {
                namespace argcheck = pypeline::util::argcheck;
                if (T <= 0) {
                    std::string msg = "Parameter[T] must be positive.";
                    throw std::runtime_error(msg);
                }
                if (!argcheck::is_odd(N_FS)) {
                    std::string msg = "Parameter[N_FS] must be odd-valued.";
                    throw std::runtime_error(msg);
                }
                const size_t N_samples = shape[axis];
                if (!((3 <= N_FS) && (N_FS <= N_samples))) {
                    std::string msg = "Parameter[N_FS] must lie in {3, ..., shape[axis]}.";
                    throw std::runtime_error(msg);
                }

                compute_modulation_vectors(T, T_c, N_FS);
            }

            /*
             * Returns
             * -------
             * data_in : std::complex<TT>*
             *     Pointer to input array.
             */
            std::complex<TT>* data_in() {
                return m_transform.data_in();
            }

            /*
             * Returns
             * -------
             * data_out : std::complex<TT>*
             *     Pointer to output array.
             *     If `inplace` was set to true, then ``data_in() == data_out()``.
             */
            std::complex<TT>* data_out() {
                return m_transform.data_out();
            }

            /*
             * Returns
             * -------
             * shape : std::vector<size_t>
             *     Dimensions of the input buffers.
             */
            std::vector<size_t> shape() {
                return m_shape;
            }

            /*
             * Returns
             * -------
             * view_in : xt::xstrided_view
             *     View on the input array.
             */
            auto view_in() {
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}

                auto view_in = xt::reshape_view(xt::adapt(data_in(), N_cells, xt::no_ownership()),
                                                std::vector<size_t>{m_shape});
                return view_in;
            }

            /*
             * Returns
             * -------
             * view_out : xt::xstrided_view
             *     View on the output array.
             */
            auto view_out() {
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}

                auto view_out = xt::reshape_view(xt::adapt(data_out(), N_cells, xt::no_ownership()),
                                                 std::vector<size_t>{m_shape});
                return view_out;
            }

            /*
             * Transform input buffer using 1D-FFS, result available in output buffer.
             *
             * It is assumed the input buffer contains function samples in the
             * same order specified by :cpp:func:`ffs_sample` along dimension `axis`.
             *
             * The output buffer will contain coefficients
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_samples}`.
             */
            void ffs() {
                view_in().multiplies_assign(m_mod_2);
                m_transform.fft();

                const bool out_of_place = (data_in() != data_out());
                if (out_of_place) {
                    // Undo in-place modulation by `m_mod_2`.
                    view_in().multiplies_assign(xt::conj(m_mod_2));
                }

                const TT N_samples = static_cast<int>(m_shape[m_axis]);
                view_out().multiplies_assign(m_mod_1 / N_samples);
            }

            /*
             * Transform output buffer using 1D-FFS, result available in input buffer.
             *
             * It is assumed the output buffer contains function samples in the
             * same order specified by :cpp:func:`ffs_sample` along dimension `axis`.
             *
             * The input buffer will contain coefficients
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_samples}`.
             */
            void ffs_r() {
                view_out().multiplies_assign(m_mod_2);
                m_transform.fft_r();

                const bool out_of_place = (data_in() != data_out());
                if (out_of_place) {
                    // Undo in-place modulation by `m_mod_2`.
                    view_out().multiplies_assign(xt::conj(m_mod_2));
                }

                const TT N_samples = static_cast<int>(m_shape[m_axis]);
                view_in().multiplies_assign(m_mod_1 / N_samples);
            }

            /*
             * Transform input buffer using 1D-iFFS, result available in output buffer.
             *
             * It is assumed the input buffer contains FS coefficients ordered as
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{samples}}`.
             * along dimension `axis`.
             *
             * The output buffer will contain the original function samples in
             * the same order specified by :cpp:func:`ffs_sample`.
             */
            void iffs() {
                view_in().multiplies_assign(xt::conj(m_mod_1));
                m_transform.ifft();

                const bool out_of_place = (data_in() != data_out());
                if (out_of_place) {
                    // Undo in-place modulataion by `m_mod_1`.
                    view_in().multiplies_assign(m_mod_1);
                }

                const TT N_samples = static_cast<int>(m_shape[m_axis]);
                view_out().multiplies_assign(xt::conj(m_mod_2) * N_samples);
            }

            /*
             * Transform output buffer using 1D-iFFS, result available in input buffer.
             *
             * It is assumed the output buffer contains FS coefficients ordered as
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{samples}}`.
             * along dimension `axis`.
             *
             * The input buffer will contain the original function samples in
             * the same order specified by :cpp:func:`ffs_sample`.
             */
            void iffs_r() {
                view_out().multiplies_assign(xt::conj(m_mod_1));
                m_transform.ifft_r();

                const bool out_of_place = (data_in() != data_out());
                if (out_of_place) {
                    // Undo in-place modulataion by `m_mod_1`.
                    view_out().multiplies_assign(m_mod_1);
                }

                const TT N_samples = static_cast<int>(m_shape[m_axis]);
                view_in().multiplies_assign(xt::conj(m_mod_2) * N_samples);
            }
    };
}}}}

#endif //PYPELINE_UTIL_MATH_FOURIER_HPP
