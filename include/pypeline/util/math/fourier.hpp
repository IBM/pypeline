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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fftw3.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xio.hpp"

#include "pypeline/util/argcheck.hpp"
#include "pypeline/util/array.hpp"

namespace pypeline { namespace util { namespace math { namespace fourier {
    enum class planning_effort: unsigned int {
        NONE = FFTW_ESTIMATE,
        MEASURE = FFTW_MEASURE
    };

    /*
     * Find FFTW transform lengths that are speed-optimal.
     *
     * The method used is surely not the fastest, but the 'slow' search speed is
     * inconsequential if only used in startup code.
     *
     * The result is available in <= 3[s] for lengths <= 50000.
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include "pypeline/util/math/fourier.hpp"
     *    namespace fourier = pypeline::util::math::fourier;
     *
     *    const size_t N = 97;
     *    const size_t N_best = fourier::FFTW_size_finder(N).next_fast_len();  // 100
     *
     */
    class FFTW_size_finder {
        private:
            size_t m_N_orig = 0;
            size_t m_N_best = 0;

            bool fast_FFTW_factors_only(std::vector<size_t> base) {
                // It is assumed base was generated from factorize().
                if (xt::all(xt::adapt(base) <= size_t(7))) {
                    return true;
                } else {
                    return false;
                }
            }

            bool is_prime(const size_t N) {
                if (N == 0) {
                    return false;
                } else if (N <= 3) {
                    return true;
                } else {
                    const size_t lim = static_cast<size_t>(sqrt(double(N)) + 1);
                    for (size_t i = 2; i < lim; ++i) {
                        if (N % i == 0) {
                            return false;
                        }
                    }
                    return true;
                }
            }

            size_t next_prime(const size_t N) {
                size_t i = 1;
                while (!is_prime(N + i)) {
                    i += 1;
                }
                return N + i;
            }

            std::tuple<std::vector<size_t>,
                       std::vector<size_t>> factorize(const size_t N) {
                std::vector<size_t> base;
                std::vector<size_t> exponent;

                size_t p = 1;
                while (p < N) {
                    p = next_prime(p);
                    if (N % p == 0) {
                        base.push_back(p);

                        size_t k = 1;
                        while (N % static_cast<size_t>(pow(p, k + 1)) == 0) {
                            k += 1;
                        }
                        exponent.push_back(k);
                    }
                }

                return std::make_tuple(base, exponent);
            }

            size_t find_next_fast_len(const size_t N) {
                std::vector<size_t> base, exponent;
                std::tie(base, exponent) = factorize(N);

                if (!fast_FFTW_factors_only(base)) {
                    std::vector<size_t> base_new, exponent_new;

                    size_t i = 0;
                    do {
                        ++i;
                        std::tie(base_new, exponent_new) = factorize(N + i);
                    } while (!fast_FFTW_factors_only(base_new));

                    base = base_new;
                    exponent = exponent_new;
                }

                // (base, exponent) contain decomposition of best N.
                size_t N_best = 1;
                for (size_t i = 0; i < base.size(); ++i) {
                    auto factor = pow(base[i], exponent[i]);
                     N_best *= static_cast<size_t>(factor);
                }
                return N_best;
            }

        public:
            /*
             * Parameters
             * ----------
             * N : size_t
             *     Length to start searching from.
             */
            FFTW_size_finder(const size_t N):
                m_N_orig(N) {
                if (N < 2) {
                    std::string msg = "Parameter[N] cannot be {0, 1}.";
                    throw std::runtime_error(msg);
                }

                m_N_best = find_next_fast_len(m_N_orig);
            }

            /*
             * Returns
             * -------
             * N_best : size_t
             *     Most efficient transform length >= `N`.
             */
            size_t next_fast_len() {
                return m_N_best;
            }
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

                assert((m_data_in  != nullptr) && "Could not allocate buffer.");
                assert((m_data_out != nullptr) && "Could not allocate buffer.");

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

                assert((m_plan_fft    != nullptr) && "Could not plan transform.");
                assert((m_plan_fft_r  != nullptr) && "Could not plan transform.");
                assert((m_plan_ifft   != nullptr) && "Could not plan transform.");
                assert((m_plan_ifft_r != nullptr) && "Could not plan transform.");
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
                xt::xtensor<std::complex<T>, 1> scale {T(1.0) / static_cast<T>(m_shape[m_axis])};
                view_out().multiplies_assign(scale);
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
                xt::xtensor<std::complex<T>, 1> scale {T(1.0) / static_cast<T>(m_shape[m_axis])};
                view_in().multiplies_assign(scale);
            }

            std::string __repr__() {
                std::stringstream msg;
                msg << "FFTW_FFT<" << ((is_float) ? "float" : "double") << ">("
                    << "shape=" << xt::adapt(m_shape) << ", "
                    << "axis=" << std::to_string(m_axis)
                    << ")";

                return msg.str();
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
            static constexpr bool is_float = std::is_same<TT, float>::value;

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

            std::string __repr__() {
                std::stringstream msg;
                msg << "FFTW_FFS<" << ((is_float) ? "float" : "double") << ">("
                    << "shape=" << xt::adapt(m_shape) << ", "
                    << "axis=" << std::to_string(m_axis)
                    << ")";

                return msg.str();
            }
    };

    /*
     * FFTW wrapper to compute the 1D Chirp Z-Transform on multidimensional tensors.
     *
     * This implementation follows the semantics defined in :ref:`CZT_def`.
     *
     * This object automatically allocates an optimally-sized buffer and
     * provides a tensor interface to the underlying memory using xtensor views.
     *
     * Examples
     * --------
     * Implementation of the DFT:
     *
     * .. literal_block::
     *
     *    #include <cmath>
     *    #include <vector>
     *    #include "pypeline/util/math/fourier.hpp"
     *    namespace fourier = pypeline::util::math::fourier;
     *
     *    // Transform parameters
     *    const size_t N_transform {3}, len_transform {10};
     *    const std::vector<size_t> shape {N_transform, len_transform};
     *    const size_t axis = 1;
     *    const size_t N_threads = 1;
     *    auto effort = fourier::planning_effort::NONE;
     *
     *    // Perform DFT by using the CZT.
     *    std::complex<double> _1j {0, 1};
     *    std::complex<double> A {1, 0};
     *    std::complex<double> W = std::exp(-_1j * (2 * M_PI / len_transform));
     *    fourier::FFTW_CZT<double> czt_transform(shape, axis, A, W, len_transform,
     *                                            N_threads, effort);
     *
     *    czt_transform.view_in() = 1;  // fill buffer
     *    czt_transform.czt();
     *    auto czt_res = czt_transform.view_out();
     *
     *    // Perform DFT using FFT directly.
     *    fourier::FFTW_FFT<double> dft_transform(shape, axis, false, N_threads, effort);
     *    dft_transform.view_in() = 1;
     *    dft_transform.fft();
     *    auto fft_res = dft_transform.view_out();
     *
     *    xt::allclose(fft_res,czt_res);  // true
     */
    template <typename T>
    class FFTW_CZT {
        private:
            static constexpr bool is_float = std::is_same<T, float>::value;

            size_t m_axis = 0;
            std::vector<size_t> m_shape {};
            size_t m_M = 0;
            size_t m_L = 0;
            FFTW_FFT<T> m_transform;
            xt::xarray<std::complex<T>> m_mod_y;
            xt::xarray<std::complex<T>> m_mod_G;
            xt::xarray<std::complex<T>> m_mod_g;

            static size_t _L(const std::vector<size_t> shape,
                             const size_t axis,
                             const size_t M) {
                const size_t N = shape[axis];
                const size_t L = FFTW_size_finder(N + M - 1).next_fast_len();
                return L;
            }

            static std::vector<size_t> _transform_shape(const std::vector<size_t> shape,
                                                        const size_t axis,
                                                        const size_t M) {
                std::vector<size_t> shape_transform = shape;
                shape_transform[axis] = _L(shape, axis, M);
                return shape_transform;
            }

            void compute_modulation_vectors(const std::complex<double> A,
                                            const std::complex<double> W) {
                using cT = std::complex<T>;
                auto n = xt::arange<double>(0, m_L);

                // m_mod_y ====================================================
                const size_t N = m_shape[m_axis];
                std::vector<size_t> shape_mod_y(m_shape.size());
                std::fill(shape_mod_y.begin(), shape_mod_y.end(), 1);
                shape_mod_y[m_axis] = N;

                xt::pow(A, xt::view(-n, xt::range(0, N)));
                auto mod_y = (xt::pow(A, xt::view(-n, xt::range(0, N))) *
                              xt::pow(W, 0.5 * xt::square(xt::view(n, xt::range(0, N)))));
                m_mod_y = xt::reshape_view(xt::cast<cT>(mod_y),
                                           std::vector<size_t> {shape_mod_y});

                // m_mod_G ====================================================
                std::vector<size_t> shape_mod_G(m_shape.size());
                std::fill(shape_mod_G.begin(), shape_mod_G.end(), 1);
                shape_mod_G[m_axis] = m_L;

                xt::xtensor<std::complex<double>, 1> mod_G = xt::zeros<double>({m_L});
                xt::view(mod_G, xt::range(0, m_M)) = xt::pow(W, -0.5 * xt::square(xt::view(n, xt::range(0, m_M))));
                xt::view(mod_G, xt::range(m_L - N + 1, m_L)) = xt::pow(W, -0.5 * xt::square(m_L - xt::view(n, xt::range(m_L - N + 1, m_L))));

                auto transform_mod_G = FFTW_FFT<double>(std::vector<size_t> {m_L}, 0, true, 1, planning_effort::NONE);
                transform_mod_G.view_in() = mod_G;
                transform_mod_G.fft();
                mod_G = transform_mod_G.view_out();

                m_mod_G = xt::reshape_view(xt::cast<cT>(mod_G),
                                           std::vector<size_t> {shape_mod_G});

                // m_mod_g ====================================================
                std::vector<size_t> shape_mod_g(m_shape.size());
                std::fill(shape_mod_g.begin(), shape_mod_g.end(), 1);
                shape_mod_g[m_axis] = m_M;

                auto mod_g = xt::pow(W, 0.5 * xt::square(xt::view(n, xt::range(0, m_M))));
                m_mod_g = xt::reshape_view(xt::cast<cT>(mod_g),
                                           std::vector<size_t> {shape_mod_g});
            }

        public:
            /*
             * Parameters
             * ----------
             * shape : std::vector<size_t>
             *     Dimensions of the input array.
             * axis : size_t
             *     Dimension along which to apply the transform.
             * A : std::complex<double>
             *     Circular offset from the positive real-axis.
             * W : std::complex<double>
             *     Circular spacing between transform points.
             * M : size_t
             *     Length of the transform.
             * N_threads : size_t
             *     Number of threads to use.
             * effort : planning_effort
             *
             * Notes
             * -----
             * Due to numerical instability when using large `M`, this implementation
             * only supports transforms where `A` and `W` have unit norm.
             */
            FFTW_CZT(const std::vector<size_t> &shape,
                     const size_t axis,
                     const std::complex<double> A,
                     const std::complex<double> W,
                     const size_t M,
                     const size_t N_threads,
                     const planning_effort effort):
                m_axis(axis), m_shape(shape),
                m_M(M), m_L(_L(shape, axis, M)),
                m_transform{_transform_shape(shape, axis, M), axis,
                            true, N_threads, effort} {
                if (M == 0) {
                    std::string msg = "Parameter[M] must be positive.";
                    throw std::runtime_error(msg);
                }
                if (!xt::allclose(std::abs(A), 1)) {
                    std::string msg = "Parameter[A] must lie on the unit circle for numerical stability.";
                    throw std::runtime_error(msg);
                }
                if (!xt::allclose(std::abs(W), 1)) {
                    std::string msg = "Parameter[W] must lie on the unit circle for numerical stability.";
                    throw std::runtime_error(msg);
                }

                compute_modulation_vectors(A, W);
            }

            /*
             * Returns
             * -------
             * shape : std::vector<size_t>
             *     Dimensions of the input.
             */
            std::vector<size_t> shape_in() {
                return m_shape;
            }

            /*
             * Returns
             * -------
             * shape : std::vectoro<size_t>
             *     Dimensions of the output.
             */
            std::vector<size_t> shape_out() {
                std::vector<size_t> shape_out = m_shape;
                shape_out[m_axis] = m_M;

                return shape_out;
            }

            /*
             * Returns
             * -------
             * view_in : xt::xstrided_view
             *     View on the input array.
             */
            auto view_in() {
                namespace array = pypeline::util::array;

                const size_t N = m_shape[m_axis];
                const auto& idx = array::index(m_shape.size(), m_axis, xt::range(0, N));
                return xt::strided_view(m_transform.view_in(), idx);
            }

            /*
             * Returns
             * -------
             * view_out : xt::xstrided_view
             *     View on the output array.
             */
            auto view_out() {
                namespace array = pypeline::util::array;

                const auto& idx = array::index(m_shape.size(), m_axis, xt::range(0, m_M));
                return xt::strided_view(m_transform.view_out(), idx);
            }

            /*
             * Transform buffer contents in-place using 1D-CZT.
             *
             * This function is meant to be used as follows:
             * * Use `view_in()` to fill up the buffer.
             * * Call `czt()` to transform buffer contents.
             * * Use `view_out()` to get the result of the transform.
             *
             * Notes
             * -----
             * The contents of `view_in()` are not preserved after calls to `czt()`.
             */
            void czt() {
                namespace array = pypeline::util::array;

                /*
                 * The user must have used `view_in()` to fill the transform buffer.
                 * Since `m_L >= m_N`, we need to zero-out the last `m_L - m_N`
                 * entries along `m_axis` for `czt()` to work.
                 */
                const size_t N = m_shape[m_axis];
                if (m_L - N > 0) {
                    const auto& idx = array::index(m_shape.size(), m_axis, xt::range(N, m_L));
                    xt::strided_view(m_transform.view_in(), idx) = 0;
                }

                view_in().multiplies_assign(m_mod_y);
                m_transform.fft();
                m_transform.view_out().multiplies_assign(m_mod_G);
                m_transform.ifft();
                view_out().multiplies_assign(m_mod_g);
            }

            std::string __repr__() {
                std::stringstream msg;
                msg << "FFTW_CZT<" << ((is_float) ? "float" : "double") << ">("
                    << "shape=" << xt::adapt(m_shape) << ", "
                    << "axis=" << std::to_string(m_axis) << ", "
                    << "M=" << std::to_string(m_M)
                    << ")";

                return msg.str();
            }
    };

    /*
     * Interpolate bandlimited periodic signal as described in :ref:`fp_interp_def`.
     *
     * If given the Fourier Series coefficients of a bandlimited periodic function
     * :math:`x(t): \mathbb{R} \to \mathbb{C}`, then :py:meth:`FFTW_FS_INTERP::fs_interp()`
     * computes the values of :math:`x(t)` at points :math:`t[k] = (a + \frac{b - a}{M - 1} k) 1_{[0,\ldots,M-1]}[k]`.
     *
     * Examples
     * --------
     * Let :math:`\{\phi_{k}^{FS}, k = -N, \ldots, N\}` be the Fourier Series (FS)
     * coefficients of a shifted Dirichlet kernel of period :math:`T`:
     *
     *.. math::
     *
     *   \phi_{k}^{FS} =
     *   \begin{cases}
     *       \exp\left( -j \frac{2 \pi}{T} k T_{c} \right) & -N \le k \le N, \\
     *       0 & \text{otherwise}.
     *   \end{cases}
     *
     * Being bandlimited, we can use :cpp:class:`FFTW_FS_INTERP` to numerically
     * evaluate :math:`\phi(t)` on the interval
     * :math:`\left[ T_{c} - \frac{T}{2}, T_{c} + \frac{T}{2} \right]`.
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
     *    // Signal Parameters
     *    const double T = M_PI;
     *    const double T_c = M_E;
     *    const size_t N_FS = 15;
     *    xt::xarray<std::complex<double>> diric_FS {dirichlet_FS_theory(T, T_c, N_FS)};
     *
     *    // Ground-truth: exact interpolated result.
     *    const double a = T_c - 0.5 * T;
     *    const double b = T_c + 0.5 * T;
     *    const size_t M = 10;  // We want a lot of interpolated points.
     *    xt::xarray<double> sample_positions {a + ((b - a) / (M - 1)) * xt::arange<int>(M)};
     *    xt::xarray<double> diric_sig_exact {dirichlet(sample_positions, T, T_c, N_FS)};
     *
     *    const std::vector<size_t> shape_transform {N_FS};
     *    const size_t axis_transform = 0;
     *    const size_t N_threads = 1;
     *    auto effort = fourier::planning_effort::NONE;
     *
     *    // Option 1
     *    // --------
     *    // No assumptions on FS spectra: use generic algorithm.
     *    fourier::FFTW_FS_INTERP<double> interpolant(shape_transform, axis_transform,
     *                                                T, a, b, M, false,
     *                                                N_threads, effort);
     *    interpolant.in(diric_FS);  // fill input
     *    interpolant.fs_interp();
     *    xt::xarray<std::complex<double>> diric_sig {interpolant.view_out()};
     *
     *    // Option 2
     *    // --------
     *    // You know that the output is real-valued: use accelerated algorithm.
     *    fourier::FFTW_FS_INTERP<double> interpolant_real(shape_transform, axis_transform,
     *                                                     T, a, b, M, true,
     *                                                     N_threads, effort);
     *    interpolant_real.in(diric_FS);  // fill input
     *    interpolant_real.fs_interp();
     *    xt::xarray<std::complex<double>> diric_sig_real {interpolant_real.view_out()};
     *    // .view_out() is always complex-valued, but its imaginary part will be 0.
     *
     *    xt::allclose(diric_sig_exact, diric_sig);       // true
     *    xt::allclose(diric_sig_exact, diric_sig_real);  // true
     */
    template <typename TT>
    class FFTW_FS_INTERP {
        private:
            static constexpr bool is_float = std::is_same<TT, float>::value;

            size_t m_axis = 0;
            std::vector<size_t> m_shape {};
            size_t m_M = 0;
            bool m_real_output = false;
            FFTW_CZT<TT> m_transform;
            xt::xarray<std::complex<TT>> m_mod;
            xt::xarray<std::complex<TT>> m_DC;

            static std::vector<size_t> _transform_shape(const std::vector<size_t> shape,
                                                        const size_t axis,
                                                        const bool real_valued_output) {
                std::vector<size_t> shape_transform = shape;

                if (real_valued_output) {
                    const size_t N_FS = shape[axis];
                    const size_t N = (N_FS - 1) / 2;

                    shape_transform[axis] = N;
                }

                return shape_transform;
            }

            static std::complex<double> _transform_A(const double T,
                                                     const double a) {
                std::complex<double> _1j(0, 1);
                std::complex<double> A = exp(-_1j * ((2 * M_PI * a) / T));
                return A;
            }

            static std::complex<double> _transform_W(const double T,
                                                     const double a,
                                                     const double b,
                                                     const size_t M) {
                std::complex<double> _1j(0, 1);
                std::complex<double> W = exp(_1j * (2 * M_PI / T) * ((b - a) / (M - 1)));
                return W;
            }

            void compute_modulation_vector(const std::complex<double> A,
                                           const std::complex<double> W) {
                auto E = xt::arange<double>(0, m_M);

                xt::xtensor<std::complex<double>, 1> mod;
                if (m_real_output) {
                    mod = 2.0 * xt::pow(W, E) / A;
                } else {
                    const size_t N_FS = m_shape[m_axis];
                    const int N = (N_FS - 1) / 2;

                    mod = xt::pow(W, -N * E) * std::pow<double>(A, N);
                }

                std::vector<size_t> shape_mod(m_shape.size());
                std::fill(shape_mod.begin(), shape_mod.end(), 1);
                shape_mod[m_axis] = m_M;

                using cTT = std::complex<TT>;
                m_mod = xt::reshape_view(xt::cast<cTT>(mod),
                                         std::vector<size_t> {shape_mod});
            }

        public:
            /*
             * Parameters
             * ----------
             * shape : std::vector<size_t>
             *     Dimensions of the input array.
             * axis : size_t
             *     Dimension along which the FS coefficients are stored.
             * T : double
             *     Function period.
             * a : double
             *     Interval LHS.
             * b : double
             *     Interval RHS.
             * M : size_t
             *     Number of points to interpolate.
             * real_valued_output : bool
             *     If true, it is assumed the interpolated signal is real-valued.
             *     In this context, only the FS coefficients corresponding to
             *     non-negative frequencies will be used, along with a more
             *     efficient interpolation algorithm.
             * N_threads : size_t
             *     Number of threads to use.
             * effort : planning_effort
             */
            FFTW_FS_INTERP(const std::vector<size_t> &shape,
                           const size_t axis,
                           const double T,
                           const double a,
                           const double b,
                           const size_t M,
                           const bool real_valued_output,
                           const size_t N_threads,
                           const planning_effort effort):
                m_axis(axis), m_shape(shape), m_M(M), m_real_output(real_valued_output),
                m_transform{_transform_shape(shape, axis, real_valued_output), axis,
                            _transform_A(T, a), _transform_W(T, a, b, M), M,
                            N_threads, effort} {
                namespace argcheck = pypeline::util::argcheck;
                if (argcheck::is_even(shape[axis])) {
                    std::string msg = "Parameter[shape] must be odd-valued along Parameter[axis].";
                    throw std::runtime_error(msg);
                }
                if (T <= 0) {
                    std::string msg = "Parameter[T] must be positive.";
                    throw std::runtime_error(msg);
                }
                if (b <= a) {
                    std::string msg = "Parameter[a] must be smaller than Parameter[b].";
                    throw std::runtime_error(msg);
                }
                if (M == 0) {
                    std::string msg = "Parameter[M] must be positive.";
                    throw std::runtime_error(msg);
                }

                compute_modulation_vector(_transform_A(T, a),
                                          _transform_W(T, a, b, M));
            }

            /*
             * Returns
             * -------
             * shape : std::vector<size_t>
             *     Dimensions of the input.
             */
            std::vector<size_t> shape_in() {
                return m_shape;
            }

            /*
             * Returns
             * -------
             * shape : std::vector<size_t>
             *     Dimensions of the output.
             */
            std::vector<size_t> shape_out() {
                std::vector<size_t> shape_out = m_shape;
                shape_out[m_axis] = m_M;

                return shape_out;
            }

            /*
             * Fill input array.
             *
             * Parameters
             * ----------
             * x : xt::xexpression
             *     (..., N_FS, ...) FS coefficients in the order
             *     :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}\right]`.
             *
             * Notes
             * -----
             * If `real_valued_output` was set to `true`, only the FS coefficients
             * corresponding to non-negative frequencies are stored.
             */
            template <typename E>
            void in(E &&x) {
                namespace argcheck = pypeline::util::argcheck;
                if (!(argcheck::has_floats(x) || argcheck::has_complex(x))) {
                    std::string msg = "Parameter[x] must be real/complex-valued.";
                    throw std::runtime_error(msg);
                }

                auto shape_x = x.shape();
                std::string shape_error_msg = "Parameter[x] must have shape (..., N_FS, ...).";
                if (shape_x.size() != m_shape.size()) {
                    throw std::runtime_error(shape_error_msg);
                }
                for (size_t i = 0; i < shape_x.size(); ++i) {
                    if (static_cast<size_t>(shape_x[i]) !=
                        static_cast<size_t>(m_shape[i])) {
                        throw std::runtime_error(shape_error_msg);
                    }
                }

                if (m_real_output) {
                    // Store DC + POSitive terms only.
                    namespace array = pypeline::util::array;
                    const size_t N_FS = m_shape[m_axis];
                    const size_t N = (N_FS - 1) / 2;

                    const auto& idx_DC = array::index(m_shape.size(), m_axis, xt::range(N, N + 1));
                    m_DC = xt::strided_view(x, idx_DC);

                    const auto& idx_POS = array::index(m_shape.size(), m_axis, xt::range(N + 1, N_FS));
                    m_transform.view_in() = xt::strided_view(x, idx_POS);
                } else {
                    m_transform.view_in() = x;
                }
            }

            /*
             * Returns
             * -------
             * view_out : xt::xstrided_view
             *     (..., M, ...) interpolated values :math:`\left[ x(t[0]), \ldots, x(t[M-1]) \right]` along the axis indicated by `axis`.
             *     If `real_valued_output` is `true`, the output's imaginary part
             *     is guaranteed to be 0.
             */
            auto view_out() {
                return m_transform.view_out();
            }

            /*
             * Interpolate bandlimited periodic signal.
             *
             * This function is meant to be used as follows:
             * * Use `in()` to fill up the buffer with FS coefficients.
             * * Call `fs_interp()` to obtain interpolated signal samples.
             * * Use `view_out()` to get the signal samples.
             */
            void fs_interp() {
                m_transform.czt();
                m_transform.view_out().multiplies_assign(m_mod);

                if (m_real_output) {
                    m_transform.view_out().plus_assign(m_DC);
                    xt::imag(m_transform.view_out()) = 0;
                }
            }

            std::string __repr__() {
                std::stringstream msg;
                msg << "FFTW_FS_INTERP<" << ((is_float) ? "float" : "double") << ">("
                    << "shape=" << xt::adapt(m_shape) << ", "
                    << "axis=" << std::to_string(m_axis) << ", "
                    << "M=" << std::to_string(m_M)
                    << ")";

                return msg.str();
            }
    };
}}}}

#endif //PYPELINE_UTIL_MATH_FOURIER_HPP
