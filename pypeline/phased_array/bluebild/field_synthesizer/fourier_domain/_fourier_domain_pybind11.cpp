// ############################################################################
// _fourier_domain_pybind11.cpp
// ============================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <complex>

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

#include "pypeline/types.hpp"
#include "pypeline/phased_array/bluebild/field_synthesizer/fourier_domain.hpp"
#include "pypeline/util/cpp_py3_interop.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace f_synth = pypeline::phased_array::bluebild::field_synthesizer::fourier_domain;

template <typename TT>
void FourierFieldSynthesizerBlock_bindings(pybind11::module &m,
                                           const std::string &class_name) {
    using cTT = std::complex<TT>;

    auto obj = pybind11::class_<f_synth::FourierFieldSynthesizerBlock<TT>>(m,
                                                                           class_name.data(),
                                                                           R"EOF()EOF");

    obj.def(pybind11::init([](const double wl,
                              pybind11::array_t<TT> grid_colat,
                              pybind11::array_t<TT> grid_lon,
                              const int N_FS,
                              const double T,
                              pybind11::array_t<TT> R,
                              const int N_eig,
                              const int N_antenna,
                              const int N_threads,
                              fourier::planning_effort effort) {
        const auto& colat_view = cpp_py3_interop::numpy_to_xview<TT>(grid_colat);
        const auto& lon_view = cpp_py3_interop::numpy_to_xview<TT>(grid_lon);
        const auto& R_view = cpp_py3_interop::numpy_to_xview<TT>(R);

        if (N_FS < 0) {
            std::string msg = "Parameter[N_FS] must be positive.";
        }

        if (N_eig < 0) {
            std::string msg = "Parameter[N_eig] must be positive.";
        }

        if (N_antenna < 0) {
            std::string msg = "Parameter[N_antenna] must be positive.";
        }

        if (N_threads < 0) {
            std::string msg = "Parameter[N_threads] must be positive.";
            throw std::runtime_error(msg);
        }

        return std::make_unique<f_synth::FourierFieldSynthesizerBlock<TT>>(
                  wl,
                  colat_view, lon_view,
                  N_FS, T, R_view,
                  N_eig, N_antenna,
                  N_threads, effort);
    }), pybind11::arg("wl").none(false),
        pybind11::arg("grid_colat").none(false),
        pybind11::arg("grid_lon").none(false),
        pybind11::arg("N_FS").none(false),
        pybind11::arg("T").none(false),
        pybind11::arg("R").none(false),
        pybind11::arg("N_eig").none(false),
        pybind11::arg("N_antenna").none(false),
        pybind11::arg("N_threads").none(false),
        pybind11::arg("effort").none(false),
        pybind11::doc(R"EOF(
__init__(wl, grid_colat, grid_lon, N_FS, T, R, N_eig, N_antenna, N_threads, effort)

Parameters
----------
wl : float
    Wave-length [m] of observations.
grid_colat : :py:class:`~numpy.ndarray`
    (N_height, 1) BFSF polar angles [rad].
grid_lon : :py:class:`~numpy.ndarray`
    (1, N_width) equi-spaced BFSF azimuthal angles [rad].
N_FS : int
    :math:`2\pi`-periodic kernel bandwidth. (odd-valued)
T : float
    Kernel periodicity [rad] to use for imaging.
R : :py:class:`~numpy.ndarray`
    (3, 3) ICRS -> BFSF rotation matrix.
N_eig : int
    Number of eigenfunctions to output.
N_antenna : int
    Number of antennas received at each time instant.
N_threads : int
    Number of threads to use.
effort : :py:class:`~pypeline.util.math.fourier.planning_effort`
    Amount of time spent finding best transform.

Notes
-----
* `grid_colat` and `grid_lon` should be generated using :py:func:`~pypeline.phased_array.util.grid.ea_grid` or :py:func:`~pypeline.phased_array.util.grid.ea_harmonic_grid`.
* `N_FS` can be optimally chosen by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.bfsf_kernel_bandwidth`.
* `R` can be obtained by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.icrs2bfsf_rot`.
)EOF"));

    obj.def("__call__", [](f_synth::FourierFieldSynthesizerBlock<TT> &field_synth,
                           pybind11::array_t<cTT> V,
                           pybind11::array_t<TT> XYZ,
                           pybind11::array_t<cTT> W) {
        xt::xtensor<cTT, 2> cpp_V {cpp_py3_interop::numpy_to_xview<cTT>(V)};
        xt::xtensor<TT, 2> cpp_XYZ {cpp_py3_interop::numpy_to_xview<TT>(XYZ)};
        xt::xtensor<cTT, 2> cpp_W {cpp_py3_interop::numpy_to_xview<cTT>(W)};

        const auto& stat = field_synth(cpp_V, cpp_XYZ, cpp_W);
        return cpp_py3_interop::xtensor_to_numpy(std::move(stat));
    }, pybind11::arg("V").none(false),
       pybind11::arg("XYZ").none(false),
       pybind11::arg("W").none(false),
       pybind11::doc("EOF()EOF"));

    obj.def("__call__", [](f_synth::FourierFieldSynthesizerBlock<TT> &field_synth,
                           pybind11::array_t<cTT> V,
                           pybind11::array_t<TT> XYZ,
                           SpMatrixXX_t<cTT> W) {
        xt::xtensor<cTT, 2> cpp_V {cpp_py3_interop::numpy_to_xview<cTT>(V)};
        xt::xtensor<TT, 2> cpp_XYZ {cpp_py3_interop::numpy_to_xview<TT>(XYZ)};

        const auto& stat = field_synth(cpp_V, cpp_XYZ, W);
        return cpp_py3_interop::xtensor_to_numpy(std::move(stat));
    }, pybind11::arg("V").none(false),
       pybind11::arg("XYZ").none(false),
       pybind11::arg("W").none(false),
       pybind11::doc("EOF()EOF"));

    obj.def("synthesize", [](f_synth::FourierFieldSynthesizerBlock<TT> &field_synth,
                             pybind11::array_t<TT> stat) {
        const auto& stat_view = cpp_py3_interop::numpy_to_xview<TT>(stat);

        const auto& field = field_synth.synthesize(stat_view);
        return cpp_py3_interop::xtensor_to_numpy(std::move(field));
    }, pybind11::arg("stat").noconvert().none(false),
       pybind11::doc("EOF()EOF"));
}

PYBIND11_MODULE(_pypeline_phased_array_bluebild_field_synthesizer_fourier_domain_pybind11, m) {
    pybind11::options options;
    options.disable_function_signatures();

    FourierFieldSynthesizerBlock_bindings<double>(m, "FourierFieldSynthesizerBlock_c128");
}
