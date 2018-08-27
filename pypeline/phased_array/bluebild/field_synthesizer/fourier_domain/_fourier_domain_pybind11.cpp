// ############################################################################
// _fourier_domain_pybind11.cpp
// ============================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include "pybind11/pybind11.h"

#include "pypeline/phased_array/bluebild/field_synthesizer/fourier_domain.hpp"
#include "pypeline/util/cpp_py3_interop.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace fourier_domain = pypeline::phased_array::bluebild::field_synthesizer::fourier_domain;

PYBIND11_MODULE(_pypeline_phased_array_bluebild_field_synthesizer_fourier_domain_pybind11, m) {
    pybind11::options options;
    options.disable_function_signatures();

}
