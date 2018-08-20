// ############################################################################
// _image_pybind11.cpp
// ===================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <memory>

#include "pybind11/pybind11.h"

#include "pypeline/phased_array/util/io/image.hpp"
#include "pypeline/util/cpp_py3_interop.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace image = pypeline::phased_array::util::io::image;

template <typename T>
void SphericalImageContainer_bindings(pybind11::module &m,
                                      const std::string &class_name) {
    auto obj = pybind11::class_<image::SphericalImageContainer<T>>(m,
                                                        class_name.data(),
                                                        R"EOF(
In-memory container for storing real-valued images defined on :math:`\mathbb{S}^{2}`.
This class is not meant to be subclassed: prefer encapsulating an instance in your objects instead.

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.sphere import pol2cart
   from pypeline.phased_array.util.io.image import SphericalImageContainer_float32

.. doctest::

   >>> N_image, N_height, N_width = 2, 6, 7
   >>> image = np.arange(N_image*N_height*N_width, dtype=np.float32).reshape(N_image, N_height, N_width)
   >>> grid =  np.stack(pol2cart(1,
   ...                           np.linspace(0, np.pi, N_height).reshape(-1, 1),
   ...                           np.linspace(0, 2 * np.pi, N_width).reshape(1, -1)),
   ...                  axis=0).astype(np.float32)
   >>> I = SphericalImageContainer_float32(image, grid)

   >>> I.image
   array([[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
           [ 7.,  8.,  9., 10., 11., 12., 13.],
           [14., 15., 16., 17., 18., 19., 20.],
           [21., 22., 23., 24., 25., 26., 27.],
           [28., 29., 30., 31., 32., 33., 34.],
           [35., 36., 37., 38., 39., 40., 41.]],
   <BLANKLINE>
          [[42., 43., 44., 45., 46., 47., 48.],
           [49., 50., 51., 52., 53., 54., 55.],
           [56., 57., 58., 59., 60., 61., 62.],
           [63., 64., 65., 66., 67., 68., 69.],
           [70., 71., 72., 73., 74., 75., 76.],
           [77., 78., 79., 80., 81., 82., 83.]]], dtype=float32)


   >>> np.around(I.grid, 2)
   array([[[ 0.  ,  0.  , -0.  , -0.  , -0.  ,  0.  ,  0.  ],
           [ 0.59,  0.29, -0.29, -0.59, -0.29,  0.29,  0.59],
           [ 0.95,  0.48, -0.48, -0.95, -0.48,  0.48,  0.95],
           [ 0.95,  0.48, -0.48, -0.95, -0.48,  0.48,  0.95],
           [ 0.59,  0.29, -0.29, -0.59, -0.29,  0.29,  0.59],
           [ 0.  ,  0.  , -0.  , -0.  , -0.  ,  0.  ,  0.  ]],
   <BLANKLINE>
          [[ 0.  ,  0.  ,  0.  ,  0.  , -0.  , -0.  , -0.  ],
           [ 0.  ,  0.51,  0.51,  0.  , -0.51, -0.51, -0.  ],
           [ 0.  ,  0.82,  0.82,  0.  , -0.82, -0.82, -0.  ],
           [ 0.  ,  0.82,  0.82,  0.  , -0.82, -0.82, -0.  ],
           [ 0.  ,  0.51,  0.51,  0.  , -0.51, -0.51, -0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  , -0.  , -0.  , -0.  ]],
   <BLANKLINE>
          [[ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
           [ 0.81,  0.81,  0.81,  0.81,  0.81,  0.81,  0.81],
           [ 0.31,  0.31,  0.31,  0.31,  0.31,  0.31,  0.31],
           [-0.31, -0.31, -0.31, -0.31, -0.31, -0.31, -0.31],
           [-0.81, -0.81, -0.81, -0.81, -0.81, -0.81, -0.81],
           [-1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ]]], dtype=float32)
)EOF");

    obj.def(pybind11::init([](pybind11::array_t<T> image,
                              pybind11::array_t<T> grid) {
        const auto& image_view = cpp_py3_interop::numpy_to_xview<T>(image);
        const auto& grid_view = cpp_py3_interop::numpy_to_xview<T>(grid);

        return std::make_unique<image::SphericalImageContainer<T>>(image_view, grid_view);
    }), pybind11::arg("image").noconvert().none(false),
        pybind11::arg("grid").noconvert().none(false),
        pybind11::doc(R"EOF(
__init__(image, grid)

Parameters
----------
image : :py:class:`~numpy.ndarray`
    multi-level (real) data-cube.

    Possible shapes are:

    * (N_height, N_width);
    * (N_image, N_height, N_width);
    * (N_points,);
    * (N_image, N_points).
grid : :py:class:`~numpy.ndarray`
    (3, ...) Cartesian coordinates of the sky on which the data points are defined.

    Possible shapes are:

    * (3, N_height, N_width);
    * (3, N_points).

Notes
-----
It is mandatory for `image` and `grid` to have the same dtype.
)EOF"));

    obj.def_property_readonly("image", [](image::SphericalImageContainer<T> &sic) {
        const auto& image = sic.image();
        return cpp_py3_interop::xtensor_to_numpy(image, false);
    }, pybind11::doc(R"EOF(
Returns
-------
:py:class:`~numpy.ndarray`
    (N_image, ...) data cube.
)EOF"));

    obj.def_property_readonly("grid", [](image::SphericalImageContainer<T> &sic) {
        const auto& grid = sic.grid();
        return cpp_py3_interop::xtensor_to_numpy(grid, false);
    }, pybind11::doc(R"EOF(
Returns
-------
grid : :py:class:`~numpy.ndarray`
    (3, ...) Cartesian coordinates of the sky on which the data points are defined.
)EOF"));

    obj.def_property_readonly("is_gridded", [](image::SphericalImageContainer<T> &sic) {
        return sic.is_gridded();
    }, pybind11::doc(R"EOF(
Returns
-------
is_gridded : bool
    :py:obj:`True` if grid has shape (3, N_height, N_width), :py:obj:`False` otherwise.
)EOF"));
}

PYBIND11_MODULE(_pypeline_phased_array_util_io_image_pybind11, m) {
    pybind11::options options;
    options.disable_function_signatures();

    SphericalImageContainer_bindings<float>(m, "SphericalImageContainer_float32");
    SphericalImageContainer_bindings<double>(m, "SphericalImageContainer_float64");
}
