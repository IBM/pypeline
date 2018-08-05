// ############################################################################
// _sphere_pybind11.cpp
// ====================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include "pybind11/pybind11.h"
#include "xtensor/xtensor.hpp"

#include "pypeline/util/cpp_py3_interop.hpp"
#include "pypeline/util/math/sphere.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace sphere = pypeline::util::math::sphere;

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> py_pol2cart(pybind11::array_t<T> r,
                                             pybind11::array_t<T> colat,
                                             pybind11::array_t<T> lon) {
        auto r_view = cpp_py3_interop::numpy_to_xview<T>(r);
        auto colat_view = cpp_py3_interop::numpy_to_xview<T>(colat);
        auto lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        const auto& [x, y, z] = sphere::pol2cart(r_view, colat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(x),
                                   cpp_py3_interop::xtensor_to_numpy(y),
                                   cpp_py3_interop::xtensor_to_numpy(z));
        return xyz;
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> py_pol2cart(double r,
                                             pybind11::array_t<T> colat,
                                             pybind11::array_t<T> lon) {
        xt::xtensor<double, 1> _r {r};
        auto colat_view = cpp_py3_interop::numpy_to_xview<T>(colat);
        auto lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        const auto& [x, y, z] = sphere::pol2cart(_r, colat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(x),
                                   cpp_py3_interop::xtensor_to_numpy(y),
                                   cpp_py3_interop::xtensor_to_numpy(z));
        return xyz;
}

template <typename T>
std::tuple<double,
           double,
           double> py_pol2cart(double r,
                               double colat,
                               double lon) {
        xt::xtensor<double, 1> _r {r};
        xt::xtensor<double, 1> _colat {colat};
        xt::xtensor<double, 1> _lon {lon};

        const auto& [x, y, z] = sphere::pol2cart(_r, _colat, _lon);
        auto xyz = std::make_tuple(x(0), y(0), z(0));
        return xyz;
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> py_eq2cart(pybind11::array_t<T> r,
                                            pybind11::array_t<T> lat,
                                            pybind11::array_t<T> lon) {
        auto r_view = cpp_py3_interop::numpy_to_xview<T>(r);
        auto lat_view = cpp_py3_interop::numpy_to_xview<T>(lat);
        auto lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        const auto& [x, y, z] = sphere::eq2cart(r_view, lat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(x),
                                   cpp_py3_interop::xtensor_to_numpy(y),
                                   cpp_py3_interop::xtensor_to_numpy(z));
        return xyz;
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> py_eq2cart(double r,
                                            pybind11::array_t<T> lat,
                                            pybind11::array_t<T> lon) {
        xt::xtensor<double, 1> _r {r};
        auto lat_view = cpp_py3_interop::numpy_to_xview<T>(lat);
        auto lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        const auto& [x, y, z] = sphere::eq2cart(_r, lat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(x),
                                   cpp_py3_interop::xtensor_to_numpy(y),
                                   cpp_py3_interop::xtensor_to_numpy(z));
        return xyz;
}

template <typename T>
std::tuple<double,
           double,
           double> py_eq2cart(double r,
                              double lat,
                              double lon) {
        xt::xtensor<double, 1> _r {r};
        xt::xtensor<double, 1> _lat {lat};
        xt::xtensor<double, 1> _lon {lon};

        const auto& [x, y, z] = sphere::eq2cart(_r, _lat, _lon);
        auto xyz = std::make_tuple(x(0), y(0), z(0));
        return xyz;
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> py_cart2pol(pybind11::array_t<T> x,
                                             pybind11::array_t<T> y,
                                             pybind11::array_t<T> z) {
    auto x_view = cpp_py3_interop::numpy_to_xview<T>(x);
    auto y_view = cpp_py3_interop::numpy_to_xview<T>(y);
    auto z_view = cpp_py3_interop::numpy_to_xview<T>(z);

    const auto& [r, colat, lon] = sphere::cart2pol(x_view, y_view, z_view);
    auto pol = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(r),
                               cpp_py3_interop::xtensor_to_numpy(colat),
                               cpp_py3_interop::xtensor_to_numpy(lon));
    return pol;
}

template <typename T>
std::tuple<double,
           double,
           double> py_cart2pol(double x,
                               double y,
                               double z) {
    xt::xtensor<double, 1> _x {x};
    xt::xtensor<double, 1> _y {y};
    xt::xtensor<double, 1> _z {z};

    const auto& [r, colat, lon] = sphere::cart2pol(_x, _y, _z);
    auto pol = std::make_tuple(r(0), colat(0), lon(0));
    return pol;
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> py_cart2eq(pybind11::array_t<T> x,
                                            pybind11::array_t<T> y,
                                            pybind11::array_t<T> z) {
    auto x_view = cpp_py3_interop::numpy_to_xview<T>(x);
    auto y_view = cpp_py3_interop::numpy_to_xview<T>(y);
    auto z_view = cpp_py3_interop::numpy_to_xview<T>(z);

    const auto& [r, lat, lon] = sphere::cart2eq(x_view, y_view, z_view);
    auto eq = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(r),
                              cpp_py3_interop::xtensor_to_numpy(lat),
                              cpp_py3_interop::xtensor_to_numpy(lon));
    return eq;
}

template <typename T>
std::tuple<double,
           double,
           double> py_cart2eq(double x,
                              double y,
                              double z) {
    xt::xtensor<double, 1> _x {x};
    xt::xtensor<double, 1> _y {y};
    xt::xtensor<double, 1> _z {z};

    const auto& [r, lat, lon] = sphere::cart2eq(_x, _y, _z);
    auto eq = std::make_tuple(r(0), lat(0), lon(0));
    return eq;
}

template <typename T>
pybind11::array_t<T> py_colat2lat(pybind11::array_t<T> colat) {
    auto colat_view = cpp_py3_interop::numpy_to_xview<T>(colat);

    auto lat = sphere::colat2lat(colat_view);
    return cpp_py3_interop::xtensor_to_numpy(lat);
}

template <typename T>
double py_colat2lat(double colat) {
    xt::xtensor<double, 1> _colat {colat};

    auto lat = sphere::colat2lat(_colat);
    return lat(0);
}

template <typename T>
pybind11::array_t<T> py_lat2colat(pybind11::array_t<T> lat) {
    auto lat_view = cpp_py3_interop::numpy_to_xview<T>(lat);

    auto colat = sphere::lat2colat(lat_view);
    return cpp_py3_interop::xtensor_to_numpy(colat);
}

template <typename T>
double py_lat2colat(double lat) {
    xt::xtensor<double, 1> _lat {lat};

    auto colat = sphere::lat2colat(_lat);
    return colat(0);
}

PYBIND11_MODULE(_pypeline_util_math_sphere_pybind11, m) {
    m.def("pol2cart",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&py_pol2cart<float>),
          pybind11::arg("r").noconvert(),
          pybind11::arg("colat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("pol2cart",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&py_pol2cart<double>),
          pybind11::arg("r").noconvert(),
          pybind11::arg("colat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("pol2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&py_pol2cart<float>),
          pybind11::arg("r"),
          pybind11::arg("colat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("pol2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&py_pol2cart<double>),
          pybind11::arg("r"),
          pybind11::arg("colat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("pol2cart",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&py_pol2cart<double>),
          pybind11::arg("r"),
          pybind11::arg("colat"),
          pybind11::arg("lon"),
          pybind11::doc(R"EOF(
Polar coordinates to Cartesian coordinates.

Parameters
----------
r : float or :py:class:`~numpy.ndarray`
    Radius.
colat : float or :py:class:`~numpy.ndarray`
    Polar/Zenith angle [rad].
lon : float or :py:class:`~numpy.ndarray`
    Longitude angle [rad].

Returns
-------
X : float or :py:class:`~numpy.ndarray`
Y : float or :py:class:`~numpy.ndarray`
Z : float or :py:class:`~numpy.ndarray`

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.sphere import pol2cart

.. doctest::

   >>> x, y, z = pol2cart(1, 0, 0)
   >>> x, y, z
   (0.0, 0.0, 1.0)
)EOF"));

    m.def("eq2cart",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&py_eq2cart<float>),
          pybind11::arg("r").noconvert(),
          pybind11::arg("lat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("eq2cart",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&py_eq2cart<double>),
          pybind11::arg("r").noconvert(),
          pybind11::arg("lat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("eq2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&py_eq2cart<float>),
          pybind11::arg("r"),
          pybind11::arg("lat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("eq2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&py_eq2cart<double>),
          pybind11::arg("r"),
          pybind11::arg("lat").noconvert(),
          pybind11::arg("lon").noconvert());
    m.def("eq2cart",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&py_eq2cart<double>),
          pybind11::arg("r"),
          pybind11::arg("lat"),
          pybind11::arg("lon"),
          pybind11::doc(R"EOF(
Equatorial coordinates to Cartesian coordinates.

Parameters
----------
r : float or :py:class:`~numpy.ndarray`
    Radius.
lat : float or :py:class:`~numpy.ndarray`
    Latitude angle [rad].
lon : float or :py:class:`~numpy.ndarray`
    Longitude angle [rad].

Returns
-------
X : float or :py:class:`~numpy.ndarray`
Y : float or :py:class:`~numpy.ndarray`
Z : float or :py:class:`~numpy.ndarray`

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.sphere import eq2cart

.. doctest::

   >>> x, y, z = eq2cart(1, 0, 0)
   >>> x, y, z
   (1.0, 0.0, 0.0)
)EOF"));

    m.def("cart2pol",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&py_cart2pol<float>),
          pybind11::arg("x").noconvert(),
          pybind11::arg("y").noconvert(),
          pybind11::arg("z").noconvert());
    m.def("cart2pol",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&py_cart2pol<double>),
          pybind11::arg("x").noconvert(),
          pybind11::arg("y").noconvert(),
          pybind11::arg("z").noconvert());
    m.def("cart2pol",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&py_cart2pol<double>),
          pybind11::arg("x"),
          pybind11::arg("y"),
          pybind11::arg("z"),
          pybind11::doc(R"EOF(
Cartesian coordinates to Polar coordinates.

Parameters
----------
x : float or :py:class:`~numpy.ndarray`
y : float or :py:class:`~numpy.ndarray`
z : float or :py:class:`~numpy.ndarray`

Returns
-------
r : float or :py:class:`~numpy.ndarray`
    Radius.
colat : float or :py:class:`~numpy.ndarray`
    Polar/Zenith angle [rad].
lon : float or :py:class:`~numpy.ndarray`
    Longitude angle [rad].

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.sphere import cart2pol

.. doctest::

   >>> r, colat, lon = cart2pol(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

   >>> np.around([r, colat, lon], 2)
   array([1.  , 0.79, 1.57])
)EOF"));

    m.def("cart2eq",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&py_cart2eq<float>),
          pybind11::arg("x").noconvert(),
          pybind11::arg("y").noconvert(),
          pybind11::arg("z").noconvert());
    m.def("cart2eq",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&py_cart2eq<double>),
          pybind11::arg("x").noconvert(),
          pybind11::arg("y").noconvert(),
          pybind11::arg("z").noconvert());
    m.def("cart2eq",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&py_cart2eq<double>),
          pybind11::arg("x"),
          pybind11::arg("y"),
          pybind11::arg("z"),
          pybind11::doc(R"EOF(
Cartesian coordinates to Equatorial coordinates.

Parameters
----------
x : float or :py:class:`~numpy.ndarray`
y : float or :py:class:`~numpy.ndarray`
z : float or :py:class:`~numpy.ndarray`

Returns
-------
r : float or :py:class:`~numpy.ndarray`
    Radius.
lat : float or :py:class:`~numpy.ndarray`
    Latitude angle [rad].
lon : float or :py:class:`~numpy.ndarray`
    Longitude angle [rad].

Examples
--------
.. testsetup::

   import numpy as np
   from pypeline.util.math.sphere import cart2eq

.. doctest::

   >>> r, lat, lon = cart2eq(0, 1 / np.sqrt(2), 1 / np.sqrt(2))

   >>> np.around([r, lat, lon], 2)
   array([1.  , 0.79, 1.57])
)EOF"));

    m.def("colat2lat",
          pybind11::overload_cast<pybind11::array_t<float>>(&py_colat2lat<float>),
          pybind11::arg("colat").noconvert());
    m.def("colat2lat",
          pybind11::overload_cast<pybind11::array_t<double>>(&py_colat2lat<double>),
          pybind11::arg("colat").noconvert());
    m.def("colat2lat",
          pybind11::overload_cast<double>(&py_colat2lat<double>),
          pybind11::arg("colat"),
          pybind11::doc(R"EOF(
Co-latitude to latitude.

Parameters
----------
colat : float or :py:class:`~numpy.ndarray`
    Polar/Zenith angle [rad].

Returns
-------
lat : float or :py:class:`~numpy.ndarray`
    Latitude angle [rad].
)EOF"));

    m.def("lat2colat",
          pybind11::overload_cast<pybind11::array_t<float>>(&py_lat2colat<float>),
          pybind11::arg("lat").noconvert());
    m.def("lat2colat",
          pybind11::overload_cast<pybind11::array_t<double>>(&py_lat2colat<double>),
          pybind11::arg("lat").noconvert());
    m.def("lat2colat",
          pybind11::overload_cast<double>(&py_lat2colat<double>),
          pybind11::arg("lat"),
          pybind11::doc(R"EOF(
Latitude to co-latitude.

Parameters
----------
lat : float or :py:class:`~numpy.ndarray`
    Latitude angle [rad].

Returns
-------
colat : float or :py:class:`~numpy.ndarray`
    Polar/Zenith angle [rad].
)EOF"));
}
