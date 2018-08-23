// ############################################################################
// _sphere_pybind11.cpp
// ====================
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <tuple>
#include <utility>

#include "pybind11/pybind11.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

#include "pypeline/util/cpp_py3_interop.hpp"
#include "pypeline/util/math/sphere.hpp"

namespace cpp_py3_interop = pypeline::util::cpp_py3_interop;
namespace sphere = pypeline::util::math::sphere;

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> _pol2cart(pybind11::array_t<T> r,
                                           pybind11::array_t<T> colat,
                                           pybind11::array_t<T> lon) {
        const auto& r_view = cpp_py3_interop::numpy_to_xview<T>(r);
        const auto& colat_view = cpp_py3_interop::numpy_to_xview<T>(colat);
        const auto& lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        xt::xarray<double> x, y, z;
        std::tie(x, y, z) = sphere::pol2cart(r_view, colat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(std::move(x)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(y)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(z)));
        return xyz;
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> _pol2cart(double r,
                                           pybind11::array_t<T> colat,
                                           pybind11::array_t<T> lon) {
        xt::xtensor<double, 1> _r {r};
        const auto& colat_view = cpp_py3_interop::numpy_to_xview<T>(colat);
        const auto& lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        xt::xarray<double> x, y, z;
        std::tie(x, y, z) = sphere::pol2cart(_r, colat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(std::move(x)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(y)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(z)));
        return xyz;
}

template <typename T>
std::tuple<double,
           double,
           double> _pol2cart(double r,
                             double colat,
                             double lon) {
        xt::xtensor<double, 1> _r {r};
        xt::xtensor<double, 1> _colat {colat};
        xt::xtensor<double, 1> _lon {lon};

        xt::xarray<double> x, y, z;
        std::tie(x, y, z) = sphere::pol2cart(_r, _colat, _lon);
        auto xyz = std::make_tuple(x(0), y(0), z(0));
        return xyz;
}

void pol2cart_bindings(pybind11::module &m) {
   m.def("pol2cart",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&_pol2cart<float>),
          pybind11::arg("r").noconvert().none(false),
          pybind11::arg("colat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("pol2cart",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&_pol2cart<double>),
          pybind11::arg("r").noconvert().none(false),
          pybind11::arg("colat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("pol2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&_pol2cart<float>),
          pybind11::arg("r").none(false),
          pybind11::arg("colat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("pol2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&_pol2cart<double>),
          pybind11::arg("r").none(false),
          pybind11::arg("colat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("pol2cart",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&_pol2cart<double>),
          pybind11::arg("r").none(false),
          pybind11::arg("colat").none(false),
          pybind11::arg("lon").none(false),
          pybind11::doc(R"EOF(
pol2cart(r, colat, lon)

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
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> _eq2cart(pybind11::array_t<T> r,
                                          pybind11::array_t<T> lat,
                                          pybind11::array_t<T> lon) {
        const auto& r_view = cpp_py3_interop::numpy_to_xview<T>(r);
        const auto& lat_view = cpp_py3_interop::numpy_to_xview<T>(lat);
        const auto& lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        xt::xarray<double> x, y, z;
        std::tie(x, y, z) = sphere::eq2cart(r_view, lat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(std::move(x)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(y)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(z)));
        return xyz;
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> _eq2cart(double r,
                                          pybind11::array_t<T> lat,
                                          pybind11::array_t<T> lon) {
        xt::xtensor<double, 1> _r {r};
        const auto& lat_view = cpp_py3_interop::numpy_to_xview<T>(lat);
        const auto& lon_view = cpp_py3_interop::numpy_to_xview<T>(lon);

        xt::xarray<double> x, y, z;
        std::tie(x, y, z) = sphere::eq2cart(_r, lat_view, lon_view);
        auto xyz = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(std::move(x)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(y)),
                                   cpp_py3_interop::xtensor_to_numpy(std::move(z)));
        return xyz;
}

template <typename T>
std::tuple<double,
           double,
           double> _eq2cart(double r,
                            double lat,
                            double lon) {
        xt::xtensor<double, 1> _r {r};
        xt::xtensor<double, 1> _lat {lat};
        xt::xtensor<double, 1> _lon {lon};

        xt::xarray<double> x, y, z;
        std::tie(x, y, z) = sphere::eq2cart(_r, _lat, _lon);
        auto xyz = std::make_tuple(x(0), y(0), z(0));
        return xyz;
}

void eq2cart_bindings(pybind11::module &m) {
    m.def("eq2cart",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&_eq2cart<float>),
          pybind11::arg("r").noconvert().none(false),
          pybind11::arg("lat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("eq2cart",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&_eq2cart<double>),
          pybind11::arg("r").noconvert().none(false),
          pybind11::arg("lat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("eq2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&_eq2cart<float>),
          pybind11::arg("r").none(false),
          pybind11::arg("lat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("eq2cart",
          pybind11::overload_cast<double,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&_eq2cart<double>),
          pybind11::arg("r").none(false),
          pybind11::arg("lat").noconvert().none(false),
          pybind11::arg("lon").noconvert().none(false));

    m.def("eq2cart",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&_eq2cart<double>),
          pybind11::arg("r").none(false),
          pybind11::arg("lat").none(false),
          pybind11::arg("lon").none(false),
          pybind11::doc(R"EOF(
eq2cart(r, lat, lon)

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
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> _cart2pol(pybind11::array_t<T> x,
                                           pybind11::array_t<T> y,
                                           pybind11::array_t<T> z) {
    const auto& x_view = cpp_py3_interop::numpy_to_xview<T>(x);
    const auto& y_view = cpp_py3_interop::numpy_to_xview<T>(y);
    const auto& z_view = cpp_py3_interop::numpy_to_xview<T>(z);

    xt::xarray<double> r, colat, lon;
    std::tie(r, colat, lon) = sphere::cart2pol(x_view, y_view, z_view);
    auto pol = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(std::move(r)),
                               cpp_py3_interop::xtensor_to_numpy(std::move(colat)),
                               cpp_py3_interop::xtensor_to_numpy(std::move(lon)));
    return pol;
}

template <typename T>
std::tuple<double,
           double,
           double> _cart2pol(double x,
                             double y,
                             double z) {
    xt::xtensor<double, 1> _x {x};
    xt::xtensor<double, 1> _y {y};
    xt::xtensor<double, 1> _z {z};

    xt::xarray<double> r, colat, lon;
    std::tie(r, colat, lon) = sphere::cart2pol(_x, _y, _z);
    auto pol = std::make_tuple(r(0), colat(0), lon(0));
    return pol;
}

void cart2pol_bindings(pybind11::module &m) {
    m.def("cart2pol",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&_cart2pol<float>),
          pybind11::arg("x").noconvert().none(false),
          pybind11::arg("y").noconvert().none(false),
          pybind11::arg("z").noconvert().none(false));

    m.def("cart2pol",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&_cart2pol<double>),
          pybind11::arg("x").noconvert().none(false),
          pybind11::arg("y").noconvert().none(false),
          pybind11::arg("z").noconvert().none(false));

    m.def("cart2pol",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&_cart2pol<double>),
          pybind11::arg("x").none(false),
          pybind11::arg("y").none(false),
          pybind11::arg("z").none(false),
          pybind11::doc(R"EOF(
cart2pol(x, y, z)

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
}

template <typename T>
std::tuple<pybind11::array_t<T>,
           pybind11::array_t<T>,
           pybind11::array_t<T>> _cart2eq(pybind11::array_t<T> x,
                                          pybind11::array_t<T> y,
                                          pybind11::array_t<T> z) {
    const auto& x_view = cpp_py3_interop::numpy_to_xview<T>(x);
    const auto& y_view = cpp_py3_interop::numpy_to_xview<T>(y);
    const auto& z_view = cpp_py3_interop::numpy_to_xview<T>(z);

    xt::xarray<double> r, lat, lon;
    std::tie(r, lat, lon) = sphere::cart2eq(x_view, y_view, z_view);
    auto eq = std::make_tuple(cpp_py3_interop::xtensor_to_numpy(std::move(r)),
                              cpp_py3_interop::xtensor_to_numpy(std::move(lat)),
                              cpp_py3_interop::xtensor_to_numpy(std::move(lon)));
    return eq;
}

template <typename T>
std::tuple<double,
           double,
           double> _cart2eq(double x,
                            double y,
                            double z) {
    xt::xtensor<double, 1> _x {x};
    xt::xtensor<double, 1> _y {y};
    xt::xtensor<double, 1> _z {z};

    xt::xarray<double> r, lat, lon;
    std::tie(r, lat, lon) = sphere::cart2eq(_x, _y, _z);
    auto eq = std::make_tuple(r(0), lat(0), lon(0));
    return eq;
}

void cart2eq_bindings(pybind11::module &m) {
    m.def("cart2eq",
          pybind11::overload_cast<pybind11::array_t<float>,
                                  pybind11::array_t<float>,
                                  pybind11::array_t<float>>(&_cart2eq<float>),
          pybind11::arg("x").noconvert().none(false),
          pybind11::arg("y").noconvert().none(false),
          pybind11::arg("z").noconvert().none(false));

    m.def("cart2eq",
          pybind11::overload_cast<pybind11::array_t<double>,
                                  pybind11::array_t<double>,
                                  pybind11::array_t<double>>(&_cart2eq<double>),
          pybind11::arg("x").noconvert().none(false),
          pybind11::arg("y").noconvert().none(false),
          pybind11::arg("z").noconvert().none(false));

    m.def("cart2eq",
          pybind11::overload_cast<double,
                                  double,
                                  double>(&_cart2eq<double>),
          pybind11::arg("x").none(false),
          pybind11::arg("y").none(false),
          pybind11::arg("z").none(false),
          pybind11::doc(R"EOF(
cart2eq(x, y, z)

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
}

template <typename T>
pybind11::array_t<T> _colat2lat(pybind11::array_t<T> colat) {
    const auto& colat_view = cpp_py3_interop::numpy_to_xview<T>(colat);

    const auto& lat = sphere::colat2lat(colat_view);
    return cpp_py3_interop::xtensor_to_numpy(std::move(lat));
}

template <typename T>
double _colat2lat(double colat) {
    xt::xtensor<double, 1> _colat {colat};

    const auto& lat = sphere::colat2lat(_colat);
    return lat(0);
}

void colat2lat_bindings(pybind11::module &m) {
    m.def("colat2lat",
          pybind11::overload_cast<pybind11::array_t<float>>(&_colat2lat<float>),
          pybind11::arg("colat").noconvert().none(false));

    m.def("colat2lat",
          pybind11::overload_cast<pybind11::array_t<double>>(&_colat2lat<double>),
          pybind11::arg("colat").noconvert().none(false));

    m.def("colat2lat",
          pybind11::overload_cast<double>(&_colat2lat<double>),
          pybind11::arg("colat").none(false),
          pybind11::doc(R"EOF(
colat2lat(colat)

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
}

template <typename T>
pybind11::array_t<T> _lat2colat(pybind11::array_t<T> lat) {
    const auto& lat_view = cpp_py3_interop::numpy_to_xview<T>(lat);

    const auto& colat = sphere::lat2colat(lat_view);
    return cpp_py3_interop::xtensor_to_numpy(std::move(colat));
}

template <typename T>
double _lat2colat(double lat) {
    xt::xtensor<double, 1> _lat {lat};

    const auto& colat = sphere::lat2colat(_lat);
    return colat(0);
}

void lat2colat_bindings(pybind11::module &m) {
    m.def("lat2colat",
          pybind11::overload_cast<pybind11::array_t<float>>(&_lat2colat<float>),
          pybind11::arg("lat").noconvert().none(false));

    m.def("lat2colat",
          pybind11::overload_cast<pybind11::array_t<double>>(&_lat2colat<double>),
          pybind11::arg("lat").noconvert().none(false));

    m.def("lat2colat",
          pybind11::overload_cast<double>(&_lat2colat<double>),
          pybind11::arg("lat").none(false),
          pybind11::doc(R"EOF(
lat2colat(lat)

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

PYBIND11_MODULE(_pypeline_util_math_sphere_pybind11, m) {
    pybind11::options options;
    options.disable_function_signatures();

    pol2cart_bindings(m);
    eq2cart_bindings(m);
    cart2pol_bindings(m);
    cart2eq_bindings(m);
    colat2lat_bindings(m);
    lat2colat_bindings(m);
}
