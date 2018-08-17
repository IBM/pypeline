// ############################################################################
// sphere.hpp
// ==========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Spherical geometry tools.
 */

#ifndef PYPELINE_UTIL_MATH_SPHERE_HPP
#define PYPELINE_UTIL_MATH_SPHERE_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xbroadcast.hpp"

#include "pypeline/util/argcheck.hpp"

namespace pypeline { namespace util { namespace math { namespace sphere {
    /*
     * Polar coordinates to Cartesian coordinates.
     *
     * Parameters
     * ----------
     * r : xt::xexpression
     *     Radius.
     * colat : xt::xexpression
     *     Polar/Zenith angle [rad].
     * lon : xt::xexpression
     *     Longitude angle [rad].
     *
     * Returns
     * -------
     * x : xt::xarray<double>
     * y : xt::xarray<double>
     * z : xt::xarray<double>
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include <tuple>
     *    #include "xtensor/xarray.hpp"
     *    #include "xtensor/xbuilder.hpp"
     *    #include "xtensor/xstrided_view.hpp"
     *    #include "pypeline/util/math/sphere.hpp"
     *
     *    namespace sphere = pypeline::util::math::sphere;
     *    auto r = xt::ones<float>({3, 1, 4});
     *    auto colat = xt::ones<double>({1, 5, 1});
     *    auto lon = xt::reshape_view(xt::linspace<double>(0, M_PI, 4), {1, 1, 4});
     *
     *    xt::xarray<double> x, y, z;
     *    std::tie(x, y, z) = sphere::pol2cart(r, colat, lon);
     */
    template <typename E1, typename E2, typename E3>
    auto pol2cart(E1 &&r, E2 &&colat, E3 &&lon) {
        namespace argcheck = pypeline::util::argcheck;
        if (!(argcheck::has_floats(r) && xt::all(r >= 0))) {
            std::string msg = "Parameter[r] must contain positive real values.";
            throw std::runtime_error(msg);
        }
        if (!(argcheck::has_floats(colat) &&
            xt::all((0 <= colat) && (colat <= M_PI)))) {
            std::string msg = "Parameter[colat] must contain real values in [0, pi].";
            throw std::runtime_error(msg);
        }
        if (!argcheck::has_floats(lon)) {
            std::string msg = "Parameter[lon] must contain real values.";
            throw std::runtime_error(msg);
        }

        xt::xarray<double> sin_colat {xt::sin(colat)};
        xt::xarray<double> x {r * sin_colat * xt::cos(lon)};
        xt::xarray<double> y {r * sin_colat * xt::sin(lon)};
        xt::xarray<double> z {xt::broadcast(r * xt::cos(colat), x.shape())};

        return std::make_tuple(x, y, z);
    }

    /*
     * Equatorial coordinates to Cartesian coordinates.
     *
     * Parameters
     * ----------
     * r : xt::xexpression
     *     Radius.
     * lat : xt::xexpression
     *     Latitude angle [rad].
     * lon : xt::xexpression
     *     Longitude angle [rad].
     *
     * Returns
     * -------
     * x : xt::xarray<double>
     * y : xt::xarray<double>
     * z : xt::xarray<double>
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include <tuple>
     *    #include "xtensor/xarray.hpp"
     *    #include "xtensor/xbuilder.hpp"
     *    #include "xtensor/xstrided_view.hpp"
     *    #include "pypeline/util/math/sphere.hpp"
     *
     *    namespace sphere = pypeline::util::math::sphere;
     *    auto r = xt::ones<float>({3, 1, 4});
     *    auto lat = xt::ones<double>({1, 5, 1});
     *    auto lon = xt::reshape_view(xt::linspace<double>(0, M_PI, 4), {1, 1, 4});
     *
     *    xt::xarray<double> x, y, z;
     *    std::tie(x, y, z) = sphere::eq2cart(r, colat, lon);
     */
    template <typename E1, typename E2, typename E3>
    auto eq2cart(E1 &&r, E2 &&lat, E3 &&lon) {
        namespace argcheck = pypeline::util::argcheck;
        if (!(argcheck::has_floats(r) && xt::all(r >= 0))) {
            std::string msg = "Parameter[r] must contain positive real values.";
            throw std::runtime_error(msg);
        }
        if (!(argcheck::has_floats(lat) &&
            xt::all(((-0.5 * M_PI) <= lat) && (lat <= (0.5 * M_PI))))) {
            std::string msg = "Parameter[lat] must contain real values in [-pi/2, pi/2].";
            throw std::runtime_error(msg);
        }
        if (!argcheck::has_floats(lon)) {
            std::string msg = "Parameter[lon] must contain real values.";
            throw std::runtime_error(msg);
        }

        xt::xarray<double> cos_lat {xt::cos(lat)};
        xt::xarray<double> x {r * cos_lat * xt::cos(lon)};
        xt::xarray<double> y {r * cos_lat * xt::sin(lon)};
        xt::xarray<double> z {xt::broadcast(r * xt::sin(lat), x.shape())};

        return std::make_tuple(x, y, z);
    }

    /*
     * Cartesian coordinates to Polar coordinates.
     *
     * Parameters
     * ----------
     * x : xexpression
     * y : xexpression
     * z : xexpression
     *
     * Returns
     * -------
     * r : xt::xarray<double>
     *     Radius.
     * colat : xt::xarray<double>
     *     Polar/Zenith angle [rad].
     * lon : xt::xarray<double>
     *     Longitude angle [rad].
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include <tuple>
     *    #include "xtensor/xarray.hpp"
     *    #include "xtensor/xbuilder.hpp"
     *    #include "pypeline/util/math/sphere.hpp"
     *
     *    namespace sphere = pypeline::util::math::sphere;
     *    auto x = xt::ones<float>({3, 1, 4});
     *    auto y = xt::zeros<double>({3, 5, 1});
     *    auto z = 5 * xt::ones<double>({1, 1, 4});
     *
     *    xt::xarray<double> r, colat, lon;
     *    std::tie(r, colat, lon) = sphere::cart2pol(x, y, z);
     */
    template <typename E1, typename E2, typename E3>
    auto cart2pol(E1 &&x, E2 &&y, E3 &&z) {
        namespace argcheck = pypeline::util::argcheck;
        if (!argcheck::has_floats(x)) {
            std::string msg = "Parameter[x] must contain real values.";
            throw std::runtime_error(msg);
        }
        if (!argcheck::has_floats(y)) {
            std::string msg = "Parameter[y] must contain real values.";
            throw std::runtime_error(msg);
        }
        if (!argcheck::has_floats(z)) {
            std::string msg = "Parameter[z] must contain real values.";
            throw std::runtime_error(msg);
        }

        xt::xarray<double> s2 {(x * x) + (y * y)};
        xt::xarray<double> r {xt::sqrt(s2 + (z * z))};
        xt::xarray<double> colat {xt::atan2(xt::sqrt(s2), z)};
        xt::xarray<double> lon {xt::broadcast(xt::atan2(y, x), r.shape())};

        return std::make_tuple(r, colat, lon);
    }

    /*
     * Cartesian coordinates to Equatorial coordinates.
     *
     * Parameters
     * ----------
     * x : xexpression
     * y : xexpression
     * z : xexpression
     *
     * Returns
     * -------
     * r : xt::xarray<double>
     *     Radius.
     * lat : xt::xarray<double>
     *     Latitude angle [rad].
     * lon : xt::xarray<double>
     *     Longitude angle [rad].
     *
     * Examples
     * --------
     * .. literal_block::
     *
     *    #include <tuple>
     *    #include "xtensor/xarray.hpp"
     *    #include "xtensor/xbuilder.hpp"
     *    #include "pypeline/util/math/sphere.hpp"
     *
     *    namespace sphere = pypeline::util::math::sphere;
     *    auto x = xt::ones<float>({3, 1, 4});
     *    auto y = xt::zeros<double>({3, 5, 1});
     *    auto z = 5 * xt::ones<double>({1, 1, 4});
     *
     *    xt::xarray<double> r, lat, lon;
     *    std::tie(r, lat, lon) = sphere::cart2eq(x, y, z);
     */
    template <typename E1, typename E2, typename E3>
    auto cart2eq(E1 &&x, E2 &&y, E3 &&z) {
        namespace argcheck = pypeline::util::argcheck;
        if (!argcheck::has_floats(x)) {
            std::string msg = "Parameter[x] must contain real values.";
            throw std::runtime_error(msg);
        }
        if (!argcheck::has_floats(y)) {
            std::string msg = "Parameter[y] must contain real values.";
            throw std::runtime_error(msg);
        }
        if (!argcheck::has_floats(z)) {
            std::string msg = "Parameter[z] must contain real values.";
            throw std::runtime_error(msg);
        }

        xt::xarray<double> s2 {(x * x) + (y * y)};
        xt::xarray<double> r {xt::sqrt(s2 + (z * z))};
        xt::xarray<double> lat {xt::atan2(z, xt::sqrt(s2))};
        xt::xarray<double> lon {xt::broadcast(xt::atan2(y, x), r.shape())};

        return std::make_tuple(r, lat, lon);
    }

    /*
     * Co-latitude to latitude.
     *
     * Parameters
     * ----------
     * colat : xt::xexpression
     *     Polar/Zenith angle [rad].
     *
     * Returns
     * -------
     * lat : xt::xarray<double>
     *     Latitude angle [rad].
     */
    template <typename E>
    xt::xarray<double> colat2lat(E &&colat) {
        namespace argcheck = pypeline::util::argcheck;
        if (!(argcheck::has_floats(colat) &&
            xt::all((0 <= colat) && (colat <= M_PI)))) {
            std::string msg = "Parameter[colat] must contain real values in [0, pi].";
            throw std::runtime_error(msg);
        }

        xt::xarray<double> lat {(0.5 * M_PI) - colat};
        return lat;
    }

    /*
     * Latitude to co-latitude.
     *
     * Parameters
     * ----------
     * lat : xt::xexpression
     *     Latitude angle [rad].
     *
     * Returns
     * -------
     * colat : xt::xarray<double>
     *     Polar/Zenith angle [rad].
     */
    template <typename E>
    xt::xarray<double> lat2colat(E &&lat) {
        namespace argcheck = pypeline::util::argcheck;
        if (!(argcheck::has_floats(lat) &&
            xt::all(((-0.5 * M_PI) <= lat) && (lat <= (0.5 * M_PI))))) {
            std::string msg = "Parameter[lat] must contain real values in [-pi/2, pi/2].";
            throw std::runtime_error(msg);
        }

        xt::xarray<double> colat {(0.5 * M_PI) - lat};
        return colat;
    }
}}}}

#endif //PYPELINE_UTIL_MATH_SPHERE_HPP
