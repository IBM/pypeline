// ############################################################################
// image.hpp
// =========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Low-level image containers.
 */

#ifndef PYPELINE_PHASED_ARRAY_UTIL_IO_IMAGE
#define PYPELINE_PHASED_ARRAY_UTIL_IO_IMAGE

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xstrided_view.hpp"

#include "pypeline/util/argcheck.hpp"

namespace pypeline { namespace phased_array { namespace util { namespace io { namespace image {
    /*
     * In-memory container for storing real-valued images defined on :math:`\mathbb{S}^{2}`.
     *
     * This class is not meant to be subclassed: prefer encapsulating an instance in your objects instead.
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
     *    #include "pypeline/phased_array/util/io/image.hpp"
     *
     *    namespace sphere = pypeline::util::math::sphere;
     *    namespace image = pypeline::phased_array::util::io::image;
     *
     *    auto im = xt::ones<double>({5, 768, 1024});                // 5 images of shape (768, 1024).
     *
     *    xt::xarray<double> r {1};
     *    auto colat = xt::reshape_view(xt::linspace<double>(0, M_PI, 768), {768, 1});
     *    auto lon = xt::reshape_view(xt::linspace<double>(0, 2 * M_PI, 1024), {1, 1024});
     *    auto gr = xt::stack(sphere::pol2cart(r, colat, lon), 0);  // grid of shape (3, 768, 1024)
     *
     *    const auto& I = image::SphericalImageContainer<float>(im, gr);  // (im, gr) internally stored as float.
     */
    template <typename T>
    class SphericalImageContainer {
        private:
            xt::xarray<T> m_image;
            xt::xarray<T> m_grid;
            bool m_is_gridded;

        public:
            /*
             * Parameters
             * ----------
             * image : xt::xexpression
             *     Multi-level real-valued data cube.
             *
             *     Possible shapes are:
             *     * (N_height, N_width)
             *     * (N_image, N_height, N_width)
             *     * (N_points,)
             *     * (N_image, N_points)
             *
             * grid : xt::xexpression
             *     (3, ...) Cartesian coordinates of the sky on which the data-points are defined.
             *
             *     Possible shapes are:
             *     * (3, N_height, N_width)
             *     * (3, N_points)
             *
             * Note
             * ----
             * The image and grid are stored internally with scalar type T.
             * Only floating-point types are accepted.
             */
            template <typename E1, typename E2>
            SphericalImageContainer(E1 &&image, E2 &&grid) {
                static_assert(std::is_floating_point<T>::value, "Only {float, double} are allowed for Type[T].");

                namespace argcheck = pypeline::util::argcheck;
                if (!argcheck::has_floats(grid)) {
                    std::string msg = "Parameter[grid] must be real-valued.";
                    throw std::runtime_error(msg);
                }
                if (!((grid.shape().size() == 2) ||
                      (grid.shape().size() == 3))) {
                    std::string msg = "Parameter[grid] must have shape (3, N_height, N_width) or (3, N_point).";
                    throw std::runtime_error(msg);
                }
                if (grid.shape()[0] != 3) {
                    std::string msg = "Parameter[grid] must have shape (3, N_height, N_width) or (3, N_point).";
                    throw std::runtime_error(msg);
                }
                m_grid = grid;

                if (!argcheck::has_floats(image)) {
                    std::string msg = "Parameter[image] must be real-valued.";
                    throw std::runtime_error(msg);
                }
                if (grid.shape().size() == 2) {  // N_point mode
                    std::string msg = "Parameter[image] must have shapae (N_point,) or (N_image, N_point).";
                    auto N_point = grid.shape()[grid.shape().size() - 1];
                    m_is_gridded = false;

                    if (image.shape().size() == 1) {
                        if (image.shape()[0] != N_point) {
                            throw std::runtime_error(msg);
                        }

                        m_image = xt::reshape_view(image, std::vector<size_t>{1, static_cast<size_t>(N_point)});
                    } else if (image.shape().size() == 2) {
                        if (image.shape()[1] != N_point) {
                            throw std::runtime_error(msg);
                        }
                        m_image = image;
                    } else {
                        throw std::runtime_error(msg);
                    }
                } else {  // (N_height, N_width) mode
                    std::string msg = "Parameter[image] must have shape (N_height, N_width) or (N_image, N_height, N_width).";
                    auto N_height = grid.shape()[1];
                    auto N_width = grid.shape()[2];
                    m_is_gridded = true;

                    if (image.shape().size() == 2) {
                        if (!((image.shape()[0] == N_height) &&
                              (image.shape()[1] == N_width))) {
                            throw std::runtime_error(msg);
                        }

                        m_image = xt::reshape_view(image, std::vector<size_t>{1, static_cast<size_t>(N_height), static_cast<size_t>(N_width)});
                    } else if (image.shape().size() == 3) {
                        if (!((image.shape()[1] == N_height) &&
                              (image.shape()[2] == N_width))) {
                            throw std::runtime_error(msg);
                        }

                        m_image = image;
                    } else {
                        throw std::runtime_error(msg);
                    }
                }
            }

            const xt::xarray<T>& image() {
                return m_image;
            }

            const xt::xarray<T>& grid() {
                return m_grid;
            }

            bool is_gridded() {
                return m_is_gridded;
            }
    };
}}}}}

#endif //PYPELINE_PHASED_ARRAY_UTIL_IO_IMAGE
