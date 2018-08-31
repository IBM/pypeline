// ############################################################################
// types.hpp
// =========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

/*
 * Global type definitions used throughout Pypeline.
 */

#ifndef PYPELINE_TYPES_HPP
#define PYPELINE_TYPES_HPP

#include <complex>

#include "eigen3/Eigen/Eigen"
#include "eigen3/Eigen/Sparse"

using cfloat_t = std::complex<float>;
using cdouble_t = std::complex<double>;

template <typename T> using ArrayX_t   = Eigen::Array  <T,              1, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
template <typename T> using ArrayXX_t  = Eigen::Array  <T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
template <typename T> using MatrixXX_t = Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;

template <typename T> using SpMatrixXX_t = Eigen::SparseMatrix<T, Eigen::ColMajor, int>;

#endif //PYPELINE_TYPES_HPP
