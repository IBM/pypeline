// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [sep@zurich.ibm.com]
// ############################################################################

#include <chrono>
#include <iostream>
#include <vector>

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#include "Eigen/Eigen"

int main() {
    size_t N(5000), M(10000), K(7000);

    xt::xtensor<double, 2> x = xt::random::randn({N, M}, 0.0, 1.0);
    xt::xtensor<double, 2> y = xt::random::randn({M, K}, 0.0, 1.0);

    Eigen::MatrixXd xx(N, M); xx.setRandom();
    Eigen::MatrixXd yy(M, K); yy.setRandom();

    auto start_eig = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd zz = xx * yy;
    auto stop_eig = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_eig = stop_eig - start_eig;
    std::cout << "Elapsed time: " << elapsed_eig.count() << " [s]" << std::endl;

    return 0;
}
