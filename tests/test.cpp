#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <chrono>

#include "../src/warps.h"

int main() {

    std::ifstream f("/home/paul/git/myNRSFM/data/kinect/2d.txt");

    int frames, dimensions, points;
    f >> frames >> dimensions >> points;

    Eigen::MatrixXd mat1(3, points);
    Eigen::MatrixXd mat2(3, points);

    for (int i = 0; i < points; i++) {
        f >> mat1(0, i) >> mat1(1, i) >> mat1(2, i);
    }
    for (int frame = 0; frame < 50; frame++) {
        for (int i = 0; i < points; i++) {
            f >> mat1(0, i) >> mat1(1, i) >> mat1(2, i);
        }
    }
    for (int i = 0; i < points; i++) {
        f >> mat2(0, i) >> mat2(1, i) >> mat2(2, i);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++)
        Warp warp(mat1, std::vector<Eigen::MatrixXd>{mat2});

    auto t2 = std::chrono::high_resolution_clock::now();

    int timems = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Took " << timems / 10 << " ms per frame"  << std::endl;;

    f.close();

    return 0;
}
