#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "bbs.h"

/**
 * @brief The Jacobian matrix of an image warp.
 *
 * Since the warp is a function from R^2 to R^2, the Jacobian at each point has
 * four elements, here denoted by a,b,c,d. Each row of the a,b,c,d matrices
 * represents one frame in the set of images of the warp, and each column
 * correspoinds to a point.
 */
class Jacobian {
    public:
        Jacobian() = default;

        Jacobian(int rows, int cols) {
            a = Eigen::MatrixXd::Zero(rows, cols);
            b = Eigen::MatrixXd::Zero(rows, cols);
            c = Eigen::MatrixXd::Zero(rows, cols);
            d = Eigen::MatrixXd::Zero(rows, cols);
        }

        Eigen::MatrixXd a;
        Eigen::MatrixXd b;
        Eigen::MatrixXd c;
        Eigen::MatrixXd d;
};


/**
 * @brief The Hessian matrix of an image warp.
 *
 * Similar to @ref Jacobian.
 */
class Hessian {
    public:
        Hessian() = default;

        Hessian(int rows, int cols) {
            uua = Eigen::MatrixXd::Zero(rows, cols);
            uub = Eigen::MatrixXd::Zero(rows, cols);
            uva = Eigen::MatrixXd::Zero(rows, cols);
            uvb = Eigen::MatrixXd::Zero(rows, cols);
            vva = Eigen::MatrixXd::Zero(rows, cols);
            vvb = Eigen::MatrixXd::Zero(rows, cols);
        }

        Eigen::MatrixXd uua;
        Eigen::MatrixXd uub;
        Eigen::MatrixXd uva;
        Eigen::MatrixXd uvb;
        Eigen::MatrixXd vva;
        Eigen::MatrixXd vvb;
};


/**
 * @brief Represents the warp between two frames.
 *
 * The class @ref Warp models the 2D warp between a reference frame and
 * multiple frames. This small structure is provided for convenience, storing
 * some information for only a pair of frame.
 */
class TwoFrameWarp {
    public:
        Eigen::VectorXd a;
        Eigen::VectorXd b;
        Eigen::VectorXd c;
        Eigen::VectorXd d;

        Eigen::VectorXd t1;
        Eigen::VectorXd t2;

        Eigen::MatrixXd asMatrix() const {
            Eigen::MatrixXd der(6, a.size());

            der.row(0) = a;
            der.row(1) = b;
            der.row(2) = c;
            der.row(3) = d;
            der.row(4) = t1;
            der.row(5) = t2;

            return der;

        }
};


/**
 * @brief Models the 2D image warp between a reference frame and a list of
 * frames, also providing first and second order derivatives.
 *
 * In this class, frames (i.e. the keypoints of a frame) are represented as
 * a 2xn matrix, where the first row represents the u coordinate and the second
 * row represents the v coordinate of the points.
 */
class Warp {

    public:
        /**
         * @brief Create an empty warp structure.
         *
         * @param n The number of frames.
         * @param p The number of points.
         */
        Warp(int n, int p) : noOfFrames(n) {
            Iu = Eigen::MatrixXd::Zero(n, p);
            Iv = Eigen::MatrixXd::Zero(n, p);

            J21 = Jacobian(n-1, p);
            J12 = Jacobian(n-1, p);
            H21 = Hessian(n-1, p);
        }

        [[deprecated]]
            explicit Warp(const std::vector<Eigen::MatrixXd> &gt2d,
                    int p = 20, int nC = 40, double er = 1e-4);

        /**
         * @brief Create a 2D image warp from a referance frame to a list of
         * frames.
         *
         * @param from The reference frame
         * @param to A list of frames
         * @param nC The number of control points in each dimension
         * @param evaluate Whether to evaluate the warp at the points of @ref
         * from
         */
        Warp(const Eigen::MatrixXd &from, const std::vector<Eigen::MatrixXd> &to,
             int nC = 40,  bool evaluate=false);


        /**
         * @brief Fit a warp between the frames and interpolate the keypoints
         * into a grid.
         *
         * It is used to convert an arbitrary spatial distribution of the
         * keypoints to a regular grid of @ref p points in each dimension.
         *
         * @param points The frames to interpolate
         * @param p The size of the resulting grid
         * @param nC The number of control points along each dimension of the
         * warp
         *
         * @return A set of interpolated keypoints.
         */
        static std::vector<Eigen::MatrixXd> makeGrid(const std::vector<Eigen::MatrixXd> &gt2d,
                int p = 20, int nC = 40,
                bool selfInterpolate = false, double er = 1e-4);

        /**
         * @brief If evaluated is set in the constructor, returns the warps
         * evaluated at the keypoints of the reference frame.
         *
         * It can be used, for example, at measuring the interpolation error
         * of the warp by comparing these interpolated keypoints with the
         * known keypoints on a frame.
         */
        std::vector<Eigen::MatrixXd> getEvaluated() const {
            if (evaluated.size() == 0)
                throw std::runtime_error("The warps have not been evaluated; pass true to the constructor");
            return evaluated;
        }

        /**
         * @brief Utility function, to get some useful information for
         * a specific frame.
         */
        TwoFrameWarp getTwoFrameWarp(int to) {
            TwoFrameWarp tfw;

            tfw.a = J21.a.row(to);
            tfw.b = J21.b.row(to);
            tfw.c = J21.c.row(to);
            tfw.d = J21.d.row(to);
            tfw.t1    = -(J12.b.row(to).array() * H21.uva.row(to).array() +
                    J12.d.row(to).array() * H21.uvb.row(to).array());
            tfw.t2    = -(J12.a.row(to).array() * H21.uva.row(to).array() +
                    J12.c.row(to).array() * H21.uvb.row(to).array());

            return tfw;
        }

        const int noOfFrames;   /// The number of frames

        Eigen::MatrixXd Iu;
        Eigen::MatrixXd Iv;

        Jacobian J21;
        Jacobian J12;
        Hessian H21;

    private:
        std::vector<Eigen::MatrixXd> evaluated;
};



/**
 * @brief Warp from 2D to 3D.
 */
class Warp3D {

    public:
        /**
         * @brief Create a warp from 2D to 3D.
         *
         * @param u The u coordinates of the 2D points
         * @param v The v coordinates of the 2D points
         * @param points 3xn matrix of 3D points
         * @param nC The number of control points
         */
        Warp3D(const Eigen::VectorXd &u, const Eigen::VectorXd &v, const Eigen::MatrixXd &points, int nC=40, double er=1e-4, double t=1e-1);

        /**
         * @brief Evaluate the warp at the 2D points from the 2xn matrix q
         */
        Eigen::MatrixXd eval(const Eigen::MatrixXd &q, int du = 0, int dv = 0) const;

        BBS bbs;
        Eigen::MatrixXd ctrlpts;
};

