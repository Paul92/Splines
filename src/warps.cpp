#include <iostream>
#include <chrono>

#include "warps.h"

double setToZeroIfSmall(double x) { return (x < 1e-10) ? 0 : x; };

Warp::Warp(const std::vector<Eigen::MatrixXd> &gt2d, int p, int nC, double er)
    : Warp(gt2d.size(), p*p)
{
    const std::size_t nframes = gt2d.size();

    Eigen::MatrixXd q1 = gt2d[0].block(0, 0, 2, gt2d[0].cols());

    BBS bbs(gt2d[0], nC, 2);
    Eigen::SparseMatrix<double> coloc = bbs.coloc(q1.row(0), q1.row(1));

    Eigen::MatrixXd ones(nC-3, nC-3);
    ones.setOnes();
    Eigen::MatrixXd lambdas = er * ones;

    auto bending = bbs.bending(lambdas);

    const auto [xv, yv] = bbs.meshgrid(p);

    Eigen::Map<const Eigen::RowVectorXd> I1u(xv.data(), xv.size());
    Eigen::Map<const Eigen::RowVectorXd> I1v(yv.data(), yv.size());

    Iu.row(0) = I1u;
    Iv.row(0) = I1v;

    // Compute BBS
    for (std::size_t i = 1; i < nframes; i++) {

        Eigen::MatrixXd q2 = gt2d[i].topRows(2);

        Eigen::MatrixXd ctrlpts = bbs.getCtrlpts(coloc, bending, q2);
#ifdef DEBUG
    std::cout << "[WARN] Debug is active, this screws up the BBS caches; you might see errors" << std::endl;
        auto estimated = bbs.eval(ctrlpts, gt2d[0].row(0).transpose(), gt2d[0].row(1).transpose(), 0, 0);
        double error = std::sqrt(((estimated.row(0) - gt2d[i].row(0)).array().pow(2) +
                                  (estimated.row(1) - gt2d[i].row(1)).array().pow(2)).mean());

        std::cout << "[ETA] Internal Rep error = " << error << std::endl;
#endif

        Eigen::MatrixXd q;
        if (i == 1)
            q = bbs.eval(ctrlpts, Iu.row(0).transpose(), Iv.row(0).transpose(), 0, 0);
        else
            q = bbs.eval(ctrlpts, 0, 0);

        Iu.row(i) = q.row(0);
        Iv.row(i) = q.row(1);

    }
    for (std::size_t i = 1; i < nframes; i++) {

        Eigen::Matrix<double, 2, Eigen::Dynamic> q1(2, Iu.cols());
        Eigen::Matrix<double, 2, Eigen::Dynamic> q2(2, Iu.cols());

        q1 << Iu.row(0), Iv.row(0);
        q2 << Iu.row(i), Iv.row(i);

        BBS bbs(q2, nC, 2);

        coloc = bbs.coloc(q2.row(0), q2.row(1));

        lambdas = er * ones;
        bending = bbs.bending(lambdas);

        const Eigen::MatrixXd ctrlpts = bbs.getCtrlpts(coloc, bending, q1);

        const auto [xv, yv] = bbs.meshgrid(p);

        const Eigen::Map<const Eigen::VectorXd> xv_map(xv.data(), xv.size());
        const Eigen::Map<const Eigen::VectorXd> yv_map(yv.data(), yv.size());

        Eigen::MatrixXd c(xv_map.rows(), 2);

        c << xv_map, yv_map;

        const auto dqu = bbs.eval(ctrlpts, 1, 0);
        const auto dqv = bbs.eval(ctrlpts, 0, 1);
        const auto dquv = bbs.eval(ctrlpts, 1, 1);
        const auto dquu = bbs.eval(ctrlpts, 2, 0);
        const auto dqvv = bbs.eval(ctrlpts, 0, 2);

        J21.a.row(i-1) = dqu.row(0);
        J21.b.row(i-1) = dqu.row(1);
        J21.c.row(i-1) = dqv.row(0);
        J21.d.row(i-1) = dqv.row(1);

        const auto normalizer = dqu.row(0).array() * dqv.row(1).array() - dqv.row(0).array() * dqu.row(1).array();
        J12.a.row(i-1) =  dqv.row(1).array() / normalizer;
        J12.b.row(i-1) = -dqu.row(1).array() / normalizer;
        J12.c.row(i-1) = -dqv.row(0).array() / normalizer;
        J12.d.row(i-1) =  dqu.row(0).array() / normalizer;

        H21.uua.row(i-1) = dquu.row(0);
        H21.uub.row(i-1) = dquu.row(1);

        H21.uva.row(i-1) = dquv.row(0);
        H21.uvb.row(i-1) = dquv.row(1);

        H21.vva.row(i-1) = dqvv.row(0);
        H21.vvb.row(i-1) = dqvv.row(1);

#ifdef DEBUG
        std::vector<Eigen::MatrixXd> qw2(nframes - 1);
        qw2[i-1] = bbs.eval(ctrlpts, q2.row(0).transpose(), q2.row(1).transpose(), 0, 0);
        const auto err = std::sqrt((((qw2[i-1].row(0) - q1.row(0)).array()).pow(2) +
                                    ((qw2[i-1].row(1) - q1.row(1)).array()).pow(2)).mean());
        Writer::writeDense(qw2[i-1], "errors.txt");
        std::cout << "[ETA - Warps] Internal Rep error " << err << std::endl;
#endif
    }

}

Warp::Warp(const Eigen::MatrixXd &from, const std::vector<Eigen::MatrixXd> &to, int nC, bool evaluate)
    : Warp(to.size() + 1, to[0].cols())
{
    double er = 1e-5;
    double  t = 1e-1;

#pragma omp parallel for num_threads(NTHREADS)
    for (std::size_t i = 0; i < to.size(); i++) {

        auto q1 = from;
        auto q2 = to[i];

        double umin = std::min(q1.row(0).minCoeff(), q2.row(0).minCoeff()) - t;
        double umax = std::max(q1.row(0).maxCoeff(), q2.row(0).maxCoeff()) + t;
        double vmin = std::min(q1.row(1).minCoeff(), q2.row(1).minCoeff()) - t;
        double vmax = std::max(q1.row(1).maxCoeff(), q2.row(1).maxCoeff()) + t;

        BBS bbs(umin, umax, nC, vmin, vmax, nC, 2);

        Eigen::SparseMatrix<double> coloc = bbs.coloc(q2.row(0), q2.row(1));

        Eigen::MatrixXd ones(nC-3, nC-3);
        ones.setOnes();
        Eigen::MatrixXd lambdas = er * ones;

        auto bending = bbs.bending(lambdas);

        Eigen::MatrixXd ctrlpts = bbs.getCtrlpts(coloc, bending, q1.topRows(2));

        if (evaluate) {
            evaluated.push_back(bbs.eval(ctrlpts, q2.row(0).transpose(), q2.row(1).transpose(), 0, 0));
        }

        auto dqu  = bbs.eval(ctrlpts, q2.row(0).transpose(), q2.row(1).transpose(), 1, 0);

        auto dqv  = bbs.eval(ctrlpts, 0, 1);
        auto dquv = bbs.eval(ctrlpts, 1, 1);

        J21.a.row(i) = dqu.row(0);
        J21.b.row(i) = dqu.row(1);
        J21.c.row(i) = dqv.row(0);
        J21.d.row(i) = dqv.row(1);

        const auto normalizer = dqu.row(0).array() * dqv.row(1).array() - dqv.row(0).array() * dqu.row(1).array();
        J12.a.row(i) =  dqv.row(1).array() / normalizer;
        J12.b.row(i) = -dqu.row(1).array() / normalizer;
        J12.c.row(i) = -dqv.row(0).array() / normalizer;
        J12.d.row(i) =  dqu.row(0).array() / normalizer;

        H21.uva.row(i) = dquv.row(0);
        H21.uvb.row(i) = dquv.row(1);
    }

}

std::vector<Eigen::MatrixXd> Warp::makeGrid(const std::vector<Eigen::MatrixXd> &points,
                                            int p, int nC, bool selfInterpolate, double er) {

    const std::size_t nframes = points.size();

    Eigen::MatrixXd q1 = points[0].block(0, 0, 2, points[0].cols());

    BBS bbs(points[0], nC, 2);

    Eigen::SparseMatrix<double> coloc = bbs.coloc(q1.row(0), q1.row(1));

    Eigen::MatrixXd ones(nC-3, nC-3);
    ones.setOnes();
    Eigen::MatrixXd lambdas = er * ones;

    auto bending = bbs.bending(lambdas);

    const auto [xv, yv] = bbs.meshgrid(p);

    Eigen::Map<const Eigen::RowVectorXd> I1u(xv.data(), xv.size());
    Eigen::Map<const Eigen::RowVectorXd> I1v(yv.data(), yv.size());

    Eigen::MatrixXd firstGrid(2, I1u.size());

    if (selfInterpolate) {
        firstGrid = points[0].topRows(2);
    } else {
        firstGrid.row(0) = I1u;
        firstGrid.row(1) = I1v;
    }

    std::vector<Eigen::MatrixXd> grid{firstGrid};

    // Compute BBS
    for (std::size_t i = 1; i < nframes; i++) {

        Eigen::MatrixXd q2 = points[i].topRows(2);

        Eigen::MatrixXd ctrlpts = bbs.getCtrlpts(coloc, bending, q2);

        Eigen::MatrixXd q;

        if (i == 1) {
            if (selfInterpolate) {
                q = bbs.eval(ctrlpts, points[0].row(0).transpose(), points[0].row(1).transpose(), 0, 0);
            } else {
                q = bbs.eval(ctrlpts, I1u.transpose(), I1v.transpose(), 0, 0);
            }
        } else {
            q = bbs.eval(ctrlpts, 0, 0);
        }

        grid.push_back(q);

    }

    return grid;
}



Warp3D::Warp3D(const Eigen::VectorXd &u, const Eigen::VectorXd &v, const Eigen::MatrixXd &points, int nC, double er, double t)
    : bbs(u.minCoeff() - t, u.maxCoeff() + t, nC, v.minCoeff() - t, v.maxCoeff() + t, nC, 3)
{

    Eigen::SparseMatrix<double> coloc = bbs.coloc(u, v);

    Eigen::MatrixXd ones(nC-3, nC-3);
    ones.setOnes();
    Eigen::MatrixXd lambdas = er * ones;

    auto bending = bbs.bending(lambdas);
    ctrlpts = bbs.getCtrlpts(coloc, bending, points);

}


Eigen::MatrixXd Warp3D::eval(const Eigen::MatrixXd &q, int du, int dv) const {
    const auto qw = bbs.eval(ctrlpts, q.row(0).transpose(), q.row(1).transpose(), du, dv);
    return qw;
}

