#include <iostream>
#include <cstdlib>
#include <cmath>
#include <exception>

#include <omp.h>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/UmfPackSupport>

#include "bbs.h"

BBS::BBS(double umin, double umax,
         int nptsu,
         double vmin, double vmax,
         int nptsv, int valdim)
    : umin(umin), umax(umax),
      nptsu(nptsu),
      vmin(vmin), vmax(vmax),
      nptsv(nptsv), valdim(valdim)
{
    if (nptsu < 4 || nptsv < 4)
        throw std::invalid_argument("We need at least 4 control points in each dimension");
}

std::tuple<double, double, double, double> getUVBoundaries(const Eigen::MatrixXd &mat, double t) {

    const auto mins = mat.rowwise().minCoeff();
    const auto maxs = mat.rowwise().maxCoeff();

    const double umin = mins(0) - t;
    const double umax = maxs(0) + t;

    const double vmin = mins(1) - t;
    const double vmax = maxs(1) + t;

    return std::make_tuple(umin, umax, vmin, vmax);

}

BBS::BBS(const Eigen::MatrixXd &mat, int npts, int valdim, double t) : nptsu(npts), nptsv(npts), valdim(valdim) {

    std::tie(umin, umax, vmin, vmax) = getUVBoundaries(mat, t);

    if (nptsu < 4 || nptsv < 4)
        throw std::invalid_argument("We need at least 4 control points in each dimension");
}


constexpr double __bxx_coeff[] = {1.0/756.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/504.0, 1.0/252.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/504.0, 1.0/252.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/1512.0, 0.0, -1.0/504.0, 1.0/756.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 43.0/5040.0, -43.0/3360.0, 0.0, 43.0/10080.0, 11.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -43.0/3360.0, 43.0/1680.0, -43.0/3360.0, 0.0, -33.0/280.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -43.0/3360.0, 43.0/1680.0, -43.0/3360.0, 0.0, -33.0/280.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 43.0/10080.0, 0.0, -43.0/3360.0, 43.0/5040.0, 11.0/280.0, 0.0, -33.0/280.0, 11.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/252.0, -1.0/168.0, 0.0, 1.0/504.0, 311.0/5040.0, -311.0/3360.0, 0.0, 311.0/10080.0, 11.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/168.0, 1.0/84.0, -1.0/168.0, 0.0, -311.0/3360.0, 311.0/1680.0, -311.0/3360.0, 0.0, -33.0/280.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/168.0, 1.0/84.0, -1.0/168.0, 0.0, -311.0/3360.0, 311.0/1680.0, -311.0/3360.0, 0.0, -33.0/280.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/504.0, 0.0, -1.0/168.0, 1.0/252.0, 311.0/10080.0, 0.0, -311.0/3360.0, 311.0/5040.0, 11.0/280.0, 0.0, -33.0/280.0, 11.0/140.0, 0.0, 0.0, 0.0, 0.0, 1.0/15120.0, -1.0/10080.0, 0.0, 1.0/30240.0, 1.0/252.0, -1.0/168.0, 0.0, 1.0/504.0, 43.0/5040.0, -43.0/3360.0, 0.0, 43.0/10080.0, 1.0/756.0, 0.0, 0.0, 0.0, -1.0/10080.0, 1.0/5040.0, -1.0/10080.0, 0.0, -1.0/168.0, 1.0/84.0, -1.0/168.0, 0.0, -43.0/3360.0, 43.0/1680.0, -43.0/3360.0, 0.0, -1.0/504.0, 1.0/252.0, 0.0, 0.0, 0.0, -1.0/10080.0, 1.0/5040.0, -1.0/10080.0, 0.0, -1.0/168.0, 1.0/84.0, -1.0/168.0, 0.0, -43.0/3360.0, 43.0/1680.0, -43.0/3360.0, 0.0, -1.0/504.0, 1.0/252.0, 0.0, 1.0/30240.0, 0.0, -1.0/10080.0, 1.0/15120.0, 1.0/504.0, 0.0, -1.0/168.0, 1.0/252.0, 43.0/10080.0, 0.0, -43.0/3360.0, 43.0/5040.0, 1.0/1512.0, 0.0, -1.0/504.0, 1.0/756.0};
constexpr double __bxy_coeff[] = {1.0/200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0/1200.0, 17.0/600.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/100.0, -29.0/1200.0, 17.0/600.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/1200.0, -1.0/100.0, 7.0/1200.0, 1.0/200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0/1200.0, 49.0/7200.0, -7.0/600.0, -7.0/7200.0, 17.0/600.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 49.0/7200.0, 119.0/3600.0, -203.0/7200.0, -7.0/600.0, 119.0/3600.0, 289.0/1800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.0/600.0, -203.0/7200.0, 119.0/3600.0, 49.0/7200.0, -17.0/300.0, -493.0/3600.0, 289.0/1800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.0/7200.0, -7.0/600.0, 49.0/7200.0, 7.0/1200.0, -17.0/3600.0, -17.0/300.0, 119.0/3600.0, 17.0/600.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/100.0, -7.0/600.0, 1.0/50.0, 1.0/600.0, -29.0/1200.0, -203.0/7200.0, 29.0/600.0, 29.0/7200.0, 17.0/600.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.0/600.0, -17.0/300.0, 29.0/600.0, 1.0/50.0, -203.0/7200.0, -493.0/3600.0, 841.0/7200.0, 29.0/600.0, 119.0/3600.0, 289.0/1800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/50.0, 29.0/600.0, -17.0/300.0, -7.0/600.0, 29.0/600.0, 841.0/7200.0, -493.0/3600.0, -203.0/7200.0, -17.0/300.0, -493.0/3600.0, 289.0/1800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/600.0, 1.0/50.0, -7.0/600.0, -1.0/100.0, 29.0/7200.0, 29.0/600.0, -203.0/7200.0, -29.0/1200.0, -17.0/3600.0, -17.0/300.0, 119.0/3600.0, 17.0/600.0, 0.0, 0.0, 0.0, 0.0, -1.0/1200.0, -7.0/7200.0, 1.0/600.0, 1.0/7200.0, -1.0/100.0, -7.0/600.0, 1.0/50.0, 1.0/600.0, 7.0/1200.0, 49.0/7200.0, -7.0/600.0, -7.0/7200.0, 1.0/200.0, 0.0, 0.0, 0.0, -7.0/7200.0, -17.0/3600.0, 29.0/7200.0, 1.0/600.0, -7.0/600.0, -17.0/300.0, 29.0/600.0, 1.0/50.0, 49.0/7200.0, 119.0/3600.0, -203.0/7200.0, -7.0/600.0, 7.0/1200.0, 17.0/600.0, 0.0, 0.0, 1.0/600.0, 29.0/7200.0, -17.0/3600.0, -7.0/7200.0, 1.0/50.0, 29.0/600.0, -17.0/300.0, -7.0/600.0, -7.0/600.0, -203.0/7200.0, 119.0/3600.0, 49.0/7200.0, -1.0/100.0, -29.0/1200.0, 17.0/600.0, 0.0, 1.0/7200.0, 1.0/600.0, -7.0/7200.0, -1.0/1200.0, 1.0/600.0, 1.0/50.0, -7.0/600.0, -1.0/100.0, -7.0/7200.0, -7.0/600.0, 49.0/7200.0, 7.0/1200.0, -1.0/1200.0, -1.0/100.0, 7.0/1200.0, 1.0/200.0};
constexpr double __byy_coeff[] = {1.0/756.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 43.0/5040.0, 11.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/252.0, 311.0/5040.0, 11.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/15120.0, 1.0/252.0, 43.0/5040.0, 1.0/756.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/504.0, -43.0/3360.0, -1.0/168.0, -1.0/10080.0, 1.0/252.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -43.0/3360.0, -33.0/280.0, -311.0/3360.0, -1.0/168.0, 43.0/1680.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/168.0, -311.0/3360.0, -33.0/280.0, -43.0/3360.0, 1.0/84.0, 311.0/1680.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/10080.0, -1.0/168.0, -43.0/3360.0, -1.0/504.0, 1.0/5040.0, 1.0/84.0, 43.0/1680.0, 1.0/252.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/504.0, -43.0/3360.0, -1.0/168.0, -1.0/10080.0, 1.0/252.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -43.0/3360.0, -33.0/280.0, -311.0/3360.0, -1.0/168.0, 43.0/1680.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/168.0, -311.0/3360.0, -33.0/280.0, -43.0/3360.0, 1.0/84.0, 311.0/1680.0, 33.0/140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/10080.0, -1.0/168.0, -43.0/3360.0, -1.0/504.0, 1.0/5040.0, 1.0/84.0, 43.0/1680.0, 1.0/252.0, 0.0, 0.0, 0.0, 0.0, 1.0/1512.0, 43.0/10080.0, 1.0/504.0, 1.0/30240.0, 0.0, 0.0, 0.0, 0.0, -1.0/504.0, -43.0/3360.0, -1.0/168.0, -1.0/10080.0, 1.0/756.0, 0.0, 0.0, 0.0, 43.0/10080.0, 11.0/280.0, 311.0/10080.0, 1.0/504.0, 0.0, 0.0, 0.0, 0.0, -43.0/3360.0, -33.0/280.0, -311.0/3360.0, -1.0/168.0, 43.0/5040.0, 11.0/140.0, 0.0, 0.0, 1.0/504.0, 311.0/10080.0, 11.0/280.0, 43.0/10080.0, 0.0, 0.0, 0.0, 0.0, -1.0/168.0, -311.0/3360.0, -33.0/280.0, -43.0/3360.0, 1.0/252.0, 311.0/5040.0, 11.0/140.0, 0.0, 1.0/30240.0, 1.0/504.0, 43.0/10080.0, 1.0/1512.0, 0.0, 0.0, 0.0, 0.0, -1.0/10080.0, -1.0/168.0, -43.0/3360.0, -1.0/504.0, 1.0/15120.0, 1.0/252.0, 43.0/5040.0, 1.0/756.0};

static const Eigen::Map<const Eigen::Matrix<double, 16, 16, Eigen::RowMajor>> ____bxx_coeff(__bxx_coeff);
static const Eigen::Map<const Eigen::Matrix<double, 16, 16, Eigen::RowMajor>> ____bxy_coeff(__bxy_coeff);
static const Eigen::Map<const Eigen::Matrix<double, 16, 16, Eigen::RowMajor>> ____byy_coeff(__byy_coeff);

void BBS::eval_basis(double nx, double *val_basis) {
    double nx2 = nx * nx;
    double nx3 = nx2 * nx;
    val_basis[0] = (-nx3 + 3.0 * nx2 - 3.0 * nx + 1.0) / 6.0;
    val_basis[1] = (3.0 * nx3 - 6.0 * nx2 + 4.0) / 6.0;
    val_basis[2] = (-3.0 * nx3 + 3.0 * nx2 + 3.0 * nx + 1.0) / 6.0;
    val_basis[3] = nx3 / 6.0;
}

void BBS::eval_basis_d(double nx, double *val_basis) {
    double nx2 = nx * nx;
    val_basis[0] = (-nx2 + 2 * nx - 1) / 2.0;
    val_basis[1] = (3.0 * nx2 - 4.0 * nx) / 2.0;
    val_basis[2] = (-3 * nx2 + 2 * nx + 1) / 2.0;
    val_basis[3] = nx2 / 2.0;
}

void BBS::eval_basis_dd(double nx, double *val_basis) {
    val_basis[0] = -nx + 1.0;
    val_basis[1] = 3.0 * nx - 2.0;
    val_basis[2] = -3.0 * nx + 1.0;
    val_basis[3] = nx;
}

void BBS::normalize_with_inter(double xmin, double xmax, int npts,
                               const double *x, int nb_x,
                               double *nx, int *inter) const {
    const int ninter = npts - 3;
    const double width_inter = (xmax - xmin) / ninter;

#pragma omp parallel for num_threads(NTHREADS), schedule(guided), default(none), firstprivate(nb_x, x, xmax, xmin, width_inter, ninter), shared(nx, inter)
    for (int i = 0; i < nb_x; ++i) {
        if (x[i] == xmax) {
            nx[i] = 1.0;
            inter[i] = ninter - 1;
        } else if (x[i] < xmin) {
            nx[i] = (x[i] - xmin) / width_inter;
            inter[i] = -1;
        } else if (x[i] > xmax) {
            nx[i] = (x[i] - xmin) / width_inter - ninter;
            inter[i] = ninter;
        } else {
            double scaled = (x[i] - xmin) / width_inter;
            inter[i] = static_cast<int>(std::floor(scaled));
            nx[i] = scaled - inter[i];
        }
    }
}

Eigen::SparseMatrix<double> BBS::coloc(const Eigen::VectorXd &u,
                                       const Eigen::VectorXd &v) const {

    assert(u.size() == v.size());

    const int nrRows = u.size();
    const int nrCols = nptsu * nptsv;
    const std::size_t nsites = u.size();

    if (nu_cache.size() != nsites) {
        nu_cache.resize(nsites);
        nv_cache.resize(nsites);
        interu_cache.resize(nsites);
        interv_cache.resize(nsites);
    }

    double basis_u[4];
    double basis_v[4];

    // Compute the normalized evaluation values and their interval numbers
    normalize_with_inter(umin, umax, nptsu, u.data(), nsites, nu_cache.data(), interu_cache.data());
    normalize_with_inter(vmin, vmax, nptsv, v.data(), nsites, nv_cache.data(), interv_cache.data());

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(nsites);

    for (std::size_t k = 0; k < nsites; ++k) {
        if (interu_cache[k] < 0 || interu_cache[k] > nptsu - 4 || interv_cache[k] < 0 || interv_cache[k] > nptsv - 4)
            throw std::runtime_error("Colocation site out of range");

        eval_basis(nu_cache[k], basis_u);
        eval_basis(nv_cache[k], basis_v);
        for (int iu = 0; iu < 4; ++iu)
            for (int iv = 0; iv < 4; ++iv) {
                int col = (iu + interu_cache[k]) * nptsv + iv + interv_cache[k];
                tripletList.emplace_back(k, col, basis_u[iu] * basis_v[iv]);
            }
    }

    Eigen::SparseMatrix<double> mat(nrRows, nrCols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

Eigen::SparseMatrix<double> BBS::bending_ur(const Eigen::MatrixXd &lambdas) const {

    int ny = nptsu; // Order of the dimensions are switched due to a previous version of the code
    int nx = nptsv;

    double sy = (umax - umin) / (nptsu - 3);
    double sx = (vmax - vmin) / (nptsv - 3);

    int column = 0;

    std::vector<Eigen::Triplet<double>> tripletList;

    /* Location of the non-zeros coefficients of the matrix. */
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {

            for (int b = std::max(j-3,0) ; b <= j-1 ; ++b)
                for (int a = std::max(0,i-3) ; a <= std::min(nx-1,i+3) ; ++a)
                    tripletList.emplace_back(b*nx + a, column, 0);

            for (int a = std::max(0,i-3); a <= i; ++a)
                tripletList.emplace_back(j*nx + a, column, 0);

            column++;
        }
    }

    // Build the coefficients of the B matrix with the scales taken into account
    const Eigen::Matrix<double, 16, 16> _coeff_b = sy * ____bxx_coeff / (sx * sx * sx)
                                                 +      ____bxy_coeff / (sx*sy)
                                                 + sx * ____byy_coeff / (sy * sy * sy);

    Eigen::SparseMatrix<double> mat(nptsu * nptsv, nptsu * nptsv);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    // Put the right coefficients at the right locations (one knot domain at a time
    // and with the scaling given by lambda)
    for (int b = 0; b < ny-3 ; ++b) {
        for (int a = 0; a < nx-3; ++a) {
            double lbd = lambdas(b,a);
            for (int c = 0; c < 16; ++c) {
                for (int d = c; d < 16; ++d) {
                    int i = (b+c/4)*nx+a+c%4;
                    int j = (b+d/4)*nx+a+d%4;

                    // See (notebook 2, p. 61) for the tricky formulas
                    int nb = i/nx - std::max(j/nx-3,0);
                    int sb = std::min( std::min(4+(j%nx), 3+nx-(j%nx)) , std::min(nx, 7) );

                    int rowOffset = nb*sb + (i%nx) - std::max((j%nx)-3, 0);

                    // Access nonzero element rowOffset on column j
                    mat.valuePtr()[mat.outerIndexPtr()[j]+rowOffset] += lbd*_coeff_b(d,c);
                }
            }
        }
    }

    return mat;

}

Eigen::SparseMatrix<double> BBS::bending(const Eigen::MatrixXd &lambdas) const {
    Eigen::SparseMatrix<double> bend_ur = bending_ur(lambdas);
    return bend_ur.selfadjointView<Eigen::Upper>();
}


double BBS::get_deriv_fact(int uorder, int vorder) const {
    double su = (umax - umin) / (nptsu - 3);
    double sv = (vmax - vmin) / (nptsv - 3);

    double a = std::pow(su, uorder);
    double b = std::pow(sv, vorder);
    double p = a * b;
    
    return 1.0 / p;
}

void BBS::eval_int(const double *ctrlpts,
              const double *nu, const double *nv, const int *interu, const int *interv,
              int nval, double *val, int du, int dv) const {

    #pragma omp parallel for num_threads(NTHREADS), schedule(guided)
    for (int k = 0; k < nval; ++k) {
        int iu = 0, iv = 0, d = 0, ind = 0;
        double basis_u[4];
        double basis_v[4];
        double bu = 0, bas = 0;
        basis_func_t b_func_u = get_basis_ptr(du);
        basis_func_t b_func_v = get_basis_ptr(dv);
        double fact = get_deriv_fact(du, dv);

        b_func_u(nu[k], basis_u);
        b_func_v(nv[k], basis_v);

        for (d = 0; d < valdim; ++d)
            val[valdim*k+d] = 0.0;

        for (iu = 0; iu < 4; ++iu) {
            bu = basis_u[iu];
            for (iv = 0; iv < 4; ++iv) {
                bas = bu * basis_v[iv];
                ind = valdim * ((iu+interu[k])*nptsv + iv+interv[k]);
                for (d = 0; d < valdim; ++d)
                    val[valdim*k+d] += ctrlpts[ind++] * bas;
            }
        }

        for (d = 0; d < valdim; ++d)
            val[valdim*k+d] *= fact;
    }
}

Eigen::MatrixXd BBS::eval(const Eigen::MatrixXd &ctrlpts,
                          const Eigen::VectorXd &u, const Eigen::VectorXd &v,
                          int du, int dv) const {

    if (u.size() != v.size())
        throw std::invalid_argument("u and v are not the same size");

    std::size_t nval = u.size();

    Eigen::MatrixXd ret(valdim, nval);

    if (nu_cache.size() != nval) {
        nu_cache.resize(nval);
        nv_cache.resize(nval);
        interu_cache.resize(nval);
        interv_cache.resize(nval);
    }

    // Compute the normalized evaluation values and their interval numbers
    normalize_with_inter(umin, umax, nptsu, u.data(), nval, nu_cache.data(), interu_cache.data());
    normalize_with_inter(vmin, vmax, nptsv, v.data(), nval, nv_cache.data(), interv_cache.data());

    eval_int(ctrlpts.data(), nu_cache.data(), nv_cache.data(), interu_cache.data(), interv_cache.data(), nval, ret.data(), du, dv);

    return ret;

}

Eigen::MatrixXd BBS::eval(const Eigen::MatrixXd &ctrlpts, int du, int dv) const {

    Eigen::MatrixXd ret(valdim, nu_cache.size());

    eval_int(ctrlpts.data(), nu_cache.data(), nv_cache.data(), interu_cache.data(), interv_cache.data(), nu_cache.size(), ret.data(), du, dv);

    return ret;
}



Eigen::SparseMatrix<double> BBS::coloc_deriv_int(const double *nu, const double *nv, const int *interu, const int *interv,
                                                 const int du, const int dv) const {

    const int nrCols = nptsu * nptsv;
    const int nsites = nu_cache.size();
    const int nrRows = nsites;

    basis_func_t b_func_u = get_basis_ptr(du);
    basis_func_t b_func_v = get_basis_ptr(dv);

    double fact = get_deriv_fact(du, dv);
    double basis_u[4];
    double basis_v[4];

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(nsites);

    for (int k = 0; k < nsites; ++k) {
        b_func_u(nu[k], basis_u);
        b_func_v(nv[k], basis_v);
        for (int iu = 0; iu < 4; ++iu)
            for (int iv = 0; iv < 4; ++iv) {
                int col = (iu+interu[k])*nptsv + iv + interv[k];
                tripletList.emplace_back(k, col, fact * basis_u[iu] * basis_v[iv]);
            }
    }

    Eigen::SparseMatrix<double> mat(nrRows, nrCols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

Eigen::SparseMatrix<double> BBS::coloc_deriv(const Eigen::VectorXd &u,
                                             const Eigen::VectorXd &v,
                                             const int du, const int dv) const {

    if (u.size() != v.size())
        throw std::invalid_argument("u and v are not the same size");

    const std::size_t nrRows = u.size();

    std::size_t nsites = nrRows;

    if (nu_cache.size() != nsites) {
        nu_cache.resize(nsites);
        nv_cache.resize(nsites);
        interu_cache.resize(nsites);
        interv_cache.resize(nsites);
    }

    // Compute the normalized evaluation values and their interval numbers
    normalize_with_inter(umin, umax, nptsu, u.data(), nsites, nu_cache.data(), interu_cache.data());
    normalize_with_inter(vmin, vmax, nptsv, v.data(), nsites, nv_cache.data(), interv_cache.data());

    return coloc_deriv_int(nu_cache.data(), nv_cache.data(), interu_cache.data(), interv_cache.data(), du, dv);
}

Eigen::SparseMatrix<double> BBS::coloc_deriv(const int du, const int dv) const {
    return coloc_deriv_int(nu_cache.data(), nv_cache.data(), interu_cache.data(), interv_cache.data(), du, dv);
}

Eigen::MatrixXd BBS::getCtrlpts(const Eigen::SparseMatrix<double> &coloc,
                                const Eigen::SparseMatrix<double> &bending,
                                const Eigen::MatrixXd &points) {

    Eigen::SparseMatrix<double> A = (coloc.transpose() * coloc + bending);

    Eigen::MatrixXd B = coloc.transpose() * points.transpose();

    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> solver;

    solver.compute(A);
    Eigen::MatrixXd cpts = solver.solve(B);


    return cpts.transpose();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> BBS::meshgrid(int count) const {
    auto linspaceu = Eigen::RowVectorXd::LinSpaced(count, umin, umax);
    auto linspacev = Eigen::VectorXd::LinSpaced(count, vmin, vmax);

    Eigen::MatrixXd xv = linspaceu.replicate(linspacev.size(),1);
    Eigen::MatrixXd yv = linspacev.replicate(1,linspaceu.size());

    return std::make_tuple(xv, yv);
}

BBS::basis_func_t BBS::get_basis_ptr(int order) const {
    switch (order) {
        case 0:  return &eval_basis;
        case 1:  return &eval_basis_d;
        case 2:  return &eval_basis_dd;
        default: throw std::invalid_argument("Can evaluate B-Splines of order 0, 1 or 2 only");
    }
}

