#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

/**
 * Class for creating and evaluating Bicubic B-Spline interpolations.
 *
 * In general, a B-Spline is parametrized by the domain (umin, umax, vmin, vmax),
 * number of points in each dimension and the number of dimensions. A direct constructor
 * that takes this parameters and a constructor which determins the domain
 * from a set of points are provided.
 *
 * In general, the algorithm to perform a BBS interpolation is:
 * - create the object
 * - find the colocation matrix for the points
 * - construct the bending matrix
 * - determine the control points
 * - evaluate the interpolation at points of interest
 *
 * Note that the functions for colocation, coloc_deriv and evaluation require
 * the points to be normalized (a preprocessing step done by @ref normalize_with_inter).
 * They use a shared cache to store the preprocessed values. All of them come in 2 versions:
 * one version that takes the point coordinates as parameter and ovewrites the cache,
 * and one of them that uses the previously set cache (except for @ref coloc, which
 * is the first function to call, so there can't be anything in the cache).
 *
 * @warn The cache makes BBS not thread safe: do not call different methods
 *       of the same object on different threads. Embarrasing parallel functions,
 *       such as eval, provide OpenMP parallelization.
 */
class BBS {

public:
    /**
     *  Create a structure containing the parameters of a bidimensional cubic
     *  B-Spline. It can mono or multi valued depending on the "valdim"
     *  parameter.
     *
     *  @note: By convention, the (u,v) coordinates are used for the input
     *  space of the spline, and the (x,y) coordinatese for the output space
     *  of the spline.
     *
     *  @param umin, umax Limits of the domain along the first input dimension
     *  @param vmin, vmax Limit of the domain along the second input dimension
     *
     *  @param nptsu, nptsv Size of the control points grid. The total
     *                      number of control points is nptsu*nptsv. The total
     *                      number of parameters is valdim*nptsu*nptsv.
     *
     *  @param valdim Dimension of the B-Spline values. "valdim=1" corresponds
     *                to a monovalued spline such as a range surface.
     *                "valdim=2" corresponds to, for example, an image warp.
     */
    BBS(double umin, double umax, int nptsu, double vmin, double vmax, int nptsv, int valdim);

    /**
     * Initialize the parameters of a spline based on the points in @ref mat.
     *
     * Simply determines the ranges of the points in @ref mat and calls the other constructor.
     *
     * @param mat
     * @param npts
     * @param valdim
     */
    BBS(const Eigen::MatrixXd &mat, int npts, int valdim, double t = 1e-3);

    /**
     * Create colocation matrix
     * @param u, v The coordinates of the points
     * @return
     */
    Eigen::SparseMatrix<double> coloc(const Eigen::VectorXd &u,
                                      const Eigen::VectorXd &v) const;

    Eigen::SparseMatrix<double> coloc_deriv(const Eigen::VectorXd &u,
                                            const Eigen::VectorXd &v,
                                            const int du, const int dv) const;

    Eigen::SparseMatrix<double> coloc_deriv(const int du, const int dv) const;

    /**
     * Compute the "bending matrix".
     *
     * Implementation based on @ref bending_ur
     *
     * @param lambdas: (nptsv-3)x(nptsu-3) matrix
     *                 lambdas[iu,iv] is the regularization parameter over the knot interval #(iu,iv).
     */
    Eigen::SparseMatrix<double> bending(const Eigen::MatrixXd &lambdas) const;


    /**
     * Evaluate the B-Spline at certain points.
     *
     * Implemented using @ref eval_int
     *
     * @param ctrlpts The control points of the B-spline (valdim x npts elements)
     * @param u, v    The points of interest
     * @param du, dv  The order of derivation in the 2 directions (0, 1 or 2)
     *
     * @return Matrix of interpolated points (valdim x npts)
     */
    Eigen::MatrixXd eval(const Eigen::MatrixXd &ctrlpts,
                         const Eigen::VectorXd &u, const Eigen::VectorXd &v,
                         int du = 0, int dv = 0) const;

    /**
     * Same as the above, but uses cached values from @ref coloc to avoid
     * recomputing the normalized u and v.
     *
     * Can be used whenever the points of evaluation are the same as
     * the input points (e.g. to determine derivatives).
     */
    Eigen::MatrixXd eval(const Eigen::MatrixXd &ctrlpts, int du = 0, int dv = 0) const;

    /**
     * Compute the control points.
     * @param coloc
     * @param bending
     * @param points
     * @return
     */
    Eigen::MatrixXd getCtrlpts(const Eigen::SparseMatrix<double> &coloc,
                               const Eigen::SparseMatrix<double> &bending,
                               const Eigen::MatrixXd &points);

    /**
     * Return a linearly spaced grid, with @ref count intervals in both
     * u and v dimensions, replicated. The extrema are umin, umax, vmin, vmax.
     *
     * @param count
     * @return
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(int count) const;

private:

    double umin;
    double umax;
    int nptsu;
    double vmin;
    double vmax;
    int nptsv;
    int valdim;


    /// Cache variables, used to store results of @ref normalize_with_inter for coloc and eval
    mutable std::vector<double> nu_cache, nv_cache;
    mutable std::vector<int> interu_cache, interv_cache;


    /**
     * Evaluate the 4 basis of the B-Spline.
     *
     * @param nx Evaluation point. Must be normalized (see @ref normalize_with_inter)
     * @param val_basis Output the 4 basis. Must have size 4.
     */
    static void eval_basis(double nx, double *val_basis);

    /**
     * Evaluate the 4 basis first derivative of the B-Spline.
     *
     * @param nx Evaluation point. Must be normalized (see @ref normalize_with_inter)
     * @param val_basis Output the 4 basis. Must have size 4.
     */
    static void eval_basis_d(double nx, double *val_basis);

    /**
     * Evaluate the 4 basis second derivative of the B-Spline.
     *
     * @param nx Evaluation point. Must be normalized (see @ref normalize_with_inter)
     * @param val_basis Output the 4 basis. Must have size 4.
     */
    static void eval_basis_dd(double nx, double *val_basis);


    /**
     * Normalize points in the output domain in discrete intervals.
     *
     * @param xmin, xmax The limits of the domain
     * @param npts       The number of intervals (look at implementation)
     * @param x          The points
     * @param nb_x       Number of points in @ref x
     * @param nx         Output offset for each point within the interval
     * @param inter      Output the interval number for each point
     */
    void normalize_with_inter(double xmin, double xmax,
                              int npts, const double *x, int nb_x, double *nx,
                              int *inter) const;




    /**
     * This function computes the upper right part of the "bending matrix".
     *
     * @param lambdas: (nptsv-3)x(nptsu-3) matrix
     *                 lambdas[iu,iv] is the regularization parameter over the knot interval #(iu,iv).
     */
    Eigen::SparseMatrix<double> bending_ur(const Eigen::MatrixXd &lambdas) const;

    /**
     * Internal implementation for evaluating the B-Spline.
     *
     * @param ctrlpts         Pointer to the control points matrix, column storage
     * @param nu, nv          (u,v) coordinates of the evaluation points
     * @param interu, interv  The intervals of the points (see @ref normalize_with_inter)
     * @param nval            The number of points
     * @param val             Output array (must be zero initialized)
     * @param du, dv          Derivation order
     */
    void eval_int(const double *ctrlpts,
                  const double *nu, const double *nv, const int *interu, const int *interv,
                  int nval, double *val, int du, int dv) const;


    Eigen::SparseMatrix<double> coloc_deriv_int(const double *nu, const double *nv,
                                                const int *interu, const int *interv,
                                                const int du, const int dv) const;

    std::vector<Eigen::Triplet<double>> colocint(const double *u, const double *v, int nsites) const;


    /**
     * Compute multiplicative factor used in some operations.
     * @param uorder, vorder Derivation order
     * @return
     */
    double get_deriv_fact(int uorder, int vorder) const;

    /// Pointer type for a basis function evaluation
    using basis_func_t = void (*)(double, double*);

    /**
     * Return a basis evaluation function
     * @param order The order of derivation. Can be 0, 1, 2.
     * @throw std::invalid_argument if the order is out of range.
     */
    basis_func_t get_basis_ptr(int order) const;
};

