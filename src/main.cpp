#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "bbs.h"
#include "warps.h"

#include <Eigen/Core>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

PYBIND11_MODULE(splines, m) {

    py::class_<BBS>(m, "BBS")
        .def(py::init<double, double, int, double, double, int, int>())
        .def("coloc", &BBS::coloc)
        .def("coloc_deriv", static_cast<Eigen::SparseMatrix<double> (BBS::*)(const Eigen::VectorXd &, const Eigen::VectorXd &, const int, const int) const>(&BBS::coloc_deriv))
        .def("coloc_deriv", static_cast<Eigen::SparseMatrix<double> (BBS::*)(int, int) const>(&BBS::coloc_deriv))
        .def("bending", &BBS::bending)
        .def("eval", static_cast<Eigen::MatrixXd (BBS::*)(const Eigen::MatrixXd &, const Eigen::VectorXd &, const Eigen::VectorXd &, int, int) const>(&BBS::eval))
        .def("eval", static_cast<Eigen::MatrixXd (BBS::*)(const Eigen::MatrixXd &, int, int) const>(&BBS::eval))
        .def("getCtrlpts", &BBS::getCtrlpts)
        .def("meshgrid", &BBS::meshgrid);


    py::class_<Jacobian>(m, "Jacobian")
        .def_readwrite("a", &Jacobian::a)
        .def_readwrite("b", &Jacobian::b)
        .def_readwrite("c", &Jacobian::c)
        .def_readwrite("d", &Jacobian::d);

    py::class_<Hessian>(m, "Hessian")
        .def_readwrite("uua", &Hessian::uua)
        .def_readwrite("uub", &Hessian::uub)
        .def_readwrite("uva", &Hessian::uva)
        .def_readwrite("uvb", &Hessian::uvb)
        .def_readwrite("vva", &Hessian::vva)
        .def_readwrite("vvb", &Hessian::vvb);

    py::class_<TwoFrameWarp>(m, "TwoFrameWarp")
        .def_readwrite("a", &TwoFrameWarp::a)
        .def_readwrite("b", &TwoFrameWarp::b)
        .def_readwrite("c", &TwoFrameWarp::c)
        .def_readwrite("d", &TwoFrameWarp::d)
        .def_readwrite("t1", &TwoFrameWarp::t1)
        .def_readwrite("t2", &TwoFrameWarp::t2);


    py::class_<Warp>(m, "Warp")
        .def(py::init<const Eigen::MatrixXd &, const std::vector<Eigen::MatrixXd> &, bool>())
        .def_static("makeGrid", &Warp::makeGrid)
        .def("getEvaluated", &Warp::getEvaluated)
        .def("getTwoFrameWarp", &Warp::getTwoFrameWarp)
        .def_readwrite("Iu", &Warp::Iu)
        .def_readwrite("Iv", &Warp::Iv)
        .def_readwrite("J21", &Warp::J21)
        .def_readwrite("J12", &Warp::J12)
        .def_readwrite("H21", &Warp::H21);



#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}



