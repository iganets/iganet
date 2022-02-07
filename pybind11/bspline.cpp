
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <bspline.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pyiganet, m) {
  
  py::class_<iganet::UniformBSpline<double,1,1>>(m, "UniformBSpline1d")    
    .def(py::init< std::array<int64_t,1> >())
    
    .def_static("parDim", &iganet::UniformBSpline<double,1,1>::parDim)
    .def_static("geoDim", &iganet::UniformBSpline<double,1,1>::geoDim)
    .def_static("degrees", &iganet::UniformBSpline<double,1,1>::degrees)
    .def_static("degree", &iganet::UniformBSpline<double,1,1>::degree)

    .def("knots", static_cast
         <std::array<torch::Tensor,1>&
         (iganet::UniformBSpline<double,1,1>::*)
         ()
         >(&iganet::UniformBSpline<double,1,1>::knots)
         )
    ;
}
