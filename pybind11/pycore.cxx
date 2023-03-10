/**
   @file pybind11/pycore.cxx

   @brief Pybind11 core components

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <pyconfig.hpp>

#include <core.hpp>

namespace py = pybind11;

void init_core(py::module_ &m) {
  py::class_<iganet::core<iganet::pybind11::real_t>>(m, "Core")
    
    // Constructors
    .def(py::init<>())
    .def(py::init<c10::DeviceType>())
    .def(py::init<bool>())
    .def(py::init<c10::DeviceType, bool>())
    
    // Member functions
    .def("options", &iganet::core<iganet::pybind11::real_t>::options)
    .def("to_json", &iganet::core<iganet::pybind11::real_t>::to_json);
}
