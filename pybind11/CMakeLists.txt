########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2022 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

# Define remote repository
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/heads/master.zip
  )

FetchContent_MakeAvailable(pybind11)

########################################################################
# Options
########################################################################

if(NOT PYIGANET_COEFF_TYPE)
  set(PYIGANET_COEFF_TYPE double CACHE STRING
    "Coefficient type(float, double)" FORCE)
endif()
set_property(CACHE PYIGANET_COEFF_TYPE PROPERTY STRINGS "float" "double")

message("Options:")
message("PYIGANET_COEFF_TYPE................: ${PYIGANET_COEFF_TYPE}")

########################################################################
# Auto-generate source files
########################################################################

# Config
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyconfig.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/pyconfig.hpp)

# Create list of PyBind11 init subroutines
list(APPEND inits "init_core")

# Explicit instantiation of templated classes
foreach(GEODIM RANGE 1 4)
  foreach(PARDIM IN ITEMS
      "1" "1,1" "1,1,1" "1,1,1,1"
      "2" "2,2" "2,2,2" "2,2,2,2"
      "3" "3,3" "3,3,3" "3,3,3,3"
      "4" "4,4" "4,4,4" "4,4,4,4"
      "5" "5,5" "5,5,5" "5,5,5,5")

    set(SUFFIX "${GEODIM}d_${PARDIM}")
    string(REPLACE "," "_" SUFFIX ${SUFFIX})
    #list(APPEND inits "init_UniformBSpline${SUFFIX}")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pybspline.cxx.in
      ${CMAKE_CURRENT_BINARY_DIR}/pybspline${SUFFIX}.cxx)
  endforeach()
endforeach()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
  "/**
   @file pybind11/pyiganet.cxx

   @brief Pybind11 IgANet

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <pybind11/pybind11.h>

#include <pyconfig.hpp>

namespace py = pybind11;")

foreach(init IN LISTS inits)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
    "void ${init}(py::module_ &);\n")
endforeach()

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
  "PYBIND11_MODULE(pyiganet, m) {\n")

foreach(init IN LISTS inits)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
    "${init}(m);\n")
endforeach()

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
    "}\n")

# Include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx ${CMAKE_CURRENT_BINARY_DIR}/*.cxx)

pybind11_add_module(pyiganet MODULE ${SRCS})
target_link_libraries(pyiganet PUBLIC "${TORCH_LIBRARIES}")
  
if (IGANET_WITH_MATPLOT)
  target_link_libraries(pyiganet PUBLIC Matplot++::matplot)
endif()
