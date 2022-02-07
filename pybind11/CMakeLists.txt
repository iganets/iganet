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

pybind11_add_module(pyiganet MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/bspline.cpp)
target_link_libraries(pyiganet PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(pyiganet PUBLIC Matplot++::matplot)
