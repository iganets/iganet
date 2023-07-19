########################################################################
# gismo.cmake
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

########################################################################
# G+Smo
########################################################################

include(FetchContent)
FetchContent_Declare(gismo
  URL https://github.com/gismo/gismo/archive/refs/heads/stable.zip
  PATCH_COMMAND patch -p1 -N -d ${PROJECT_BINARY_DIR}/_deps/gismo-src < ${PROJECT_SOURCE_DIR}/cmake/gismo.patch
  )

set(GISMO_BUILD_EXAMPLES 0 CACHE BOOL "")
set(BUILD_TESTING        0 CACHE BOOL "")
FetchContent_MakeAvailable(gismo)
include_directories(${GISMO_INCLUDE_DIRS})
