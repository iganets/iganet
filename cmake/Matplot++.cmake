########################################################################
# Matplot++.cmake
#
# Author: Matthias Moller
# Copyright (C) 2021-2023 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

########################################################################
# Matplot++
########################################################################

include(FetchContent)
FetchContent_Declare(matplotplusplus
  URL https://github.com/alandefreitas/matplotplusplus/archive/refs/tags/v1.1.0.zip
  )

set(BUILD_EXAMPLES 0 CACHE BOOL "")
set(BUILD_TESTING  0 CACHE BOOL "")
set(BUILD_TESTS    0 CACHE BOOL "")
FetchContent_MakeAvailable(matplotplusplus)
