########################################################################
# gismo.cmake
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgaNet authors
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
  URL https://github.com/gismo/gismo/archive/refs/tags/v25.07.0.zip
)

set(BUILD_TESTING        OFF CACHE INTERNAL "")
set(GISMO_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(GISMO_BUILD_LIB      OFF CACHE INTERNAL "" FORCE)
set(LIB_INSTALL_DIR      "${CMAKE_INSTALL_LIBDIR}/iganet" CACHE STRING
  "Private G+Smo library installation directory" FORCE)
set(INCLUDE_INSTALL_DIR  "${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}" CACHE STRING
  "Private G+Smo header installation directory" FORCE)
set(GISMO_OPTIONAL       "gsHLBFGS;gsKLShell;gsElasticity" CACHE INTERNAL "")
set(GISMO_SHORT_TYPE     "int" CACHE INTERNAL "")
set(GISMO_WITH_OPENMP    ${IGANET_WITH_OPENMP} CACHE INTERNAL "")
set(GISMO_WITH_MPI       ${IGANET_WITH_MPI} CACHE INTERNAL "")
set(NOSNIPPETS           ON  CACHE INTERNAL "")
FetchContent_MakeAvailable(gismo)
# IgANet installs the selected target and headers into its private dependency
# layout. Suppress G+Smo's own top-level include/lib installation rules.
set_property(DIRECTORY "${gismo_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
