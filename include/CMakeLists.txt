########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

target_include_directories(iganet_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(iganet_core INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

set(IGANET_WITH_CUDA OFF)
set(IGANET_WITH_HIP OFF)

if(DEFINED CUDA_VERSION AND NOT "${CUDA_VERSION}" STREQUAL "")
  set(IGANET_WITH_CUDA ON)
endif()

if(DEFINED HIP_VERSION AND NOT "${HIP_VERSION}" STREQUAL "")
  set(IGANET_WITH_HIP ON)
endif()  

########################################################################
# Config file
########################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/core/config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/core/config.hpp" @ONLY)

########################################################################
# JIT compiler
########################################################################
# Reconfigure when a public header changes and fingerprint their contents so
# cached JIT libraries cannot outlive the header implementation they use.
file(GLOB_RECURSE IGANET_JIT_CACHE_HEADERS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/*.h"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/include/*.hpp.in")
set(JIT_CACHE_FINGERPRINT_INPUT "")
foreach(JIT_CACHE_HEADER IN LISTS IGANET_JIT_CACHE_HEADERS)
  file(SHA256 "${JIT_CACHE_HEADER}" JIT_CACHE_HEADER_HASH)
  string(APPEND JIT_CACHE_FINGERPRINT_INPUT "${JIT_CACHE_HEADER_HASH}")
endforeach()
string(SHA256 JIT_CACHE_FINGERPRINT "${JIT_CACHE_FINGERPRINT_INPUT}")

include("${PROJECT_SOURCE_DIR}/cmake/genJIT.cmake")
genJITCompiler("${CMAKE_CURRENT_SOURCE_DIR}/utils/jit.hpp.in" iganet::core
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/utils")

########################################################################
# Precompiled headers
########################################################################
if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
  file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_BINARY_DIR}/*.hpp)

  add_executable(iganet_pch .iganet_pch.cxx iganet.h ${HEADERS})
  target_link_libraries(iganet_pch PRIVATE iganet::core)
  target_precompile_headers(iganet_pch PRIVATE iganet.h PRIVATE ${HEADERS})
endif()

########################################################################
# Install header files
########################################################################
include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)
