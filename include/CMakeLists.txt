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

target_include_directories(iganet_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set(IGANET_WITH_CUDA OFF)
set(IGANET_WITH_HIP  OFF)
set(IGANET_WITH_XPU  OFF)

if(PYTORCH_FOUND_HIP OR TARGET torch_hip)
  set(IGANET_WITH_HIP ON)
elseif(TARGET torch_cuda)
  set(IGANET_WITH_CUDA ON)
elseif(TARGET torch_xpu)
  set(IGANET_WITH_XPU ON)
endif()

########################################################################
# Config file
########################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/iganet/core/config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/iganet/core/config.hpp" @ONLY)

########################################################################
# JIT compiler
########################################################################
# Reconfigure when a public header changes and fingerprint their contents so
# cached JIT libraries cannot outlive the header implementation they use.
file(GLOB_RECURSE IGANET_JIT_CACHE_HEADERS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/iganet/*.h"
  "${PROJECT_SOURCE_DIR}/include/iganet/*.hpp"
  "${PROJECT_SOURCE_DIR}/include/iganet/*.hpp.in")
set(JIT_CACHE_FINGERPRINT_INPUT "")
foreach(JIT_CACHE_HEADER IN LISTS IGANET_JIT_CACHE_HEADERS)
  file(SHA256 "${JIT_CACHE_HEADER}" JIT_CACHE_HEADER_HASH)
  string(APPEND JIT_CACHE_FINGERPRINT_INPUT "${JIT_CACHE_HEADER_HASH}")
endforeach()
string(SHA256 JIT_CACHE_FINGERPRINT "${JIT_CACHE_FINGERPRINT_INPUT}")

include("${PROJECT_SOURCE_DIR}/cmake/genJIT.cmake")
# Generate the install configuration first so source-built executables can
# embed it as their runtime alternative.
genJITCompiler("${CMAKE_CURRENT_SOURCE_DIR}/iganet/utils/jit.hpp.in" iganet::core
  INSTALL_TREE
  OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/install/include/iganet/utils")

genJITCompiler("${CMAKE_CURRENT_SOURCE_DIR}/iganet/utils/jit.hpp.in" iganet::core
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/iganet/utils")

########################################################################
# Precompiled headers
########################################################################
if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
  file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_BINARY_DIR}/*.hpp)

  add_executable(iganet_pch .iganet_pch.cxx iganet/iganet.h ${HEADERS})
  target_link_libraries(iganet_pch PRIVATE iganet::core)
  target_precompile_headers(iganet_pch PRIVATE iganet/iganet.h PRIVATE ${HEADERS})
endif()

########################################################################
# Install header files
########################################################################
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/iganet/
  DESTINATION ${IGANET_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)

# Generated headers are not present in the source tree and must be installed
# explicitly.  Keep this list selective so CMake's build metadata is excluded.
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/iganet/core/config.hpp"
  DESTINATION "${IGANET_INSTALL_INCLUDEDIR}/core")
install(FILES
  "${PROJECT_BINARY_DIR}/install/include/iganet/utils/jit.hpp"
  DESTINATION "${IGANET_INSTALL_INCLUDEDIR}/utils")
