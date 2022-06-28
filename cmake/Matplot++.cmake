########################################################################
# Matplot++.cmake
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
# Matplot++
########################################################################

include(FetchContent)
FetchContent_Declare(
  matplotplusplus
  URL https://github.com/alandefreitas/matplotplusplus/archive/refs/tags/v1.1.0.zip
  )

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
  message(STATUS "Fetching Matplot++")
  FetchContent_Populate(matplotplusplus)
  message(STATUS "Fetching Matplot++ -- done")
endif()

set(MATPLOTPLUSPLUS_BINARY_DIR ${matplotplusplus_BINARY_DIR})
set(MATPLOTPLUSPLUS_SOURCE_DIR ${matplotplusplus_SOURCE_DIR})

set(BUILD_EXAMPLES 0 CACHE BOOL "")
set(BUILD_TESTING  0 CACHE BOOL "")
set(BUILD_TESTS    0 CACHE BOOL "")

# Add include directory
include_directories("${MATPLOTPLUSPLUS_SOURCE_DIR}/source/3rd_party/cimg")
include_directories("${MATPLOTPLUSPLUS_SOURCE_DIR}/source/3rd_party/nodesoup/include")
include_directories("${MATPLOTPLUSPLUS_SOURCE_DIR}/source/matplot")

# Process Matplot++ project
if (NOT TARGET matplot)
  add_subdirectory(${MATPLOTPLUSPLUS_SOURCE_DIR} ${MATPLOTPLUSPLUS_BINARY_DIR})
endif()

# Get compile definitions
get_target_property(MATPLOTPLUSPLUS_COMPILE_DEFINITIONS matplot COMPILE_DEFINITIONS)

# Add include directory
include_directories("${MATPLOTPLUSPLUS_SOURCE_DIR}/source/matplot")

# Add link directory
link_directories("${MATPLOTPLUSPLUS_BINARY_DIR}/source/3rd_party")
link_directories("${MATPLOTPLUSPLUS_BINARY_DIR}/source/matplot")

# Fix error handling on MacOSX using Clang compiler
if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_compile_options(matplot PRIVATE -Wno-unused-but-set-variable)
endif()
