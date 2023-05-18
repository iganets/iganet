########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2022 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

########################################################################
# Force CMake version 3.24 or above
########################################################################
cmake_minimum_required (VERSION 3.24)

# Define remote repositoies
include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest  
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# SplineLib
FetchContent_Declare(
  splinelib
  URL https://github.com/tataratat/SplineLib/archive/refs/heads/tataratat.zip
  FIND_PACKAGE_ARGS
  )
set(SPLINELIB_BUILD_PUGIXML OFF CACHE BOOL "" FORCE)
set(pugixml_DIR ${pugixml_BINARY_DIR})
FetchContent_MakeAvailable(googletest splinelib)

# Collect unittest sources
file(GLOB UNITTESTS *.cxx)

foreach(unittest ${UNITTESTS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${unittest})

  if (CMAKE_CUDA_COMPILER)
    set_source_files_properties(${filename} PROPERTIES LANGUAGE CUDA)
  endif()
  
  string(REPLACE ".cxx" "" name ${filename})
  add_executable(${name} ${filename}) 
  target_link_libraries(${name} PUBLIC "${TORCH_LIBRARIES}")

  # SplineLib
  target_link_libraries(${name} PUBLIC pugixml)
  target_link_libraries(${name} PUBLIC splines)

  # Google test
  target_link_libraries(${name} PUBLIC gtest_main)
  
  if (IGANET_WITH_MATPLOT)
    target_link_libraries(${name} PUBLIC Matplot++::matplot)
  endif()
  target_compile_definitions(${name} PUBLIC UNITTEST_DISABLE_SHORT_MACROS)
  add_test(${name} ${name})
endforeach()

