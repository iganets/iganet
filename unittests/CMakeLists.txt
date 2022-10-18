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
  googletest  
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Compile TinySpline
include_directories("${PROJECT_SOURCE_DIR}/external/tinyspline")
add_library(tinyspline STATIC
  ${PROJECT_SOURCE_DIR}/external/tinyspline/parson.c
  ${PROJECT_SOURCE_DIR}/external/tinyspline/tinyspline.c
  ${PROJECT_SOURCE_DIR}/external/tinyspline/tinysplinecxx.cxx)

file(GLOB UNITTESTS *.cxx)

foreach(unittest ${UNITTESTS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${unittest})
  string(REPLACE ".cxx" "" name ${filename})
  add_executable(${name} ${filename})
  target_link_libraries(${name} PUBLIC "${TORCH_LIBRARIES}")
  target_link_libraries(${name} PUBLIC Matplot++::matplot)
  target_link_libraries(${name} PUBLIC tinyspline)
  target_link_libraries(${name} PUBLIC gtest_main)
  target_compile_definitions(${name} PUBLIC UNITTEST_DISABLE_SHORT_MACROS)
  add_test(${name} ${name})
endforeach()

