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

file(GLOB UNITTESTS *.cxx)

foreach(unittest ${UNITTESTS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${unittest})
  string(REPLACE ".cxx" "" app ${filename})
  add_executable(${app} ${filename})
  target_link_libraries(${app} PUBLIC "${TORCH_LIBRARIES}")
  target_link_libraries(${app} PUBLIC Matplot++::matplot)
  target_link_libraries(${app} PUBLIC gtest_main)
  target_compile_definitions(${app} PUBLIC UNITTEST_DISABLE_SHORT_MACROS)
  set_property(TARGET ${app} PROPERTY CXX_STANDARD 17)
  add_test(${app} ${app})
endforeach()

