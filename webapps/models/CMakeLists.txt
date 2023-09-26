########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2023 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

########################################################################
# Models
########################################################################
file(GLOB MODELS *.cxx)
foreach(model ${MODELS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${model})
  string(REPLACE ".cxx" "" name ${filename})
  add_library(${name} SHARED ${filename})
  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${name} PUBLIC pugixml)

  install(TARGETS ${name} DESTINATION models)
endforeach()

file(GLOB SOURCES *.hpp)
install(FILES ${SOURCES} DESTINATION include/iganet/webapp/models)

########################################################################
# Include directories
########################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

########################################################################
# JIT compiler
########################################################################
include("${PROJECT_SOURCE_DIR}/cmake/genJIT.cmake")
genJITCompiler("jit.hpp.in")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/jit.hpp
  DESTINATION include/iganet/webapp)
