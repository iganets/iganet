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

  if (CMAKE_CUDA_COMPILER)
    set_source_files_properties(${filename} PROPERTIES LANGUAGE CUDA)
  endif()

  string(REPLACE ".cxx" "" name ${filename})
  add_library(${name} SHARED ${filename})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${ZLIB_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${CMAKE_DL_LIBS})
  target_link_libraries(${name} PUBLIC pugixml)

  if (IGANET_WITH_GISMO)
    target_link_libraries(${name} PUBLIC gismo_static)
  endif()
  
  if (IGANET_WITH_OPENMP)
    target_link_libraries(${name} PUBLIC OpenMP::OpenMP_CXX)
  endif()

  install(TARGETS ${name} DESTINATION models)
endforeach()

file(GLOB SOURCES *.hpp)
install(FILES ${SOURCES} DESTINATION include/iganet/webapp/models)

########################################################################
# Include directories
########################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

########################################################################
# JIT compiler (using last target as base)
########################################################################
include("${PROJECT_SOURCE_DIR}/cmake/genJIT.cmake")
genJITCompiler("jit.hpp.in" ${name})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/jit.hpp
  DESTINATION include/iganet/webapp)
