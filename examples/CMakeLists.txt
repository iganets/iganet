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

add_custom_target(iganet_examples)

file(GLOB EXAMPLES *.cxx)

foreach(example ${EXAMPLES})

  get_filename_component(name ${example} NAME_WE)
  add_executable(${name} ${example})
  add_dependencies(iganet_examples ${name})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_compile_options(${name} PUBLIC "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")
  target_link_libraries(${name} PRIVATE iganet::core)
  
endforeach()
