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

########################################################################
# Config file
########################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/config.hpp" @ONLY)

########################################################################
# Precompiled headers
########################################################################
if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
  file(GLOB HEADERS *.hpp ${CMAKE_CURRENT_BINARY_DIR}/*.hpp)

  add_executable(iganet_pch .iganet_pch.cxx iganet.h ${HEADERS})
  target_link_libraries(iganet_pch PRIVATE iganet::core)
  target_precompile_headers(iganet_pch PRIVATE iganet.h PRIVATE ${HEADERS})
endif()

########################################################################
# Install header files
########################################################################
file(GLOB SOURCES ${CMAKE_CURRENT_BINARY_DIR}/*.hpp ${CMAKE_CURRENT_BINARY_DIR}/*.h)
install(FILES ${SOURCES} DESTINATION include/iganet)

file(GLOB SOURCES *.hpp *.h "${CMAKE_CURRENT_BINARY_DIR}/*.hpp")
install(FILES ${SOURCES} DESTINATION include/iganet)

file(GLOB SOURCES utils/*.hpp utils/*.h)
install(FILES ${SOURCES} DESTINATION include/iganet/utils)
