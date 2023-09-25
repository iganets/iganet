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

# Set a default coefficient numeric types if not specified
if(NOT IGANET_WEBAPPS_COEFF_TYPE)
  set (IGANET_WEBAPPS_COEFF_TYPE "float" CACHE STRING
    "Coefficient type(float, double)" FORCE)
endif()
set_property(CACHE IGANET_WEBAPPS_COEFF_TYPE PROPERTY STRINGS
"float" "double")

########################################################################
# Option list
########################################################################
option(IGANET_WITH_OPENSSL      "With OpenSSL suppoer"      OFF)
option(IGANET_WITH_LIBUV        "With LibUV support"        OFF)

########################################################################
# Prerequisites
########################################################################
include(popl)
include(uwebsockets)

########################################################################
# Config file
########################################################################
configure_file(${PROJECT_SOURCE_DIR}/webapps/config.hpp.in
  ${PROJECT_BINARY_DIR}/webapps/config.hpp @ONLY)

install(FILES ${PROJECT_BINARY_DIR}/webapps/config.hpp
  DESTINATION include/iganet/webapp)

########################################################################
# Include directories
########################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

########################################################################
# Compression library
########################################################################
find_package(ZLIB REQUIRED)

########################################################################
# WebApps
########################################################################

file(GLOB WEBAPPS *.cxx)

foreach(webapp ${WEBAPPS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${webapp})

  if (CMAKE_CUDA_COMPILER)
    set_source_files_properties(${filename} PROPERTIES LANGUAGE CUDA)
  endif()

  string(REPLACE ".cxx" "" name ${filename})
  add_executable(${name} ${filename})

  if(NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${ZLIB_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${CMAKE_DL_LIBS})
  target_link_libraries(${name} PUBLIC usockets)
  target_link_libraries(${name} PUBLIC pugixml)

  install(TARGETS ${name} DESTINATION bin)
endforeach()

install(FILES server.cfg DESTINATION etc)

file(GLOB SOURCES *.hpp)
install(FILES ${SOURCES} DESTINATION include/iganet/webapp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models)
