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
include(websockets)

########################################################################
# Config file
########################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
              "${CMAKE_CURRENT_BINARY_DIR}/config.hpp" @ONLY)

########################################################################
# WebApps
########################################################################
find_package(ZLIB REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB WEBAPPS *.cxx)
foreach(webapp ${WEBAPPS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${webapp})
  string(REPLACE ".cxx" "" name ${filename})
  add_executable(${name} ${filename})
  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${ZLIB_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${CMAKE_DL_LIBS})
  target_link_libraries(${name} PUBLIC usockets)
  target_link_libraries(${name} PUBLIC pugixml)
endforeach()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models)
