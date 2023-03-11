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
# Prerequisites
########################################################################
include(popl)
include(websockets)

########################################################################
# WebApps
########################################################################
file(GLOB WEBAPPS *.cxx)

find_package(ZLIB REQUIRED)

foreach(webapp ${WEBAPPS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${webapp})
  string(REPLACE ".cxx" "" name ${filename})
  add_executable(${name} ${filename})
  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${ZLIB_LIBRARIES})
  target_link_libraries(${name} PUBLIC usockets)
endforeach()


add_library(libtest SHARED test.cpp)
target_link_libraries(libtest PUBLIC ${TORCH_LIBRARIES})
