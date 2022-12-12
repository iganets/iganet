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

file(GLOB EXAMPLES *.cxx)

foreach(example ${EXAMPLES})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${example})
  string(REPLACE ".cxx" "" name ${filename})
  add_executable(${name} ${filename})
  target_link_libraries(${name} PUBLIC "${TORCH_LIBRARIES}")

  if (IGANET_WITH_MATPLOT)
    target_link_libraries(${name} PUBLIC Matplot++::matplot)
  endif()
endforeach()
