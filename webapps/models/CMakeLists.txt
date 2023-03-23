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
# Modles
########################################################################
file(GLOB MODELS *.cxx)
foreach(model ${MODELS})
  file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${model})
  string(REPLACE ".cxx" "" name ${filename})
  add_library(${name} SHARED ${filename})
  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
endforeach()
