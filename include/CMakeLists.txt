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

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
  file(GLOB HEADERS *.hpp)

  add_executable(iganet_pch .iganet_pch.cxx iganet.h ${HEADERS})
  target_link_libraries(iganet_pch PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(iganet_pch PUBLIC pugixml)

  if (IGANET_WITH_GISMO)
    target_compile_definitions(iganet_pch PUBLIC WITH_GISMO)
    target_link_libraries(iganet_pch PUBLIC gismo)
  endif()

  if (IGANET_WITH_MATPLOT)
    target_compile_definitions(iganet_pch PUBLIC WITH_MATPLOT)
    target_link_libraries(iganet_pch PUBLIC Matplot++::matplot)
  endif()

  target_precompile_headers(iganet_pch PUBLIC iganet.h PRIVATE ${HEADERS})
endif()
