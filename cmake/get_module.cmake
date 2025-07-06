########################################################################
# get_module.cmake
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

########################################################################
# Get an optional module
########################################################################

include(FetchContent)

function(get_module name)

  cmake_parse_arguments(PARSE_ARGV 1 arg
    "" "GIT_REPOSITORY;GIT_TAG;URL" "")

  if(arg_URL)
    FetchContent_Declare(${name}
      URL ${arg_URL}
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/optional/${name}
    )
    
  elseif(arg_GIT_REPOSITORY)
    
    if(arg_GIT_TAG)
      FetchContent_Declare(${name}
        GIT_REPOSITORY ${arg_GIT_REPOSITORY}
        GIT_TAG ${arg_GIT_TAG}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/optional/${name}
      )
      
    else()
      FetchContent_Declare(${name}
        GIT_REPOSITORY ${arg_GIT_REPOSITORY}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/optional/${name}
      )

    endif()
    
  else()
    message(FATAL_ERROR "Either GIT_REPOSITORY or URL argument mustbe specified")
  endif()

  FetchContent_MakeAvailable(${name})
  
endfunction()
    
