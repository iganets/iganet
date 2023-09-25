########################################################################
# genJIT.cmake
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

#
# CMake macro: generate just-in-time compiler configuration
#
# Remark: The source files must be given with relative paths
#
macro(genJITCompiler)
  
  # Set JIT compiler command
  set(JIT_CXX_COMPILER ${CMAKE_CXX_COMPILER})

  # Get build-type as upper-case string
  string(TOUPPER ${CMAKE_BUILD_TYPE} JIT_BUILD_TYPE)

  # Set JIT compiler flags (build-type dependent)
  set(JIT_CXX_FLAGS ${CMAKE_CXX_FLAGS_${JIT_BUILD_TYPE}})

  # Set additional global compile definitions
  get_directory_property(JIT_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)  
  if (JIT_COMPILE_DEFINITIONS)
    foreach (flag ${JIT_COMPILE_DEFINITIONS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -D${flag}")
    endforeach()
  endif()
  
  # Set additional target-specific compile options
  if (TARGET iganet_pch)
    get_target_property(JIT_COMPILE_OPTIONS iganet_pch COMPILE_OPTIONS)
    if (JIT_COMPILE_OPTIONS)
      foreach (flag ${JIT_COMPILE_OPTIONS})
        set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${flag}")
      endforeach()
    endif()   
  endif()
  
  # Set SYSROOT on MacOS
  if (APPLE)
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
  endif()
  
  # Create a set of shared library variable specific to C++
  # For 90% of the systems, these are the same flags as the C versions
  # so if these are not set just copy the flags from the C version
  if(NOT DEFINED CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS)
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX_FLAGS} ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
  else()
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX_FLAGS} ${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")
  endif()

  # Add C++ standard and PIC (position independent code)
  if(NOT DEFINED CMAKE_CXX_EXTENSIONS OR NOT CMAKE_CXX_EXTENSIONS)
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX${CMAKE_CXX_STANDARD}_STANDARD_COMPILE_OPTION} ${CMAKE_CXX_COMPILE_OPTIONS_PIC}")
  else()
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX${CMAKE_CXX_STANDARD}_EXTENSION_COMPILE_OPTION} ${CMAKE_CXX_COMPILE_OPTIONS_PIC}")
  endif()

  # Fix visibility
  string(REPLACE "-fvisibility=hidden"         "" JIT_CXX_FLAGS ${JIT_CXX_FLAGS})
  string(REPLACE "-fvisibility-inlines-hidden" "" JIT_CXX_FLAGS ${JIT_CXX_FLAGS})
  
  # Generate list of global include directories
  get_property(IGANET_INCLUDE_DIRECTORIES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  if(IGANET_INCLUDE_DIRECTORIES)
    string(REPLACE ";" " -I" JIT_INCLUDE_DIRECTORIES "-I${IGANET_INCLUDE_DIRECTORIES}")
    string(REPLACE ";" " /I" JIT_INCLUDE_DIRECTORIES_WIN32 "/I${IGANET_INCLUDE_DIRECTORIES}")
  endif()

  # Generate list of target-specific include directories  
  if (TARGET iganet_pch)
    get_target_property(IGANET_INCLUDE_DIRECTORIES iganet_pch INCLUDE_DIRECTORIES)
    if (IGANET_INCLUDE_DIRECTORIES)
      foreach (dir ${IGANET_INCLUDE_DIRECTORIES})
        set (JIT_INCLUDE_DIRECTORIES "${JIT_INCLUDE_DIRECTORIES} -I${dir}")
        set (JIT_INCLUDE_DIRECTORIES_WIN32 "${JIT_INCLUDE_DIRECTORIES_WIN32} /I${dir}")
      endforeach()
    endif()   
  endif()
  
  # Generate list of global external libraries
  get_property(IGANET_LINK_DIRECTORIES DIRECTORY PROPERTY LINK_DIRECTORIES)  
  if(IGANET_LINK_DIRECTORIES)
    string(REPLACE ";" " -L" JIT_LIBRARIES "-L${IGANET__LINK_DIRECTORIES}")
    string(REPLACE ";" " /L" JIT_LIBRARIES_WIN32 "/L${IGANET_LINK_DIRECTORIES}")
  endif()

  # Generate list of target-specific external libraries
  if (TARGET iganet_pch)
    get_target_property(IGANET_LINK_LIBRARIES iganet_pch LINK_DIRECTORIES)
    if (IGANET_LINK_LIBRARIES)
      foreach (lib ${IGANET_LINK_LIBRARIES})
        set (JIT_LIBRARIES "${JIT_LIBRARIES} -L${lib}")
        set (JIT_LIBRARIES_WIN32 "${JIT_LIBRARIES_WIN32} /L${lib}")
      endforeach()
    endif()
  endif()

  if (0)
    
  # Generate list of target-specific external libraries
  if (TARGET iganet_pch)
    get_target_property(IGANET_LINK_LIBRARIES iganet_pch LINK_LIBRARIES)
    if (IGANET_LINK_LIBRARIES)
      foreach (lib ${IGANET_LINK_LIBRARIES})
        if (NOT IS_ABSOLUTE ${lib})
          set(JIT_LIBRARIES "${JIT_LIBRARIES} -l${lib}")
          set(JIT_LIBRARIES_WIN32 "${JIT_LIBRARIES_WIN32} /l${lib}")
        else()
          set(JIT_LIBRARIES "${JIT_LIBRARIES} ${lib}")
          set(JIT_LIBRARIES_WIN32 "${JIT_LIBRARIES_WIN32} ${lib}")
        endif()
      endforeach()
    endif()
  endif()

  
  
  # Set (extra) link libraries
  # foreach(lib ${LIBKET_CXX_TARGET_LINK_LIBRARIES})
  #   if(lib STREQUAL "Threads::Threads")
  #     # Fix imported target Threads::Threads
  #     continue()
  #   endif()

  #   if (NOT IS_ABSOLUTE ${lib})
  #     set(JIT_LIBRARIES "${JIT_LIBRARIES} -l${lib}")
  #     set(JIT_LIBRARIES_WIN32 "${JIT_LIBRARIES_WIN32} /l${lib}")
  #   else()
  #     set(JIT_LIBRARIES "${JIT_LIBRARIES} ${lib}")
  #     set(JIT_LIBRARIES_WIN32 "${JIT_LIBRARIES_WIN32} ${lib}")
  #   endif()
  # endforeach()
endif()
  # Generate source file
  configure_file(${PROJECT_SOURCE_DIR}/webapps/models/jit.hpp.in
    ${PROJECT_BINARY_DIR}/webapps/models/jit.hpp @ONLY)
endmacro()
