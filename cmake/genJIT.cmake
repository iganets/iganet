########################################################################
# genJIT.cmake
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

#
# CMake function: generate just-in-time compiler configuration
#
# SOURCE_FILES may contain absolute paths or paths relative to
# CMAKE_CURRENT_SOURCE_DIR. Generated files are written to
# CMAKE_CURRENT_BINARY_DIR by default. Use OUTPUT_DIRECTORY (or OUTPUT) to
# select a different directory.
#
function(genJITCompiler SOURCE_FILES SOURCE_TARGET)

  set(options INSTALL_TREE)
  set(oneValueArgs OUTPUT OUTPUT_DIRECTORY)
  cmake_parse_arguments(JIT "${options}" "${oneValueArgs}" "" ${ARGN})

  # Valid defaults for the build-tree header; its flags contain no marker, so
  # the replacement helper returns before these values are used.
  set(JIT_INSTALL_PREFIX_DEPTH 0)
  set(JIT_INSTALL_FALLBACK_PREFIX "")

  if (JIT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "genJITCompiler received unknown arguments: ${JIT_UNPARSED_ARGUMENTS}")
  endif()

  if (JIT_OUTPUT AND JIT_OUTPUT_DIRECTORY)
    message(FATAL_ERROR
      "genJITCompiler accepts either OUTPUT or OUTPUT_DIRECTORY, not both")
  endif()

  if (JIT_OUTPUT_DIRECTORY)
    set(JIT_GENERATED_OUTPUT_DIRECTORY "${JIT_OUTPUT_DIRECTORY}")
  elseif (JIT_OUTPUT)
    set(JIT_GENERATED_OUTPUT_DIRECTORY "${JIT_OUTPUT}")
  else()
    set(JIT_GENERATED_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  if (NOT IS_ABSOLUTE "${JIT_GENERATED_OUTPUT_DIRECTORY}")
    get_filename_component(JIT_GENERATED_OUTPUT_DIRECTORY
      "${JIT_GENERATED_OUTPUT_DIRECTORY}" ABSOLUTE
      BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  file(MAKE_DIRECTORY "${JIT_GENERATED_OUTPUT_DIRECTORY}")

  #
  # Set JIT compiler command
  #
  set(JIT_CXX_COMPILER ${CMAKE_CXX_COMPILER})

  # ====================================================================

  #
  # Set JIT compiler input/output flag
  #
  if (MSVC)
    set(JIT_CXX_INCLUDE_FLAG       "/I")
    set(JIT_CXX_LINKER_FLAG        "/l")
    set(JIT_CXX_LINKER_SEARCH_FLAG "/L")
    set(JIT_CXX_OUTPUT_FLAG        "/Fo")
  else()
    set(JIT_CXX_INCLUDE_FLAG       "-I")
    set(JIT_CXX_LINKER_FLAG        "-l")
    set(JIT_CXX_LINKER_SEARCH_FLAG "-L")
    set(JIT_CXX_OUTPUT_FLAG        "-o ")
  endif()

  # ====================================================================

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

  # Set additional global compile options
  get_directory_property(JIT_COMPILE_OPTIONS COMPILE_OPTIONS)
  if (JIT_COMPILE_OPTIONS)
    foreach (flag ${JIT_COMPILE_OPTIONS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -D${flag}")
    endforeach()
  endif()

  # Set additional target-specific compile definitions and options (if available)
  if (TARGET ${SOURCE_TARGET})
    get_target_property(JIT_COMPILE_DEFINITIONS ${SOURCE_TARGET} INTERFACE_COMPILE_DEFINITIONS)
    if (JIT_COMPILE_DEFINITIONS)
      foreach (flag ${JIT_COMPILE_DEFINITIONS})
        set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -D${flag}")
      endforeach()
    endif()

    get_target_property(JIT_COMPILE_OPTIONS ${SOURCE_TARGET} INTERFACE_COMPILE_OPTIONS)
    if (JIT_COMPILE_OPTIONS)
      foreach (flag ${JIT_COMPILE_OPTIONS})
        set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${flag}")
      endforeach()
    endif()
  endif()

  # Set Torch-specific compile flags
  if (TORCH_CXX_FLAGS)
    foreach (flag ${TORCH_CXX_FLAGS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${flag}")
    endforeach()
  endif()

  # Set OpenMP-specific compiler flags
  if (OpenMP_CXX_FLAGS)
    foreach (flag ${OpenMP_CXX_FLAGS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${flag}")
    endforeach()
  endif()

  # ====================================================================

  # Set SYSROOT on MacOS
  if (APPLE AND CMAKE_OSX_SYSROOT)
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
  endif()

  # ====================================================================

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

  # ====================================================================

  if(JIT_INSTALL_TREE)
    # The install-tree header must never retain source, build, or _deps paths.
    # Keep IgANet-owned paths relocatable.  The generated header replaces this
    # marker with the prefix derived from its own installed location.
    set(JIT_INSTALL_PREFIX_MARKER "__IGANET_INSTALL_PREFIX__")
    set(JIT_INSTALL_INCLUDE_DIRECTORIES
      "${JIT_INSTALL_PREFIX_MARKER}/${CMAKE_INSTALL_INCLUDEDIR}"
      "${JIT_INSTALL_PREFIX_MARKER}/${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    file(TO_CMAKE_PATH "${IGANET_INSTALL_INCLUDEDIR}" JIT_INSTALL_INCLUDEDIR_NORMALIZED)
    string(REPLACE "/" ";" JIT_INSTALL_INCLUDEDIR_PARTS
      "${JIT_INSTALL_INCLUDEDIR_NORMALIZED}")
    list(LENGTH JIT_INSTALL_INCLUDEDIR_PARTS JIT_INSTALL_PREFIX_DEPTH)
    math(EXPR JIT_INSTALL_PREFIX_DEPTH "${JIT_INSTALL_PREFIX_DEPTH} + 2")
    set(JIT_INSTALL_FALLBACK_PREFIX "${CMAKE_INSTALL_PREFIX}")
    # Preserve additional public include directories advertised specifically
    # for installed consumers (for example G+Smo's include/gismo directory).
    # Relative INSTALL_INTERFACE paths are relative to the install prefix.
    if(TARGET ${SOURCE_TARGET})
      get_target_property(IGANET_INCLUDE_DIRECTORIES ${SOURCE_TARGET}
        INTERFACE_INCLUDE_DIRECTORIES)
      if(IGANET_INCLUDE_DIRECTORIES)
        foreach(dir IN LISTS IGANET_INCLUDE_DIRECTORIES)
          if(NOT dir MATCHES "^\\$<INSTALL_INTERFACE:(.*)>$")
            continue()
          endif()
          set(dir "${CMAKE_MATCH_1}")
          if(NOT IS_ABSOLUTE "${dir}")
            set(dir "${JIT_INSTALL_PREFIX_MARKER}/${dir}")
          endif()
          list(APPEND JIT_INSTALL_INCLUDE_DIRECTORIES "${dir}")
        endforeach()
      endif()
    endif()
    list(REMOVE_DUPLICATES JIT_INSTALL_INCLUDE_DIRECTORIES)
    foreach(dir IN LISTS JIT_INSTALL_INCLUDE_DIRECTORIES)
      set(JIT_INCLUDE_DIRECTORIES
        "${JIT_INCLUDE_DIRECTORIES} ${JIT_CXX_INCLUDE_FLAG}${dir}")
    endforeach()
  else()
    # Generate list of global include directories
    get_property(IGANET_INCLUDE_DIRECTORIES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
    if(IGANET_INCLUDE_DIRECTORIES)
      string(REPLACE ";" " ${JIT_CXX_INCLUDE_FLAG}"
        JIT_INCLUDE_DIRECTORIES
        "${JIT_CXX_INCLUDE_FLAG}${IGANET_INCLUDE_DIRECTORIES}")
    endif()

    # Generate list of target-specific build-tree include directories.
    if (TARGET ${SOURCE_TARGET})
      get_target_property(IGANET_INCLUDE_DIRECTORIES ${SOURCE_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
      if (IGANET_INCLUDE_DIRECTORIES)
        foreach (dir ${IGANET_INCLUDE_DIRECTORIES})
          if(dir MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
            set(dir "${CMAKE_MATCH_1}")
          elseif(dir MATCHES "^\\$<INSTALL_INTERFACE:.*>$")
            continue()
          endif()
          set(JIT_INCLUDE_DIRECTORIES
            "${JIT_INCLUDE_DIRECTORIES} ${JIT_CXX_INCLUDE_FLAG}${dir}")
        endforeach()
      endif()
    endif()
  endif()

  # Add Torch-specific include directories
  if (TORCH_INCLUDE_DIRS)
    foreach (dir ${TORCH_INCLUDE_DIRS})
      set (JIT_INCLUDE_DIRECTORIES
        "${JIT_INCLUDE_DIRECTORIES} ${JIT_CXX_INCLUDE_FLAG}${dir}")
    endforeach()
  endif()

  # ====================================================================

  # Generate list of global external libraries
  get_property(IGANET_LINK_DIRECTORIES DIRECTORY PROPERTY LINK_DIRECTORIES)
  if(IGANET_LINK_DIRECTORIES)
    string(REPLACE ";" " ${JIT_CXX_LINKER_SEARCH_FLAG}"
      JIT_LIBRARIES
      "${JIT_CXX_LINKER_SEARCH_FLAG}${IGANET_LINK_DIRECTORIES}")
  endif()

  if(JIT_INSTALL_TREE)
    set(JIT_LIBRARIES
      "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${JIT_INSTALL_PREFIX_MARKER}/${CMAKE_INSTALL_LIBDIR}/iganet")
  endif()

  # Generate list of target-specific external libraries
  if (TARGET ${SOURCE_TARGET})
    get_target_property(IGANET_LINK_LIBRARIES ${SOURCE_TARGET} INTERFACE_LINK_DIRECTORIES)
    if (IGANET_LINK_LIBRARIES)
      foreach (lib ${IGANET_LINK_LIBRARIES})
        set (JIT_LIBRARIES
          "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${lib}")
      endforeach()
    endif()
  endif()

  # Generate list of target-specific external libraries
  if (TARGET ${SOURCE_TARGET})
    get_target_property(IGANET_LINK_LIBRARIES ${SOURCE_TARGET} INTERFACE_LINK_LIBRARIES)
    if (IGANET_LINK_LIBRARIES)

      # Generate include and link directories
      foreach (lib ${IGANET_LINK_LIBRARIES})

        # Select the usage requirement for the tree being generated.  This is
        # intentionally done here because configure_file cannot evaluate
        # BUILD_INTERFACE/INSTALL_INTERFACE generator expressions.
        if(lib MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
          if(JIT_INSTALL_TREE)
            continue()
          endif()
          set(lib "${CMAKE_MATCH_1}")
        elseif(lib MATCHES "^\\$<INSTALL_INTERFACE:(.*)>$")
          if(NOT JIT_INSTALL_TREE)
            continue()
          endif()
          set(lib "${CMAKE_MATCH_1}")
        endif()

        if(lib STREQUAL "gismo_static" OR lib STREQUAL "iganet::gismo_static")
          # Link the static target explicitly.  Using -lgismo is ambiguous
          # when G+Smo also builds a shared library and can leave a JIT library
          # with an unresolved runtime dependency on libgismo.
          if(JIT_INSTALL_TREE)
            set(JIT_GISMO_LIBRARY
              "${JIT_INSTALL_PREFIX_MARKER}/${CMAKE_INSTALL_LIBDIR}/iganet/${CMAKE_STATIC_LIBRARY_PREFIX}gismo${CMAKE_STATIC_LIBRARY_SUFFIX}")
          else()
            set(JIT_GISMO_LIBRARY
              "${PROJECT_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gismo${CMAKE_STATIC_LIBRARY_SUFFIX}")
          endif()
          set(JIT_LIBRARIES
            "${JIT_LIBRARIES} ${JIT_GISMO_LIBRARY}")

        elseif(lib STREQUAL "pugixml" OR lib STREQUAL "iganet::pugixml")
          if(NOT JIT_INSTALL_TREE)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${pugixml_BINARY_DIR}")
          endif()

          list(APPEND LIBS pugixml)

        elseif (lib STREQUAL "torch")
          if (WIN32)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}\\..\\..\\..\\lib")
          else()
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}/../../../lib")
          endif()

          list(APPEND LIBS torch)

        elseif(lib STREQUAL "torch_library")
          if (Torch_CUDA_FOUND)

            if (WIN32)
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}\\..\\..\\..\\lib")
            else()
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}/../../../lib")
            endif()

            list(APPEND LIBS torch_cuda)

          else()

            if (WIN32)
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}\\..\\..\\..\\lib")
            else()
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}/../../../lib")
            endif()

            list(APPEND LIBS torch_cpu)
          endif()

        elseif(lib STREQUAL "Matplot++::matplot")
          set(JIT_LIBRARIES
            "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${matplotplusplus_BINARY_DIR}/source/matplot")

          list(APPEND LIBS matplot)

        elseif(lib STREQUAL "OpenMP::OpenMP_CXX")

          if (OpenMP_CXX_INCLUDE_DIR)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_INCLUDE_FLAG}${OpenMP_CXX_INCLUDE_DIR}")
          endif()

          foreach (libname ${OpenMP_CXX_LIB_NAMES})
            list(APPEND LIBS ${OpenMP_${libname}_LIBRARY})
          endforeach()

        else()

          list(APPEND LIBS ${lib})

        endif()

      endforeach()

      # Generate linking directives
      foreach(lib ${LIBS})

        if (IS_ABSOLUTE ${lib})

          if (lib STREQUAL "pugixml" AND UNIX AND NOT APPLE)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive")
          else()
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${lib}")
          endif()

        else()

          if (lib STREQUAL "pugixml" AND UNIX AND NOT APPLE)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} -Wl,--whole-archive ${JIT_CXX_LINKER_FLAG}${lib} -Wl,--no-whole-archive")
          else()
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_LINKER_FLAG}${lib}")
          endif()
        endif()

      endforeach()
    endif()
  endif()

  # ====================================================================

  set(JIT_CONFIGURED_FLAGS
    "${JIT_CXX_FLAGS} ${JIT_INCLUDE_DIRECTORIES} ${JIT_LIBRARIES}")
  set(JIT_ALTERNATE_FLAGS "")
  if(JIT_INSTALL_TREE)
    set(JIT_RUNTIME_INSTALL_FLAGS "${JIT_CONFIGURED_FLAGS}" PARENT_SCOPE)
  elseif(JIT_RUNTIME_INSTALL_FLAGS)
    set(JIT_ALTERNATE_FLAGS "${JIT_RUNTIME_INSTALL_FLAGS}")
  endif()

  # ====================================================================

  # Generate source files
  foreach (input_file IN LISTS SOURCE_FILES)
    if (IS_ABSOLUTE "${input_file}")
      set(input_path "${input_file}")
    else()
      get_filename_component(input_path "${input_file}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    if (NOT EXISTS "${input_path}")
      message(FATAL_ERROR "JIT source file does not exist: ${input_path}")
    endif()

    get_filename_component(output_file "${input_path}" NAME_WLE)
    configure_file("${input_path}"
      "${JIT_GENERATED_OUTPUT_DIRECTORY}/${output_file}" @ONLY)
  endforeach()

endfunction()
