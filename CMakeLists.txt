########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2022 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# CMakeLists.txt accepts the following command line parameters
#
# IGANET_BUILD_DOCS
# IGANET_BUILD_EXAMPLES
# IGANET_BUILD_PYBIND11
# IGANET_BUILD_UNITTESTS
#
# IGANET_WITH_OPENMP
#
########################################################################

########################################################################
# Force CMake version 3.14 or above
########################################################################
cmake_minimum_required (VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

########################################################################
# Set RPATH on MacOSX
########################################################################
if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  IF ("${isSystemDir}" STREQUAL "-1")
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  ENDIF ()
endif()

########################################################################
# Use solution folders for Visual Studio
########################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # Set default build type to RelWithDebInfo
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Type of build (None Debug Release RelWithDebInfo MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "RelWithDebInfo" "MinSizeRel")
endif()

########################################################################
# We do not support in-source build
########################################################################
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not permitted.\nPlease make a separate folder for building, otherwise type \nmake\nthat will create a ./build folder and will compile in that folder. Cmake has created garbage files/dirs (you may manually remove them):\nCMakeCache.txt CMakeFiles")
endif()

########################################################################
#
# Project: IgaNet
#
########################################################################
project(IgaNet)

cmake_policy(SET CMP0135 NEW)

########################################################################
# Append path to additional modules
########################################################################
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

########################################################################
# Option list
########################################################################
option(IGANET_BUILD_DOCS        "Build documentation"       OFF)
option(IGANET_BUILD_EXAMPLES    "Build examples"            ON )
option(IGANET_BUILD_PYBIND11    "Build Python bindings"     OFF)
option(IGANET_BUILD_UNITTESTS   "Build unittests"           OFF)
option(IGANET_WITH_OPENMP       "With OpenMP"               ON )

########################################################################
# Prerequisites
########################################################################
include(Matplot++)
find_package(Torch REQUIRED)

########################################################################
# Enable OpenMP support
########################################################################
if (IGANET_WITH_OPENMP)
  # Apple explicitly disabled OpenMP support in their compilers that
  # are shipped with XCode but there is an easy workaround as
  # described at https://mac.r-project.org/openmp/
  if (CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR
      CMAKE_C_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    find_path(OpenMP_C_INCLUDE_DIR
      NAMES "omp.h" PATHS /usr/local /opt /opt/local /opt/homebrew PATH_SUFFICES include)
    find_path(OpenMP_CXX_INCLUDE_DIR
      NAMES "omp.h" PATHS /usr/local /opt /opt/local /opt/homebrew PATH_SUFFICES include)
    find_library(OpenMP_libomp_LIBRARY
      NAMES "omp" PATHS /usr/local /opt /opt/local /opt/homebrew PATH_SUFFICES lib)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xclang -fopenmp -I${OpenMP_C_INCLUDE_DIR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp -I${OpenMP_CXX_INCLUDE_DIR}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_libomp_LIBRARY}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_libomp_LIBRARY}")
  else() 
    find_package(OpenMP QUIET REQUIRED)
    list(APPEND CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
    list(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
endif()

########################################################################
# Doxygen
########################################################################
if(IGANET_BUILD_DOCS)
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR ${PROJECT_BINARY_DIR}/docs/doxygen)
    set(DOXYGEN_TAGFILE ${PROJECT_BINARY_DIR}/docs/doxygen/iganet-doxygen.tag.xml)  
    configure_file(${PROJECT_SOURCE_DIR}/docs/Doxyfile.in ${PROJECT_BINARY_DIR}/docs/doxygen/Doxyfile @ONLY)
    add_custom_target(Doxygen
      COMMAND
      ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/docs/doxygen/Doxyfile
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Generating documentation with Doxygen"
      )
    
  else()
    add_custom_target(Doxygen
      ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Please install doxygen, and run cmake . \\&\\& make Doxygen to generate the Doxygen documentation."
      )
  endif()
endif()

########################################################################
# Sphinx
########################################################################
if(IGANET_BUILD_DOCS)
  find_package(Sphinx QUIET)
  if(SPHINX_FOUND)
    set(SPHINX_SOURCE ${PROJECT_SOURCE_DIR}/docs)
    set(SPHINX_BUILD ${PROJECT_BINARY_DIR}/docs/sphinx)
    set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
    add_custom_target(Sphinx
      COMMAND
      ${SPHINX_EXECUTABLE} -b html
      -Dbreathe_projects.LibKet=${PROJECT_BINARY_DIR}/docs/doxygen/xml
      ${SPHINX_SOURCE} ${SPHINX_BUILD}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Generating documentation with Sphinx"
      )
    add_dependencies(Sphinx Doxygen)
  else()
    add_custom_target(Sphinx
      ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Please install sphinx, and run cmake . \\&\\& make Sphinx to generate the Sphinx documentation."
      )
  endif()
endif()

########################################################################
# Documentation
########################################################################
add_custom_target(docs
  DEPENDS Doxygen
  DEPENDS Sphinx)

########################################################################
# Summary
########################################################################
message("Configuration:")
message("Build type.........................: ${CMAKE_BUILD_TYPE}")
message("Build shared libraries.............: ${BUILD_SHARED_LIBS}")
message("Build directory....................: ${PROJECT_BINARY_DIR}")
message("Source directory...................: ${PROJECT_SOURCE_DIR}")
message("Install directory..................: ${CMAKE_INSTALL_PREFIX}")

message("")
message("AR command.........................: ${CMAKE_AR}")
message("RANLIB command.....................: ${CMAKE_RANLIB}")

if(CMAKE_C_COMPILER)
  message("")
  message("C compiler.........................: ${CMAKE_C_COMPILER}")
  message("C compiler flags ..................: ${CMAKE_C_FLAGS}")
  message("C compiler flags (debug)...........: ${CMAKE_C_FLAGS_DEBUG}")
  message("C compiler flags (release).........: ${CMAKE_C_FLAGS_RELEASE}")
  message("C compiler flags (release+debug)...: ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
endif()

if(CMAKE_CXX_COMPILER)
  message("")
  message("CXX compiler.......................: ${CMAKE_CXX_COMPILER}")
  message("CXX standard.......................: ${CMAKE_CXX_STANDARD}")
  message("CXX compiler flags ................: ${CMAKE_CXX_FLAGS}")
  message("CXX compiler flags (debug).........: ${CMAKE_CXX_FLAGS_DEBUG}")
  message("CXX compiler flags (release).......: ${CMAKE_CXX_FLAGS_RELEASE}")
  message("CXX compiler flags (release+debug).: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

message("")
message("EXE linker flags...................: ${CMAKE_EXE_LINKER_FLAGS}")
message("EXE linker flags (debug)...........: ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
message("EXE linker flags (release).........: ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
message("EXE linker flags (release+debug)...: ${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")

if(DOXYGEN_FOUND)
  message("")
  message("Doxygen............................: ${DOXYGEN_EXECUTABLE}")

  if(DOXYGEN_DOT_FOUND)
    message("Dot................................: ${DOXYGEN_DOT_EXECUTABLE}")
  endif()
endif()

if(SPHINX_FOUND)
  message("")
  message("Sphinx.............................: ${SPHINX_EXECUTABLE}")
endif()

message("")
message("Options:")
message("IGANET_BUILD_DOCS..................: ${IGANET_BUILD_DOCS}")
message("IGANET_BUILD_EXAMPLES..............: ${IGANET_BUILD_EXAMPLES}")
message("IGANET_BUILD_PYBIND11..............: ${IGANET_BUILD_PYBIND11}")
message("IGANET_BUILD_UNITTESTS.............: ${IGANET_BUILD_UNITTESTS}")

message("")
message("Features:")
message("IGANET_WITH_OPENMP.................: ${IGANET_WITH_OPENMP}")
message("")

########################################################################
# Add kernel directory
########################################################################
include_directories(${PROJECT_SOURCE_DIR}/include)

########################################################################
# Add examples (if enabled)
########################################################################
if (IGANET_BUILD_EXAMPLES)
  add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
endif()

########################################################################
# Add Python bindings (if enabled)
########################################################################
if (IGANET_BUILD_PYBIND11)
  add_subdirectory(${PROJECT_SOURCE_DIR}/pybind11)
endif()

########################################################################
# Add unittests (if enabled)
########################################################################
if (IGANET_BUILD_UNITTESTS)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
endif()
