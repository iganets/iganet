########################################################################
# CMakeLists.txt
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
# CMakeLists.txt accepts the following command line parameters
#
# IGANET_BUILD_CPUONLY
# IGANET_BUILD_DOCS
# IGANET_BUILD_PCH
#
# IGANET_OPTIONAL
#
# IGANET_WITH_GISMO
# IGANET_WITH_MATPLOT
# IGANET_WITH_MPI
# IGANET_WITH_OPENMP
#
########################################################################

cmake_minimum_required (VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

########################################################################
# IgANet version number
########################################################################
set(iganet_VERSION_MAJOR 25) #year
set(iganet_VERSION_MINOR 06) #month
set(iganet_VERSION_PATCH 0 ) #patch
set(iganet_VERSION
  "${iganet_VERSION_MAJOR}.${iganet_VERSION_MINOR}.${iganet_VERSION_PATCH}" CACHE INTERNAL "iganet version number")
set(IGANET_VERSION "${iganet_VERSION}" CACHE INTERNAL "IgANet version number")

########################################################################
# Set RPATH on MacOSX
########################################################################
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif()
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
# Project: IgANet (core)
#
########################################################################
project(IgANet)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(IGANET_INSTALL_CMAKEDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/IgANet"
  CACHE STRING "Directory for the IgANet CMake package")
set(IGANET_INSTALL_INCLUDEDIR
  "${CMAKE_INSTALL_INCLUDEDIR}/iganet"
  CACHE STRING "Directory for public IgANet headers")
set(IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR
  "${IGANET_INSTALL_INCLUDEDIR}/third_party"
  CACHE STRING "Directory for private third-party headers")

add_library(iganet_core INTERFACE)
add_library(iganet::core ALIAS iganet_core)
set_target_properties(iganet_core PROPERTIES EXPORT_NAME core)
target_compile_features(iganet_core INTERFACE cxx_std_20)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.24)
  cmake_policy(SET CMP0135 NEW)
endif()

########################################################################
# Append path to additional modules
########################################################################
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

########################################################################
# Option and feature lists
########################################################################
option(IGANET_BUILD_CPUONLY     "Build CPU-only code"       OFF)
option(IGANET_BUILD_DOCS        "Build documentation"       OFF)
option(IGANET_BUILD_PCH         "Build precompiled headers" ON )
option(IGANET_WITH_GISMO        "With G+Smo support"        OFF)
option(IGANET_WITH_MATPLOT      "With Matplot support"      OFF)
option(IGANET_WITH_MPI          "With MPI support"          OFF)
option(IGANET_WITH_OPENMP       "With OpenMP support"       ON )

if(NOT DEFINED IGANET_OPTIONAL)
  set(IGANET_OPTIONAL "" CACHE STRING "Build optional modules")
endif()

########################################################################
# Prerequisites
########################################################################
find_package(Torch REQUIRED)
list(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
target_link_libraries(iganet_core INTERFACE ${TORCH_LIBRARIES})

find_package(IPEX QUIET)
target_link_libraries(iganet_core INTERFACE ${TORCH_IPEX_LIBRARIES})

if(NOT WIN32)
  find_package(ZLIB REQUIRED)
endif()

include(boost_preprocessor)
target_include_directories(iganet_core INTERFACE
  $<BUILD_INTERFACE:${boost_preprocessor_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}>)

include(nlohmann_json)
target_include_directories(iganet_core INTERFACE
  $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/single_include>
  $<INSTALL_INTERFACE:${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}>)
target_compile_definitions(iganet_core INTERFACE JSON_HAS_RANGES=0)

include(pugixml)
target_include_directories(iganet_core INTERFACE
  $<BUILD_INTERFACE:${pugixml_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}>)
target_link_libraries(iganet_core INTERFACE
  $<BUILD_INTERFACE:pugixml>
  $<INSTALL_INTERFACE:iganet::pugixml>)

if(IGANET_WITH_GISMO)
  include(gismo)
  target_include_directories(iganet_core INTERFACE
    "$<BUILD_INTERFACE:${GISMO_INCLUDE_DIRS}>"
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gismo>)
  target_link_libraries(iganet_core INTERFACE
    $<BUILD_INTERFACE:gismo_static>
    $<INSTALL_INTERFACE:iganet::gismo_static>)
endif()

if(IGANET_WITH_MATPLOT)
  include(Matplot++)
  target_link_libraries(iganet_core INTERFACE Matplot++::matplot)
endif()

########################################################################
# Enable MPI support
########################################################################
if(IGANET_WITH_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(iganet_core INTERFACE MPI::MPI_CXX)
endif()

########################################################################
# Enable OpenMP support
########################################################################
if(IGANET_WITH_OPENMP)
  # Apple explicitly disabled OpenMP support in their compilers that
  # are shipped with XCode but there is an easy workaround as
  # described at https://mac.r-project.org/openmp/
  if(CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR
      CMAKE_C_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    find_path(OpenMP_C_INCLUDE_DIR
      NAMES "omp.h"
      PATHS /usr/local /usr/local/opt/libomp /opt /opt/local /opt/homebrew/opt/libomp
      PATH_SUFFIXES include)
    find_path(OpenMP_CXX_INCLUDE_DIR
      NAMES "omp.h"
      PATHS /usr/local /usr/local/opt/libomp /opt /opt/local /opt/homebrew/opt/libomp
      PATH_SUFFIXES include)
    find_library(OpenMP_libomp_LIBRARY
      NAMES "omp"
      PATHS /usr/local /usr/local/opt/libomp /opt /opt/local /opt/homebrew/opt/libomp
      PATH_SUFFIXES lib)
  endif()
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(iganet_core INTERFACE OpenMP::OpenMP_CXX)
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
# Documentation
########################################################################
if(IGANET_BUILD_DOCS)
  add_custom_target(docs
    DEPENDS Doxygen)
endif()

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

message("")
message("Options:")
message("IGANET_BUILD_CPUONLY...............: ${IGANET_BUILD_CPUONLY}")
message("IGANET_BUILD_DOCS..................: ${IGANET_BUILD_DOCS}")
message("IGANET_BUILD_PCH...................: ${IGANET_BUILD_PCH}")
message("")
message("Optional modules:")
message("IGANET_OPTIONAL....................: ${IGANET_OPTIONAL}")
message("")
message("Features:")
message("IGANET_WITH_GISMO..................: ${IGANET_WITH_GISMO}")
message("IGANET_WITH_MATPLOT................: ${IGANET_WITH_MATPLOT}")
message("IGANET_WITH_MPI....................: ${IGANET_WITH_MPI}")
message("IGANET_WITH_OPENMP.................: ${IGANET_WITH_OPENMP}")
message("")

########################################################################
# Add kernel directory
########################################################################
add_subdirectory(${PROJECT_SOURCE_DIR}/include)

########################################################################
# Add documentation (if enabled)
########################################################################
if(IGANET_BUILD_DOCS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/docs)
endif()

########################################################################
# Add optional modules
########################################################################
enable_testing()
include(get_module)

foreach(entry IN LISTS IGANET_OPTIONAL)

  # Extract entries of the form "module[branch]"
  string(REGEX MATCH "^([^\[]+)\\[([^\]]+)\\]" _ "${entry}")
  if(_)
    set(module "${CMAKE_MATCH_1}")
    set(branch "${CMAKE_MATCH_2}")
  else()
    set(module "${entry}")
    set(branch "main")
  endif()

  if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    get_module(
      ${module}
      GIT_REPOSITORY https://github.com/iganets/iganet-${module}.git
      GIT_TAG ${branch})
  else()
    get_module(
      ${module}
      URL https://github.com/iganets/iganet-${module}/archive/refs/heads/${branch}.zip)
  endif()
endforeach()

########################################################################
# Install and export the self-contained core package
########################################################################
if(BUILD_SHARED_LIBS)
  set(IGANET_PUGIXML_TARGET pugixml-shared)
else()
  set(IGANET_PUGIXML_TARGET pugixml-static)
endif()

# Export the selected implementation under IgANet's namespace and private
# layout rather than exposing pugixml as a separately installed package.
set_target_properties(${IGANET_PUGIXML_TARGET} PROPERTIES EXPORT_NAME pugixml)
target_include_directories(${IGANET_PUGIXML_TARGET} PUBLIC
  $<INSTALL_INTERFACE:${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}>)

set(IGANET_EXPORT_TARGETS iganet_core ${IGANET_PUGIXML_TARGET})
if(IGANET_WITH_GISMO)
  # G+Smo does not install its generated gismoTargets export. Export the
  # selected static library with IgANet so the installed core target retains
  # the same public G+Smo API as the build-tree target.
  set_target_properties(gismo_static PROPERTIES EXPORT_NAME gismo_static)
  list(APPEND IGANET_EXPORT_TARGETS gismo_static)
endif()

install(TARGETS ${IGANET_EXPORT_TARGETS}
  EXPORT IgANetTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/iganet
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/iganet
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY "${boost_preprocessor_SOURCE_DIR}/include/boost"
  DESTINATION "${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}")
install(DIRECTORY "${nlohmann_json_SOURCE_DIR}/single_include/nlohmann"
  DESTINATION "${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}")
install(FILES
  "${pugixml_SOURCE_DIR}/src/pugiconfig.hpp"
  "${pugixml_SOURCE_DIR}/src/pugixml.hpp"
  DESTINATION "${IGANET_INSTALL_THIRD_PARTY_INCLUDEDIR}")

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/IgANetConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/IgANetConfig.cmake"
  INSTALL_DESTINATION "${IGANET_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/IgANetConfigVersion.cmake"
  VERSION "${iganet_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(EXPORT IgANetTargets
  FILE IgANetTargets.cmake
  NAMESPACE iganet::
  DESTINATION "${IGANET_INSTALL_CMAKEDIR}")
install(FILES
  "${PROJECT_BINARY_DIR}/IgANetConfig.cmake"
  "${PROJECT_BINARY_DIR}/IgANetConfigVersion.cmake"
  DESTINATION "${IGANET_INSTALL_CMAKEDIR}")
