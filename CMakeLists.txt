cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(igapinn)

find_package(Torch REQUIRED)
find_package(Matplot++ REQUIRED)

# Apple explicitly disabled OpenMP support in their compilers that
# are shipped with XCode but there is an easy workaround as
# described at https://mac.r-project.org/openmp/
if (CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_C_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
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
  find_package(OpenMP)
  list(APPEND CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
  list(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
