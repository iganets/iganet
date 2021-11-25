cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(igapinn)

find_package(Torch REQUIRED)
find_package(Matplot++ REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
