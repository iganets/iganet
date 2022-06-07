#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Matplot++::nodesoup" for configuration "Release"
set_property(TARGET Matplot++::nodesoup APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Matplot++::nodesoup PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/Matplot++/libnodesoup.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS Matplot++::nodesoup )
list(APPEND _IMPORT_CHECK_FILES_FOR_Matplot++::nodesoup "${_IMPORT_PREFIX}/lib/Matplot++/libnodesoup.a" )

# Import target "Matplot++::matplot" for configuration "Release"
set_property(TARGET Matplot++::matplot APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Matplot++::matplot PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmatplot.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS Matplot++::matplot )
list(APPEND _IMPORT_CHECK_FILES_FOR_Matplot++::matplot "${_IMPORT_PREFIX}/lib/libmatplot.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
