
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was matplot++-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(MATPLOT_BUILT_SHARED "OFF")
if(NOT ${MATPLOT_BUILT_SHARED})
    include(CMakeFindDependencyMacro)

    list(APPEND CMAKE_MODULE_PATH ${MATPLOT_CONFIG_INSTALL_DIR})
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
    find_dependency(Filesystem COMPONENTS Experimental Final)
    list(POP_BACK CMAKE_MODULE_PATH)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/Matplot++Targets.cmake")
check_required_components(Matplot++)
