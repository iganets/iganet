function(iganet_configure_torch_hip)
  if(NOT TARGET torch_hip OR NOT TARGET hip::amdhip64)
    return()
  endif()

  # Some ROCm Python packages export hip::amdhip64 without the include
  # directory of the separately packaged ROCm SDK.  Torch's public HIP
  # headers include <hip/hip_runtime.h>, so repair the imported target when
  # that header is not available through its usage requirements.
  get_target_property(_iganet_hip_include_dirs
    hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES)

  set(_iganet_has_hip_runtime_header OFF)
  foreach(_iganet_hip_include_dir IN LISTS _iganet_hip_include_dirs)
    if(NOT _iganet_hip_include_dir MATCHES "^\\$<" AND
       EXISTS "${_iganet_hip_include_dir}/hip/hip_runtime.h")
      set(_iganet_has_hip_runtime_header ON)
      break()
    endif()
  endforeach()

  if(_iganet_has_hip_runtime_header)
    return()
  endif()

  set(_iganet_hip_include_hints
    ${ROCM_INCLUDE_DIRS}
    ${hip_INCLUDE_DIRS}
    ${hip_INCLUDE_DIR}
    "${ROCM_PATH}/include"
    "$ENV{ROCM_PATH}/include"
    "$ENV{ROCM_HOME}/include"
    "/opt/rocm/include")

  # Wheels built with the split ROCm SDK install Torch and the SDK as sibling
  # Python packages: torch/share/cmake/Torch and _rocm_sdk_core/include.
  if(Torch_DIR)
    get_filename_component(_iganet_site_packages
      "${Torch_DIR}/../../../.." ABSOLUTE)
    list(APPEND _iganet_hip_include_hints
      "${_iganet_site_packages}/_rocm_sdk_core/include")
  endif()

  find_path(IGANET_HIP_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    HINTS ${_iganet_hip_include_hints})

  if(NOT IGANET_HIP_INCLUDE_DIR)
    message(FATAL_ERROR
      "Torch has ROCm/HIP support, but hip/hip_runtime.h was not found. "
      "Set ROCM_PATH or IGANET_HIP_INCLUDE_DIR to the ROCm SDK prefix or "
      "include directory, respectively.")
  endif()

  set_property(TARGET hip::amdhip64 APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${IGANET_HIP_INCLUDE_DIR}")
endfunction()
