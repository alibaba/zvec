## https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures  
## https://en.wikipedia.org/wiki/List_of_AMD_CPU_microarchitectures  
## https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html  

## Intel Microarchitectures
option(ENABLE_NEHALEM "Enable Intel Nehalem CPU microarchitecture" OFF)
option(ENABLE_SANDYBRIDGE "Enable Intel Sandy Bridge CPU microarchitecture" OFF)
option(ENABLE_HASWELL "Enable Intel Haswell CPU microarchitecture" OFF)
option(ENABLE_BROADWELL "Enable Intel Broadwell CPU microarchitecture" OFF)
option(ENABLE_SKYLAKE "Enable Intel Skylake CPU microarchitecture" OFF)
option(ENABLE_SKYLAKE_AVX512 "Enable Intel Skylake Server CPU microarchitecture" OFF)
option(ENABLE_ICELAKE "Enable Intel Icelake CPU microarchitecture" OFF)
option(ENABLE_SAPPHIRERAPIDS "Enable Intel Sapphire Rapids Server CPU microarchitecture" OFF)
option(ENABLE_EMERALDRAPIDS "Enable Intel Emerald Rapids Server CPU microarchitecture" OFF)
option(ENABLE_GRANITERAPIDS "Enable Intel Granite Rapids Server CPU microarchitecture" OFF)

option(ENABLE_NATIVE "Enable native CPU microarchitecture" OFF)

## AMD Microarchitectures
option(ENABLE_ZEN1 "Enable AMD Zen+ Family 17h CPU microarchitecture" OFF)
option(ENABLE_ZEN2 "Enable AMD Zen 2 Family 17h CPU microarchitecture" OFF)
option(ENABLE_ZEN3 "Enable AMD Zen 3 Family 19h CPU microarchitecture" OFF)

## ARM architectures
option(ENABLE_ARMV8A "Enable ARMv8-a architecture" OFF)
option(ENABLE_ARMV8.1A "Enable ARMv8.1-a architecture" OFF)
option(ENABLE_ARMV8.2A "Enable ARMv8.2-a architecture" OFF)
option(ENABLE_ARMV8.3A "Enable ARMv8.3-a architecture" OFF)
option(ENABLE_ARMV8.4A "Enable ARMv8.4-a architecture" OFF)
option(ENABLE_ARMV8.5A "Enable ARMv8.5-a architecture" OFF)
option(ENABLE_ARMV8.6A "Enable ARMv8.6-a architecture" OFF)

## OpenMP option
option(ENABLE_OPENMP "Enable OpenMP support" OFF)

set(ARCH_OPTIONS
  ENABLE_NEHALEM ENABLE_SANDYBRIDGE ENABLE_HASWELL ENABLE_BROADWELL ENABLE_SKYLAKE
  ENABLE_SKYLAKE_AVX512 ENABLE_ICELAKE ENABLE_SAPPHIRERAPIDS ENABLE_EMERALDRAPIDS ENABLE_GRANITERAPIDS
  ENABLE_ZEN1 ENABLE_ZEN2 ENABLE_ZEN3
  ENABLE_ARMV8A ENABLE_ARMV8.1A ENABLE_ARMV8.2A ENABLE_ARMV8.3A ENABLE_ARMV8.4A
  ENABLE_ARMV8.5A ENABLE_ARMV8.6A
  ENABLE_NATIVE
)

option(AUTO_DETECT_ARCH "Auto detect CPU microarchitecture" ON)
foreach(opt IN LISTS ARCH_OPTIONS)
  if(${opt})
    message(WARNING "opt ${opt}")
    set(AUTO_DETECT_ARCH OFF)
    break()
  endif()
endforeach()

include(CheckCCompilerFlag)

function(_AppendFlags _RESULT _FLAG)
  if(${_RESULT} AND NOT "${${_RESULT}}" MATCHES "${_FLAG}")
    set(${_RESULT} "${${_RESULT}} ${_FLAG}" PARENT_SCOPE)
  else()
    set(${_RESULT} "${_FLAG}" PARENT_SCOPE)
  endif()
endfunction()

macro(add_arch_flag FLAG VAR_NAME OPTION_NAME)
  if(MSVC)
    if(${ARGC} GREATER 3)
      set(_effective_flag "${ARGV3}")
    else()
      set(_effective_flag "")
    endif()
  else()
    set(_effective_flag "${FLAG}")
  endif()

  if(NOT "${_effective_flag}" STREQUAL "")
    check_c_compiler_flag("${_effective_flag}" COMPILER_SUPPORT_${VAR_NAME})
    if(COMPILER_SUPPORT_${VAR_NAME})
      _AppendFlags(CMAKE_C_FLAGS "${_effective_flag}")
      _AppendFlags(CMAKE_CXX_FLAGS "${_effective_flag}")
      set(${VAR_NAME}_ENABLED ON)
    else()
      if(${OPTION_NAME})
        message(FATAL_ERROR "Compiler does not support required flag: '${_effective_flag}' for ${OPTION_NAME}")
      else()
        set(${VAR_NAME}_ENABLED OFF)
      endif()
    endif()
  else()
    set(${VAR_NAME}_ENABLED ON)
  endif()
endmacro()

function(_setup_armv8_march)
  if(MSVC)
    return()
  endif()
  set(_arch "armv8")
  check_c_compiler_flag("-march=${_arch}" _COMP_SUPP_${_arch})
  if(_COMP_SUPP_${_arch})
    _AppendFlags(CMAKE_C_FLAGS "-march=${_arch}")
    _AppendFlags(CMAKE_CXX_FLAGS "-march=${_arch}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
    return()
  else()
    message(WARNING "No ARMv8 march flag supported by compiler.")
  endif()
endfunction()

function(_setup_x86_march)
  if(MSVC)
    return()
  endif()
  set(_arch "x86-64")
  check_c_compiler_flag("-march=${_arch}" _COMP_SUPP_${_arch})
  if(_COMP_SUPP_${_arch})
    _AppendFlags(CMAKE_C_FLAGS "-march=${_arch}")
    _AppendFlags(CMAKE_CXX_FLAGS "-march=${_arch}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
    return()
  else()
    message(WARNING "No known x86 march flag supported; falling back to generic.")
  endif()
endfunction()

function(setup_compiler_march_for_x86 VAR_NAME_SSE VAR_NAME_AVX2 VAR_NAME_AVX512 VAR_NAME_AVX512FP16)
  if(MSVC)
    #sse
    set(${VAR_NAME_SSE} "" PARENT_SCOPE)

    #avx2
    check_c_compiler_flag("/arch:AVX2" _COMP_SUPP_MSVC_AVX2)
    if(_COMP_SUPP_MSVC_AVX2)
      set(${VAR_NAME_AVX2} "/arch:AVX2" PARENT_SCOPE)
    else()
      set(${VAR_NAME_AVX2} "" PARENT_SCOPE)
    endif()

    #avx512
    check_c_compiler_flag("/arch:AVX512" _COMP_SUPP_MSVC_AVX512)
    if(_COMP_SUPP_MSVC_AVX512)
      set(${VAR_NAME_AVX512} "/arch:AVX512" PARENT_SCOPE)
    elseif(_COMP_SUPP_MSVC_AVX2)
      set(${VAR_NAME_AVX512} "/arch:AVX2" PARENT_SCOPE)
    else()
      set(${VAR_NAME_AVX512} "" PARENT_SCOPE)
    endif()

    #avx512fp16
    if(_COMP_SUPP_MSVC_AVX512)
      set(${VAR_NAME_AVX512FP16} "/arch:AVX512" PARENT_SCOPE)
    elseif(_COMP_SUPP_MSVC_AVX2)
      set(${VAR_NAME_AVX512FP16} "/arch:AVX2" PARENT_SCOPE)
    else()
      set(${VAR_NAME_AVX512FP16} "" PARENT_SCOPE)
    endif()

    return()
  endif()

  #sse
  set(${VAR_NAME_SSE} "-march=corei7" PARENT_SCOPE)

  #avx 2
  set(${VAR_NAME_AVX2} "-march=core-avx2" PARENT_SCOPE)

  #avx512
  set(_x86_flags_avx512 "icelake-server" "skylake-avx512" "core-avx2" "x86-64")
  foreach(_arch_avx512 IN LISTS _x86_flags_avx512)
    check_c_compiler_flag("-march=${_arch_avx512}" _COMP_SUPP_${_arch_avx512})
    if(_COMP_SUPP_${_arch_avx512})
      set(${VAR_NAME_AVX512} "-march=${_arch_avx512}" PARENT_SCOPE)
      break()
    endif()
  endforeach()

  #avx512fp16
  set(_x86_flags_avx512fp16
    "sapphirerapids" "icelake-server" "skylake-avx512" "core-avx2" "x86-64"
  )
  foreach(_arch_avx512fp16 IN LISTS _x86_flags_avx512fp16)
    check_c_compiler_flag("-march=${_arch_avx512fp16}" _COMP_SUPP_${_arch_avx512fp16})
    if(_COMP_SUPP_${_arch_avx512fp16})
      set(${VAR_NAME_AVX512FP16} "-march=${_arch_avx512fp16}" PARENT_SCOPE)
      break()
    endif()
  endforeach()
endfunction()

if(NOT AUTO_DETECT_ARCH)
  if(ENABLE_NATIVE)
    if (NOT MSVC)
      add_arch_flag("-march=native" NATIVE ENABLE_NATIVE)
    endif ()
  endif()

  if(ENABLE_ZEN3)
    add_arch_flag("-march=znver3" ZNVER3 ENABLE_ZEN3 "/arch:AVX2")
  endif()

  if(ENABLE_ZEN2)
    add_arch_flag("-march=znver2" ZNVER2 ENABLE_ZEN2 "/arch:AVX2")
  endif()

  if(ENABLE_ZEN1)
    add_arch_flag("-march=znver1" ZNVER1 ENABLE_ZEN1 "/arch:AVX2")
  endif()

  if(ENABLE_GRANITERAPIDS)
    add_arch_flag("-march=graniterapids" GRANITERAPIDS ENABLE_GRANITERAPIDS "/arch:AVX512")
  endif()

  if(ENABLE_EMERALDRAPIDS)
    add_arch_flag("-march=emeraldrapids" EMERALDRAPIDS ENABLE_EMERALDRAPIDS "/arch:AVX512")
  endif()

  if(ENABLE_SAPPHIRERAPIDS)
    add_arch_flag("-march=sapphirerapids" SAPPHIRERAPIDS ENABLE_SAPPHIRERAPIDS "/arch:AVX512")
  endif()

  if(ENABLE_ICELAKE)
    add_arch_flag("-march=icelake-server" ICELAKE ENABLE_ICELAKE "/arch:AVX512")
  endif()

  if(ENABLE_SKYLAKE_AVX512)
    add_arch_flag("-march=skylake-avx512" SKYLAKE_AVX512 ENABLE_SKYLAKE_AVX512 "/arch:AVX512")
  endif()

  if(ENABLE_SKYLAKE)
    add_arch_flag("-march=skylake" SKYLAKE ENABLE_SKYLAKE "/arch:AVX2")
  endif()

  if(ENABLE_BROADWELL)
    add_arch_flag("-march=broadwell" BROADWELL ENABLE_BROADWELL "/arch:AVX2")
  endif()

  if(ENABLE_HASWELL)
    add_arch_flag("-march=haswell" HASWELL ENABLE_HASWELL "/arch:AVX2")
  endif()

  if(ENABLE_SANDYBRIDGE)
    add_arch_flag("-march=sandybridge" SANDYBRIDGE ENABLE_SANDYBRIDGE "/arch:AVX")
  endif()

  if(ENABLE_NEHALEM)
    add_arch_flag("-march=nehalem" NEHALEM ENABLE_NEHALEM)
  endif()

  # ARM (newest first — allow multiple? usually only one)
  # But GCC allows only one -march=, so honor highest enabled
  if(ENABLE_ARMV8.6A)
    add_arch_flag("-march=armv8.6-a" ARMV86A ENABLE_ARMV8.6A)
  endif()
  if(ENABLE_ARMV8.5A)
    add_arch_flag("-march=armv8.5-a" ARMV85A ENABLE_ARMV8.5A)
  endif()
  if(ENABLE_ARMV8.4A)
    add_arch_flag("-march=armv8.4-a" ARMV84A ENABLE_ARMV8.4A)
  endif()
  if(ENABLE_ARMV8.3A)
    add_arch_flag("-march=armv8.3-a" ARMV83A ENABLE_ARMV8.3A)
  endif()
  if(ENABLE_ARMV8.2A)
    add_arch_flag("-march=armv8.2-a" ARMV82A ENABLE_ARMV8.2A)
  endif()
  if(ENABLE_ARMV8.1A)
    add_arch_flag("-march=armv8.1-a" ARMV81A ENABLE_ARMV8.1A)
  endif()
  if(ENABLE_ARMV8A)
    add_arch_flag("-march=armv8-a" ARMV8A ENABLE_ARMV8A)
  endif()

else()
  # AUTO DETECT
  # Heuristic: detect host architecture and probe appropriate flags
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_PROC_LOWER)

  if(SYSTEM_PROC_LOWER MATCHES "^(x86_64|amd64|x64)$")
    set(HOST_ARCH x64)
  elseif(SYSTEM_PROC_LOWER MATCHES "^(i[3-6]86|x86)$")
    set(HOST_ARCH x86)
  elseif(SYSTEM_PROC_LOWER MATCHES "^(aarch64|arm64)$")
    set(HOST_ARCH arm64)
  elseif(SYSTEM_PROC_LOWER MATCHES "^(arm|armv7|armv7-a|armv7l)$")
    set(HOST_ARCH arm)
  else()
    set(HOST_ARCH unknown)
    message(WARNING "unknown host arch: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  #  message(STATUS "host arch: ${HOST_ARCH}")

  if (HOST_ARCH MATCHES "^(arm|arm64)$")
    _setup_armv8_march()
  elseif (HOST_ARCH MATCHES "^(x86|x64)$")
    _setup_x86_march()
  else ()
    message(WARNING "unknown host arch - no -march set")
  endif ()
endif()

# -----------------------------
# OpenMP
# -----------------------------
if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_C_FLAGS)
    _AppendFlags(CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
  endif()
  if(OpenMP_CXX_FLAGS)
    _AppendFlags(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()
endif()
