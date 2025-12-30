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
option(ENABLE_SAPPHIRERAPIDS "Enable Intel Sapphire Rapids Server CPU microarchitecture" OFF)
option(ENABLE_EMERALDRAPIDS "Enable Intel Emerald Rapids Server CPU microarchitecture" OFF)
option(ENABLE_GRANITERAPIDS "Enable Intel Granite Rapids Server CPU microarchitecture" OFF)

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
  ENABLE_SKYLAKE_AVX512 ENABLE_SAPPHIRERAPIDS ENABLE_EMERALDRAPIDS ENABLE_GRANITERAPIDS
  ENABLE_ZEN1 ENABLE_ZEN2 ENABLE_ZEN3
  ENABLE_ARMV8A ENABLE_ARMV8.1A ENABLE_ARMV8.2A ENABLE_ARMV8.3A ENABLE_ARMV8.4A
  ENABLE_ARMV8.5A ENABLE_ARMV8.6A
)

set(AUTO_DETECT_ARCH ON)
foreach(opt IN LISTS ARCH_OPTIONS)
  if(${opt})
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
  check_c_compiler_flag("${FLAG}" COMPILER_SUPPORT_${VAR_NAME})
  if(COMPILER_SUPPORT_${VAR_NAME})
    _AppendFlags(CMAKE_C_FLAGS "${FLAG}")
    _AppendFlags(CMAKE_CXX_FLAGS "${FLAG}")
    set(${VAR_NAME}_ENABLED ON)
  else()
    if(${OPTION_NAME})
      message(FATAL_ERROR "Compiler does not support required flag: '${FLAG}' for ${OPTION_NAME}")
    else()
      set(${VAR_NAME}_ENABLED OFF)
    endif()
  endif()
endmacro()

function(_detect_armv8_best)
  set(_arm_flags
    "armv8.6-a" "armv8.5-a" "armv8.4-a" "armv8.3-a" "armv8.2-a" "armv8.1-a" "armv8-a" "armv8"
  )
  foreach(_ver IN LISTS _arm_flags)
    check_c_compiler_flag("-march=${_ver}" _COMP_SUPP_${_ver})
    if(_COMP_SUPP_${_ver})
      _AppendFlags(CMAKE_C_FLAGS "-march=${_ver}")
      _AppendFlags(CMAKE_CXX_FLAGS "-march=${_ver}")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" PARENT_SCOPE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(WARNING "No ARMv8 architecture flag supported by compiler.")
endfunction()

function(_detect_x86_best)
  set(_x86_flags
    "graniterapids" "emeraldrapids" "sapphirerapids"
    "skylake-avx512" "skylake"
    "broadwell" "haswell" "sandybridge" "nehalem"
    "znver3" "znver2" "znver1"
  )
  foreach(_arch IN LISTS _x86_flags)
    check_c_compiler_flag("-march=${_arch}" _COMP_SUPP_${_arch})
    if(_COMP_SUPP_${_arch})
      _AppendFlags(CMAKE_C_FLAGS "-march=${_arch}")
      _AppendFlags(CMAKE_CXX_FLAGS "-march=${_arch}")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" PARENT_SCOPE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(WARNING "No known x86 microarchitecture flag supported; falling back to generic.")
endfunction()

if(MSVC)
  # Prefer higher ISAs
  foreach(_isa IN ITEMS "AVX512" "AVX2" "AVX" "SSE2")
    check_c_compiler_flag("/arch:${_isa}" _COMP_SUPP_${_isa})
    if(_COMP_SUPP_${_isa})
      _AppendFlags(CMAKE_C_FLAGS "/arch:${_isa}")
      _AppendFlags(CMAKE_CXX_FLAGS "/arch:${_isa}")
      message(STATUS "MSVC: enabled /arch:${_isa}")
      break()
    endif()
  endforeach()
  return()
endif()


if(NOT AUTO_DETECT_ARCH)
  if(ENABLE_ZEN3)
    add_arch_flag("-march=znver3" ZNVER3 ENABLE_ZEN3)
  endif()

  if(ENABLE_ZEN2)
    add_arch_flag("-march=znver2" ZNVER2 ENABLE_ZEN2)
  endif()

  if(ENABLE_ZEN1)
    add_arch_flag("-march=znver1" ZNVER1 ENABLE_ZEN1)
  endif()

  if(ENABLE_GRANITERAPIDS)
    add_arch_flag("-march=graniterapids" GRANITERAPIDS ENABLE_GRANITERAPIDS)
  endif()

  if(ENABLE_EMERALDRAPIDS)
    add_arch_flag("-march=emeraldrapids" EMERALDRAPIDS ENABLE_EMERALDRAPIDS)
  endif()

  if(ENABLE_SAPPHIRERAPIDS)
    add_arch_flag("-march=sapphirerapids" SAPPHIRERAPIDS ENABLE_SAPPHIRERAPIDS)
  endif()

  if(ENABLE_SKYLAKE_AVX512)
    add_arch_flag("-march=skylake-avx512" SKYLAKE_AVX512 ENABLE_SKYLAKE_AVX512)
  endif()

  if(ENABLE_SKYLAKE)
    add_arch_flag("-march=skylake" SKYLAKE ENABLE_SKYLAKE)
  endif()

  if(ENABLE_BROADWELL)
    add_arch_flag("-march=broadwell" BROADWELL ENABLE_BROADWELL)
  endif()

  if(ENABLE_HASWELL)
    add_arch_flag("-march=haswell" HASWELL ENABLE_HASWELL)
  endif()

  if(ENABLE_SANDYBRIDGE)
    add_arch_flag("-march=sandybridge" SANDYBRIDGE ENABLE_SANDYBRIDGE)
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
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    _detect_armv8_best()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
    _detect_x86_best()
  else()
    message(WARNING "Unknown host architecture: ${CMAKE_SYSTEM_PROCESSOR}; no -march= set.")
  endif()
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
