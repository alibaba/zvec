# Resolve the budget when the external build runs, so a reused build tree
# honors the current environment rather than the configure-time value.
cmake_minimum_required(VERSION 3.26)

set(_jobs "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
if(_jobs STREQUAL "")
    include(ProcessorCount)
    ProcessorCount(_jobs)
    if(_jobs LESS 1)
        set(_jobs 1)
    endif()
elseif(NOT _jobs MATCHES "^[0-9]+$" OR _jobs LESS 1)
    message(FATAL_ERROR "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer")
endif()

if(ZVEC_EXTERNAL_MAKE)
    set(_command "${ZVEC_EXTERNAL_MAKE}" -j "${_jobs}")
else()
    set(_command "${CMAKE_COMMAND}" --build . --parallel "${_jobs}")
    if(NOT "${ZVEC_EXTERNAL_CONFIG}" STREQUAL "")
        list(APPEND _command --config "${ZVEC_EXTERNAL_CONFIG}")
    endif()
endif()

execute_process(COMMAND ${_command} COMMAND_ERROR_IS_FATAL ANY)
