# Promote a target's INTERFACE_INCLUDE_DIRECTORIES to be treated as SYSTEM
# includes by consumers, suppressing warnings from third-party headers.
function(mark_target_includes_system)
    foreach(_target ${ARGN})
        if(NOT TARGET ${_target})
            continue()
        endif()
        get_target_property(_aliased ${_target} ALIASED_TARGET)
        if(_aliased)
            set(_target ${_aliased})
        endif()
        get_target_property(_inc ${_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc)
            set_target_properties(${_target} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
        endif()
    endforeach()
endfunction()

function(apply_patch_once patch_name target_dir patch_file)
    set(mark_file "${target_dir}/.${patch_name}_patched")

    if(EXISTS "${mark_file}")
        #message(STATUS "Patch '${patch_name}' already applied to ${target_dir}, skipping.")
        return()
    endif()

    if(NOT EXISTS "${patch_file}")
        message(FATAL_ERROR "Patch file '${patch_file}' not found!")
    endif()

    #message(STATUS "Applying patch '${patch_name}' to ${target_dir} ...")
    execute_process(
        COMMAND git apply --ignore-space-change --ignore-whitespace "${patch_file}"
        WORKING_DIRECTORY "${target_dir}"
        RESULT_VARIABLE patch_result
        OUTPUT_VARIABLE patch_stdout
        ERROR_VARIABLE patch_stderr
    )

    if(NOT patch_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply patch '${patch_name}' to ${target_dir}:\n${patch_stderr}")
    else()
        #message(STATUS "Patch '${patch_name}' applied successfully:\n${patch_stdout}")
        file(WRITE "${mark_file}" "patched")
    endif()
endfunction()
