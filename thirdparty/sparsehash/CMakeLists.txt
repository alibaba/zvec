set(SPARSE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sparseconfig.h")
set(SPARSE_DST "${CMAKE_CURRENT_SOURCE_DIR}/sparsehash-2.0.4/src/sparsehash/internal/sparseconfig.h")
get_filename_component(DESTINATION_DIR "${SPARSE_DST}" DIRECTORY)
if(NOT EXISTS "${SPARSE_DST}")
    file(COPY "${SPARSE_SRC}" DESTINATION "${DESTINATION_DIR}")
endif()

if(MSVC)
    set(SPARSEHASH_WINDOWS_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/sparsehash.windows.patch)
    apply_patch_once("sparsehash.windows.patch" "${DESTINATION_DIR}" "${SPARSEHASH_WINDOWS_PATCH}")
endif()

add_library(sparsehash INTERFACE)
target_include_directories(
    sparsehash INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/sparsehash-2.0.4/src/"
  )
