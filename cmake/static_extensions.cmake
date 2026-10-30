# Ordinary linking helpers. Registration lists live in C++, not in CMake.
include_guard(GLOBAL)

function(zvec_resolve_component_libraries RESULT)
  # Only cc_* consumers opt into this mapping. Do not rewrite parent-project
  # targets, and resolve forward references without inspecting the target graph.
  set(COMPONENTS core_metric core_quantizer core_utility core_mix_reducer
    core_knn_cluster core_knn_flat core_knn_flat_sparse core_knn_ivf
    core_knn_hnsw core_knn_hnsw_sparse core_knn_vamana core_knn_diskann
    core_knn_hnsw_rabitq core_knn_ivf_rabitq)
  set(LIBRARIES)
  foreach(LIBRARY IN LISTS ARGN)
    if(LIBRARY IN_LIST COMPONENTS)
      list(APPEND LIBRARIES "${LIBRARY}_static")
    else()
      list(APPEND LIBRARIES "${LIBRARY}")
    endif()
  endforeach()
  set(${RESULT} "${LIBRARIES}" PARENT_SCOPE)
endfunction()

# A prebuilt archive must provide an extern C function that initializes (or
# references) its registration code. Compile the checked-in C++ adapter with
# that entry point; no generated source or dependency-graph traversal is needed.
function(zvec_link_static_extension)
  cmake_parse_arguments(EXT "" "TARGET;LIBRARY;INITIALIZER" "" ${ARGN})
  if(NOT TARGET "${EXT_TARGET}" OR NOT EXT_LIBRARY OR
     NOT EXT_INITIALIZER MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "Expected TARGET, LIBRARY and an extern C INITIALIZER")
  endif()
  string(SHA256 KEY "${EXT_TARGET};${EXT_LIBRARY};${EXT_INITIALIZER}")
  set(ADAPTER "zvec_extension_${KEY}")
  if(TARGET ${ADAPTER})
    return()
  endif()
  add_library(${ADAPTER} OBJECT EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/static_extension.cc")
  set_target_properties(${ADAPTER} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_features(${ADAPTER} PRIVATE cxx_std_11)
  target_compile_definitions(${ADAPTER} PRIVATE
    ZVEC_EXTENSION_INITIALIZER=${EXT_INITIALIZER})
  # Object references propagate through static/interface wrappers, including
  # conditional CMake links, using CMake's own dependency handling.
  get_target_property(TYPE ${EXT_TARGET} TYPE)
  if(TYPE STREQUAL "INTERFACE_LIBRARY")
    set_property(TARGET ${EXT_TARGET} APPEND PROPERTY INTERFACE_LINK_LIBRARIES
      ${EXT_LIBRARY} $<TARGET_OBJECTS:${ADAPTER}>)
  else()
    set_property(TARGET ${EXT_TARGET} APPEND PROPERTY LINK_LIBRARIES ${EXT_LIBRARY})
    if(TYPE STREQUAL "STATIC_LIBRARY")
      set_property(TARGET ${EXT_TARGET} APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        "$<LINK_ONLY:${EXT_LIBRARY}>" $<TARGET_OBJECTS:${ADAPTER}>)
    else()
      target_sources(${EXT_TARGET} PRIVATE $<TARGET_OBJECTS:${ADAPTER}>)
    endif()
  endif()
endfunction()
