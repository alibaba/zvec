# SPDX-License-Identifier: Apache-2.0
# Arrow/Gandiva are built by the existing zvec Arrow ExternalProject. In
# particular, do not find_package(Arrow) here: a second Arrow instance would
# cross C++ object ownership and ABI boundaries in the predicate callback.
add_library(zvec_gandiva_scalar STATIC
    src/scalar_predicate.cc
    src/scalar_predicate_cache.cc)
add_library(Zvec::gandiva_scalar ALIAS zvec_gandiva_scalar)
target_compile_features(zvec_gandiva_scalar PUBLIC cxx_std_17)
target_include_directories(zvec_gandiva_scalar PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(zvec_gandiva_scalar PUBLIC ZVEC_ENABLE_SCALAR_JIT=1)
target_link_libraries(zvec_gandiva_scalar PUBLIC
    Gandiva::gandiva_static Arrow::arrow_static Threads::Threads)
add_dependencies(zvec_gandiva_scalar ARROW.BUILD)
set_target_properties(zvec_gandiva_scalar PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Keep lookup/binding code on the speed-oriented Release flags used by zvec,
# rather than inheriting thirdparty/'s global -Os override.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(zvec_gandiva_scalar PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

install(TARGETS zvec_gandiva_scalar ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/gandiva DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE NOTICE PROVENANCE.json
    DESTINATION share/zvec/thirdparty/gandiva_scalar)
