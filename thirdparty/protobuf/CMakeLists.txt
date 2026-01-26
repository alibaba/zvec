set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable testing in protobuf" FORCE)
set(protobuf_WITH_ZLIB ON CACHE BOOL "Disable zlib support in protobuf" FORCE)

set(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR})
add_subdirectory(protobuf-3.21.12/cmake protobuf-3.21.12)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(libprotobuf PRIVATE 
        -Wno-deprecated-declarations 
        -Wno-invalid-noreturn 
        -Wno-unused-function
    )
    target_compile_options(libprotoc PRIVATE 
        -Wno-unused-private-field 
        -Wno-unused-function
    )
    target_compile_options(protoc PRIVATE 
        -Wno-unused-private-field 
        -Wno-unused-function
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(libprotobuf PRIVATE 
        -Wno-deprecated-declarations 
        -Wno-unused-function
        -Wno-maybe-uninitialized
        -Wno-sign-compare
        -Wno-return-type
        -Wno-stringop-overflow
    )
    target_compile_options(libprotoc PRIVATE 
        -Wno-unused-private-field 
        -Wno-unused-function
        -Wno-unused-but-set-variable
        -Wno-sign-compare
    )
    target_compile_options(protoc PRIVATE 
        -Wno-unused-private-field 
        -Wno-unused-function
        -Wno-unused-but-set-variable
        -Wno-sign-compare
    )
endif()

get_target_property(libprotobuf_SOURCE_DIR libprotobuf SOURCE_DIR)
get_filename_component(libprotobuf_INCLUDE_DIR ${libprotobuf_SOURCE_DIR}/../src ABSOLUTE)

set(PROTOBUF_FOUND TRUE PARENT_SCOPE)
set(PROTOBUF_INCLUDE_DIR ${libprotobuf_INCLUDE_DIR} PARENT_SCOPE)
set(PROTOBUF_INCLUDE_DIRS ${libprotobuf_INCLUDE_DIR} PARENT_SCOPE)

set(PROTOBUF_LIBRARY $<TARGET_FILE:libprotobuf> PARENT_SCOPE)
set(PROTOBUF_LIBRARIES $<TARGET_FILE:libprotobuf> PARENT_SCOPE)

set(PROTOBUF_LITE_LIBRARY $<TARGET_FILE:libprotobuf-lite> PARENT_SCOPE)
set(PROTOBUF_LITE_LIBRARIES $<TARGET_FILE:libprotobuf-lite> PARENT_SCOPE)

set(PROTOBUF_PROTOC_LIBRARY $<TARGET_FILE:libprotoc> PARENT_SCOPE)
set(PROTOBUF_PROTOC_LIBRARIES $<TARGET_FILE:libprotoc> PARENT_SCOPE)
set(PROTOBUF_PROTOC_EXECUTABLE $<TARGET_FILE:protoc> PARENT_SCOPE)
