cmake_minimum_required(VERSION 3.26)
project(zvec_fastpfor_portable_test LANGUAGES C CXX)

# Configure independently of zvec's other third-party dependencies:
# cmake -S thirdparty/FastPFOR/tests -B build/fastpfor-portable \
#   -DZVEC_FASTPFOR_PORTABLE=ON -DFETCHCONTENT_SOURCE_DIR_SIMDE=/path/to/simde
# cmake --build build/fastpfor-portable
# ctest --test-dir build/fastpfor-portable --output-on-failure
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(CC_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include("${PROJECT_ROOT_DIR}/cmake/utils.cmake")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." fastpfor)

add_executable(fastpfor_portable_roundtrip portable_roundtrip.cc)
target_link_libraries(fastpfor_portable_roundtrip PRIVATE FastPFOR)

# Per-source options come after target-wide flags, including the native
# integration's -msse4.1, so the regression cannot silently raise its baseline.
set(_fastpfor_test_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
    set(_fastpfor_test_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(_fastpfor_test_arch MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$"
        AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    get_target_property(_fastpfor_sources FastPFOR SOURCES)
    get_target_property(_fastpfor_source_dir FastPFOR SOURCE_DIR)
    foreach(_source IN LISTS _fastpfor_sources)
        get_filename_component(_source_path "${_source}" ABSOLUTE
            BASE_DIR "${_fastpfor_source_dir}")
        set_property(SOURCE "${_source_path}" TARGET_DIRECTORY FastPFOR
            APPEND PROPERTY COMPILE_OPTIONS -march=core2 -mno-sse4.1)
    endforeach()
    set_property(SOURCE portable_roundtrip.cc APPEND PROPERTY
        COMPILE_OPTIONS -march=core2 -mno-sse4.1)
endif()

enable_testing()
add_test(NAME fastpfor_portable_roundtrip COMMAND fastpfor_portable_roundtrip)
