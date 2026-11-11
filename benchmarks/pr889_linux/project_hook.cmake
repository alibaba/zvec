# Optional build-only hook: -DCMAKE_PROJECT_zvec_INCLUDE=<absolute path to this file>
# Defers adding the benchmark until the root has defined zvec::static.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL include "${CMAKE_SOURCE_DIR}/benchmarks/pr889_linux/CMakeLists.txt")
endif()
