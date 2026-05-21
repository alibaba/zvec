add_library(magic_enum INTERFACE)
target_include_directories(
    magic_enum SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/magic_enum-0.9.7/include/"
  )
