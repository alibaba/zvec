include(ExternalProject)

set(SNOWBALL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/snowball-3.1.1")
set(SNOWBALL_LIBSTEMMER "${SNOWBALL_SOURCE_DIR}/libstemmer.a")

ExternalProject_Add(snowball_ext
  SOURCE_DIR        ${SNOWBALL_SOURCE_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     make libstemmer.a "CFLAGS=-fPIC -O2"
  BUILD_IN_SOURCE   TRUE
  INSTALL_COMMAND   ""
  BUILD_BYPRODUCTS  ${SNOWBALL_LIBSTEMMER}
)

if(NOT TARGET snowball)
  add_library(snowball STATIC IMPORTED GLOBAL)
  set_target_properties(snowball PROPERTIES
    IMPORTED_LOCATION ${SNOWBALL_LIBSTEMMER}
  )
  target_include_directories(snowball SYSTEM INTERFACE
    ${SNOWBALL_SOURCE_DIR}/include
  )
  add_dependencies(snowball snowball_ext)
endif()

set(snowball_FOUND TRUE PARENT_SCOPE)
