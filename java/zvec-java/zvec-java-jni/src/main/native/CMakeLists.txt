set(ZVEC_JAVA_HOME "$ENV{JAVA_HOME}" CACHE PATH "JDK home used for JNI headers")
if(NOT ZVEC_JAVA_HOME)
    message(FATAL_ERROR "JAVA_HOME must be set to build zvec_java_jni")
endif()

set(ZVEC_JAVA_JNI_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "zvec Java JNI source directory")

if(WIN32)
    set(ZVEC_JNI_PLATFORM_INCLUDE_DIR win32)
elseif(APPLE)
    set(ZVEC_JNI_PLATFORM_INCLUDE_DIR darwin)
elseif(UNIX)
    set(ZVEC_JNI_PLATFORM_INCLUDE_DIR linux)
else()
    message(FATAL_ERROR "Unsupported JNI platform: ${CMAKE_SYSTEM_NAME}")
endif()

set(ZVEC_JNI_INCLUDE_DIRS
    "${ZVEC_JAVA_HOME}/include"
    "${ZVEC_JAVA_HOME}/include/${ZVEC_JNI_PLATFORM_INCLUDE_DIR}"
)

foreach(ZVEC_JNI_INCLUDE_DIR IN LISTS ZVEC_JNI_INCLUDE_DIRS)
    if(NOT EXISTS "${ZVEC_JNI_INCLUDE_DIR}")
        message(FATAL_ERROR "JNI include directory not found: ${ZVEC_JNI_INCLUDE_DIR}")
    endif()
endforeach()

add_library(zvec_java_jni SHARED
    "${ZVEC_JAVA_JNI_SOURCE_DIR}/zvec_java_jni.cc"
)

set_target_properties(zvec_java_jni PROPERTIES
    OUTPUT_NAME "zvec_java_jni"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(zvec_java_jni
    PRIVATE
        ${ZVEC_JNI_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_BINARY_DIR}/src/generated
)

target_link_libraries(zvec_java_jni
    PRIVATE
        zvec_c_api
)

if(APPLE)
    set_target_properties(zvec_java_jni PROPERTIES
        MACOSX_RPATH ON
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(zvec_java_jni PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()
