add_subdirectory(OMEGALib)

# Set C++ standard for omega target to fix Eigen compilation issues
if(TARGET omega)
  set_target_properties(omega PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endif()
