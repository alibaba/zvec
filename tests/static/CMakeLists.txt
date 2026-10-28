# A real archive consumer: no component shared libraries or registration roots
# supplied by this test. The shipped core archive must be self-contained.
find_package(Threads REQUIRED)
add_executable(index_factory_static_test EXCLUDE_FROM_ALL index_factory_static_test.cc)
zvec_configure_ios_test(index_factory_static_test)
target_compile_features(index_factory_static_test PRIVATE cxx_std_17)
target_link_libraries(index_factory_static_test PRIVATE
  zvec_core zvec_turbo zvec_ailego Threads::Threads gtest gtest_main)
add_dependencies(unittest index_factory_static_test)
add_test(NAME index_factory_static_test COMMAND index_factory_static_test)
