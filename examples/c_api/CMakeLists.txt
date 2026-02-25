# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Basic example
add_executable(c_api_basic_example basic_example.c)
target_link_libraries(c_api_basic_example PRIVATE zvec_c_api)
target_include_directories(c_api_basic_example PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
)
set_target_properties(c_api_basic_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/c_api
)


# Schema example
add_executable(c_api_collection_schema_example collection_schema_example.c)
target_link_libraries(c_api_collection_schema_example PRIVATE zvec_c_api)
target_include_directories(c_api_collection_schema_example PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
)
set_target_properties(c_api_collection_schema_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/c_api
)

# Struct document example
add_executable(c_api_doc_example doc_example.c)
target_link_libraries(c_api_doc_example PRIVATE zvec_c_api)
target_include_directories(c_api_doc_example PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
)
set_target_properties(c_api_doc_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/c_api
)

# Index example
add_executable(c_api_index_example index_example.c)
target_link_libraries(c_api_index_example PRIVATE zvec_c_api)
set_target_properties(c_api_index_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples/c_api
)

# Newly added field schema example
add_executable(c_api_field_schema_example field_schema_example.c)
target_link_libraries(c_api_field_schema_example PRIVATE zvec_c_api)
set_target_properties(c_api_field_schema_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples/c_api
)

# Optimized example
add_executable(c_api_optimized_example optimized_example.c)
target_link_libraries(c_api_optimized_example PRIVATE zvec_c_api)
set_target_properties(c_api_optimized_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples/c_api
)
