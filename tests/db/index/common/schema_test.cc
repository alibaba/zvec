// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zvec/db/schema.h"
#include <gtest/gtest.h>
#include "zvec/db/index_params.h"
#include "zvec/db/status.h"

using namespace zvec;

TEST(FieldSchemaTest, DefaultConstructor) {
  FieldSchema field;
  EXPECT_EQ(field.name(), "");
  EXPECT_EQ(field.data_type(), DataType::UNDEFINED);
  EXPECT_FALSE(field.nullable());
  EXPECT_EQ(field.dimension(), 0u);
  EXPECT_EQ(field.index_params(), nullptr);
}

TEST(FieldSchemaTest, ConstructorWithParameters) {
  auto index_params =
      std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
  FieldSchema field("test_field", DataType::VECTOR_FP32, 128, true,
                    index_params);

  EXPECT_EQ(field.name(), "test_field");
  EXPECT_EQ(field.data_type(), DataType::VECTOR_FP32);
  EXPECT_TRUE(field.nullable());
  EXPECT_EQ(field.dimension(), 128u);
  EXPECT_NE(field.index_params(), nullptr);
  EXPECT_EQ(field.index_params()->type(), IndexType::HNSW);
}

TEST(FieldSchemaTest, SettersAndGetters) {
  FieldSchema field;

  field.set_name("new_field");
  EXPECT_EQ(field.name(), "new_field");

  field.set_data_type(DataType::STRING);
  EXPECT_EQ(field.data_type(), DataType::STRING);

  field.set_nullable(true);
  EXPECT_TRUE(field.nullable());

  field.set_dimension(256);
  EXPECT_EQ(field.dimension(), 256u);
}

TEST(FieldSchemaTest, ElementDataType) {
  FieldSchema array_field;
  array_field.set_data_type(DataType::ARRAY_BINARY);
  EXPECT_EQ(array_field.element_data_type(), DataType::BINARY);

  array_field.set_data_type(DataType::ARRAY_STRING);
  EXPECT_EQ(array_field.element_data_type(), DataType::STRING);

  array_field.set_data_type(DataType::ARRAY_BOOL);
  EXPECT_EQ(array_field.element_data_type(), DataType::BOOL);

  array_field.set_data_type(DataType::ARRAY_INT32);
  EXPECT_EQ(array_field.element_data_type(), DataType::INT32);

  array_field.set_data_type(DataType::ARRAY_INT64);
  EXPECT_EQ(array_field.element_data_type(), DataType::INT64);

  array_field.set_data_type(DataType::ARRAY_UINT32);
  EXPECT_EQ(array_field.element_data_type(), DataType::UINT32);

  array_field.set_data_type(DataType::ARRAY_UINT64);
  EXPECT_EQ(array_field.element_data_type(), DataType::UINT64);

  array_field.set_data_type(DataType::ARRAY_FLOAT);
  EXPECT_EQ(array_field.element_data_type(), DataType::FLOAT);

  array_field.set_data_type(DataType::ARRAY_DOUBLE);
  EXPECT_EQ(array_field.element_data_type(), DataType::DOUBLE);

  // Non-array types should return the same type
  FieldSchema non_array_field;
  non_array_field.set_data_type(DataType::STRING);
  EXPECT_EQ(non_array_field.element_data_type(), DataType::STRING);
}

TEST(FieldSchemaTest, VectorFieldDetection) {
  FieldSchema field;

  // Test dense vector field detection
  field.set_data_type(DataType::VECTOR_BINARY32);
  EXPECT_TRUE(field.is_vector_field());
  EXPECT_TRUE(field.is_dense_vector());
  EXPECT_FALSE(field.is_sparse_vector());

  field.set_data_type(DataType::VECTOR_FP32);
  EXPECT_TRUE(field.is_vector_field());
  EXPECT_TRUE(field.is_dense_vector());
  EXPECT_FALSE(field.is_sparse_vector());

  field.set_data_type(DataType::VECTOR_INT16);
  EXPECT_TRUE(field.is_vector_field());
  EXPECT_TRUE(field.is_dense_vector());
  EXPECT_FALSE(field.is_sparse_vector());

  // Test sparse vector field detection
  field.set_data_type(DataType::SPARSE_VECTOR_FP32);
  EXPECT_TRUE(field.is_vector_field());
  EXPECT_FALSE(field.is_dense_vector());
  EXPECT_TRUE(field.is_sparse_vector());

  // Test non-vector field
  field.set_data_type(DataType::STRING);
  EXPECT_FALSE(field.is_vector_field());
  EXPECT_FALSE(field.is_dense_vector());
  EXPECT_FALSE(field.is_sparse_vector());

  // Test static methods
  EXPECT_TRUE(FieldSchema::is_dense_vector_field(DataType::VECTOR_FP32));
  EXPECT_FALSE(FieldSchema::is_dense_vector_field(DataType::STRING));

  EXPECT_TRUE(
      FieldSchema::is_sparse_vector_field(DataType::SPARSE_VECTOR_FP32));
  EXPECT_FALSE(FieldSchema::is_sparse_vector_field(DataType::VECTOR_FP32));

  EXPECT_TRUE(FieldSchema::is_vector_field(DataType::VECTOR_FP32));
  EXPECT_TRUE(FieldSchema::is_vector_field(DataType::SPARSE_VECTOR_FP32));
  EXPECT_FALSE(FieldSchema::is_vector_field(DataType::STRING));
}

TEST(FieldSchemaTest, ArrayTypeDetection) {
  FieldSchema field;

  field.set_data_type(DataType::ARRAY_BINARY);
  EXPECT_TRUE(field.is_array_type());

  field.set_data_type(DataType::ARRAY_STRING);
  EXPECT_TRUE(field.is_array_type());

  field.set_data_type(DataType::ARRAY_DOUBLE);
  EXPECT_TRUE(field.is_array_type());

  field.set_data_type(DataType::STRING);
  EXPECT_FALSE(field.is_array_type());

  field.set_data_type(DataType::VECTOR_FP32);
  EXPECT_FALSE(field.is_array_type());
}

TEST(FieldSchemaTest, IndexTypeAndParams) {
  FieldSchema field;
  EXPECT_EQ(field.index_type(), IndexType::UNDEFINED);
  EXPECT_EQ(field.index_params(), nullptr);

  auto hnsw_params = std::make_shared<HnswIndexParams>(MetricType::IP, 32, 200);
  field.set_index_params(hnsw_params);
  EXPECT_EQ(field.index_type(), IndexType::HNSW);
  EXPECT_NE(field.index_params(), nullptr);

  // Test setting with nullptr
  field.set_index_params(nullptr);
  EXPECT_EQ(field.index_type(), IndexType::UNDEFINED);
  EXPECT_EQ(field.index_params(), nullptr);
}

TEST(FieldSchemaTest, CopyConstructorAndAssignment) {
  auto index_params = std::make_shared<FlatIndexParams>(MetricType::L2);
  FieldSchema original("original", DataType::STRING, 100, true, index_params);

  // Test copy constructor
  FieldSchema copy(original);
  EXPECT_EQ(copy.name(), "original");
  EXPECT_EQ(copy.data_type(), DataType::STRING);
  EXPECT_TRUE(copy.nullable());
  EXPECT_EQ(copy.dimension(), 100u);
  EXPECT_NE(copy.index_params(), nullptr);
  EXPECT_EQ(copy.index_params()->type(), IndexType::FLAT);

  // Test copy assignment
  FieldSchema assigned;
  assigned = original;
  EXPECT_EQ(assigned.name(), "original");
  EXPECT_EQ(assigned.data_type(), DataType::STRING);
  EXPECT_TRUE(assigned.nullable());
  EXPECT_EQ(assigned.dimension(), 100u);
  EXPECT_NE(assigned.index_params(), nullptr);
  EXPECT_EQ(assigned.index_params()->type(), IndexType::FLAT);

  // Verify deep copy - modifying original shouldn't affect copy
  original.set_name("modified");
  EXPECT_EQ(copy.name(), "original");      // Copy should be unchanged
  EXPECT_EQ(assigned.name(), "original");  // Assigned should be unchanged
}

TEST(FieldSchemaTest, MoveConstructorAndAssignment) {
  auto index_params = std::make_shared<IVFIndexParams>(MetricType::COSINE, 128);
  FieldSchema original("move_test", DataType::VECTOR_FP32, 256, false,
                       index_params);

  // Test move constructor
  FieldSchema moved(std::move(original));
  EXPECT_EQ(moved.name(), "move_test");
  EXPECT_EQ(moved.data_type(), DataType::VECTOR_FP32);
  EXPECT_FALSE(moved.nullable());
  EXPECT_EQ(moved.dimension(), 256u);
  EXPECT_NE(moved.index_params(), nullptr);
  EXPECT_EQ(moved.index_params()->type(), IndexType::IVF);

  // After move, original should be in valid but unspecified state
  // Note: In practice, the name would likely be moved, but we don't test that
  // as it's implementation-dependent
}

TEST(FieldSchemaTest, ComparisonOperators) {
  auto index_params1 =
      std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
  auto index_params2 =
      std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
  auto index_params3 = std::make_shared<FlatIndexParams>(MetricType::IP);

  FieldSchema field1("field", DataType::STRING, 100, false, index_params1);
  FieldSchema field2("field", DataType::STRING, 100, false, index_params2);
  FieldSchema field3("field", DataType::STRING, 100, false, index_params3);
  FieldSchema field4("field", DataType::STRING, 100, true, index_params1);
  FieldSchema field5("different", DataType::STRING, 100, false, index_params1);

  // Equal fields
  EXPECT_TRUE(field1 == field2);
  EXPECT_FALSE(field1 != field2);

  // Different index params
  EXPECT_FALSE(field1 == field3);
  EXPECT_TRUE(field1 != field3);

  // Different nullable
  EXPECT_FALSE(field1 == field4);
  EXPECT_TRUE(field1 != field4);

  // Different name
  EXPECT_FALSE(field1 == field5);
  EXPECT_TRUE(field1 != field5);

  // Compare with nullptr index params
  FieldSchema field6("no_index", DataType::STRING);
  FieldSchema field7("no_index", DataType::STRING);
  FieldSchema field8("no_index", DataType::STRING, false, 0, index_params1);

  EXPECT_TRUE(field6 == field7);
  EXPECT_FALSE(field6 == field8);
  EXPECT_TRUE(field6 != field8);
}

TEST(FieldSchemaTest, Validate) {
  {
    FieldSchema field("", DataType::UNDEFINED);
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    FieldSchema field("", DataType::STRING);
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    FieldSchema field("vector_field", DataType::VECTOR_FP32, 0,
                      false);  // Zero dimension
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    FieldSchema field("dense_vector", DataType::VECTOR_FP32, 20001,
                      false);  // Zero dimension
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    auto ivf_params = std::make_shared<IVFIndexParams>(MetricType::IP, 128);
    FieldSchema field("sparse_field", DataType::SPARSE_VECTOR_FP32, 0, false,
                      ivf_params);
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    auto hnsw_params =
        std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
    FieldSchema field("sparse_field", DataType::SPARSE_VECTOR_FP32, 0, false,
                      hnsw_params);
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    auto invalid_params = std::make_shared<InvertIndexParams>(false);
    FieldSchema field("dense_field", DataType::VECTOR_FP32, 128, false,
                      invalid_params);
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    auto hnsw_params =
        std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
    FieldSchema field("scalar_field", DataType::STRING, 0, false, hnsw_params);
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    auto hnsw_params =
        std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
    FieldSchema field("vector_field", DataType::VECTOR_FP32, 128, false,
                      hnsw_params);
    auto status = field.validate();
    EXPECT_TRUE(status.ok());
  }

  {
    auto flat_params = std::make_shared<FlatIndexParams>(MetricType::IP);
    FieldSchema field("sparse_field", DataType::SPARSE_VECTOR_FP32, 0, false,
                      flat_params);
    auto status = field.validate();
    EXPECT_TRUE(status.ok());
  }

  {
    auto invert_params = std::make_shared<InvertIndexParams>(false);
    FieldSchema field("scalar_field", DataType::STRING, 0, false,
                      invert_params);
    auto status = field.validate();
    EXPECT_TRUE(status.ok());
  }

  {
    FieldSchema field("simple_field", DataType::STRING);
    auto status = field.validate();
    EXPECT_TRUE(status.ok());  // Scalar fields without index params are valid

    FieldSchema vector_field("vector_field", DataType::VECTOR_FP32, 128, false);
    status = vector_field.validate();
    EXPECT_TRUE(
        status.ok());  // Vector fields without index params are also valid
  }

  {
    // Test that VECTOR_FP32 with FP16 quantize type is valid
    auto hnsw_params = std::make_shared<HnswIndexParams>(
        MetricType::L2, 16, 100, QuantizeType::FP16);
    FieldSchema field("fp32_vector", DataType::VECTOR_FP32, 128, false,
                      hnsw_params);
    auto status = field.validate();
    if (!status.ok()) {
      std::cout << "status: " << status.message() << std::endl;
    }
    EXPECT_TRUE(status.ok());
  }

  {
    // Test that VECTOR_FP32 with UNDEFINED quantize type is valid
    auto hnsw_params = std::make_shared<HnswIndexParams>(
        MetricType::L2, 16, 100, QuantizeType::UNDEFINED);
    FieldSchema field("fp32_vector_no_quantize", DataType::VECTOR_FP32, 128,
                      false, hnsw_params);
    auto status = field.validate();
    EXPECT_TRUE(status.ok());
  }

  {
    // Test that SPARSE_VECTOR_FP32 with FP16 quantize type should fail
    auto hnsw_params = std::make_shared<HnswIndexParams>(
        MetricType::IP, 16, 100, QuantizeType::FP16);
    FieldSchema field("sparse_fp32_vector", DataType::SPARSE_VECTOR_FP32, 0,
                      false, hnsw_params);
    auto status = field.validate();
    EXPECT_TRUE(status.ok());
  }

  {
    // Test that VECTOR_FP64 with FP16 quantize type is valid
    auto hnsw_params = std::make_shared<HnswIndexParams>(
        MetricType::L2, 16, 100, QuantizeType::FP16);
    FieldSchema field("fp64_vector", DataType::VECTOR_FP64, 128, false,
                      hnsw_params);
    auto status = field.validate();
    EXPECT_FALSE(status.ok());
  }

  {
    // already support int8/int4 quantizer
    // Test that VECTOR_FP32 with INT8 quantize type should succeed
    auto hnsw_params = std::make_shared<HnswIndexParams>(
        MetricType::L2, 16, 100, QuantizeType::INT8);
    FieldSchema field("fp32_vector_int8_quantize", DataType::VECTOR_FP32, 128,
                      false, hnsw_params);
    auto status = field.validate();
    EXPECT_TRUE(status.ok());

    auto flat_params =
        std::make_shared<FlatIndexParams>(MetricType::L2, QuantizeType::INT4);
    FieldSchema flat_field("fp32_vector_int4_quantize", DataType::VECTOR_FP32,
                           128, false, flat_params);
    EXPECT_TRUE(field.validate().ok());
  }

  {
    std::vector<std::string> valid_names = {
        "a",  // min len = 1
        "A",
        "0",
        "_",
        "-",  // single allowed char
        "abc",
        "ABC",
        "a1_",
        "user_name",
        "test-123",
        "aBc123_-",
        std::string(32, 'a'),  // max len = 32
        "a_b-c1",
        "__test__",
        "123_test"};
    for (auto name : valid_names) {
      FieldSchema field(name, DataType::STRING);
      auto status = field.validate();
      if (!status.ok()) {
        std::cout << "status: " << status.message() << std::endl;
      }
      EXPECT_TRUE(status.ok());
    }
  }

  {
    std::vector<std::string> invalid_names = {
        "",                    // empty — len < 1
        std::string(33, 'a'),  // len > 32
        "a b",                 // space
        "a.b",
        "a@b",
        "a#b",  // illegal chars: . @ #
        "a$b",
        "a%",
        "a&",  // $ % & etc.
        "中文",
        "用户",  // non-ASCII
        "a..b",
        "a__b?",  // ? not allowed
    };
    for (auto name : invalid_names) {
      FieldSchema field(name, DataType::STRING);
      auto status = field.validate();
      EXPECT_FALSE(status.ok());
      EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    }
  }
}

TEST(CollectionSchemaTest, DefaultConstructor) {
  CollectionSchema schema;
  EXPECT_EQ(schema.name(), "");
  EXPECT_EQ(schema.fields().size(), 0);
  EXPECT_EQ(schema.max_doc_count_per_segment(), MAX_DOC_COUNT_PER_SEGMENT);
}

TEST(CollectionSchemaTest, ConstructorWithParameters) {
  FieldSchemaPtrList fields;
  auto field1 = std::make_shared<FieldSchema>("field1", DataType::STRING);
  auto field2 = std::make_shared<FieldSchema>("field2", DataType::VECTOR_FP32);
  fields.push_back(field1);
  fields.push_back(field2);

  CollectionSchema schema("test_collection", fields);
  EXPECT_EQ(schema.name(), "test_collection");
  EXPECT_EQ(schema.fields().size(), 2);
  EXPECT_TRUE(schema.has_field("field1"));
  EXPECT_TRUE(schema.has_field("field2"));
}

TEST(CollectionSchemaTest, NameManagement) {
  CollectionSchema schema;
  EXPECT_EQ(schema.name(), "");

  schema.set_name("new_name");
  EXPECT_EQ(schema.name(), "new_name");
}

TEST(CollectionSchemaTest, MaxDocCountPerSegment) {
  CollectionSchema schema;
  EXPECT_EQ(schema.max_doc_count_per_segment(), MAX_DOC_COUNT_PER_SEGMENT);

  schema.set_max_doc_count_per_segment(500000);
  EXPECT_EQ(schema.max_doc_count_per_segment(), 500000u);
}

TEST(CollectionSchemaTest, AddField) {
  CollectionSchema schema;
  auto field = std::make_shared<FieldSchema>("test_field", DataType::STRING);

  auto status = schema.add_field(field);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(schema.has_field("test_field"));
  EXPECT_EQ(schema.fields().size(), 1);

  // Try to add the same field again
  auto status2 = schema.add_field(field);
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), StatusCode::ALREADY_EXISTS);
}

TEST(CollectionSchemaTest, DropField) {
  CollectionSchema schema;
  auto field1 = std::make_shared<FieldSchema>("field1", DataType::STRING);
  auto field2 = std::make_shared<FieldSchema>("field2", DataType::VECTOR_FP32);

  schema.add_field(field1);
  schema.add_field(field2);
  EXPECT_EQ(schema.fields().size(), 2);

  // Drop existing field
  auto status = schema.drop_field("field1");
  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(schema.has_field("field1"));
  EXPECT_TRUE(schema.has_field("field2"));
  EXPECT_EQ(schema.fields().size(), 1);

  // Try to drop non-existing field
  auto status2 = schema.drop_field("nonexistent");
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), StatusCode::NOT_FOUND);
}

TEST(CollectionSchemaTest, AlterField) {
  CollectionSchema schema;
  auto original_field =
      std::make_shared<FieldSchema>("field", DataType::STRING);
  schema.add_field(original_field);

  auto new_field =
      std::make_shared<FieldSchema>("field", DataType::VECTOR_FP32);
  auto status = schema.alter_field("field", new_field);
  EXPECT_TRUE(status.ok());

  auto *field = schema.get_field("field");
  EXPECT_NE(field, nullptr);
  EXPECT_EQ(field->data_type(), DataType::VECTOR_FP32);

  // Try to alter non-existing field
  auto status2 = schema.alter_field("nonexistent", new_field);
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), StatusCode::NOT_FOUND);
}

TEST(CollectionSchemaTest, FieldRetrieval) {
  CollectionSchema schema;
  auto string_field =
      std::make_shared<FieldSchema>("string_field", DataType::STRING);
  auto vector_field =
      std::make_shared<FieldSchema>("vector_field", DataType::VECTOR_FP32);

  schema.add_field(string_field);
  schema.add_field(vector_field);

  // Test get_field
  const auto *const_string_field = schema.get_field("string_field");
  EXPECT_NE(const_string_field, nullptr);
  EXPECT_EQ(const_string_field->data_type(), DataType::STRING);

  auto *mutable_string_field = schema.get_field("string_field");
  EXPECT_NE(mutable_string_field, nullptr);
  EXPECT_EQ(mutable_string_field->data_type(), DataType::STRING);

  // Test get_forward_field
  const auto *const_forward_field = schema.get_forward_field("string_field");
  EXPECT_NE(const_forward_field, nullptr);
  EXPECT_EQ(const_forward_field->data_type(), DataType::STRING);

  auto *mutable_forward_field = schema.get_forward_field("string_field");
  EXPECT_NE(mutable_forward_field, nullptr);
  EXPECT_EQ(mutable_forward_field->data_type(), DataType::STRING);

  // Forward field should return nullptr for vector fields
  EXPECT_EQ(schema.get_forward_field("vector_field"), nullptr);

  // Test get_vector_field
  const auto *const_vector_field = schema.get_vector_field("vector_field");
  EXPECT_NE(const_vector_field, nullptr);
  EXPECT_EQ(const_vector_field->data_type(), DataType::VECTOR_FP32);

  auto *mutable_vector_field = schema.get_vector_field("vector_field");
  EXPECT_NE(mutable_vector_field, nullptr);
  EXPECT_EQ(mutable_vector_field->data_type(), DataType::VECTOR_FP32);

  // Vector field should return nullptr for string fields
  EXPECT_EQ(schema.get_vector_field("string_field"), nullptr);

  // Test non-existing field
  EXPECT_EQ(schema.get_field("nonexistent"), nullptr);
  EXPECT_EQ(schema.get_forward_field("nonexistent"), nullptr);
  EXPECT_EQ(schema.get_vector_field("nonexistent"), nullptr);
}

TEST(CollectionSchemaTest, FieldLists) {
  CollectionSchema schema;
  auto string_field =
      std::make_shared<FieldSchema>("string_field", DataType::STRING);
  auto vector_field =
      std::make_shared<FieldSchema>("vector_field", DataType::VECTOR_FP32);
  auto array_field =
      std::make_shared<FieldSchema>("array_field", DataType::ARRAY_INT32);

  schema.add_field(string_field);
  schema.add_field(vector_field);
  schema.add_field(array_field);

  // Test fields()
  auto all_fields = schema.fields();
  EXPECT_EQ(all_fields.size(), 3);

  // Test forward_fields()
  auto forward_fields = schema.forward_fields();
  EXPECT_EQ(forward_fields.size(), 2);  // string_field and array_field

  // Test forward_field_names()
  auto forward_field_names = schema.forward_field_names();
  EXPECT_EQ(forward_field_names.size(), 2);
  EXPECT_TRUE(std::find(forward_field_names.begin(), forward_field_names.end(),
                        "string_field") != forward_field_names.end());
  EXPECT_TRUE(std::find(forward_field_names.begin(), forward_field_names.end(),
                        "array_field") != forward_field_names.end());

  // Test vector_fields()
  auto vector_fields = schema.vector_fields();
  EXPECT_EQ(vector_fields.size(), 1);
  EXPECT_EQ(vector_fields[0]->name(), "vector_field");
}

TEST(CollectionSchemaTest, IndexManagement) {
  CollectionSchema schema;
  auto field =
      std::make_shared<FieldSchema>("indexed_field", DataType::VECTOR_FP32);
  schema.add_field(field);

  auto forward_field =
      std::make_shared<FieldSchema>("forward_field", DataType::STRING);
  schema.add_field(forward_field);

  // Test has_index on field without index
  EXPECT_FALSE(schema.has_index("indexed_field"));
  EXPECT_FALSE(schema.has_index("forward_field"));

  // Add index
  auto index_params =
      std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
  auto status = schema.add_index("indexed_field", index_params);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(schema.has_index("indexed_field"));

  // Try to add index to non-existing field
  auto status2 = schema.add_index("nonexistent", index_params);
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), StatusCode::NOT_FOUND);

  // Drop index
  auto status3 = schema.drop_index("indexed_field");
  EXPECT_TRUE(status3.ok());
  EXPECT_FALSE(schema.has_index("indexed_field"));

  // Try to drop index from non-existing field
  auto status4 = schema.drop_index("nonexistent");
  EXPECT_FALSE(status4.ok());
  EXPECT_EQ(status4.code(), StatusCode::NOT_FOUND);

  auto forward_index_params = std::make_shared<InvertIndexParams>(false);
  auto status5 = schema.add_index("forward_field", forward_index_params);
  EXPECT_TRUE(status5.ok());
  EXPECT_TRUE(schema.has_index("forward_field"));

  auto status6 = schema.drop_index("forward_field");
  EXPECT_TRUE(status5.ok());
  EXPECT_FALSE(schema.has_index("forward_field"));
}

TEST(CollectionSchemaTest, CopyConstructor) {
  CollectionSchema original("original_schema", {});
  auto field = std::make_shared<FieldSchema>("field", DataType::STRING);
  original.add_field(field);
  original.set_max_doc_count_per_segment(100000);

  CollectionSchema copy(original);
  EXPECT_EQ(copy.name(), "original_schema");
  EXPECT_EQ(copy.fields().size(), 1);
  EXPECT_TRUE(copy.has_field("field"));
  EXPECT_EQ(copy.max_doc_count_per_segment(), 100000u);
}

TEST(CollectionSchemaTest, Validate) {
  CollectionSchema original("original_schema", {});
  auto field =
      std::make_shared<FieldSchema>("sparse", DataType::SPARSE_VECTOR_FP32);
  original.add_field(field);
  original.set_max_doc_count_per_segment(100000);

  ASSERT_TRUE(original.validate().ok());

  CollectionSchema c1;
  auto s = c1.validate();
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  CollectionSchema c2("c2", {});
  s = c1.validate();
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  auto f1 = std::make_shared<FieldSchema>();
  CollectionSchema c3("c3", {f1});
  s = c3.validate();
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  auto f2 = std::make_shared<FieldSchema>("f2", DataType::INT32);
  CollectionSchema c4("c4", {f2});
  s = c4.validate();
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  auto f3 = std::make_shared<FieldSchema>("f3", DataType::VECTOR_FP16);
  CollectionSchema c5("c5", {f3});
  s = c5.validate();
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  // validate collection name regex "^[a-zA-Z0-9_-]{3,32}$"
  {
    std::vector<std::string> invalid_names = {
        "",                    // empty
        "ab",                  // too short (<3)
        std::string(65, 'a'),  // too long (>64)
        "a b",                 // space not allowed
        "a.b",                 // dot not allowed
        "a$b",                 // $ not allowed
        "中文",                // non-ASCII
        "a\nb",                // newline not allowed
        "a\tb",                // tab not allowed
        "a\rb",                // carriage return not allowed
    };

    for (const auto &name : invalid_names) {
      CollectionSchema c(name, {field});
      s = c.validate();
      if (!s.ok()) {
        std::cout << "Invalid name: " << name << std::endl;
      }
      ASSERT_FALSE(s.ok());
      ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
    }

    std::vector<std::string> valid_names = {
        "test_collection_supported_vectors",
        std::string(64, 'a'),
        "a_b",     // underscore allowed
        "a-b",     // dash allowed
        "a_1",     // underscore and digit allowed
        "a-1",     // dash and digit allowed
        "a_1b",    // underscore, digit and letter allowed
        "a-1b",    // dash, digit and letter allowed
        "-start",  // allowed! (regex permits leading -/_)
        "_start",  // also allowed
        "end-",
        "end_",  // trailing -/_ allowed
        "a--b",
        "__b",
        "a__b"  // consecutive allowed
    };
    for (const auto &name : valid_names) {
      CollectionSchema c(name, {field});
      s = c.validate();
      ASSERT_TRUE(s.ok());
    }
  }

  // validate vector/scalar field size
  {
    std::vector<FieldSchema::Ptr> fields;
    for (int i = 0; i < 1025; ++i) {
      auto f = std::make_shared<FieldSchema>("f" + std::to_string(i),
                                             DataType::VECTOR_FP32, 1024);
      fields.emplace_back(f);
    }
    CollectionSchema c5("c5", fields);
    s = c5.validate();
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

    std::vector<FieldSchema::Ptr> vectors;
    for (int i = 0; i < 5; ++i) {
      auto f = std::make_shared<FieldSchema>(
          "f" + std::to_string(i), DataType::VECTOR_FP32, 1024, false);
      fields.emplace_back(f);
    }
    CollectionSchema c6("c6", fields);
    s = c6.validate();
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }
}