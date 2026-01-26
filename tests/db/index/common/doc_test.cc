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

#include "zvec/db/doc.h"
#include <cstdint>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include "utils/utils.h"
#include "zvec/db/status.h"
#include "zvec/db/type.h"


using namespace zvec;

class DocDetailedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_doc_ = std::make_shared<Doc>();
    test_doc_->set_pk("test_pk");
    test_doc_->set_doc_id(12345);
    test_doc_->set_score(0.95f);
    test_doc_->set_operator(Operator::INSERT);
  }

  Doc::Ptr test_doc_;
};

// Test serialization and deserialization of basic data types
TEST_F(DocDetailedTest, BasicTypeSerializationDeserialization) {
  // Test boundary values
  test_doc_->set("bool_true", true);
  test_doc_->set("bool_false", false);
  test_doc_->set("int32_min", std::numeric_limits<int32_t>::min());
  test_doc_->set("int32_max", std::numeric_limits<int32_t>::max());
  test_doc_->set("uint32_min", std::numeric_limits<uint32_t>::min());
  test_doc_->set("uint32_max", std::numeric_limits<uint32_t>::max());
  test_doc_->set("int64_min", std::numeric_limits<int64_t>::min());
  test_doc_->set("int64_max", std::numeric_limits<int64_t>::max());
  test_doc_->set("uint64_min", std::numeric_limits<uint64_t>::min());
  test_doc_->set("uint64_max", std::numeric_limits<uint64_t>::max());
  test_doc_->set("float_min", std::numeric_limits<float>::min());
  test_doc_->set("float_max", std::numeric_limits<float>::max());
  test_doc_->set("float_lowest", std::numeric_limits<float>::lowest());
  test_doc_->set("double_min", std::numeric_limits<double>::min());
  test_doc_->set("double_max", std::numeric_limits<double>::max());
  test_doc_->set("double_lowest", std::numeric_limits<double>::lowest());

  auto serialized = test_doc_->serialize();
  ASSERT_FALSE(serialized.empty());

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  EXPECT_EQ(deserialized_doc->get<bool>("bool_true").value(), true);
  EXPECT_EQ(deserialized_doc->get<bool>("bool_false").value(), false);
  EXPECT_EQ(deserialized_doc->get<int32_t>("int32_min").value(),
            std::numeric_limits<int32_t>::min());
  EXPECT_EQ(deserialized_doc->get<int32_t>("int32_max").value(),
            std::numeric_limits<int32_t>::max());
  EXPECT_EQ(deserialized_doc->get<uint32_t>("uint32_min").value(),
            std::numeric_limits<uint32_t>::min());
  EXPECT_EQ(deserialized_doc->get<uint32_t>("uint32_max").value(),
            std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(deserialized_doc->get<int64_t>("int64_min").value(),
            std::numeric_limits<int64_t>::min());
  EXPECT_EQ(deserialized_doc->get<int64_t>("int64_max").value(),
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(deserialized_doc->get<uint64_t>("uint64_min").value(),
            std::numeric_limits<uint64_t>::min());
  EXPECT_EQ(deserialized_doc->get<uint64_t>("uint64_max").value(),
            std::numeric_limits<uint64_t>::max());

  // For floating point numbers, use approximate comparison
  EXPECT_FLOAT_EQ(deserialized_doc->get<float>("float_min").value(),
                  std::numeric_limits<float>::min());
  EXPECT_FLOAT_EQ(deserialized_doc->get<float>("float_max").value(),
                  std::numeric_limits<float>::max());
  EXPECT_FLOAT_EQ(deserialized_doc->get<float>("float_lowest").value(),
                  std::numeric_limits<float>::lowest());
  EXPECT_DOUBLE_EQ(deserialized_doc->get<double>("double_min").value(),
                   std::numeric_limits<double>::min());
  EXPECT_DOUBLE_EQ(deserialized_doc->get<double>("double_max").value(),
                   std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(deserialized_doc->get<double>("double_lowest").value(),
                   std::numeric_limits<double>::lowest());
}

// Test various cases of string types
TEST_F(DocDetailedTest, StringTypeSerializationDeserialization) {
  // Test empty string
  test_doc_->set("empty_string", std::string(""));

  // Test long string
  std::string long_string(10000, 'a');
  test_doc_->set("long_string", long_string);

  // Test string with special characters
  test_doc_->set("special_chars",
                 std::string("Special characters\t\n\r\0included", 15));

  // Test string with binary data
  std::string binary_string;
  for (int i = 0; i < 256; ++i) {
    binary_string.push_back(static_cast<char>(i));
  }
  test_doc_->set("binary_string", binary_string);

  auto serialized = test_doc_->serialize();
  ASSERT_FALSE(serialized.empty());

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  EXPECT_EQ(deserialized_doc->get<std::string>("empty_string").value(), "");
  EXPECT_EQ(deserialized_doc->get<std::string>("long_string").value(),
            long_string);
  EXPECT_EQ(deserialized_doc->get<std::string>("special_chars").value(),
            std::string("Special characters\t\n\r\0included", 15));
  EXPECT_EQ(deserialized_doc->get<std::string>("binary_string").value(),
            binary_string);
}


// Test vector<bool> type
TEST_F(DocDetailedTest, VectorBoolSerializationDeserialization) {
  std::vector<bool> bool_vec;
  // Create a vector<bool> with a large number of elements
  for (int i = 0; i < 1000; ++i) {
    bool_vec.push_back(i % 2 == 0);
  }
  test_doc_->set("bool_vec", bool_vec);

  auto serialized = test_doc_->serialize();
  ASSERT_FALSE(serialized.empty());

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  auto deserialized_vec =
      deserialized_doc->get<std::vector<bool>>("bool_vec").value();

  ASSERT_EQ(deserialized_vec.size(), bool_vec.size());
  for (size_t i = 0; i < bool_vec.size(); ++i) {
    EXPECT_EQ(deserialized_vec[i], bool_vec[i]) << "Mismatch at index " << i;
  }
}

// Test numeric vector types
TEST_F(DocDetailedTest, NumericVectorSerializationDeserialization) {
  // Test int8_t vector
  std::vector<int8_t> int8_vec = {std::numeric_limits<int8_t>::min(), -1, 0, 1,
                                  std::numeric_limits<int8_t>::max()};
  test_doc_->set("int8_vec", int8_vec);

  // Test int16_t vector
  std::vector<int16_t> int16_vec = {std::numeric_limits<int16_t>::min(), -1, 0,
                                    1, std::numeric_limits<int16_t>::max()};
  test_doc_->set("int16_vec", int16_vec);

  // Test int32_t vector
  std::vector<int32_t> int32_vec = {std::numeric_limits<int32_t>::min(), -1, 0,
                                    1, std::numeric_limits<int32_t>::max()};
  test_doc_->set("int32_vec", int32_vec);

  // Test int64_t vector
  std::vector<int64_t> int64_vec = {std::numeric_limits<int64_t>::min(), -1, 0,
                                    1, std::numeric_limits<int64_t>::max()};
  test_doc_->set("int64_vec", int64_vec);

  // Test uint32_t vector
  std::vector<uint32_t> uint32_vec = {std::numeric_limits<uint32_t>::min(), 1,
                                      100,
                                      std::numeric_limits<uint32_t>::max()};
  test_doc_->set("uint32_vec", uint32_vec);

  // Test uint64_t vector
  std::vector<uint64_t> uint64_vec = {std::numeric_limits<uint64_t>::min(), 1,
                                      100,
                                      std::numeric_limits<uint64_t>::max()};
  test_doc_->set("uint64_vec", uint64_vec);

  // Test float vector
  std::vector<float> float_vec = {std::numeric_limits<float>::min(), -1.0f,
                                  0.0f, 1.0f,
                                  std::numeric_limits<float>::max()};
  test_doc_->set("float_vec", float_vec);

  // Test double vector
  std::vector<double> double_vec = {std::numeric_limits<double>::min(), -1.0,
                                    0.0, 1.0,
                                    std::numeric_limits<double>::max()};
  test_doc_->set("double_vec", double_vec);

  auto serialized = test_doc_->serialize();
  ASSERT_FALSE(serialized.empty());
  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  EXPECT_EQ(deserialized_doc->get<std::vector<int8_t>>("int8_vec").value(),
            int8_vec);
  EXPECT_EQ(deserialized_doc->get<std::vector<int16_t>>("int16_vec").value(),
            int16_vec);
  EXPECT_EQ(deserialized_doc->get<std::vector<int32_t>>("int32_vec").value(),
            int32_vec);
  EXPECT_EQ(deserialized_doc->get<std::vector<int64_t>>("int64_vec").value(),
            int64_vec);
  EXPECT_EQ(deserialized_doc->get<std::vector<uint32_t>>("uint32_vec").value(),
            uint32_vec);
  EXPECT_EQ(deserialized_doc->get<std::vector<uint64_t>>("uint64_vec").value(),
            uint64_vec);


  // Floating point numbers use approximate comparison
  auto deserialized_float_vec =
      deserialized_doc->get<std::vector<float>>("float_vec").value();

  ASSERT_EQ(deserialized_float_vec.size(), float_vec.size());
  for (size_t i = 0; i < float_vec.size(); ++i) {
    EXPECT_FLOAT_EQ(deserialized_float_vec[i], float_vec[i])
        << "Mismatch at index " << i;
  }

  auto deserialized_double_vec =
      deserialized_doc->get<std::vector<double>>("double_vec").value();
  ASSERT_EQ(deserialized_double_vec.size(), double_vec.size());
  for (size_t i = 0; i < double_vec.size(); ++i) {
    EXPECT_DOUBLE_EQ(deserialized_double_vec[i], double_vec[i])
        << "Mismatch at index " << i;
  }
}

// Test string vector types
TEST_F(DocDetailedTest, StringVectorSerializationDeserialization) {
  std::vector<std::string> string_vec;
  string_vec.push_back("");  // Empty string
  string_vec.push_back("normal string");
  string_vec.push_back(std::string(1000, 'x'));  // Long string
  string_vec.push_back("Special character test");
  string_vec.push_back(
      std::string("binary\0data", 11));  // Contains binary data

  test_doc_->set("string_vec", string_vec);

  auto serialized = test_doc_->serialize();
  ASSERT_FALSE(serialized.empty());

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  auto deserialized_vec =
      deserialized_doc->get<std::vector<std::string>>("string_vec").value();
  ASSERT_EQ(deserialized_vec.size(), string_vec.size());
  for (size_t i = 0; i < string_vec.size(); ++i) {
    EXPECT_EQ(deserialized_vec[i], string_vec[i]) << "Mismatch at index " << i;
  }
}

// Test sparse vector types
TEST_F(DocDetailedTest, SparseVectorSerializationDeserialization) {
  // Test float type sparse vector
  std::pair<std::vector<uint32_t>, std::vector<float>> sparse_float_vec;
  sparse_float_vec.first = {0, 100, 1000, 10000};
  sparse_float_vec.second = {0.1f, 100.5f, -200.7f,
                             std::numeric_limits<float>::max()};

  test_doc_->set("sparse_float_vec", sparse_float_vec);

  // Test ailego::Float16 type sparse vector
  std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>
      sparse_float16_vec;
  sparse_float16_vec.first = {1, 50, 500};
  sparse_float16_vec.second = {ailego::Float16(0.5f), ailego::Float16(-10.25f),
                               ailego::Float16(1000.0f)};

  test_doc_->set("sparse_float16_vec", sparse_float16_vec);

  auto serialized = test_doc_->serialize();
  ASSERT_FALSE(serialized.empty());

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  // Verify float sparse vector
  auto deserialized_float_vec =
      deserialized_doc
          ->get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              "sparse_float_vec")
          .value();

  EXPECT_EQ(deserialized_float_vec.first, sparse_float_vec.first);
  ASSERT_EQ(deserialized_float_vec.second.size(),
            sparse_float_vec.second.size());
  for (size_t i = 0; i < sparse_float_vec.second.size(); ++i) {
    EXPECT_FLOAT_EQ(deserialized_float_vec.second[i],
                    sparse_float_vec.second[i])
        << "Mismatch at index " << i;
  }

  // Verify float16 sparse vector
  auto deserialized_float16_vec =
      deserialized_doc
          ->get<std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>>(
              "sparse_float16_vec")
          .value();

  EXPECT_EQ(deserialized_float16_vec.first, sparse_float16_vec.first);
  EXPECT_EQ(deserialized_float16_vec.second, sparse_float16_vec.second);
}

// Test case with many fields
TEST_F(DocDetailedTest, ManyFieldsSerializationDeserialization) {
  const int field_count = 1000;
  for (int i = 0; i < field_count; ++i) {
    test_doc_->set("field_" + std::to_string(i), i);
  }

  auto serialized = test_doc_->serialize();
  ASSERT_FALSE(serialized.empty());

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  for (int i = 0; i < field_count; ++i) {
    std::string field_name = "field_" + std::to_string(i);
    EXPECT_EQ(deserialized_doc->get<int32_t>(field_name).value(), i);
  }
}

// Test empty document
TEST_F(DocDetailedTest, EmptyDocSerializationDeserialization) {
  Doc::Ptr empty_doc = std::make_shared<Doc>();
  empty_doc->set_pk("");  // Empty primary key

  auto serialized = empty_doc->serialize();
  ASSERT_FALSE(serialized.empty());

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);
  EXPECT_EQ(deserialized_doc->pk(), "");
}

// Test large document
TEST_F(DocDetailedTest, LargeDocSerializationDeserialization) {
  // Create a document with a large amount of data
  std::string large_string(100000, 'A');
  test_doc_->set("large_string", large_string);

  std::vector<int32_t> large_vector(50000);
  std::iota(large_vector.begin(), large_vector.end(), 0);
  test_doc_->set("large_vector", large_vector);

  auto serialized = test_doc_->serialize();
  EXPECT_GT(serialized.size(), 100000);  // Should be a large document

  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());
  ASSERT_NE(deserialized_doc, nullptr);

  EXPECT_EQ(deserialized_doc->get<std::string>("large_string").value(),
            large_string);
  EXPECT_EQ(deserialized_doc->get<std::vector<int32_t>>("large_vector").value(),
            large_vector);
}

// Test memory usage calculation
TEST_F(DocDetailedTest, MemoryUsageCalculation) {
  size_t initial_usage = test_doc_->memory_usage();

  // Add some fields
  test_doc_->set("small_string", std::string("small"));
  test_doc_->set("int_field", int32_t(42));
  test_doc_->set("float_field", 3.14f);

  size_t usage_with_fields = test_doc_->memory_usage();
  EXPECT_GT(usage_with_fields, initial_usage);

  // Add a large field
  std::string large_string(10000, 'B');
  test_doc_->set("large_string", large_string);

  size_t usage_with_large_field = test_doc_->memory_usage();
  EXPECT_GT(usage_with_large_field, usage_with_fields);
}

// Test detailed string representation
TEST_F(DocDetailedTest, DetailStringRepresentation) {
  test_doc_->set("test_bool", true);
  test_doc_->set("test_int", int32_t(-42));
  test_doc_->set("test_string", std::string("hello"));

  std::vector<float> float_vec = {1.1f, 2.2f, 3.3f};
  test_doc_->set("test_float_vec", float_vec);

  std::string detail_str = test_doc_->to_detail_string();
  EXPECT_FALSE(detail_str.empty());
  EXPECT_NE(detail_str.find("test_pk"), std::string::npos);
  EXPECT_NE(detail_str.find("test_bool"), std::string::npos);
  EXPECT_NE(detail_str.find("test_int"), std::string::npos);
  EXPECT_NE(detail_str.find("test_string"), std::string::npos);
  EXPECT_NE(detail_str.find("test_float_vec"), std::string::npos);
}

// Test operator types
TEST_F(DocDetailedTest, OperatorTypes) {
  test_doc_->set_operator(Operator::INSERT);
  EXPECT_EQ(test_doc_->get_operator(), Operator::INSERT);

  test_doc_->set_operator(Operator::DELETE);
  EXPECT_EQ(test_doc_->get_operator(), Operator::DELETE);

  test_doc_->set_operator(Operator::UPDATE);
  EXPECT_EQ(test_doc_->get_operator(), Operator::UPDATE);
}

// Test document ID and score
TEST_F(DocDetailedTest, DocIdAndScore) {
  test_doc_->set_doc_id(0);
  EXPECT_EQ(test_doc_->doc_id(), 0);

  test_doc_->set_doc_id(std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(test_doc_->doc_id(), std::numeric_limits<uint64_t>::max());

  test_doc_->set_score(0.0f);
  EXPECT_FLOAT_EQ(test_doc_->score(), 0.0f);

  test_doc_->set_score(1.0f);
  EXPECT_FLOAT_EQ(test_doc_->score(), 1.0f);

  test_doc_->set_score(-1.0f);
  EXPECT_FLOAT_EQ(test_doc_->score(), -1.0f);

  test_doc_->set_score(std::numeric_limits<float>::max());
  EXPECT_FLOAT_EQ(test_doc_->score(), std::numeric_limits<float>::max());
}

// Test primary key
TEST_F(DocDetailedTest, PrimaryKey) {
  test_doc_->set_pk("");
  EXPECT_EQ(test_doc_->pk(), "");

  std::string long_pk(10000, 'X');
  test_doc_->set_pk(long_pk);
  EXPECT_EQ(test_doc_->pk(), long_pk);

  test_doc_->set_pk("normal_pk");
  EXPECT_EQ(test_doc_->pk(), "normal_pk");
}

// Test duplicate field names (should overwrite old values)
TEST_F(DocDetailedTest, DuplicateFieldNames) {
  test_doc_->set("duplicate_field", int32_t(1));
  test_doc_->set("duplicate_field", int32_t(2));  // Overwrite old value

  auto serialized = test_doc_->serialize();
  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(deserialized_doc->get<int32_t>("duplicate_field").value(), 2);
}

// Test combination of various data types
TEST_F(DocDetailedTest, MixedDataTypes) {
  test_doc_->set("bool_field", true);
  test_doc_->set("int_field", int32_t(-1000));
  test_doc_->set("uint_field", uint32_t(2000));
  test_doc_->set("float_field", 3.14159f);
  test_doc_->set("double_field", 2.718281828459045);
  test_doc_->set("string_field", std::string("Hello, World!"));

  std::vector<int32_t> int_vec = {1, 2, 3, 4, 5};
  test_doc_->set("int_vec", int_vec);

  std::vector<float> float_vec = {1.1f, 2.2f, 3.3f};
  test_doc_->set("float_vec", float_vec);

  std::vector<std::string> string_vec = {"apple", "banana", "cherry"};
  test_doc_->set("string_vec", string_vec);

  std::pair<std::vector<uint32_t>, std::vector<float>> sparse_vec;
  sparse_vec.first = {1, 10, 100};
  sparse_vec.second = {0.1f, 1.0f, 10.0f};
  test_doc_->set("sparse_vec", sparse_vec);

  auto serialized = test_doc_->serialize();
  auto deserialized_doc =
      Doc::deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(deserialized_doc->get<bool>("bool_field").value(), true);
  EXPECT_EQ(deserialized_doc->get<int32_t>("int_field").value(), -1000);
  EXPECT_EQ(deserialized_doc->get<uint32_t>("uint_field").value(), 2000);
  EXPECT_FLOAT_EQ(deserialized_doc->get<float>("float_field").value(),
                  3.14159f);
  EXPECT_DOUBLE_EQ(deserialized_doc->get<double>("double_field").value(),
                   2.718281828459045);
  EXPECT_EQ(deserialized_doc->get<std::string>("string_field").value(),
            "Hello, World!");
  EXPECT_EQ(deserialized_doc->get<std::vector<int32_t>>("int_vec").value(),
            int_vec);
  EXPECT_EQ(deserialized_doc->get<std::vector<float>>("float_vec").value(),
            float_vec);
  EXPECT_EQ(
      deserialized_doc->get<std::vector<std::string>>("string_vec").value(),
      string_vec);

  auto deserialized_sparse =
      deserialized_doc
          ->get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              "sparse_vec")
          .value();
  EXPECT_EQ(deserialized_sparse.first, sparse_vec.first);
  EXPECT_EQ(deserialized_sparse.second, sparse_vec.second);
}

// Test doc validate with schema
TEST_F(DocDetailedTest, Validate) {
  // test schema nullable=false, but doc's field is null
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    auto doc = test::TestHelper::CreateDoc(1, *schema);

    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    doc = test::TestHelper::CreateDocNull(1, *schema);
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  {
    auto schema = test::TestHelper::CreateNormalSchema(true);
    auto doc = test::TestHelper::CreateDoc(1, *schema);

    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    doc = test::TestHelper::CreateDocNull(1, *schema);
    s = doc.validate(schema);
    ASSERT_TRUE(s.ok());
  }

  // doc contained another field which not contained in schema
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    auto doc = test::TestHelper::CreateDoc(1, *schema);
    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    doc.set("another_field", 1);
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // doc contained a mismatch scalar field
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    auto doc = test::TestHelper::CreateDoc(1, *schema);
    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    doc.set("int32", std::string("1"));
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // doc contained a mismatch type vector field
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    auto doc = test::TestHelper::CreateDoc(1, *schema);
    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    std::string field = "dense_fp32";
    auto field_schema = schema->get_field(field);
    ASSERT_NE(field_schema, nullptr);

    doc.set(field, std::vector<int16_t>(field_schema->dimension(), 1));
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // doc contained a vector field with invalid dimension
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    auto doc = test::TestHelper::CreateDoc(1, *schema);
    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    std::string field = "dense_fp32";
    auto field_schema = schema->get_field(field);
    ASSERT_NE(field_schema, nullptr);

    doc.set(field, std::vector<float>(field_schema->dimension() - 1, 1.0));
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

    doc.set(field, std::vector<float>());
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // doc contained a sparse vector field with mismatch type
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    auto doc = test::TestHelper::CreateDoc(1, *schema);
    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    std::string field = "sparse_fp32";
    auto field_schema = schema->get_field(field);
    ASSERT_NE(field_schema, nullptr);

    doc.set(field, std::vector<int16_t>(field_schema->dimension(), 1));
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // doc contained a sparse vector field with indices/values size mismatch
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    auto doc = test::TestHelper::CreateDoc(1, *schema);
    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());

    std::string field = "sparse_fp32";
    auto field_schema = schema->get_field(field);
    ASSERT_NE(field_schema, nullptr);

    std::vector<uint32_t> indices;
    std::vector<float> values;
    for (uint32_t i = 0; i < 100; i++) {
      indices.push_back(i);
      values.push_back(float(0.1));
    }
    values.push_back(float(0.1));
    std::pair<std::vector<uint32_t>, std::vector<float>> sparse_float_vec{
        indices, values};
    doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
        field, sparse_float_vec);
    s = doc.validate(schema);
    ASSERT_FALSE(s.ok());
    ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // doc validate error
  {
    Doc doc;
    // schema is null
    auto s = doc.validate(nullptr);
    EXPECT_EQ(s.code(), StatusCode::INTERNAL_ERROR);

    // pk is null
    auto schema = test::TestHelper::CreateNormalSchema(false);
    s = doc.validate(schema);
    EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

    // field type is undefined
    schema->add_field(
        std::make_shared<FieldSchema>("undefined", DataType::UNDEFINED, true));
    s = doc.validate(schema);
    EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // doc validate more data type
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    schema->add_field(
        std::make_shared<FieldSchema>("binary", DataType::BINARY, false));

    schema->add_field(std::make_shared<FieldSchema>(
        "array_binary", DataType::ARRAY_BINARY, false));

    schema->add_field(std::make_shared<FieldSchema>(
        "vector_binary32", DataType::VECTOR_BINARY32, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));

    schema->add_field(std::make_shared<FieldSchema>(
        "vector_binary64", DataType::VECTOR_BINARY64, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));

    schema->add_field(std::make_shared<FieldSchema>(
        "vector_int8", DataType::VECTOR_INT8, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));

    schema->add_field(std::make_shared<FieldSchema>(
        "vector_int8", DataType::VECTOR_INT8, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));

    schema->add_field(std::make_shared<FieldSchema>(
        "vector_int16", DataType::VECTOR_INT16, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));

    schema->add_field(std::make_shared<FieldSchema>(
        "dense_fp16", DataType::VECTOR_FP16, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));
    schema->add_field(std::make_shared<FieldSchema>(
        "dense_fp64", DataType::VECTOR_FP64, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));
    schema->add_field(std::make_shared<FieldSchema>(
        "sparse_fp16", DataType::SPARSE_VECTOR_FP16, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));
    schema->add_field(std::make_shared<FieldSchema>(
        "sparse_fp32", DataType::SPARSE_VECTOR_FP32, 128, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));

    auto doc = test::TestHelper::CreateDoc(1, *schema);

    auto s = doc.validate(schema);
    ASSERT_TRUE(s.ok());
  }
  // doc validate pk
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    std::vector<std::string> valid_names = {
        // Min length = 1
        "a",
        "Z",
        "0",
        "_",
        "-",
        "!",
        "@",
        "#",
        "$",
        "%",
        "+",
        "=",
        ".",

        // Mixed
        "a1_",
        "user.name",
        "test@example",
        "log_2025!@#",
        "metric+=value",
        "score%change",

        "user.name",        // '.' allowed
        "test@example",     // '@' allowed
        "log_2025!@#",      // !@# allowed
        "metric+=value",    // + = allowed
        "score%change",     // % allowed
        "file-name_v1.2",   // -, _, . allowed
        "a-b_c.d!@#$%+=.",  // all specials in one

        // Max length = 64
        std::string(64, 'a'),
        std::string(63, 'a') + "_",
        "_" + std::string(62, 'x') + ".",
        "!" + std::string(62, '0') + "@",
    };
    for (auto pk : valid_names) {
      auto doc = test::TestHelper::CreateDoc(1, *schema, pk);
      auto s = doc.validate(schema);
      ASSERT_TRUE(s.ok());
    }
  }
  {
    auto schema = test::TestHelper::CreateNormalSchema(false);
    std::vector<std::string> invalid_names = {
        // Too long (>64)
        std::string(65, 'a'), std::string(64, 'a') + "_",

        // Illegal characters
        "a b",   // space
        "a&b",   // & not in set
        "a*b",   // *
        "a(b)",  // ( )
        "a:b",   // :
        "a;b",   // ;
        "a/b",   // /
        "a\\b",  // backslash
        "a\"b",  // "
        "a'b",   // '
        "a<b",
        "a>b",  // < >
        "a?b",  // ?
        "a~b",  // ~
        "a`b",  // `
        "a[b",
        "a]b",  // [ ]
        "a{b",
        "a}b",     // { }
        "a|b",     // |
        "a^b",     // ^
        "a,b",     // ,
        "用户",    // non-ASCII (Chinese)
        "αβγ",     // Greek
        "résumé",  // accented chars (é not in [a-zA-Z])
    };
    for (auto pk : invalid_names) {
      auto doc = test::TestHelper::CreateDoc(1, *schema, pk);
      auto s = doc.validate(schema);
      if (s.ok()) std::cout << "pk:" << pk << std::endl;
      ASSERT_FALSE(s.ok());
    }
  }
}

TEST_F(DocDetailedTest, GetValueTypeNameCoverage) {
  Doc::Value bool_val = true;
  EXPECT_EQ(get_value_type_name(bool_val, false), "BOOL");

  Doc::Value int32_val = int32_t(42);
  EXPECT_EQ(get_value_type_name(int32_val, false), "INT32");

  Doc::Value uint32_val = uint32_t(42);
  EXPECT_EQ(get_value_type_name(uint32_val, false), "UINT32");

  Doc::Value int64_val = int64_t(42);
  EXPECT_EQ(get_value_type_name(int64_val, false), "INT64");

  Doc::Value uint64_val = uint64_t(42);
  EXPECT_EQ(get_value_type_name(uint64_val, false), "UINT64");

  Doc::Value float_val = 3.14f;
  EXPECT_EQ(get_value_type_name(float_val, false), "FLOAT");

  Doc::Value double_val = 3.14;
  EXPECT_EQ(get_value_type_name(double_val, false), "DOUBLE");

  Doc::Value string_val = std::string("test");
  EXPECT_EQ(get_value_type_name(string_val, false), "STRING");

  Doc::Value vector_bool_val = std::vector<bool>{true, false};
  EXPECT_EQ(get_value_type_name(vector_bool_val, false), "ARRAY_BOOL");

  Doc::Value vector_int8_val = std::vector<int8_t>{1, 2, 3};
  EXPECT_EQ(get_value_type_name(vector_int8_val, true), "VECTOR_INT8");

  Doc::Value vector_int16_val = std::vector<int16_t>{10, 20, 30};
  EXPECT_EQ(get_value_type_name(vector_int16_val, true), "VECTOR_INT16");

  Doc::Value vector_int32_val = std::vector<int32_t>{100, 200, 300};
  EXPECT_EQ(get_value_type_name(vector_int32_val, true), "VECTOR_INT32");

  Doc::Value vector_int64_val = std::vector<int64_t>{1000, 2000, 3000};
  EXPECT_EQ(get_value_type_name(vector_int64_val, true), "VECTOR_INT64");

  Doc::Value vector_uint32_val = std::vector<uint32_t>{10, 20, 30};
  EXPECT_EQ(get_value_type_name(vector_uint32_val, true), "VECTOR_UINT32");

  Doc::Value vector_uint64_val = std::vector<uint64_t>{100, 200, 300};
  EXPECT_EQ(get_value_type_name(vector_uint64_val, true), "VECTOR_UINT64");

  Doc::Value vector_float_val = std::vector<float>{1.1f, 2.2f, 3.3f};
  EXPECT_EQ(get_value_type_name(vector_float_val, true), "VECTOR_FP32");

  Doc::Value vector_double_val = std::vector<double>{1.1, 2.2, 3.3};
  EXPECT_EQ(get_value_type_name(vector_double_val, true), "VECTOR_FP64");

  Doc::Value vector_float16_val = std::vector<ailego::Float16>{
      ailego::Float16(1.1f), ailego::Float16(2.2f), ailego::Float16(3.3f)};
  EXPECT_EQ(get_value_type_name(vector_float16_val, true), "VECTOR_FP16");

  Doc::Value vector_string_val = std::vector<std::string>{"a", "b", "c"};
  EXPECT_EQ(get_value_type_name(vector_string_val, false), "ARRAY_STRING");

  Doc::Value sparse_fp32_val =
      std::pair<std::vector<uint32_t>, std::vector<float>>(
          std::vector<uint32_t>{1, 2, 3}, std::vector<float>{1.1f, 2.2f, 3.3f});
  EXPECT_EQ(get_value_type_name(sparse_fp32_val, true), "SPARSE_VECTOR_FP32");

  Doc::Value sparse_fp16_val =
      std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>(
          std::vector<uint32_t>{1, 2, 3},
          std::vector<ailego::Float16>{ailego::Float16(1.1f),
                                       ailego::Float16(2.2f),
                                       ailego::Float16(3.3f)});
  EXPECT_EQ(get_value_type_name(sparse_fp16_val, true), "SPARSE_VECTOR_FP16");

  // Test monostate (null) value
  Doc::Value null_val = std::monostate{};
  EXPECT_EQ(get_value_type_name(null_val, false), "EMPTY");
}

TEST_F(DocDetailedTest, SerializeValueCoverage) {
  Doc doc;

  doc.set<bool>("bool_field", true);
  doc.set<int32_t>("int32_field", 42);
  doc.set<uint32_t>("uint32_field", 42);
  doc.set<int64_t>("int64_field", 42);
  doc.set<uint64_t>("uint64_field", 42);
  doc.set<float>("float_field", 3.14f);
  doc.set<double>("double_field", 3.14);
  doc.set<std::string>("string_field", "test");

  std::vector<bool> bool_vec = {true, false};
  doc.set<std::vector<bool>>("vector_bool_field", bool_vec);

  std::vector<int8_t> int8_vec = {1, 2, 3};
  doc.set<std::vector<int8_t>>("vector_int8_field", int8_vec);

  std::vector<int16_t> int16_vec = {10, 20, 30};
  doc.set<std::vector<int16_t>>("vector_int16_field", int16_vec);

  std::vector<int32_t> int32_vec = {100, 200, 300};
  doc.set<std::vector<int32_t>>("vector_int32_field", int32_vec);

  std::vector<int64_t> int64_vec = {1000, 2000, 3000};
  doc.set<std::vector<int64_t>>("vector_int64_field", int64_vec);

  std::vector<uint32_t> uint32_vec = {10, 20, 30};
  doc.set<std::vector<uint32_t>>("vector_uint32_field", uint32_vec);

  std::vector<uint64_t> uint64_vec = {100, 200, 300};
  doc.set<std::vector<uint64_t>>("vector_uint64_field", uint64_vec);

  std::vector<float> float_vec = {1.1f, 2.2f, 3.3f};
  doc.set<std::vector<float>>("vector_float_field", float_vec);

  std::vector<double> double_vec = {1.1, 2.2, 3.3};
  doc.set<std::vector<double>>("vector_double_field", double_vec);

  std::vector<ailego::Float16> float16_vec = {
      ailego::Float16(1.1f), ailego::Float16(2.2f), ailego::Float16(3.3f)};
  doc.set<std::vector<ailego::Float16>>("vector_float16_field", float16_vec);

  std::vector<std::string> string_vec = {"a", "b", "c"};
  doc.set<std::vector<std::string>>("vector_string_field", string_vec);

  std::pair<std::vector<uint32_t>, std::vector<float>> sparse_fp32(
      std::vector<uint32_t>{1, 2, 3}, std::vector<float>{1.1f, 2.2f, 3.3f});
  doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
      "sparse_fp32_field", sparse_fp32);

  std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>> sparse_fp16(
      std::vector<uint32_t>{1, 2, 3},
      std::vector<ailego::Float16>{ailego::Float16(1.1f), ailego::Float16(2.2f),
                                   ailego::Float16(3.3f)});
  doc.set<std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>>(
      "sparse_fp16_field", sparse_fp16);

  // Test null value
  doc.set_null("null_field");

  // for code coverage
  EXPECT_GT(doc.to_detail_string().size(), doc.to_string().size());

  auto buffer = doc.serialize();
  EXPECT_FALSE(buffer.empty());

  auto deserialized_doc = Doc::deserialize(buffer.data(), buffer.size());
  EXPECT_NE(deserialized_doc, nullptr);

  EXPECT_EQ(deserialized_doc->get<bool>("bool_field"), true);
  EXPECT_EQ(deserialized_doc->get<int32_t>("int32_field"), 42);
  EXPECT_EQ(deserialized_doc->get<uint32_t>("uint32_field"), 42u);
  EXPECT_EQ(deserialized_doc->get<int64_t>("int64_field"), 42);
  EXPECT_EQ(deserialized_doc->get<uint64_t>("uint64_field"), 42u);
  EXPECT_FLOAT_EQ(deserialized_doc->get<float>("float_field").value(), 3.14f);
  EXPECT_DOUBLE_EQ(deserialized_doc->get<double>("double_field").value(), 3.14);
  EXPECT_EQ(deserialized_doc->get<std::string>("string_field"), "test");

  // Test null value deserialization
  EXPECT_TRUE(deserialized_doc->is_null("null_field"));
  EXPECT_FALSE(deserialized_doc->has_value("null_field"));
  EXPECT_TRUE(deserialized_doc->has("null_field"));
}

TEST_F(DocDetailedTest, ToDetailStringCoverage) {
  Doc doc;
  doc.set_pk("test_pk");
  doc.set_doc_id(1);
  doc.set_score(0.95f);

  doc.set<bool>("bool_field", true);
  doc.set<int32_t>("int32_field", 42);
  doc.set<uint32_t>("uint32_field", 42);
  doc.set<int64_t>("int64_field", 42);
  doc.set<uint64_t>("uint64_field", 42);
  doc.set<float>("float_field", 3.14f);
  doc.set<double>("double_field", 3.14);
  doc.set<std::string>("string_field", "test");

  std::vector<bool> bool_vec = {true, false};
  doc.set<std::vector<bool>>("vector_bool_field", bool_vec);

  std::vector<int8_t> int8_vec = {1, 2};
  doc.set<std::vector<int8_t>>("vector_int8_field", int8_vec);

  std::vector<int16_t> int16_vec = {10, 20};
  doc.set<std::vector<int16_t>>("vector_int16_field", int16_vec);

  std::vector<int32_t> int32_vec = {100, 200};
  doc.set<std::vector<int32_t>>("vector_int32_field", int32_vec);

  std::vector<int64_t> int64_vec = {1000, 2000};
  doc.set<std::vector<int64_t>>("vector_int64_field", int64_vec);

  std::vector<uint32_t> uint32_vec = {10, 20};
  doc.set<std::vector<uint32_t>>("vector_uint32_field", uint32_vec);

  std::vector<uint64_t> uint64_vec = {100, 200};
  doc.set<std::vector<uint64_t>>("vector_uint64_field", uint64_vec);

  std::vector<float> float_vec = {1.1f, 2.2f};
  doc.set<std::vector<float>>("vector_float_field", float_vec);

  std::vector<double> double_vec = {1.1, 2.2};
  doc.set<std::vector<double>>("vector_double_field", double_vec);

  std::vector<ailego::Float16> float16_vec = {ailego::Float16(1.1f),
                                              ailego::Float16(2.2f)};
  doc.set<std::vector<ailego::Float16>>("vector_float16_field", float16_vec);

  std::vector<std::string> string_vec = {"a", "b"};
  doc.set<std::vector<std::string>>("vector_string_field", string_vec);

  std::pair<std::vector<uint32_t>, std::vector<float>> sparse_fp32(
      std::vector<uint32_t>{1, 2}, std::vector<float>{1.1f, 2.2f});
  doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
      "sparse_fp32_field", sparse_fp32);

  std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>> sparse_fp16(
      std::vector<uint32_t>{1, 2},
      std::vector<ailego::Float16>{ailego::Float16(1.1f),
                                   ailego::Float16(2.2f)});
  doc.set<std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>>(
      "sparse_fp16_field", sparse_fp16);

  // Test null value in detail string
  doc.set_null("null_field");

  std::string detail_str = doc.to_detail_string();
  EXPECT_FALSE(detail_str.empty());
  EXPECT_NE(detail_str.find("bool_field"), std::string::npos);
  EXPECT_NE(detail_str.find("int32_field"), std::string::npos);
  EXPECT_NE(detail_str.find("vector_float_field"), std::string::npos);
  EXPECT_NE(detail_str.find("null"),
            std::string::npos);  // Should contain "null" for null field
}

TEST_F(DocDetailedTest, EqualityOperatorCoverage) {
  Doc doc1, doc2;
  doc1.set_pk("test_pk");
  doc2.set_pk("test_pk");

  doc1.set_doc_id(1);
  doc2.set_doc_id(1);

  doc1.set<bool>("bool_field", true);
  doc2.set<bool>("bool_field", true);

  doc1.set<int32_t>("int32_field", 42);
  doc2.set<int32_t>("int32_field", 42);

  doc1.set<uint32_t>("uint32_field", 42);
  doc2.set<uint32_t>("uint32_field", 42);

  doc1.set<int64_t>("int64_field", 42);
  doc2.set<int64_t>("int64_field", 42);

  doc1.set<uint64_t>("uint64_field", 42);
  doc2.set<uint64_t>("uint64_field", 42);

  doc1.set<float>("float_field", 3.14f);
  doc2.set<float>("float_field", 3.14f);

  doc1.set<double>("double_field", 3.14);
  doc2.set<double>("double_field", 3.14);

  doc1.set<std::string>("string_field", "test");
  doc2.set<std::string>("string_field", "test");

  std::vector<bool> bool_vec = {true, false};
  doc1.set<std::vector<bool>>("vector_bool_field", bool_vec);
  doc2.set<std::vector<bool>>("vector_bool_field", bool_vec);

  std::vector<int8_t> int8_vec = {1, 2};
  doc1.set<std::vector<int8_t>>("vector_int8_field", int8_vec);
  doc2.set<std::vector<int8_t>>("vector_int8_field", int8_vec);

  std::vector<int16_t> int16_vec = {10, 20};
  doc1.set<std::vector<int16_t>>("vector_int16_field", int16_vec);
  doc2.set<std::vector<int16_t>>("vector_int16_field", int16_vec);

  std::vector<int32_t> int32_vec = {100, 200};
  doc1.set<std::vector<int32_t>>("vector_int32_field", int32_vec);
  doc2.set<std::vector<int32_t>>("vector_int32_field", int32_vec);

  std::vector<int64_t> int64_vec = {1000, 2000};
  doc1.set<std::vector<int64_t>>("vector_int64_field", int64_vec);
  doc2.set<std::vector<int64_t>>("vector_int64_field", int64_vec);

  std::vector<uint32_t> uint32_vec = {10, 20};
  doc1.set<std::vector<uint32_t>>("vector_uint32_field", uint32_vec);
  doc2.set<std::vector<uint32_t>>("vector_uint32_field", uint32_vec);

  std::vector<uint64_t> uint64_vec = {100, 200};
  doc1.set<std::vector<uint64_t>>("vector_uint64_field", uint64_vec);
  doc2.set<std::vector<uint64_t>>("vector_uint64_field", uint64_vec);

  std::vector<float> float_vec = {1.1f, 2.2f};
  doc1.set<std::vector<float>>("vector_float_field", float_vec);
  doc2.set<std::vector<float>>("vector_float_field", float_vec);

  std::vector<double> double_vec = {1.1, 2.2};
  doc1.set<std::vector<double>>("vector_double_field", double_vec);
  doc2.set<std::vector<double>>("vector_double_field", double_vec);

  std::vector<ailego::Float16> float16_vec = {ailego::Float16(1.1f),
                                              ailego::Float16(2.2f)};
  doc1.set<std::vector<ailego::Float16>>("vector_float16_field", float16_vec);
  doc2.set<std::vector<ailego::Float16>>("vector_float16_field", float16_vec);

  std::vector<std::string> string_vec = {"a", "b"};
  doc1.set<std::vector<std::string>>("vector_string_field", string_vec);
  doc2.set<std::vector<std::string>>("vector_string_field", string_vec);

  std::pair<std::vector<uint32_t>, std::vector<float>> sparse_fp32(
      std::vector<uint32_t>{1, 2}, std::vector<float>{1.1f, 2.2f});
  doc1.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
      "sparse_fp32_field", sparse_fp32);
  doc2.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
      "sparse_fp32_field", sparse_fp32);

  std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>> sparse_fp16(
      std::vector<uint32_t>{1, 2},
      std::vector<ailego::Float16>{ailego::Float16(1.1f),
                                   ailego::Float16(2.2f)});
  doc1.set<std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>>(
      "sparse_fp16_field", sparse_fp16);
  doc2.set<std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>>(
      "sparse_fp16_field", sparse_fp16);

  // Test equality with null values
  doc1.set_null("null_field");
  doc2.set_null("null_field");

  EXPECT_TRUE(doc1 == doc2);

  doc2.set<int32_t>("int32_field", 43);
  EXPECT_FALSE(doc1 == doc2);

  doc1.set_pk("test_pk1");
  EXPECT_FALSE(doc1 == doc2);

  doc1.set_pk("test_pk");
  doc1.set<uint32_t>("int32_field", 42);
  EXPECT_FALSE(doc1 == doc2);

  doc1.set<int32_t>("int32_field", 42);
  doc1.set<int32_t>("rename_int32_field", 42);
  EXPECT_FALSE(doc1 == doc2);

  // Test inequality with different null values
  Doc doc3, doc4;
  doc3.set_pk("test");
  doc4.set_pk("test");
  doc3.set_null("null_field");
  doc4.set<int32_t>("null_field", 42);
  EXPECT_FALSE(doc3 == doc4);
}


TEST(VectorQuery, Validate) {
  // field schema is null when query without vector
  {
    VectorQuery query;
    query.topk_ = 10;
    query.field_name_ = "field_name";
    auto s = query.validate(nullptr);
    EXPECT_TRUE(s.ok());
  }

  // field schema is null when query without vector
  {
    VectorQuery query;
    query.topk_ = 10;
    query.field_name_ = "field_name";
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string query_vector_str =
        std::string(reinterpret_cast<char *>(query_vector.data()),
                    query_vector.size() * sizeof(float));
    query.query_vector_ = query_vector_str;
    auto s = query.validate(nullptr);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }
  // vector_query exceed topk
  {
    VectorQuery query;
    query.field_name_ = "field_name";
    query.topk_ = 1000;
    FieldSchema schema =
        FieldSchema("field_name", DataType::VECTOR_FP32, 128, true);
    auto s = query.validate(&schema);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }
  // vector_query output_fields size exceed
  {
    VectorQuery query;
    query.field_name_ = "field_name";
    query.topk_ = 10;
    query.output_fields_ = std::vector<std::string>(1025);
    FieldSchema schema = FieldSchema("field_name", DataType::INT32);
    auto s = query.validate(&schema);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // validate dense vector dimension
  {
    VectorQuery query;
    query.field_name_ = "field_name";
    query.topk_ = 100;
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string query_vector_str =
        std::string(reinterpret_cast<char *>(query_vector.data()),
                    query_vector.size() * sizeof(float));
    query.query_vector_ = query_vector_str;
    FieldSchema schema =
        FieldSchema("field_name", DataType::VECTOR_FP32, 4, true);
    auto s = query.validate(&schema);
    EXPECT_TRUE(s.ok());

    query.query_vector_ = query_vector_str.substr(0, 3);
    s = query.validate(&schema);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
  }

  // validate sparse indices
  {
    VectorQuery query;
    query.field_name_ = "field_name";
    query.topk_ = 100;
    std::vector<uint32_t> query_indices = std::vector<uint32_t>(4097);
    std::string query_indices_str =
        std::string(reinterpret_cast<char *>(query_indices.data()),
                    query_indices.size() * sizeof(uint32_t));
    query.query_sparse_indices_ = query_indices_str;
    FieldSchema schema =
        FieldSchema("field_name", DataType::SPARSE_VECTOR_FP32);
    auto s = query.validate(&schema);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

    query.query_sparse_indices_ = query_indices_str.substr(0, 3);
    s = query.validate(&schema);
    EXPECT_TRUE(s.ok());
  }
}

// Test null value
TEST_F(DocDetailedTest, NullValue) {
  Doc doc;

  // Test setting null value
  doc.set_null("null_field");
  EXPECT_TRUE(doc.is_null("null_field"));
  EXPECT_FALSE(doc.has_value("null_field"));
  EXPECT_TRUE(doc.has("null_field"));

  // Test get_field with null field
  auto result = doc.get_field<int32_t>("null_field");
  EXPECT_EQ(result.status(), Doc::FieldGetStatus::IS_NULL);
  EXPECT_FALSE(result.ok());

  // Test get with null field
  auto opt_result = doc.get<int32_t>("null_field");
  EXPECT_FALSE(opt_result.has_value());

  // Test overwriting null with actual value
  doc.set<int32_t>("null_field", 42);
  EXPECT_FALSE(doc.is_null("null_field"));
  EXPECT_TRUE(doc.has_value("null_field"));
  EXPECT_TRUE(doc.has("null_field"));
  EXPECT_EQ(doc.get<int32_t>("null_field").value(), 42);

  // Test overwriting value with null
  doc.set_null("null_field");
  EXPECT_TRUE(doc.is_null("null_field"));
  EXPECT_FALSE(doc.has_value("null_field"));
  EXPECT_TRUE(doc.has("null_field"));

  // Test serialization/deserialization of null values
  auto buffer = doc.serialize();
  auto deserialized_doc = Doc::deserialize(buffer.data(), buffer.size());
  EXPECT_NE(deserialized_doc, nullptr);
  EXPECT_TRUE(deserialized_doc->is_null("null_field"));
  EXPECT_FALSE(deserialized_doc->has_value("null_field"));
  EXPECT_TRUE(deserialized_doc->has("null_field"));
}

// Test field existence checks
TEST_F(DocDetailedTest, FieldExistenceChecks) {
  Doc doc;

  // Test non-existent field
  EXPECT_FALSE(doc.has("nonexistent"));
  EXPECT_FALSE(doc.has_value("nonexistent"));
  EXPECT_FALSE(doc.is_null("nonexistent"));

  // Test get_field with non-existent field
  auto result = doc.get_field<int32_t>("nonexistent");
  EXPECT_EQ(result.status(), Doc::FieldGetStatus::NOT_FOUND);
  EXPECT_FALSE(result.ok());

  // Test get with non-existent field
  auto opt_result = doc.get<int32_t>("nonexistent");
  EXPECT_FALSE(opt_result.has_value());

  // Add a field and test again
  doc.set<int32_t>("existent", 123);
  EXPECT_TRUE(doc.has("existent"));
  EXPECT_TRUE(doc.has_value("existent"));
  EXPECT_FALSE(doc.is_null("existent"));

  // Test type mismatch
  auto type_mismatch_result = doc.get_field<std::string>("existent");
  EXPECT_EQ(type_mismatch_result.status(), Doc::FieldGetStatus::TYPE_MISMATCH);
  EXPECT_FALSE(type_mismatch_result.ok());

  auto type_mismatch_opt = doc.get<std::string>("existent");
  EXPECT_FALSE(type_mismatch_opt.has_value());
}