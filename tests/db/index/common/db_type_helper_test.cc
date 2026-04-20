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

#include <gtest/gtest.h>
#include "db/index/common/type_helper.h"

using namespace zvec;

TEST(IndexTypeCodeBookTest, CppToStringConversion) {
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::HNSW), "HNSW");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::HNSW_RABITQ), "HNSW_RABITQ");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::FLAT), "FLAT");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::IVF), "IVF");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::INVERT), "INVERT");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(IndexTypeCodeBook::AsString(static_cast<IndexType>(999)),
            "UNDEFINED");
}

TEST(DataTypeCodeBookTest, IsArrayType) {
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::BINARY));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::STRING));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::BOOL));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::INT32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::INT64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::UINT32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::UINT64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::FLOAT));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::DOUBLE));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_BINARY32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_BINARY64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_FP16));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_FP32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_FP64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_INT4));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_INT8));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::VECTOR_INT16));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(DataType::SPARSE_VECTOR_FP32));

  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_BINARY));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_STRING));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_BOOL));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_INT32));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_INT64));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_UINT32));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_UINT64));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_FLOAT));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(DataType::ARRAY_DOUBLE));
}

TEST(DataTypeCodeBookTest, CppToStringConversion) {
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::BINARY), "BINARY");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::STRING), "STRING");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::BOOL), "BOOL");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::INT32), "INT32");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::INT64), "INT64");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::UINT32), "UINT32");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::UINT64), "UINT64");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::FLOAT), "FLOAT");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::DOUBLE), "DOUBLE");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_BINARY32),
            "VECTOR_BINARY32");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_BINARY64),
            "VECTOR_BINARY64");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_FP16), "VECTOR_FP16");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_FP32), "VECTOR_FP32");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_FP64), "VECTOR_FP64");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_INT4), "VECTOR_INT4");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_INT8), "VECTOR_INT8");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::VECTOR_INT16), "VECTOR_INT16");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::SPARSE_VECTOR_FP16),
            "SPARSE_VECTOR_FP16");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::SPARSE_VECTOR_FP32),
            "SPARSE_VECTOR_FP32");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_BINARY), "ARRAY_BINARY");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_STRING), "ARRAY_STRING");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_BOOL), "ARRAY_BOOL");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_INT32), "ARRAY_INT32");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_INT64), "ARRAY_INT64");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_UINT32), "ARRAY_UINT32");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_UINT64), "ARRAY_UINT64");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_FLOAT), "ARRAY_FLOAT");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::ARRAY_DOUBLE), "ARRAY_DOUBLE");
  EXPECT_EQ(DataTypeCodeBook::AsString(DataType::UNDEFINED), "");
  EXPECT_EQ(DataTypeCodeBook::AsString(static_cast<DataType>(999)), "");
}

TEST(MetricTypeCodeBookTest, CppToStringConversion) {
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::IP), "IP");
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::L2), "L2");
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::COSINE), "COSINE");
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(MetricTypeCodeBook::AsString(static_cast<MetricType>(999)),
            "UNDEFINED");
}

TEST(QuantizeTypeCodeBookTest, CppToStringConversion) {
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::FP16), "FP16");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::INT4), "INT4");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::INT8), "INT8");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::RABITQ), "RABITQ");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::UNDEFINED),
            "UNDEFINED");
}

TEST(BlockTypeCodeBookTest, CppToStringConversion) {
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::SCALAR), "SCALAR");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::SCALAR_INDEX),
            "SCALAR_INDEX");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::VECTOR_INDEX),
            "VECTOR_INDEX");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::VECTOR_INDEX_QUANTIZE),
            "VECTOR_INDEX_QUANTIZE");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(BlockTypeCodeBook::AsString(static_cast<BlockType>(999)),
            "UNDEFINED");
}
