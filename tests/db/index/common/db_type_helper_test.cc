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

// These conversion tests intentionally pass invalid enum sentinels.
// NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)

TEST(IndexTypeCodeBookTest, ProtoToCppConversion) {
  // Test conversion from protobuf to C++ IndexType
  EXPECT_EQ(IndexTypeCodeBook::Get(proto::IT_HNSW), IndexType::HNSW);
  EXPECT_EQ(IndexTypeCodeBook::Get(proto::IT_FLAT), IndexType::FLAT);
  EXPECT_EQ(IndexTypeCodeBook::Get(proto::IT_IVF), IndexType::IVF);
  EXPECT_EQ(IndexTypeCodeBook::Get(proto::IT_INVERT), IndexType::INVERT);
  EXPECT_EQ(IndexTypeCodeBook::Get(proto::IT_UNDEFINED), IndexType::UNDEFINED);
  EXPECT_EQ(IndexTypeCodeBook::Get(static_cast<proto::IndexType>(999)),
            IndexType::UNDEFINED);
}

TEST(IndexTypeCodeBookTest, CppToProtoConversion) {
  // Test conversion from C++ IndexType to protobuf IndexType
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::HNSW), proto::IT_HNSW);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::FLAT), proto::IT_FLAT);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::IVF), proto::IT_IVF);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::INVERT), proto::IT_INVERT);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::UNDEFINED), proto::IT_UNDEFINED);
  EXPECT_EQ(IndexTypeCodeBook::Get(static_cast<IndexType>(999)),
            proto::IT_UNDEFINED);
}

TEST(IndexTypeCodeBookTest, CppToStringConversion) {
  // Test conversion from C++ IndexType to string
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::HNSW), "HNSW");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::INVERT), "INVERT");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(IndexTypeCodeBook::AsString(static_cast<IndexType>(999)),
            "UNDEFINED");
}

TEST(DataTypeCodeBookTest, IsArrayType) {
  // Test array type detection
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_BINARY));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_STRING));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_BOOL));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_INT32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_INT64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_UINT32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_UINT64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_FLOAT));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_DOUBLE));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_BINARY32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_BINARY64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_FP16));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_FP32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_FP64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_INT4));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_INT8));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_VECTOR_INT16));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(proto::DT_SPARSE_VECTOR_FP32));

  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_BINARY));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_STRING));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_BOOL));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_INT32));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_INT64));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_UINT32));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_UINT64));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_FLOAT));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(proto::DT_ARRAY_DOUBLE));
}

TEST(DataTypeCodeBookTest, ProtoToCppConversion) {
  // Test conversion from protobuf to C++ DataType
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_BINARY), DataType::BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_STRING), DataType::STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_BOOL), DataType::BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_INT32), DataType::INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_INT64), DataType::INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_UINT32), DataType::UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_UINT64), DataType::UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_FLOAT), DataType::FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_DOUBLE), DataType::DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_BINARY32),
            DataType::VECTOR_BINARY32);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_BINARY64),
            DataType::VECTOR_BINARY64);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_FP16),
            DataType::VECTOR_FP16);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_FP32),
            DataType::VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_FP64),
            DataType::VECTOR_FP64);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_INT4),
            DataType::VECTOR_INT4);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_INT8),
            DataType::VECTOR_INT8);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_VECTOR_INT16),
            DataType::VECTOR_INT16);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_SPARSE_VECTOR_FP32),
            DataType::SPARSE_VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_BINARY),
            DataType::ARRAY_BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_STRING),
            DataType::ARRAY_STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_BOOL), DataType::ARRAY_BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_INT32),
            DataType::ARRAY_INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_INT64),
            DataType::ARRAY_INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_UINT32),
            DataType::ARRAY_UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_UINT64),
            DataType::ARRAY_UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_FLOAT),
            DataType::ARRAY_FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_ARRAY_DOUBLE),
            DataType::ARRAY_DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(proto::DT_UNDEFINED), DataType::UNDEFINED);
  EXPECT_EQ(DataTypeCodeBook::Get(static_cast<proto::DataType>(999)),
            DataType::UNDEFINED);
}

TEST(DataTypeCodeBookTest, CppToProtoConversion) {
  // Test conversion from C++ DataType to protobuf DataType
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::BINARY), proto::DT_BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::STRING), proto::DT_STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::BOOL), proto::DT_BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::INT32), proto::DT_INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::INT64), proto::DT_INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::UINT32), proto::DT_UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::UINT64), proto::DT_UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::FLOAT), proto::DT_FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::DOUBLE), proto::DT_DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_BINARY32),
            proto::DT_VECTOR_BINARY32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_BINARY64),
            proto::DT_VECTOR_BINARY64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_FP16),
            proto::DT_VECTOR_FP16);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_FP32),
            proto::DT_VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_FP64),
            proto::DT_VECTOR_FP64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_INT4),
            proto::DT_VECTOR_INT4);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_INT8),
            proto::DT_VECTOR_INT8);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_INT16),
            proto::DT_VECTOR_INT16);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::SPARSE_VECTOR_FP16),
            proto::DT_SPARSE_VECTOR_FP16);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::SPARSE_VECTOR_FP32),
            proto::DT_SPARSE_VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_BINARY),
            proto::DT_ARRAY_BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_STRING),
            proto::DT_ARRAY_STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_BOOL), proto::DT_ARRAY_BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_INT32),
            proto::DT_ARRAY_INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_INT64),
            proto::DT_ARRAY_INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_UINT32),
            proto::DT_ARRAY_UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_UINT64),
            proto::DT_ARRAY_UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_FLOAT),
            proto::DT_ARRAY_FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_DOUBLE),
            proto::DT_ARRAY_DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::UNDEFINED), proto::DT_UNDEFINED);
  EXPECT_EQ(DataTypeCodeBook::Get(static_cast<DataType>(999)),
            proto::DT_UNDEFINED);
}

TEST(DataTypeCodeBookTest, CppToStringConversion) {
  // Test conversion from C++ DataType to string
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

TEST(MetricTypeCodeBookTest, ProtoToCppConversion) {
  // Test conversion from protobuf to C++ MetricType
  EXPECT_EQ(MetricTypeCodeBook::Get(proto::MT_IP), MetricType::IP);
  EXPECT_EQ(MetricTypeCodeBook::Get(proto::MT_L2), MetricType::L2);
  EXPECT_EQ(MetricTypeCodeBook::Get(proto::MT_COSINE), MetricType::COSINE);
  EXPECT_EQ(MetricTypeCodeBook::Get(proto::MT_UNDEFINED),
            MetricType::UNDEFINED);
  EXPECT_EQ(MetricTypeCodeBook::Get(static_cast<proto::MetricType>(999)),
            MetricType::UNDEFINED);
}

TEST(MetricTypeCodeBookTest, CppToProtoConversion) {
  // Test conversion from C++ MetricType to protobuf MetricType
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::IP), proto::MT_IP);
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::L2), proto::MT_L2);
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::COSINE), proto::MT_COSINE);
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::UNDEFINED),
            proto::MT_UNDEFINED);
  EXPECT_EQ(MetricTypeCodeBook::Get(static_cast<MetricType>(999)),
            proto::MT_UNDEFINED);
}

TEST(QuantizeTypeCodeBookTest, ProtoToCppConversion) {
  // Test conversion from protobuf to C++ QuantizeType
  EXPECT_EQ(QuantizeTypeCodeBook::Get(proto::QT_FP16), QuantizeType::FP16);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(proto::QT_INT4), QuantizeType::INT4);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(proto::QT_INT8), QuantizeType::INT8);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(proto::QT_UNDEFINED),
            QuantizeType::UNDEFINED);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(static_cast<proto::QuantizeType>(999)),
            QuantizeType::UNDEFINED);
}

TEST(QuantizeTypeCodeBookTest, CppToProtoConversion) {
  // Test conversion from C++ QuantizeType to protobuf QuantizeType
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::FP16), proto::QT_FP16);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::INT4), proto::QT_INT4);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::INT8), proto::QT_INT8);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::UNDEFINED),
            proto::QT_UNDEFINED);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(static_cast<QuantizeType>(999)),
            proto::QT_UNDEFINED);
}

TEST(BlockTypeCodeBookTest, ProtoToCppConversion) {
  // Test conversion from protobuf to C++ BlockType
  EXPECT_EQ(BlockTypeCodeBook::Get(proto::BT_SCALAR), BlockType::SCALAR);
  EXPECT_EQ(BlockTypeCodeBook::Get(proto::BT_SCALAR_INDEX),
            BlockType::SCALAR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(proto::BT_VECTOR_INDEX),
            BlockType::VECTOR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(proto::BT_VECTOR_INDEX_QUANTIZE),
            BlockType::VECTOR_INDEX_QUANTIZE);
  EXPECT_EQ(BlockTypeCodeBook::Get(proto::BT_UNDEFINED), BlockType::UNDEFINED);
  EXPECT_EQ(BlockTypeCodeBook::Get(static_cast<proto::BlockType>(999)),
            BlockType::UNDEFINED);
}

TEST(BlockTypeCodeBookTest, CppToProtoConversion) {
  // Test conversion from C++ BlockType to protobuf BlockType
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::SCALAR), proto::BT_SCALAR);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::SCALAR_INDEX),
            proto::BT_SCALAR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::VECTOR_INDEX),
            proto::BT_VECTOR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::VECTOR_INDEX_QUANTIZE),
            proto::BT_VECTOR_INDEX_QUANTIZE);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::UNDEFINED), proto::BT_UNDEFINED);
  EXPECT_EQ(BlockTypeCodeBook::Get(static_cast<BlockType>(999)),
            proto::BT_UNDEFINED);
}

// NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
