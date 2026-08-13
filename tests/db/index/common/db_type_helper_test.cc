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

#include <set>
#include <gtest/gtest.h>
#include "db/index/common/type_helper.h"

using namespace zvec;

TEST(IndexTypeCodeBookTest, WireToCppConversion) {
  // Test conversion from the wire format to C++ IndexType
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_HNSW), IndexType::HNSW);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_HNSW_RABITQ),
            IndexType::HNSW_RABITQ);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_IVF_RABITQ),
            IndexType::IVF_RABITQ);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_FLAT), IndexType::FLAT);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_IVF), IndexType::IVF);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_VAMANA),
            IndexType::VAMANA);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_INVERT),
            IndexType::INVERT);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_DISKANN),
            IndexType::DISKANN);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_FTS), IndexType::FTS);
  EXPECT_EQ(IndexTypeCodeBook::Get(wire::IndexType::IT_UNDEFINED),
            IndexType::UNDEFINED);
  EXPECT_EQ(IndexTypeCodeBook::Get(static_cast<wire::IndexType>(999)),
            IndexType::UNDEFINED);
}

TEST(IndexTypeCodeBookTest, CppToWireConversion) {
  // Test conversion from C++ IndexType to the wire format
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::HNSW), wire::IndexType::IT_HNSW);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::HNSW_RABITQ),
            wire::IndexType::IT_HNSW_RABITQ);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::IVF_RABITQ),
            wire::IndexType::IT_IVF_RABITQ);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::FLAT), wire::IndexType::IT_FLAT);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::IVF), wire::IndexType::IT_IVF);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::VAMANA),
            wire::IndexType::IT_VAMANA);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::INVERT),
            wire::IndexType::IT_INVERT);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::DISKANN),
            wire::IndexType::IT_DISKANN);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::FTS), wire::IndexType::IT_FTS);
  EXPECT_EQ(IndexTypeCodeBook::Get(IndexType::UNDEFINED),
            wire::IndexType::IT_UNDEFINED);
  EXPECT_EQ(IndexTypeCodeBook::Get(static_cast<IndexType>(999)),
            wire::IndexType::IT_UNDEFINED);
}

TEST(IndexTypeCodeBookTest, CppToStringConversion) {
  // Test conversion from C++ IndexType to string
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::HNSW), "HNSW");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::HNSW_RABITQ), "HNSW_RABITQ");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::IVF_RABITQ), "IVF_RABITQ");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::FLAT), "FLAT");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::IVF), "IVF");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::VAMANA), "VAMANA");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::DISKANN), "DISKANN");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::INVERT), "INVERT");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::FTS), "FTS");
  EXPECT_EQ(IndexTypeCodeBook::AsString(IndexType::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(IndexTypeCodeBook::AsString(static_cast<IndexType>(999)),
            "UNDEFINED");
}

TEST(DataTypeCodeBookTest, IsArrayType) {
  // Test array type detection
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_BINARY));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_STRING));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_BOOL));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_INT32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_INT64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_UINT32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_UINT64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_FLOAT));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_DOUBLE));
  EXPECT_FALSE(
      DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_BINARY32));
  EXPECT_FALSE(
      DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_BINARY64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_FP16));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_FP32));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_FP64));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_INT4));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_INT8));
  EXPECT_FALSE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_VECTOR_INT16));
  EXPECT_FALSE(
      DataTypeCodeBook::IsArrayType(wire::DataType::DT_SPARSE_VECTOR_FP32));
  EXPECT_FALSE(
      DataTypeCodeBook::IsArrayType(wire::DataType::DT_SPARSE_VECTOR_FP16));

  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_BINARY));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_STRING));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_BOOL));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_INT32));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_INT64));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_UINT32));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_UINT64));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_FLOAT));
  EXPECT_TRUE(DataTypeCodeBook::IsArrayType(wire::DataType::DT_ARRAY_DOUBLE));
}

TEST(DataTypeCodeBookTest, WireToCppConversion) {
  // Test conversion from the wire format to C++ DataType
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_BINARY), DataType::BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_STRING), DataType::STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_BOOL), DataType::BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_INT32), DataType::INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_INT64), DataType::INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_UINT32), DataType::UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_UINT64), DataType::UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_FLOAT), DataType::FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_DOUBLE), DataType::DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_BINARY32),
            DataType::VECTOR_BINARY32);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_BINARY64),
            DataType::VECTOR_BINARY64);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_FP16),
            DataType::VECTOR_FP16);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_FP32),
            DataType::VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_FP64),
            DataType::VECTOR_FP64);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_INT4),
            DataType::VECTOR_INT4);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_INT8),
            DataType::VECTOR_INT8);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_VECTOR_INT16),
            DataType::VECTOR_INT16);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_SPARSE_VECTOR_FP16),
            DataType::SPARSE_VECTOR_FP16);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_SPARSE_VECTOR_FP32),
            DataType::SPARSE_VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_BINARY),
            DataType::ARRAY_BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_STRING),
            DataType::ARRAY_STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_BOOL),
            DataType::ARRAY_BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_INT32),
            DataType::ARRAY_INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_INT64),
            DataType::ARRAY_INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_UINT32),
            DataType::ARRAY_UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_UINT64),
            DataType::ARRAY_UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_FLOAT),
            DataType::ARRAY_FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_ARRAY_DOUBLE),
            DataType::ARRAY_DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(wire::DataType::DT_UNDEFINED),
            DataType::UNDEFINED);
  EXPECT_EQ(DataTypeCodeBook::Get(static_cast<wire::DataType>(999)),
            DataType::UNDEFINED);
}

TEST(DataTypeCodeBookTest, CppToWireConversion) {
  // Test conversion from C++ DataType to the wire format
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::BINARY), wire::DataType::DT_BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::STRING), wire::DataType::DT_STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::BOOL), wire::DataType::DT_BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::INT32), wire::DataType::DT_INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::INT64), wire::DataType::DT_INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::UINT32), wire::DataType::DT_UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::UINT64), wire::DataType::DT_UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::FLOAT), wire::DataType::DT_FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::DOUBLE), wire::DataType::DT_DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_BINARY32),
            wire::DataType::DT_VECTOR_BINARY32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_BINARY64),
            wire::DataType::DT_VECTOR_BINARY64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_FP16),
            wire::DataType::DT_VECTOR_FP16);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_FP32),
            wire::DataType::DT_VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_FP64),
            wire::DataType::DT_VECTOR_FP64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_INT4),
            wire::DataType::DT_VECTOR_INT4);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_INT8),
            wire::DataType::DT_VECTOR_INT8);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::VECTOR_INT16),
            wire::DataType::DT_VECTOR_INT16);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::SPARSE_VECTOR_FP16),
            wire::DataType::DT_SPARSE_VECTOR_FP16);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::SPARSE_VECTOR_FP32),
            wire::DataType::DT_SPARSE_VECTOR_FP32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_BINARY),
            wire::DataType::DT_ARRAY_BINARY);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_STRING),
            wire::DataType::DT_ARRAY_STRING);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_BOOL),
            wire::DataType::DT_ARRAY_BOOL);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_INT32),
            wire::DataType::DT_ARRAY_INT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_INT64),
            wire::DataType::DT_ARRAY_INT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_UINT32),
            wire::DataType::DT_ARRAY_UINT32);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_UINT64),
            wire::DataType::DT_ARRAY_UINT64);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_FLOAT),
            wire::DataType::DT_ARRAY_FLOAT);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::ARRAY_DOUBLE),
            wire::DataType::DT_ARRAY_DOUBLE);
  EXPECT_EQ(DataTypeCodeBook::Get(DataType::UNDEFINED),
            wire::DataType::DT_UNDEFINED);
  EXPECT_EQ(DataTypeCodeBook::Get(static_cast<DataType>(999)),
            wire::DataType::DT_UNDEFINED);
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

TEST(MetricTypeCodeBookTest, WireToCppConversion) {
  // Test conversion from the wire format to C++ MetricType
  EXPECT_EQ(MetricTypeCodeBook::Get(wire::MetricType::MT_IP), MetricType::IP);
  EXPECT_EQ(MetricTypeCodeBook::Get(wire::MetricType::MT_L2), MetricType::L2);
  EXPECT_EQ(MetricTypeCodeBook::Get(wire::MetricType::MT_COSINE),
            MetricType::COSINE);
  EXPECT_EQ(MetricTypeCodeBook::Get(wire::MetricType::MT_UNDEFINED),
            MetricType::UNDEFINED);
  EXPECT_EQ(MetricTypeCodeBook::Get(static_cast<wire::MetricType>(999)),
            MetricType::UNDEFINED);
}

TEST(MetricTypeCodeBookTest, CppToWireConversion) {
  // Test conversion from C++ MetricType to the wire format
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::IP), wire::MetricType::MT_IP);
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::L2), wire::MetricType::MT_L2);
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::COSINE),
            wire::MetricType::MT_COSINE);
  // MIPSL2 is a C++-only metric type without a wire-format counterpart.
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::MIPSL2),
            wire::MetricType::MT_UNDEFINED);
  EXPECT_EQ(MetricTypeCodeBook::Get(MetricType::UNDEFINED),
            wire::MetricType::MT_UNDEFINED);
  EXPECT_EQ(MetricTypeCodeBook::Get(static_cast<MetricType>(999)),
            wire::MetricType::MT_UNDEFINED);
}

TEST(MetricTypeCodeBookTest, CppToStringConversion) {
  // Test conversion from C++ MetricType to string
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::IP), "IP");
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::L2), "L2");
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::COSINE), "COSINE");
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::MIPSL2), "UNDEFINED");
  EXPECT_EQ(MetricTypeCodeBook::AsString(MetricType::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(MetricTypeCodeBook::AsString(static_cast<MetricType>(999)),
            "UNDEFINED");
}

TEST(QuantizeTypeCodeBookTest, WireToCppConversion) {
  // Test conversion from the wire format to C++ QuantizeType
  EXPECT_EQ(QuantizeTypeCodeBook::Get(wire::QuantizeType::QT_FP16),
            QuantizeType::FP16);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(wire::QuantizeType::QT_INT4),
            QuantizeType::INT4);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(wire::QuantizeType::QT_INT8),
            QuantizeType::INT8);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(wire::QuantizeType::QT_RABITQ),
            QuantizeType::RABITQ);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(wire::QuantizeType::QT_UNDEFINED),
            QuantizeType::UNDEFINED);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(static_cast<wire::QuantizeType>(999)),
            QuantizeType::UNDEFINED);
}

TEST(QuantizeTypeCodeBookTest, CppToWireConversion) {
  // Test conversion from C++ QuantizeType to the wire format
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::FP16),
            wire::QuantizeType::QT_FP16);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::INT4),
            wire::QuantizeType::QT_INT4);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::INT8),
            wire::QuantizeType::QT_INT8);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::RABITQ),
            wire::QuantizeType::QT_RABITQ);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(QuantizeType::UNDEFINED),
            wire::QuantizeType::QT_UNDEFINED);
  EXPECT_EQ(QuantizeTypeCodeBook::Get(static_cast<QuantizeType>(999)),
            wire::QuantizeType::QT_UNDEFINED);
}

TEST(QuantizeTypeCodeBookTest, CppToStringConversion) {
  // Test conversion from C++ QuantizeType to string
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::FP16), "FP16");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::INT4), "INT4");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::INT8), "INT8");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::RABITQ), "RABITQ");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(QuantizeType::UNDEFINED),
            "UNDEFINED");
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(static_cast<QuantizeType>(999)),
            "UNDEFINED");

  // The set overload joins the sorted quantize types with commas.
  EXPECT_EQ(QuantizeTypeCodeBook::AsString(std::set<QuantizeType>{
                QuantizeType::RABITQ, QuantizeType::FP16, QuantizeType::INT8}),
            "FP16,INT8,RABITQ");
}

TEST(BlockTypeCodeBookTest, WireToCppConversion) {
  // Test conversion from the wire format to C++ BlockType
  EXPECT_EQ(BlockTypeCodeBook::Get(wire::BlockType::BT_SCALAR),
            BlockType::SCALAR);
  EXPECT_EQ(BlockTypeCodeBook::Get(wire::BlockType::BT_SCALAR_INDEX),
            BlockType::SCALAR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(wire::BlockType::BT_VECTOR_INDEX),
            BlockType::VECTOR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(wire::BlockType::BT_VECTOR_INDEX_QUANTIZE),
            BlockType::VECTOR_INDEX_QUANTIZE);
  EXPECT_EQ(BlockTypeCodeBook::Get(wire::BlockType::BT_FTS_INDEX),
            BlockType::FTS_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(wire::BlockType::BT_UNDEFINED),
            BlockType::UNDEFINED);
  EXPECT_EQ(BlockTypeCodeBook::Get(static_cast<wire::BlockType>(999)),
            BlockType::UNDEFINED);
}

TEST(BlockTypeCodeBookTest, CppToWireConversion) {
  // Test conversion from C++ BlockType to the wire format
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::SCALAR),
            wire::BlockType::BT_SCALAR);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::SCALAR_INDEX),
            wire::BlockType::BT_SCALAR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::VECTOR_INDEX),
            wire::BlockType::BT_VECTOR_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::VECTOR_INDEX_QUANTIZE),
            wire::BlockType::BT_VECTOR_INDEX_QUANTIZE);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::FTS_INDEX),
            wire::BlockType::BT_FTS_INDEX);
  EXPECT_EQ(BlockTypeCodeBook::Get(BlockType::UNDEFINED),
            wire::BlockType::BT_UNDEFINED);
  EXPECT_EQ(BlockTypeCodeBook::Get(static_cast<BlockType>(999)),
            wire::BlockType::BT_UNDEFINED);
}

TEST(BlockTypeCodeBookTest, CppToStringConversion) {
  // Test conversion from C++ BlockType to string
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::SCALAR), "SCALAR");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::SCALAR_INDEX),
            "SCALAR_INDEX");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::VECTOR_INDEX),
            "VECTOR_INDEX");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::VECTOR_INDEX_QUANTIZE),
            "VECTOR_INDEX_QUANTIZE");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::FTS_INDEX), "FTS_INDEX");
  EXPECT_EQ(BlockTypeCodeBook::AsString(BlockType::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(BlockTypeCodeBook::AsString(static_cast<BlockType>(999)),
            "UNDEFINED");
}
