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

#include <array>
#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "tests/test_util.h"

using namespace zvec::core_interface;

namespace {

struct CloseTestCase {
  const char *name;
  IndexType type;
  bool sparse;
  bool mmap;
  bool contiguous;
};

class ReadOnlyCloseTest : public testing::TestWithParam<CloseTestCase> {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, zvec::ailego::MemoryLimitPool::get_instance().init(
                     256UL * 1024 * 1024));
    path_ = std::string("readonly_close_") + GetParam().name + ".index";
    zvec::test_util::RemoveTestFiles(path_);
  }

  void TearDown() override {
    zvec::test_util::RemoveTestFiles(path_);
  }

  Index::Pointer make_index() const {
    const auto &test = GetParam();
    BaseIndexParam::Pointer param;
    if (test.type == IndexType::kFlat) {
      param = FlatIndexParamBuilder()
                  .with_metric_type(MetricType::kCosine)
                  .with_data_type(DataType::DT_FP32)
                  .with_dimension(4)
                  .with_use_contiguous_memory(test.contiguous)
                  .build();
    } else if (test.type == IndexType::kHNSW) {
      param = HNSWIndexParamBuilder()
                  .with_metric_type(test.sparse ? MetricType::kInnerProduct
                                                : MetricType::kCosine)
                  .with_data_type(DataType::DT_FP32)
                  .with_dimension(4)
                  .with_is_sparse(test.sparse)
                  .with_use_contiguous_memory(test.contiguous)
                  .build();
    } else {
      param = VamanaIndexParamBuilder()
                  .with_metric_type(MetricType::kCosine)
                  .with_data_type(DataType::DT_FP32)
                  .with_dimension(4)
                  .with_use_contiguous_memory(test.contiguous)
                  .build();
    }
    return IndexFactory::CreateAndInitIndex(*param);
  }

  StorageOptions options(bool create, bool read_only = false) const {
    return {GetParam().mmap ? StorageOptions::StorageType::kMMAP
                           : StorageOptions::StorageType::kBufferPool,
            create, read_only};
  }

  VectorData vector() const {
    if (GetParam().sparse) {
      return {SparseVector{1, indices_.data(), values_.data()}};
    }
    return {DenseVector{values_.data()}};
  }

  void check_vectors(const Index::Pointer &index, uint32_t count) const {
    for (uint32_t id = 0; id < count; ++id) {
      VectorDataBuffer fetched;
      ASSERT_EQ(0, index->fetch(id, &fetched));
      const auto expected = std::string(
          reinterpret_cast<const char *>(values_.data()),
          GetParam().sparse ? sizeof(float) : sizeof(values_));
      if (GetParam().sparse) {
        const auto &sparse =
            std::get<SparseVectorBuffer>(fetched.vector_buffer);
        EXPECT_EQ(1U, sparse.count);
        EXPECT_EQ(expected, sparse.values);
      } else {
        EXPECT_EQ(expected,
                  std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
      }
    }
    if (count != 0) {
      BaseIndexQueryParam::Pointer query;
      if (GetParam().type == IndexType::kFlat) {
        query = FlatQueryParamBuilder().with_topk(count).build();
      } else if (GetParam().type == IndexType::kHNSW) {
        query = HNSWQueryParamBuilder().with_topk(count).build();
      } else {
        query = VamanaQueryParamBuilder().with_topk(count).build();
      }
      SearchResult result;
      ASSERT_EQ(0, index->search(vector(), query, &result));
      EXPECT_EQ(count, result.doc_list_.size());
    }
  }

  std::string snapshot() const {
    std::ifstream input(path_, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

  std::string path_;
  const std::array<float, 4> values_{1.0f, 0.0f, 0.0f, 0.0f};
  const std::array<uint32_t, 1> indices_{0};
};

TEST_P(ReadOnlyCloseTest, PreservesEveryByteAcrossOpenAndClose) {
  // Cover empty indexes as well as the minimal populated issue reproducer.
  for (uint32_t count : {0U, 2U}) {
    SCOPED_TRACE(count);
    zvec::test_util::RemoveTestFiles(path_);
    auto writer = make_index();
    ASSERT_NE(nullptr, writer);
    ASSERT_EQ(0, writer->open(path_, options(true)));
    for (uint32_t id = 0; id < count; ++id) {
      ASSERT_EQ(0, writer->add(vector(), id));
    }
    ASSERT_EQ(0, writer->flush());
    ASSERT_EQ(0, writer->close());
    writer.reset();
    const auto before = snapshot();
    ASSERT_FALSE(before.empty());

    for (int cycle = 0; cycle < 2; ++cycle) {
      SCOPED_TRACE(cycle);
      // Metadata timestamps use seconds; immediate cycles can hide writes.
      std::this_thread::sleep_for(std::chrono::milliseconds(1100));
      auto reader = make_index();
      ASSERT_NE(nullptr, reader);
      ASSERT_EQ(0, reader->open(path_, options(false, true)));
      EXPECT_TRUE(before == snapshot()) << "open changed persisted bytes";
      if (cycle == 1) {
        ASSERT_NO_FATAL_FAILURE(check_vectors(reader, count));
      }
      EXPECT_TRUE(before == snapshot()) << "reads changed persisted bytes";
      ASSERT_EQ(0, reader->close());
      reader.reset();
      EXPECT_TRUE(before == snapshot()) << "close changed persisted bytes";
    }
  }
}

TEST_P(ReadOnlyCloseTest, WritableClosePersistsWritesAfterFlushAndReopen) {
  auto writer = make_index();
  ASSERT_NE(nullptr, writer);
  ASSERT_EQ(0, writer->open(path_, options(true)));
  ASSERT_EQ(0, writer->add(vector(), 0));
  ASSERT_EQ(0, writer->flush());
  ASSERT_EQ(0, writer->add(vector(), 1));
  ASSERT_EQ(0, writer->close());
  writer.reset();

  writer = make_index();
  ASSERT_NE(nullptr, writer);
  // BufferStorage uses create_if_missing as write intent, including reopen.
  ASSERT_EQ(0, writer->open(path_, options(true)));
  ASSERT_NO_FATAL_FAILURE(check_vectors(writer, 2));
  ASSERT_EQ(0, writer->add(vector(), 2));
  ASSERT_EQ(0, writer->close());
  writer.reset();

  auto reader = make_index();
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->open(path_, options(false, true)));
  ASSERT_NO_FATAL_FAILURE(check_vectors(reader, 3));
  ASSERT_EQ(0, reader->close());
}

INSTANTIATE_TEST_SUITE_P(
    StorageAndIndexTypes, ReadOnlyCloseTest,
    testing::Values(
        CloseTestCase{"FlatMmap", IndexType::kFlat, false, true, false},
        CloseTestCase{"FlatBuffer", IndexType::kFlat, false, false, false},
        CloseTestCase{"FlatContiguous", IndexType::kFlat, false, true, true},
        CloseTestCase{"HnswMmap", IndexType::kHNSW, false, true, false},
        CloseTestCase{"HnswBuffer", IndexType::kHNSW, false, false, false},
        CloseTestCase{"HnswContiguous", IndexType::kHNSW, false, true, true},
        CloseTestCase{"SparseHnswMmap", IndexType::kHNSW, true, true, false},
        CloseTestCase{"SparseHnswBuffer", IndexType::kHNSW, true, false, false},
        CloseTestCase{"VamanaMmap", IndexType::kVamana, false, true, false},
        CloseTestCase{"VamanaBuffer", IndexType::kVamana, false, false, false},
        CloseTestCase{"VamanaContiguous", IndexType::kVamana, false, true, true}),
    [](const testing::TestParamInfo<CloseTestCase> &info) {
      return info.param.name;
    });

}  // namespace
