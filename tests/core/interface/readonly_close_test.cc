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
#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_error.h>
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
    zvec::test_util::RemoveTestFiles(path_ + ".source");
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
      const auto expected =
          std::string(reinterpret_cast<const char *>(values_.data()),
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
      if (GetParam().type == IndexType::kHNSW && !GetParam().sparse) {
        auto hnsw = std::dynamic_pointer_cast<HNSWIndex>(reader);
        ASSERT_NE(nullptr, hnsw);
        EXPECT_EQ(GetParam().mmap
                      ? (GetParam().contiguous ? "contiguous" : "mmap")
                      : "buffer_pool",
                  hnsw->storage_mode());
      }
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
  uint32_t count = 3;
  if (GetParam().type == IndexType::kHNSW && !GetParam().sparse) {
    auto hnsw = std::dynamic_pointer_cast<HNSWIndex>(writer);
    ASSERT_NE(nullptr, hnsw);
    ASSERT_EQ(GetParam().mmap ? "mmap" : "buffer_pool", hnsw->storage_mode());
    ASSERT_NO_FATAL_FAILURE(check_vectors(writer, count));
    for (; count < 32; ++count) {
      ASSERT_EQ(0, writer->add(vector(), count));
      ASSERT_NO_FATAL_FAILURE(check_vectors(writer, count + 1));
    }
  }
  ASSERT_EQ(0, writer->close());
  writer.reset();

  auto reader = make_index();
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->open(path_, options(false, true)));
  ASSERT_NO_FATAL_FAILURE(check_vectors(reader, count));
  ASSERT_EQ(0, reader->close());
}

TEST_P(ReadOnlyCloseTest, ReadOnlyCannotCreateStorage) {
  auto reader = make_index();
  ASSERT_NE(nullptr, reader);
  EXPECT_NE(0, reader->open(path_, options(true, true)));
  EXPECT_FALSE(zvec::ailego::File::IsExist(path_));
}

TEST_P(ReadOnlyCloseTest, ReadOnlyRejectsMergeWithoutChangingData) {
  auto writer = make_index();
  ASSERT_NE(nullptr, writer);
  ASSERT_EQ(0, writer->open(path_, options(true)));
  ASSERT_EQ(0, writer->add(vector(), 0));
  ASSERT_EQ(0, writer->add(vector(), 1));
  ASSERT_EQ(0, writer->close());
  writer.reset();
  const auto before = snapshot();

  auto source = make_index();
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(0, source->open(path_ + ".source", options(true)));
  ASSERT_EQ(0, source->add(vector(), 0));
  ASSERT_EQ(0, source->flush());

  auto reader = make_index();
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->open(path_, options(false, true)));
  EXPECT_NE(0, reader->merge({source}, {}));
  ASSERT_NO_FATAL_FAILURE(check_vectors(reader, 2));
  EXPECT_TRUE(before == snapshot());
  ASSERT_EQ(0, reader->close());
  EXPECT_TRUE(before == snapshot());
  ASSERT_EQ(0, source->close());
}

std::vector<BaseIndexParam::Pointer> OfflineIndexParams() {
  std::vector<BaseIndexParam::Pointer> params;
  params.push_back(IVFIndexParamBuilder()
                       .with_dimension(64)
                       .with_data_type(DataType::DT_FP32)
                       .with_metric_type(MetricType::kL2sq)
                       .with_n_list(4)
                       .with_n_iters(2)
                       .build());
#if RABITQ_SUPPORTED
  params.push_back(IVFRabitqIndexParamBuilder()
                       .with_dimension(64)
                       .with_data_type(DataType::DT_FP32)
                       .with_metric_type(MetricType::kL2sq)
                       .with_n_list(4)
                       .with_total_bits(7)
                       .build());
#endif
#if DISKANN_SUPPORTED
  params.push_back(DiskAnnIndexParamBuilder()
                       .with_dimension(64)
                       .with_data_type(DataType::DT_FP32)
                       .with_metric_type(MetricType::kL2sq)
                       .with_max_degree(8)
                       .with_list_size(16)
                       .with_pq_chunk_num(4)
                       .build());
#endif
  return params;
}

TEST(ReadOnlyOfflineIndexTest, CannotCreateOrMergeIntoReadOnlyStorage) {
  ASSERT_EQ(0, zvec::ailego::MemoryLimitPool::get_instance().init(256UL * 1024 *
                                                                  1024));
  const auto params = OfflineIndexParams();
  for (const auto &param : params) {
    for (auto storage : {StorageOptions::StorageType::kMMAP,
                         StorageOptions::StorageType::kBufferPool}) {
      SCOPED_TRACE(::testing::Message()
                   << int(param->index_type) << " " << int(storage));
      const std::string path = "readonly_offline_target.index";
      const std::string source_path = path + ".source";
      zvec::test_util::RemoveTestFiles(path);
      zvec::test_util::RemoveTestFiles(source_path);
      auto snapshot = [&]() {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>());
      };

      auto rejected = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, rejected);
      EXPECT_NE(0, rejected->open(path, {storage, true, true}));
      EXPECT_FALSE(zvec::ailego::File::IsExist(path));
      rejected.reset();

      auto flat = FlatIndexParamBuilder()
                      .with_dimension(64)
                      .with_data_type(DataType::DT_FP32)
                      .with_metric_type(MetricType::kL2sq)
                      .build();
      auto source = IndexFactory::CreateAndInitIndex(*flat);
      ASSERT_NE(nullptr, source);
      ASSERT_EQ(0, source->open(source_path, {storage, true, false}));
      auto writer = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, writer);
      ASSERT_EQ(0, writer->open(path, {storage, true, false}));
      for (uint32_t id = 0; id < 128; ++id) {
        std::array<float, 64> data;
        for (size_t dim = 0; dim < data.size(); ++dim) {
          data[dim] = std::sin(float(id * 13 + dim + 1));
        }
        VectorData vector{DenseVector{data.data()}};
        ASSERT_EQ(0, source->add(vector, id));
        if (id < 64) {
          ASSERT_EQ(0, writer->add(vector, id));
        }
      }
      ASSERT_EQ(0, source->flush());
      ASSERT_EQ(0, writer->train());
      ASSERT_EQ(0, writer->close());
      writer.reset();
      const auto before = snapshot();
      ASSERT_FALSE(before.empty());

      rejected = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, rejected);
      EXPECT_NE(0, rejected->open(path, {storage, true, true}));
      rejected.reset();
      EXPECT_TRUE(before == snapshot());

      auto reader = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, reader);
      ASSERT_EQ(0, reader->open(path, {storage, false, true}));
      EXPECT_NE(0, reader->merge({source}, {}));
      EXPECT_TRUE(before == snapshot());
      EXPECT_EQ(64U, reader->get_doc_count());
      ASSERT_EQ(0, reader->close());
      reader.reset();
      EXPECT_TRUE(before == snapshot());
      ASSERT_EQ(0, source->close());
      source.reset();
      zvec::test_util::RemoveTestFiles(path);
      zvec::test_util::RemoveTestFiles(source_path);
    }
  }
}


struct OfflineMutationCase {
  std::string name;
  BaseIndexParam::Pointer param;
  StorageOptions::StorageType storage;
};

std::vector<OfflineMutationCase> OfflineMutationCases() {
  std::vector<OfflineMutationCase> cases;
  for (const auto &param : OfflineIndexParams()) {
    const std::string name = param->index_type == IndexType::kIVF ? "IVF"
                             : param->index_type == IndexType::kIVFRabitq
                                 ? "IVFRabitq"
                                 : "DiskAnn";
    cases.push_back({name + "Mmap", param, StorageOptions::StorageType::kMMAP});
    cases.push_back(
        {name + "Buffer", param, StorageOptions::StorageType::kBufferPool});
  }
  return cases;
}

class ReadOnlyOfflineMutationTest
    : public testing::TestWithParam<OfflineMutationCase> {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, zvec::ailego::MemoryLimitPool::get_instance().init(
                     256UL * 1024 * 1024));
    path_ = "readonly_offline_mutation_" + GetParam().name + ".index";
    zvec::test_util::RemoveTestFiles(path_);
  }

  void TearDown() override {
    zvec::test_util::RemoveTestFiles(path_);
  }

  Index::Pointer make_index() const {
    return IndexFactory::CreateAndInitIndex(*GetParam().param);
  }

  StorageOptions options(bool create_new, bool read_only = false) const {
    return {GetParam().storage, create_new, read_only};
  }

  int add(const Index::Pointer &index, uint32_t id) const {
    std::array<float, 64> data;
    for (size_t dim = 0; dim < data.size(); ++dim) {
      data[dim] = std::sin(float(id * 13 + dim + 1));
    }
    return index->add(VectorData{DenseVector{data.data()}}, id);
  }

  void reject_adds(const Index::Pointer &index) const {
    // Still call train() if add() incorrectly caches vectors.
    for (uint32_t id = 0; id < 32; ++id) {
      EXPECT_EQ(zvec::core::IndexError_NoReady, add(index, id));
    }
  }

  void populate(const Index::Pointer &index) const {
    for (uint32_t id = 0; id < 128; ++id) {
      ASSERT_EQ(0, add(index, id));
    }
    ASSERT_EQ(0, index->train());
    ASSERT_EQ(128U, index->get_doc_count());
  }

  std::string snapshot() const {
    std::ifstream input(path_, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

  std::string path_;
};

TEST_P(ReadOnlyOfflineMutationTest, RejectsMutationsBeforeOpen) {
  auto index = make_index();
  ASSERT_NE(nullptr, index);
  reject_adds(index);
  EXPECT_EQ(zvec::core::IndexError_NoReady, index->train());
  EXPECT_FALSE(zvec::ailego::File::IsExist(path_));
}

TEST_P(ReadOnlyOfflineMutationTest, FailedReadOnlyOpenCannotWriteTarget) {
  for (bool existing : {false, true}) {
    SCOPED_TRACE(existing);
    zvec::test_util::RemoveTestFiles(path_);
    if (existing) {
      std::ofstream output(path_, std::ios::binary);
      output << "existing index contents - invalid header";
      ASSERT_TRUE(output.good());
    }
    const auto before = snapshot();
    auto index = make_index();
    ASSERT_NE(nullptr, index);
    ASSERT_NE(0, index->open(path_, options(false, true)));
    reject_adds(index);
    EXPECT_EQ(zvec::core::IndexError_NoReady, index->train());
    EXPECT_EQ(existing, zvec::ailego::File::IsExist(path_));
    EXPECT_TRUE(before == snapshot());
    index.reset();
    EXPECT_EQ(existing, zvec::ailego::File::IsExist(path_));
    EXPECT_TRUE(before == snapshot());
  }
}

TEST_P(ReadOnlyOfflineMutationTest, ReadOnlyTrainingPreservesExistingData) {
  auto writer = make_index();
  ASSERT_NE(nullptr, writer);
  ASSERT_EQ(0, writer->open(path_, options(true)));
  ASSERT_NO_FATAL_FAILURE(populate(writer));
  ASSERT_EQ(0, writer->close());
  writer.reset();
  const auto before = snapshot();
  ASSERT_FALSE(before.empty());

  auto reader = make_index();
  ASSERT_NE(nullptr, reader);
  reject_adds(reader);
  ASSERT_EQ(0, reader->open(path_, options(false, true)));
  EXPECT_EQ(128U, reader->get_doc_count());
  reject_adds(reader);
  // Training an already trained index is a no-op.
  EXPECT_EQ(0, reader->train());
  EXPECT_TRUE(before == snapshot());
  ASSERT_EQ(0, reader->close());
  reader.reset();
  EXPECT_TRUE(before == snapshot());

  auto reopened = make_index();
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(0, reopened->open(path_, options(false, true)));
  EXPECT_EQ(128U, reopened->get_doc_count());
  ASSERT_EQ(0, reopened->close());
  reopened.reset();
  EXPECT_TRUE(before == snapshot());
}

TEST_P(ReadOnlyOfflineMutationTest,
       WritableTrainIsIdempotentAndCloseStopsWrites) {
  auto writer = make_index();
  ASSERT_NE(nullptr, writer);
  ASSERT_EQ(0, writer->open(path_, options(true)));
  ASSERT_NO_FATAL_FAILURE(populate(writer));
  const auto trained = snapshot();
  ASSERT_FALSE(trained.empty());
  ASSERT_EQ(0, writer->train());
  EXPECT_TRUE(trained == snapshot());
  EXPECT_EQ(128U, writer->get_doc_count());
  ASSERT_EQ(0, writer->close());
  const auto closed = snapshot();
  reject_adds(writer);
  EXPECT_EQ(0, writer->train());
  writer.reset();
  EXPECT_TRUE(closed == snapshot());

  auto reader = make_index();
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->open(path_, options(false, true)));
  EXPECT_EQ(128U, reader->get_doc_count());
  ASSERT_EQ(0, reader->close());
}

INSTANTIATE_TEST_SUITE_P(
    StorageAndOfflineIndexTypes, ReadOnlyOfflineMutationTest,
    testing::ValuesIn(OfflineMutationCases()),
    [](const testing::TestParamInfo<OfflineMutationCase> &info) {
      return info.param.name;
    });

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
        CloseTestCase{"VamanaContiguous", IndexType::kVamana, false, true,
                      true}),
    [](const testing::TestParamInfo<CloseTestCase> &info) {
      return info.param.name;
    });

}  // namespace
