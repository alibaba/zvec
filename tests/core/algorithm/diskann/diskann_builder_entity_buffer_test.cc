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

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include "diskann_builder_entity.h"

namespace zvec::core {
namespace {

class RecordingDumper : public IndexDumper {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int create(const std::string &) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  size_t write(const void *data, size_t size) override {
    largest_write = std::max(largest_write, size);
    if (fail_write) return 0;
    if (size != 0) bytes.append(static_cast<const char *>(data), size);
    return size;
  }
  int append(const std::string &, size_t size, size_t padding,
             uint32_t checksum) override {
    data_size = size;
    padding_size = padding;
    crc = checksum;
    return 0;
  }

  bool fail_write{false};
  size_t largest_write{0};
  size_t data_size{0};
  size_t padding_size{0};
  uint32_t crc{0};
  std::string bytes;
};

class DiskAnnBuilderEntityBufferTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Writable pools also charge their stable page table, lock stripes and
    // 128-page writeback staging. On macOS staging alone is 2 MiB per file.
    // Leave only 1 MiB for resident data across our two temporary storages,
    // irrespective of the platform's page size and metadata requirements.
    const size_t metadata =
        ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
    ailego::MemoryLimitPool::get_instance().init(2 * metadata +
                                                 1024UL * 1024UL);
  }

  void SetUp() override {
    static std::atomic<size_t> counter{0};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    directory_ = "diskann_builder_entity_buffer_" + std::to_string(stamp) +
                 "_" + std::to_string(counter.fetch_add(1));
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }

  void TearDown() override {
    // The entity owns and removes only its scratch files. No unknown files
    // are removed recursively, even in this test fixture.
    std::error_code error;
    EXPECT_TRUE(std::filesystem::is_empty(directory_, error));
    EXPECT_TRUE(std::filesystem::remove(directory_, error));
  }

  void initialize(DiskAnnBuilderEntity *entity, uint32_t dimension,
                  uint32_t count, bool buffered = true) {
    IndexMeta meta(IndexMeta::DataType::DT_FP32, dimension);
    ASSERT_EQ(0, entity->init(meta, 8, 16, 0, 2));
    if (buffered)
      ASSERT_EQ(0, entity->enable_buffered_build(directory_ + "/build"));
    ASSERT_EQ(0, entity->reserve_space(count));
    std::vector<float> vector(dimension);
    for (uint32_t id = 0; id < count; ++id) {
      for (uint32_t col = 0; col < dimension; ++col)
        vector[col] = id + col * 0.25F;
      ASSERT_EQ(0, entity->add_vector(id + 17, vector.data()));
    }
  }

  std::string directory_;
};

TEST_F(DiskAnnBuilderEntityBufferTest,
       StagedVectorsSurvivePressureAndOtherReads) {
  DiskAnnBuilderEntity entity;
  constexpr uint32_t kDimension = 257;
  constexpr uint32_t kCount = 4096;
  initialize(&entity, kDimension, kCount);
  ASSERT_EQ(kCount, entity.doc_cnt());
  EXPECT_TRUE(entity.buffered_build());
  EXPECT_EQ(nullptr, entity.get_vector(0));

  IndexStorage::MemoryBlock first;
  ASSERT_EQ(0, entity.read_vector(0, first));
  for (uint32_t id = 1; id < kCount; id += 17) {
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, entity.read_vector(id, block));
    const auto *vector = static_cast<const float *>(block.data());
    EXPECT_FLOAT_EQ(static_cast<float>(id), vector[0]);
    EXPECT_FLOAT_EQ(id + (kDimension - 1) * 0.25F, vector[kDimension - 1]);
  }
  EXPECT_FLOAT_EQ(0.0F, static_cast<const float *>(first.data())[0]);
  EXPECT_FLOAT_EQ(64.0F, static_cast<const float *>(first.data())[256]);
  first.reset();

  ASSERT_EQ(0, entity.set_neighbors(0, {1, 2}));
  entity.release_vectors();
  EXPECT_EQ(IndexError_ReadData, entity.read_vector(0, first));
  std::vector<diskann_id_t> neighbors;
  ASSERT_EQ(0, entity.read_neighbors(0, &neighbors));
  EXPECT_EQ((std::vector<diskann_id_t>{1, 2}), neighbors);
  EXPECT_EQ(17U, entity.get_key(0));
}

TEST_F(DiskAnnBuilderEntityBufferTest,
       MutableGraphMatchesInMemoryAndChecksBounds) {
  DiskAnnBuilderEntity buffered;
  DiskAnnBuilderEntity memory;
  initialize(&buffered, 16, 32);
  initialize(&memory, 16, 32, false);
  for (diskann_id_t id = 0; id < 32; ++id) {
    const std::vector<diskann_id_t> expected{(id + 1) % 32, (id + 2) % 32};
    ASSERT_EQ(0, buffered.set_neighbors(id, expected));
    ASSERT_EQ(0, memory.set_neighbors(id, expected));
    ASSERT_EQ(0, buffered.add_neighbor(id, (id + 3) % 32));
    ASSERT_EQ(0, memory.add_neighbor(id, (id + 3) % 32));
    std::vector<diskann_id_t> lhs;
    std::vector<diskann_id_t> rhs;
    ASSERT_EQ(0, buffered.read_neighbors(id, &lhs));
    ASSERT_EQ(0, memory.read_neighbors(id, &rhs));
    EXPECT_EQ(lhs, rhs);
  }
  EXPECT_EQ(IndexError_InvalidArgument, buffered.set_neighbors(32, {0}));
  EXPECT_EQ(IndexError_InvalidArgument, buffered.set_neighbors(0, {32}));
  EXPECT_EQ(IndexError_InvalidArgument,
            buffered.set_neighbors(0, std::vector<diskann_id_t>(32, 1)));
  EXPECT_EQ(IndexError_InvalidArgument, buffered.add_neighbor(32, 0));
  EXPECT_EQ(IndexError_InvalidArgument, buffered.read_neighbors(0, nullptr));
  IndexStorage::MemoryBlock block;
  EXPECT_EQ(IndexError_InvalidArgument, buffered.read_vector(32, block));
  EXPECT_EQ(kInvalidKey, buffered.get_key(32));
  const float vector[16]{};
  EXPECT_EQ(IndexError_InvalidArgument, buffered.add_vector(100, vector));
}

TEST_F(DiskAnnBuilderEntityBufferTest,
       ConcurrentGraphUpdatesKeepIndependentRows) {
  DiskAnnBuilderEntity entity;
  initialize(&entity, 32, 512);
  std::atomic<int> error{0};
  std::vector<std::thread> workers;
  for (diskann_id_t worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&, worker]() {
      for (diskann_id_t id = worker; id < 512; id += 4) {
        const std::vector<diskann_id_t> neighbors{(id + 1) % 512,
                                                  (id + 7) % 512};
        int ret = entity.set_neighbors(id, neighbors);
        if (ret == 0) {
          std::vector<diskann_id_t> snapshot;
          ret = entity.read_neighbors(id, &snapshot);
          if (snapshot != neighbors) ret = IndexError_Mismatch;
        }
        if (ret != 0) error.store(ret);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0, error.load());
}

TEST_F(DiskAnnBuilderEntityBufferTest,
       StreamedCodesMatchFormatAndPermitDumpRetry) {
  DiskAnnBuilderEntity buffered;
  DiskAnnBuilderEntity memory;
  initialize(&buffered, 4, 32);
  initialize(&memory, 4, 32, false);
  constexpr size_t kChunkSize = 65537;
  constexpr size_t kBytes = 32 * kChunkSize;
  buffered.mutable_pq_meta()->chunk_num = kChunkSize;
  memory.mutable_pq_meta()->chunk_num = kChunkSize;
  std::string expected(kBytes, '\0');
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = static_cast<char>(i % 251);
  ASSERT_EQ(0, buffered.prepare_codes(kBytes));
  ASSERT_EQ(0, memory.prepare_codes(kBytes));
  for (size_t offset = 0; offset < kBytes;) {
    const size_t size = std::min<size_t>(100003, kBytes - offset);
    ASSERT_EQ(0, buffered.append_codes(expected.data() + offset, size));
    ASSERT_EQ(0, memory.append_codes(expected.data() + offset, size));
    offset += size;
  }
  EXPECT_TRUE(buffered.block_compressed_data().empty());
  EXPECT_EQ(IndexError_InvalidArgument, buffered.append_codes("x", 1));
  auto failed = std::make_shared<RecordingDumper>();
  failed->fail_write = true;
  EXPECT_EQ(IndexError_WriteData, buffered.dump_pq_data_segment(failed));
  auto reference = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, memory.dump_pq_data_segment(reference));
  for (size_t repetition = 0; repetition < 2; ++repetition) {
    auto result = std::make_shared<RecordingDumper>();
    ASSERT_EQ(0, buffered.dump_pq_data_segment(result));
    EXPECT_EQ(reference->bytes, result->bytes);
    EXPECT_EQ(reference->data_size, result->data_size);
    EXPECT_EQ(reference->padding_size, result->padding_size);
    EXPECT_EQ(reference->crc, result->crc);
    EXPECT_LE(result->largest_write, 1024UL * 1024UL);
  }
}

TEST_F(DiskAnnBuilderEntityBufferTest, IncompleteCodesFailBeforeDump) {
  DiskAnnBuilderEntity entity;
  initialize(&entity, 4, 8);
  entity.mutable_pq_meta()->chunk_num = 2;
  ASSERT_EQ(0, entity.prepare_codes(16));
  ASSERT_EQ(0, entity.append_codes("abc", 3));
  auto dumper = std::make_shared<RecordingDumper>();
  EXPECT_EQ(IndexError_Mismatch, entity.dump_pq_data_segment(dumper));
  EXPECT_TRUE(dumper->bytes.empty());
  EXPECT_EQ(IndexError_InvalidArgument, entity.append_codes(nullptr, 1));
}

TEST_F(DiskAnnBuilderEntityBufferTest, ScratchFailureDoesNotRemoveCallerFiles) {
  const std::string marker = directory_ + "/keep";
  ailego::File file;
  ASSERT_TRUE(file.create(marker, 0));
  file.close();
  {
    DiskAnnBuilderEntity entity;
    IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
    ASSERT_EQ(0, entity.init(meta, 8, 16, 0, 1));
    ASSERT_EQ(0, entity.enable_buffered_build(directory_ + "/missing/build"));
    EXPECT_TRUE(entity.buffered_build());
    EXPECT_EQ(IndexError_OpenFile, entity.reserve_space(8));
    EXPECT_TRUE(std::filesystem::exists(marker));
    entity.clear();
    ASSERT_EQ(0, entity.init(meta, 8, 16, 0, 1));
    ASSERT_EQ(0, entity.enable_buffered_build(directory_ + "/build"));
    EXPECT_EQ(1, std::distance(std::filesystem::directory_iterator(directory_),
                               std::filesystem::directory_iterator()));
    EXPECT_EQ(IndexError_InvalidArgument,
              entity.enable_buffered_build(directory_ + "/other"));
    ASSERT_EQ(0, entity.reserve_space(8));
    entity.clear();
    EXPECT_FALSE(entity.buffered_build());
  }
  EXPECT_TRUE(std::filesystem::exists(marker));
  ASSERT_TRUE(ailego::File::Delete(marker));
}

TEST_F(DiskAnnBuilderEntityBufferTest, BufferedModeDoesNotCreateUnusedScratch) {
  {
    DiskAnnBuilderEntity entity;
    IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
    ASSERT_EQ(0, entity.init(meta, 8, 16, 0, 1));
    ASSERT_EQ(0, entity.enable_buffered_build(directory_ + "/build"));
    EXPECT_TRUE(entity.buffered_build());
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
    entity.clear();
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
  }
  EXPECT_TRUE(std::filesystem::is_empty(directory_));
}

TEST_F(DiskAnnBuilderEntityBufferTest,
       MetadataPressureFailsCleanlyAndCanRetry) {
  DiskAnnBuilderEntity entity;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
  ASSERT_EQ(0, entity.init(meta, 8, 16, 0, 1));
  ASSERT_EQ(0, entity.enable_buffered_build(directory_ + "/build"));
  auto &pool = ailego::MemoryLimitPool::get_instance();
  {
    ASSERT_GT(pool.available(), ailego::kVectorPageSize);
    const size_t reservation = pool.available() - ailego::kVectorPageSize;
    ASSERT_TRUE(pool.try_charge_external(reservation));
    AILEGO_DEFER([&]() { pool.release_external(reservation); });
    EXPECT_EQ(IndexError_NoMemory, entity.reserve_space(8));
    EXPECT_TRUE(entity.buffered_build());
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
  }
  ASSERT_EQ(0, entity.reserve_space(8));
  const float vector[4] = {1, 2, 3, 4};
  ASSERT_EQ(0, entity.add_vector(17, vector));
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(0, entity.read_vector(0, block));
  EXPECT_FLOAT_EQ(4.0F, static_cast<const float *>(block.data())[3]);
}

}  // namespace
}  // namespace zvec::core
