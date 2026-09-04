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

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_builder.h"
#include "diskann_file_reader.h"
#include "diskann_pq_trainer.h"
#include "diskann_util.h"

namespace zvec::core {
namespace {

class TemporaryFile {
 public:
  TemporaryFile() : fd_(::mkstemp(path_)) {}

  ~TemporaryFile() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    ::unlink(path_);
  }

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

  int fd() const {
    return fd_;
  }

  const char *path() const {
    return path_;
  }

  void release_descriptor_and_unlink() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    ::unlink(path_);
  }

 private:
  char path_[64] = "DiskAnnMobileCompatTest.XXXXXX";
  int fd_{-1};
};

TEST(DiskAnnMobileCompatTest, AlignedAllocationSupportsUnroundedSize) {
  constexpr size_t kSize = 400;
  constexpr size_t kAlignment = 256;

  void *buffer = nullptr;
  DiskAnnUtil::alloc_aligned(&buffer, kSize, kAlignment);

  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer) % kAlignment, 0u);
  std::memset(buffer, 0xa5, kSize);
  DiskAnnUtil::free_aligned(buffer);
}

template <typename T>
void ExpectExactPqPivotCopy(IndexMeta::DataType data_type) {
  constexpr uint32_t kDimension = 4;
  constexpr uint32_t kCenterCount = 2;
  constexpr uint32_t kChunkCount = 2;
  const std::vector<uint32_t> chunk_dims{2, 2};
  const std::vector<uint32_t> chunk_offsets{0, 2, 4};
  const std::array<std::array<float, 2>, 4> values{{
      {{1.0F, 2.0F}},
      {{5.0F, 6.0F}},
      {{3.0F, 4.0F}},
      {{7.0F, 8.0F}},
  }};

  IndexCluster::CentroidList centroids(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const std::array<T, 2> feature{{T(values[i][0]), T(values[i][1])}};
    centroids[i].set_feature(feature.data(), sizeof(feature));
  }

  IndexMeta meta(data_type, kDimension);
  std::vector<uint8_t> pivots;
  ASSERT_EQ(DiskAnnPqTrainer::convert_pivot_data<T>(
                meta, kCenterCount, kChunkCount, chunk_dims, chunk_offsets,
                centroids, pivots),
            0);
  ASSERT_EQ(pivots.size(), kCenterCount * meta.element_size());

  std::array<T, kCenterCount * kDimension> actual{};
  std::memcpy(actual.data(), pivots.data(), pivots.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(actual[i]), static_cast<float>(i + 1));
  }
}

TEST(DiskAnnMobileCompatTest, PqPivotConversionCopiesExactChunkWidths) {
  ExpectExactPqPivotCopy<float>(IndexMeta::DataType::DT_FP32);
  ExpectExactPqPivotCopy<ailego::Float16>(IndexMeta::DataType::DT_FP16);
}

TEST(DiskAnnMobileCompatTest, PortableReaderReadsAlignedBatch) {
  constexpr size_t kBlockSize = 4096;
  constexpr size_t kBlockCount = 2;
  constexpr size_t kDataSize = kBlockSize * kBlockCount;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> expected(kDataSize);
  std::fill(expected.begin(), expected.begin() + kBlockSize, 0x3c);
  std::fill(expected.begin() + kBlockSize, expected.end(), 0xc3);
  ASSERT_EQ(::pwrite(file.fd(), expected.data(), expected.size(), 0),
            static_cast<ssize_t>(expected.size()));

  void *output = nullptr;
  DiskAnnUtil::alloc_aligned(&output, kDataSize, kBlockSize);
  ASSERT_NE(output, nullptr);
  std::memset(output, 0, kDataSize);

  LinuxAlignedFileReader reader;
  reader.open(file.path());
  IOContext context{};
  std::vector<AlignedRead> requests;
  requests.emplace_back(0, kBlockSize, output);
  requests.emplace_back(kBlockSize, kBlockSize,
                        static_cast<uint8_t *>(output) + kBlockSize);

  EXPECT_EQ(reader.read(requests, context), 0);
  EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

  reader.close();
  DiskAnnUtil::free_aligned(output);
}

TEST(DiskAnnMobileCompatTest, PortableReaderRejectsShortRead) {
  constexpr size_t kBlockSize = 4096;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> expected(kBlockSize, 0x5a);
  ASSERT_EQ(::pwrite(file.fd(), expected.data(), expected.size(), 0),
            static_cast<ssize_t>(expected.size()));

  void *output = nullptr;
  DiskAnnUtil::alloc_aligned(&output, kBlockSize * 2, kBlockSize);
  ASSERT_NE(output, nullptr);

  LinuxAlignedFileReader reader;
  reader.open(file.path());
  IOContext context{};
  std::vector<AlignedRead> requests;
  requests.emplace_back(0, kBlockSize * 2, output);

  EXPECT_NE(reader.read(requests, context), 0);

  requests.clear();
  requests.emplace_back(0, kBlockSize, output);
  EXPECT_EQ(reader.read(requests, context), 0);
  EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

  reader.close();
  DiskAnnUtil::free_aligned(output);
}

TEST(DiskAnnMobileCompatTest, PortableReaderRecoversAfterOpenFailure) {
  constexpr size_t kBlockSize = 4096;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> expected(kBlockSize, 0x6b);
  ASSERT_EQ(::pwrite(file.fd(), expected.data(), expected.size(), 0),
            static_cast<ssize_t>(expected.size()));

  void *output = nullptr;
  DiskAnnUtil::alloc_aligned(&output, kBlockSize, kBlockSize);
  ASSERT_NE(output, nullptr);

  LinuxAlignedFileReader reader;
  reader.open("DiskAnnMobileCompatTest.missing");
  IOContext context{};
  std::vector<AlignedRead> requests;
  requests.emplace_back(0, kBlockSize, output);
  EXPECT_NE(reader.read(requests, context), 0);

  reader.open(file.path());
  EXPECT_EQ(reader.read(requests, context), 0);
  EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

  reader.close();
  DiskAnnUtil::free_aligned(output);
}

TEST(DiskAnnMobileCompatTest, PortableReaderSupportsConcurrentReads) {
  constexpr size_t kBlockSize = 4096;
  constexpr size_t kThreadCount = 4;
  constexpr size_t kDataSize = kBlockSize * kThreadCount;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> expected(kDataSize);
  for (size_t i = 0; i < kThreadCount; ++i) {
    std::fill(expected.begin() + i * kBlockSize,
              expected.begin() + (i + 1) * kBlockSize,
              static_cast<uint8_t>(i + 1));
  }
  ASSERT_EQ(::pwrite(file.fd(), expected.data(), expected.size(), 0),
            static_cast<ssize_t>(expected.size()));

  std::array<void *, kThreadCount> outputs{};
  for (void *&output : outputs) {
    DiskAnnUtil::alloc_aligned(&output, kBlockSize, kBlockSize);
    ASSERT_NE(output, nullptr);
  }

  LinuxAlignedFileReader reader;
  reader.open(file.path());
  std::array<int, kThreadCount> statuses{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      IOContext context{};
      std::vector<AlignedRead> requests;
      requests.emplace_back(i * kBlockSize, kBlockSize, outputs[i]);
      statuses[i] = reader.read(requests, context);
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  for (size_t i = 0; i < kThreadCount; ++i) {
    EXPECT_EQ(statuses[i], 0);
    EXPECT_EQ(
        std::memcmp(outputs[i], expected.data() + i * kBlockSize, kBlockSize),
        0);
    DiskAnnUtil::free_aligned(outputs[i]);
  }
  reader.close();
}

TEST(DiskAnnMobileCompatTest, BuildDumpLoadAndSearch) {
  constexpr size_t kDimension = 10;
  constexpr size_t kDocCount = 64;
  constexpr uint64_t kExpectedKey = 12;

  TemporaryFile index_file;
  ASSERT_GE(index_file.fd(), 0);
  index_file.release_descriptor_and_unlink();

  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  for (size_t i = 0; i < kDocCount; ++i) {
    ailego::NumericalVector<float> vector(kDimension, static_cast<float>(i));
    ASSERT_TRUE(holder->emplace(i, vector));
  }

  ailego::Params build_params;
  build_params.set("zvec.diskann.builder.max_degree", 16);
  build_params.set("zvec.diskann.builder.list_size", 32);
  build_params.set("zvec.diskann.builder.max_pq_chunk_num", 2);
  build_params.set("zvec.diskann.builder.threads", 2);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);
  ASSERT_EQ(builder->init(meta, build_params), 0);
  ASSERT_EQ(builder->train(holder), 0);
  ASSERT_EQ(builder->build(holder), 0);

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(dumper->create(index_file.path()), 0);
  ASSERT_EQ(builder->dump(dumper), 0);
  ASSERT_EQ(dumper->close(), 0);

  int snapshot_fd = ::open(index_file.path(), O_RDONLY);
  ASSERT_GE(snapshot_fd, 0);
  struct stat snapshot_stat {};
  ASSERT_EQ(::fstat(snapshot_fd, &snapshot_stat), 0);
  ASSERT_GT(snapshot_stat.st_size, 4096);
  std::vector<uint8_t> snapshot(static_cast<size_t>(snapshot_stat.st_size));
  ASSERT_EQ(::pread(snapshot_fd, snapshot.data(), snapshot.size(), 0),
            static_cast<ssize_t>(snapshot.size()));
  ASSERT_EQ(::close(snapshot_fd), 0);

  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 64);
  ASSERT_EQ(searcher->init(search_params), 0);

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(storage->open(index_file.path(), false), 0);
  ASSERT_EQ(searcher->load(storage, IndexMetric::Pointer()), 0);

  auto context = searcher->create_context();
  ASSERT_NE(context, nullptr);
  context->set_topk(5);

  ailego::NumericalVector<float> query(kDimension, 12.1f);
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDimension);
  ASSERT_EQ(searcher->search_impl(query.data(), query_meta, context), 0);

  const auto &result = context->result();
  ASSERT_FALSE(result.empty());
  EXPECT_NE(
      std::find_if(result.begin(), result.end(),
                   [](const auto &item) { return item.key() == kExpectedKey; }),
      result.end());

  IndexStreamer::Pointer first_streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(first_streamer, nullptr);
  ASSERT_EQ(first_streamer->init(meta, search_params), 0);
  auto first_streamer_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(first_streamer_storage, nullptr);
  ASSERT_EQ(first_streamer_storage->open(index_file.path(), false), 0);
  ASSERT_EQ(first_streamer->open(first_streamer_storage), 0);

  IndexStreamer::Pointer second_streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(second_streamer, nullptr);
  ASSERT_EQ(second_streamer->init(meta, search_params), 0);
  auto second_streamer_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(second_streamer_storage, nullptr);
  ASSERT_EQ(second_streamer_storage->open(index_file.path(), false), 0);
  ASSERT_EQ(second_streamer->open(second_streamer_storage), 0);

  auto switching_context = first_streamer->create_context();
  ASSERT_NE(switching_context, nullptr);
  switching_context->set_topk(5);
  switching_context->set_filter(
      [](uint64_t key) { return key != kExpectedKey; });
  ASSERT_EQ(
      second_streamer->search_impl(query.data(), query_meta, switching_context),
      0);
  ASSERT_EQ(switching_context->result().size(), 1u);
  EXPECT_EQ(switching_context->result().front().key(), kExpectedKey);

  context.reset();
  searcher.reset();
  storage.reset();

  ASSERT_EQ(::truncate(index_file.path(), snapshot.size() - 4096), 0);
  searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(searcher->init(search_params), 0);
  storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  int corrupt_open_result = storage->open(index_file.path(), false);
  bool corrupt_index_rejected = corrupt_open_result != 0;
  if (corrupt_open_result == 0) {
    corrupt_index_rejected =
        searcher->load(storage, IndexMetric::Pointer()) != 0;
  }
  EXPECT_TRUE(corrupt_index_rejected);

  searcher.reset();
  storage.reset();
  int restore_fd = ::open(index_file.path(), O_WRONLY | O_TRUNC);
  ASSERT_GE(restore_fd, 0);
  ASSERT_EQ(::pwrite(restore_fd, snapshot.data(), snapshot.size(), 0),
            static_cast<ssize_t>(snapshot.size()));
  ASSERT_EQ(::fsync(restore_fd), 0);
  ASSERT_EQ(::close(restore_fd), 0);

  searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(searcher->init(search_params), 0);
  storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(storage->open(index_file.path(), false), 0);
  ASSERT_EQ(searcher->load(storage, IndexMetric::Pointer()), 0);
  context = searcher->create_context();
  ASSERT_NE(context, nullptr);
  context->set_topk(5);
  ASSERT_EQ(searcher->search_impl(query.data(), query_meta, context), 0);
  EXPECT_NE(
      std::find_if(context->result().begin(), context->result().end(),
                   [](const auto &item) { return item.key() == kExpectedKey; }),
      context->result().end());
}

}  // namespace
}  // namespace zvec::core
