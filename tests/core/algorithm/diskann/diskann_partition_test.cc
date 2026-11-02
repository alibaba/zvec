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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <ailego/math/euclidean_distance_matrix.h>
#include <gtest/gtest.h>
#include "tests/test_util.h"
#include "diskann_builder.h"
#include "diskann_params.h"
#ifndef _WIN32
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#endif

namespace zvec::core {
namespace {
uint32_t Hash(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  return x ^ (x >> 16);
}
void Generate(size_t id, size_t dimension, float *data,
              bool identical = false) {
  for (size_t j = 0; j < dimension; ++j) {
    const float center =
        (int(Hash((id % 16) * dimension + j) % 2001) - 1000) / 100.0f;
    const float noise =
        (int(Hash(id * dimension + j + 1234567) % 2001) - 1000) / 1000.0f;
    data[j] = identical ? 1.0f : center + noise;
  }
}
// Reused iterator scratch and non-contiguous, reverse-ordered keys exercise
// the streaming input contract without retaining the dataset in the test.
class GeneratedHolder : public IndexHolder {
 public:
  GeneratedHolder(size_t count, size_t dim,
                  IndexMeta::DataType type = IndexMeta::DT_FP32,
                  bool identical = false)
      : count_(count), dim_(dim), type_(type), identical_(identical) {}
  class Iterator : public IndexHolder::Iterator {
   public:
    explicit Iterator(GeneratedHolder &holder)
        : holder_(holder), fp32_(holder.dim_), fp16_(holder.dim_) {
      fill();
    }
    const void *data() const override {
      if (holder_.on_read_) holder_.on_read_();
      return holder_.type_ == IndexMeta::DT_FP32
                 ? static_cast<const void *>(fp32_.data())
                 : fp16_.data();
    }
    bool is_valid() const override {
      return id_ < holder_.count_ && status() == 0;
    }
    int status() const override {
      return holder_.fail_ && id_ == 3 ? IndexError_ReadData : 0;
    }
    uint64_t key() const override {
      return holder_.key(id_);
    }
    void next() override {
      ++id_;
      fill();
    }

   private:
    void fill() {
      if (id_ >= holder_.count_) return;
      Generate(id_, holder_.dim_, fp32_.data(), holder_.identical_);
      if (holder_.type_ == IndexMeta::DT_FP16)
        for (size_t j = 0; j < holder_.dim_; ++j)
          fp16_[j] = ailego::Float16(fp32_[j]);
    }
    GeneratedHolder &holder_;
    size_t id_{0};
    std::vector<float> fp32_;
    std::vector<ailego::Float16> fp16_;
  };
  uint64_t key(size_t id) const {
    return (uint64_t{1} << 40) + 7 * (count_ - id);
  }
  size_t count() const override {
    return count_;
  }
  size_t dimension() const override {
    return dim_;
  }
  IndexMeta::DataType data_type() const override {
    return type_;
  }
  size_t element_size() const override {
    return dim_ * (type_ == IndexMeta::DT_FP32 ? 4 : 2);
  }
  bool multipass() const override {
    return true;
  }
  IndexHolder::Iterator::Pointer create_iterator() override {
    return std::make_unique<Iterator>(*this);
  }
  bool fail_{false};
  std::function<void()> on_read_;

 private:
  size_t count_, dim_;
  IndexMeta::DataType type_;
  bool identical_;
};

ailego::Params BuildParams(bool partition = true) {
  ailego::Params params;
  params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 16);
  params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 48);
  params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT, 2);
  params.set(PARAM_DISKANN_BUILDER_MAX_TRAIN_SAMPLE_COUNT, 512);
  params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 4);
  if (partition) params.set(PARAM_DISKANN_BUILDER_MEMORY_BUDGET, 1.0 / 64);
  return params;
}

struct TestPath {
  explicit TestPath(const std::string &suffix)
      : path("DiskAnnPartitionTest/" + suffix) {}
  ~TestPath() {
    zvec::test_util::RemoveTestPath(path);
  }
  std::string path;
};

void Dump(IndexBuilder::Pointer builder, const std::string &path) {
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
}
std::vector<uint8_t> Segment(IndexStorage::Pointer storage,
                             const std::string &name) {
  auto segment = storage->get(name);
  if (!segment) {
    ADD_FAILURE() << "Missing segment " << name;
    return {};
  }
  IndexStorage::MemoryBlock block;
  const size_t bytes = segment->data_size();
  if (segment->read(0, block, bytes) != bytes) {
    ADD_FAILURE() << "Read " << name;
    return {};
  }
  const auto *data = static_cast<const uint8_t *>(block.data());
  return {data, data + bytes};
}
void CheckGraph(const std::string &path, GeneratedHolder &holder) {
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  auto bytes = Segment(storage, DiskAnnEntity::kDiskAnnMetaSegmentId);
  ASSERT_EQ(sizeof(DiskAnnMetaHeader), bytes.size());
  DiskAnnMetaHeader meta;
  memcpy(&meta, bytes.data(), sizeof(meta));
  ASSERT_EQ(holder.count(), meta.doc_cnt);
  ASSERT_LE(meta.max_degree, 16u);
  auto graph = Segment(storage, DiskAnnEntity::kDiskAnnVectorSegmentId);
  auto keys = Segment(storage, DiskAnnEntity::kDiskAnnKeySegmentId);
  auto mapping = Segment(storage, DiskAnnEntity::kDiskAnnKeyMappingSegmentId);
  ASSERT_EQ(holder.count() * sizeof(uint64_t), keys.size());
  ASSERT_EQ(holder.count() * sizeof(uint32_t), mapping.size());
  for (uint32_t id = 0; id < holder.count(); ++id) {
    uint64_t key = 0;
    uint32_t mapped = 0;
    memcpy(&key, keys.data() + id * sizeof(key), sizeof(key));
    memcpy(&mapped, mapping.data() + id * sizeof(mapped), sizeof(mapped));
    ASSERT_EQ(holder.key(id), key);
    ASSERT_EQ(holder.count() - 1 - id, mapped);
    size_t offset = meta.node_per_sector
                        ? (id / meta.node_per_sector) * 4096 +
                              (id % meta.node_per_sector) * meta.max_node_size
                        : id * ((meta.max_node_size + 4095) / 4096) * 4096;
    ASSERT_LE(offset + meta.max_node_size, graph.size());
    uint32_t degree = 0;
    memcpy(&degree, graph.data() + offset + holder.element_size(),
           sizeof(degree));
    ASSERT_GT(degree, 0u);
    ASSERT_LE(degree, meta.max_degree);
    std::vector<uint32_t> edges(degree);
    memcpy(edges.data(),
           graph.data() + offset + holder.element_size() + sizeof(degree),
           degree * sizeof(uint32_t));
    EXPECT_NE(edges.end(),
              std::find(edges.begin(), edges.end(), (id + 1) % holder.count()));
    for (auto edge : edges) ASSERT_LT(edge, holder.count());
    std::sort(edges.begin(), edges.end());
    EXPECT_EQ(edges.end(), std::unique(edges.begin(), edges.end()));
  }
}

void RoundTrip(size_t count, size_t dim, IndexMeta::DataType type,
               bool identical = false) {
  auto holder = std::make_shared<GeneratedHolder>(count, dim, type, identical);
  IndexMeta meta(type, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_EQ(0, builder->init(meta, BuildParams()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  TestPath first("first"), second("second");
  Dump(builder, first.path);
  CheckGraph(first.path, *holder);
  std::weak_ptr<GeneratedHolder> weak = holder;
  holder.reset();
  EXPECT_TRUE(weak.expired());
  Dump(builder, second.path);
  auto a = IndexFactory::CreateStorage("FileReadStorage"),
       b = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, a->open(first.path, false));
  ASSERT_EQ(0, b->open(second.path, false));
  for (const auto &segment : {DiskAnnEntity::kDiskAnnVectorSegmentId,
                              DiskAnnEntity::kDiskAnnPqDataSegmentId,
                              DiskAnnEntity::kDiskAnnKeyMappingSegmentId})
    EXPECT_EQ(Segment(a, segment), Segment(b, segment));
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 128);
  ASSERT_EQ(0, searcher->init(params));
  ASSERT_EQ(0, searcher->load(b, nullptr));
  auto context = searcher->create_context();
  context->set_topk(1);
  for (size_t id : {size_t{0}, count / 2, count - 1}) {
    std::vector<float> fp32(dim);
    std::vector<ailego::Float16> fp16(dim);
    Generate(id, dim, fp32.data(), identical);
    const void *query = fp32.data();
    if (type == IndexMeta::DT_FP16) {
      for (size_t j = 0; j < dim; ++j) fp16[j] = ailego::Float16(fp32[j]);
      query = fp16.data();
    }
    ASSERT_EQ(0,
              searcher->search_impl(query, IndexQueryMeta(type, dim), context));
    ASSERT_EQ(1u, context->result().size());
    EXPECT_NEAR(0.0f, context->result()[0].score(), 0.001f);
  }
}

TEST(DiskAnnPartition, StreamingFp32AndRepeatDump) {
  RoundTrip(513, 32, IndexMeta::DT_FP32);
}
TEST(DiskAnnPartition, StreamingFp16) {
  RoundTrip(513, 32, IndexMeta::DT_FP16);
}
TEST(DiskAnnPartition, MultiSectorNodes) {
  RoundTrip(65, 1056, IndexMeta::DT_FP32);
}
TEST(DiskAnnPartition, Singleton) {
  RoundTrip(1, 8, IndexMeta::DT_FP32);
}
TEST(DiskAnnPartition, SkewedOversizedPartitionsAndExternalKeySort) {
  RoundTrip(8193, 128, IndexMeta::DT_FP32, true);
}

TEST(DiskAnnPartition, InvalidBudgets) {
  IndexMeta meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  for (double budget :
       {-1.0, 0.0, 0.001, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
    auto params = BuildParams();
    params.set(PARAM_DISKANN_BUILDER_MEMORY_BUDGET, budget);
    EXPECT_EQ(IndexError_InvalidArgument, builder->init(meta, params));
  }
}

TEST(DiskAnnPartition, ExplicitBuildMode) {
  IndexMeta meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  // A retained budget can be overridden without changing the input or graph
  // parameters. Partitioned builds release the input; legacy dumps retain it.
  for (bool enabled : {false, true}) {
    auto holder = std::make_shared<GeneratedHolder>(65, 16);
    std::weak_ptr<GeneratedHolder> weak = holder;
    auto params = BuildParams();
    params.set(PARAM_DISKANN_BUILDER_PARTITIONED_BUILD, enabled);
    auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
    ASSERT_EQ(0, builder->init(meta, params));
    ASSERT_EQ(0, builder->train(holder));
    ASSERT_EQ(0, builder->build(holder));
    holder.reset();
    EXPECT_EQ(enabled, weak.expired());
    TestPath path(enabled ? "explicit_partition" : "explicit_legacy");
    Dump(builder, path.path);
    ASSERT_EQ(0, builder->cleanup());
    EXPECT_TRUE(weak.expired());
  }
}

TEST(DiskAnnPartition, ExplicitBuildModeValidation) {
  IndexMeta meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  auto params = BuildParams(false);
  params.set(PARAM_DISKANN_BUILDER_PARTITIONED_BUILD, true);
  EXPECT_EQ(IndexError_InvalidArgument, builder->init(meta, params));
  params.set(PARAM_DISKANN_BUILDER_PARTITIONED_BUILD, std::string("invalid"));
  EXPECT_EQ(IndexError_InvalidArgument, builder->init(meta, params));
  params.set(PARAM_DISKANN_BUILDER_PARTITIONED_BUILD, false);
  EXPECT_EQ(0, builder->init(meta, params));
}

#ifndef _WIN32
// POSIX scratch files are already unlinked, so directory enumeration alone
// cannot prove where they were created or whether their descriptors leaked.
size_t ScratchDescriptors(const std::string &directory) {
  size_t count = 0;
  struct rlimit limit;
  if (getrlimit(RLIMIT_NOFILE, &limit) != 0) return 0;
  for (int fd = 0;
       fd < static_cast<int>(std::min<rlim_t>(limit.rlim_cur, 4096)); ++fd) {
    if (fcntl(fd, F_GETFD) < 0) continue;
    std::string path;
    if (ailego::FileHelper::GetFilePath(fd, &path) &&
        path.find(directory + "/zvec-diskann-") == 0) {
      EXPECT_NE(0, fcntl(fd, F_GETFD) & FD_CLOEXEC);
      ++count;
    }
  }
  return count;
}
#endif

TEST(DiskAnnPartition, TempDirectoryAndCleanup) {
  TestPath path("scratch space");
  const auto native_path = ailego::FileHelper::PathFromUtf8(path.path);
  std::filesystem::create_directories(native_path);
  const auto directory = std::filesystem::canonical(native_path).u8string();
  auto params = BuildParams();
  params.set(PARAM_DISKANN_BUILDER_TEMP_DIR, directory);
  auto holder = std::make_shared<GeneratedHolder>(513, 32);
  IndexMeta meta(IndexMeta::DT_FP32, 32);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
#ifndef _WIN32
  bool observed = false;
  holder->on_read_ = [&]() {
    if (!observed) {
      // Five persistent files plus the initial key-sort run.
      EXPECT_GE(ScratchDescriptors(directory), 6u);
      observed = true;
    }
  };
#endif
  holder->fail_ = true;
  EXPECT_EQ(IndexError_ReadData, builder->build(holder));
  EXPECT_TRUE(std::filesystem::is_empty(native_path));
#ifndef _WIN32
  EXPECT_TRUE(observed);
  EXPECT_EQ(0u, ScratchDescriptors(directory));
#endif
  holder->fail_ = false;
  ASSERT_EQ(0, builder->build(holder));
#ifndef _WIN32
  EXPECT_EQ(5u, ScratchDescriptors(directory));
#endif
  TestPath first("temp_dir_dump"), second("temp_dir_repeat");
  Dump(builder, first.path);
  Dump(builder, second.path);
  CheckGraph(first.path, *holder);
  builder.reset();
  EXPECT_TRUE(std::filesystem::is_empty(native_path));
#ifndef _WIN32
  EXPECT_EQ(0u, ScratchDescriptors(directory));
#endif
}

TEST(DiskAnnPartition, UnusableTempDirectoryDoesNotFallBack) {
  TestPath parent("missing_scratch");
  zvec::test_util::RemoveTestPath(parent.path);
  auto params = BuildParams();
  params.set(PARAM_DISKANN_BUILDER_TEMP_DIR, parent.path);
  auto holder = std::make_shared<GeneratedHolder>(65, 16);
  IndexMeta meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_EQ(0, builder->init(meta, params));
  size_t reads = 0;
  holder->on_read_ = [&]() { ++reads; };
  EXPECT_EQ(IndexError_OpenFile, builder->train(IndexTrainer::Pointer{}));
  EXPECT_EQ(IndexError_OpenFile, builder->train(holder));
  EXPECT_EQ(0u, reads);
  EXPECT_FALSE(std::filesystem::exists(parent.path));
  std::filesystem::create_directories(parent.path);
#ifndef _WIN32
  if (geteuid() != 0) {
    std::filesystem::permissions(parent.path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_exec);
    const int ret = builder->train(holder);
    std::filesystem::permissions(parent.path,
                                 std::filesystem::perms::owner_all);
    EXPECT_EQ(IndexError_OpenFile, ret);
    EXPECT_EQ(0u, reads);
    EXPECT_TRUE(std::filesystem::is_empty(parent.path));
  }
#endif
  ASSERT_EQ(0, builder->train(holder));
  EXPECT_GT(reads, 0u);
  EXPECT_TRUE(std::filesystem::is_empty(parent.path));
#ifndef _WIN32
  EXPECT_EQ(0u, ScratchDescriptors(
                    std::filesystem::canonical(parent.path).u8string()));
#endif
  // A successful preflight cannot replace checks at build time.
  std::filesystem::remove(parent.path);
  EXPECT_EQ(IndexError_OpenFile, builder->build(holder));
  std::filesystem::create_directories(parent.path);
  ASSERT_EQ(0, builder->build(holder));
  ASSERT_EQ(0, builder->cleanup());
  EXPECT_TRUE(std::filesystem::is_empty(parent.path));

  const auto file_path = parent.path + "/regular_file";
  {
    std::ofstream file(file_path);
    file << "keep";
  }
  params.set(PARAM_DISKANN_BUILDER_TEMP_DIR, file_path);
  ASSERT_EQ(0, builder->init(meta, params));
  reads = 0;
  EXPECT_EQ(IndexError_OpenFile, builder->train(holder));
  EXPECT_EQ(0u, reads);
  EXPECT_EQ(4u, std::filesystem::file_size(file_path));
  ASSERT_EQ(0, builder->cleanup());

  // A disabled partition path must not require access to its saved directory.
  params.set(PARAM_DISKANN_BUILDER_PARTITIONED_BUILD, false);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  EXPECT_EQ(0, builder->build(holder));
}

TEST(DiskAnnPartition, InvalidTempDirectoryParameter) {
  IndexMeta meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  auto params = BuildParams();
  params.set(PARAM_DISKANN_BUILDER_TEMP_DIR, 42);
  EXPECT_EQ(IndexError_InvalidArgument, builder->init(meta, params));
  params.set(PARAM_DISKANN_BUILDER_TEMP_DIR, std::string("path\0suffix", 11));
  EXPECT_EQ(IndexError_InvalidArgument, builder->init(meta, params));
}

#if !defined(_WIN32) && GTEST_HAS_DEATH_TEST
TEST(DiskAnnPartition, ScratchWriteFailureCleansUpAndCanRetry) {
  // Limit only an exec'ed child: exercise a real filesystem write/flush
  // failure without filling the machine's disk or altering the test runner.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  TestPath path("write_failure_scratch");
  std::filesystem::create_directories(path.path);
  const auto directory = std::filesystem::canonical(path.path).u8string();
  for (bool before_training : {false, true}) {
    ASSERT_EXIT(
        {
          auto params = BuildParams();
          params.set(PARAM_DISKANN_BUILDER_TEMP_DIR, directory);
          auto holder = std::make_shared<GeneratedHolder>(65, 32);
          IndexMeta meta(IndexMeta::DT_FP32, 32);
          meta.set_metric("SquaredEuclidean", 0, ailego::Params());
          auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
          if (builder->init(meta, params)) _exit(1);
          size_t reads = 0;
          holder->on_read_ = [&]() { ++reads; };
          if (!before_training && builder->train(holder)) _exit(1);
          struct rlimit previous;
          if (getrlimit(RLIMIT_FSIZE, &previous)) _exit(2);
          auto limited = previous;
          limited.rlim_cur = 1024;
          signal(SIGXFSZ, SIG_IGN);
          if (setrlimit(RLIMIT_FSIZE, &limited)) _exit(3);
          const int ret =
              before_training ? builder->train(holder) : builder->build(holder);
          if (setrlimit(RLIMIT_FSIZE, &previous)) _exit(4);
          if (ret != IndexError_WriteData || ScratchDescriptors(directory))
            _exit(5);
          if (before_training) {
            if (reads != 0 || builder->build(holder) != IndexError_NoReady)
              _exit(8);
            if (builder->train(holder)) _exit(9);
          }
          if (builder->build(holder) || builder->cleanup()) _exit(6);
          _exit(ScratchDescriptors(directory) == 0 ? 0 : 7);
        },
        ::testing::ExitedWithCode(0),
        "DiskAnn temporary file (write|flush) failed");
  }
  EXPECT_TRUE(std::filesystem::is_empty(path.path));
}
#endif

TEST(DiskAnnPartition, RejectsOversizedWorkerScratchWithoutOverflow) {
  auto holder = std::make_shared<GeneratedHolder>(8, 16);
  IndexMeta meta(IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto params = BuildParams();
  params.set(PARAM_DISKANN_BUILDER_LIST_SIZE,
             std::numeric_limits<uint32_t>::max());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  EXPECT_EQ(IndexError_NoMemory, builder->build(holder));
}

class FailDumper : public IndexDumper {
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
  int append(const std::string &, size_t, size_t, uint32_t) override {
    return 0;
  }
  size_t write(const void *, size_t bytes) override {
    size_ += bytes;
    return size_ > 8192 ? 0 : bytes;
  }
  uint32_t magic() const override {
    return 0;
  }
  size_t size() const override {
    return size_;
  }

 private:
  size_t size_{0};
};
TEST(DiskAnnPartition, InputFailureAndDumpFailureCanBeRetried) {
  auto holder = std::make_shared<GeneratedHolder>(65, 32);
  IndexMeta meta(IndexMeta::DT_FP32, 32);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_EQ(0, builder->init(meta, BuildParams()));
  ASSERT_EQ(0, builder->train(holder));
  holder->fail_ = true;
  EXPECT_EQ(IndexError_ReadData, builder->build(holder));
  holder->fail_ = false;
  ASSERT_EQ(0, builder->build(holder));
  EXPECT_EQ(IndexError_WriteData,
            builder->dump(std::make_shared<FailDumper>()));
  TestPath path("retry");
  Dump(builder, path.path);
  CheckGraph(path.path, *holder);
  ASSERT_EQ(0, builder->cleanup());
  ASSERT_EQ(0, builder->init(meta, BuildParams(false)));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
}

// Explicitly opt in; run each mode in its own process for comparable peak RSS.
// ZVEC_PARTITION_BENCH_BUDGET=0 selects the original builder; otherwise GiB.
// ZVEC_PARTITION_BENCH_COUNT defaults to 100000.
TEST(DiskAnnPartition, DISABLED_MemoryRecallBenchmark) {
  const char *budget_env = std::getenv("ZVEC_PARTITION_BENCH_BUDGET");
  const char *count_env = std::getenv("ZVEC_PARTITION_BENCH_COUNT");
  const double budget = budget_env ? std::atof(budget_env) : 1.0 / 32;
  const size_t count =
      count_env ? std::strtoull(count_env, nullptr, 10) : 100000;
  constexpr size_t dim = 128, queries = 100, topk = 10;
  auto holder = std::make_shared<GeneratedHolder>(count, dim);
  IndexMeta meta(IndexMeta::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto params = BuildParams(false);
  params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 32);
  params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 64);
  params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT, 4);
  params.set(PARAM_DISKANN_BUILDER_MAX_TRAIN_SAMPLE_COUNT, 2048);
  params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 32);
  if (budget) params.set(PARAM_DISKANN_BUILDER_MEMORY_BUDGET, budget);
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_EQ(0, builder->init(meta, params));
  auto start = std::chrono::steady_clock::now();
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  TestPath path(budget ? "partition_benchmark" : "baseline_benchmark");
  Dump(builder, path.path);
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  double rss_mib = 0;
#ifndef _WIN32
  struct rusage usage {};
  ASSERT_EQ(0, getrusage(RUSAGE_SELF, &usage));
#ifdef __APPLE__
  rss_mib = usage.ru_maxrss / (1024.0 * 1024);
#else
  rss_mib = usage.ru_maxrss / 1024.0;
#endif
#endif
  builder.reset();
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path.path, false));
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ailego::Params search_params;
  search_params.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 128);
  ASSERT_EQ(0, searcher->init(search_params));
  ASSERT_EQ(0, searcher->load(storage, nullptr));
  auto context = searcher->create_context();
  context->set_topk(topk);
  std::vector<std::vector<float>> query_data(queries, std::vector<float>(dim));
  std::vector<std::vector<std::pair<float, uint64_t>>> truth(queries);
  for (size_t q = 0; q < queries; ++q)
    Generate(count + q + 999, dim, query_data[q].data());
  std::vector<float> vector(dim);
  for (size_t id = 0; id < count; ++id) {
    Generate(id, dim, vector.data());
    for (size_t q = 0; q < queries; ++q) {
      float dist = 0;
      ailego::SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
          vector.data(), query_data[q].data(), dim, &dist);
      auto &best = truth[q];
      if (best.size() < topk || dist < best.back().first) {
        best.emplace_back(dist, holder->key(id));
        std::sort(best.begin(), best.end());
        if (best.size() > topk) best.pop_back();
      }
    }
  }
  size_t hits = 0;
  start = std::chrono::steady_clock::now();
  for (size_t q = 0; q < queries; ++q) {
    ASSERT_EQ(0, searcher->search_impl(query_data[q].data(),
                                       IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                       context));
    for (const auto &item : context->result())
      for (const auto &expected : truth[q])
        if (item.key() == expected.second) ++hits;
  }
  double query_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count() /
                    queries;
  std::printf(
      "BENCHMARK count=%zu dim=%zu threads=4 budget_gib=%.6f build_dump_s=%.3f "
      "peak_rss_mib=%.3f recall_at_10=%.4f query_ms=%.3f\n",
      count, dim, budget, seconds, rss_mib, double(hits) / (queries * topk),
      query_ms);
}
}  // namespace
}  // namespace zvec::core
