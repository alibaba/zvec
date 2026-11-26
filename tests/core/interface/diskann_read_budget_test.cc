// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#if DISKANN_SUPPORTED

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "../../../src/core/algorithm/diskann/diskann_entity.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace {
thread_local size_t refused_aligned_size = 0;
}  // namespace

void *operator new(size_t size, std::align_val_t alignment) {
  if (refused_aligned_size && size == refused_aligned_size)
    throw std::bad_alloc();
  void *memory =
      ailego_aligned_malloc(size ? size : 1, static_cast<size_t>(alignment));
  if (!memory) throw std::bad_alloc();
  return memory;
}

void operator delete(void *memory, std::align_val_t) noexcept {
  ailego_aligned_free(memory);
}

void operator delete(void *memory, size_t,
                     std::align_val_t alignment) noexcept {
  ::operator delete(memory, alignment);
}

namespace zvec::ailego {
struct QueueResidentSampleTestPeer {
  static bool configure_fixed_rss(MemoryLimitPool &pool, size_t limit) {
    return pool.process_rss_.configure(
        limit, +[]() -> size_t { return 32UL << 20; });
  }
  static bool configure_headroom(MemoryLimitPool &pool, size_t headroom) {
    constexpr size_t kMeasurementLimit = 128UL << 20;
    return configure_fixed_rss(pool, kMeasurementLimit) &&
           configure_fixed_rss(pool,
                               kMeasurementLimit - available(pool) + headroom);
  }
  static size_t available(MemoryLimitPool &pool) {
    return pool.process_rss_.available();
  }
  static bool pause_background(MemoryLimitPool &pool) {
    const bool running = pool.bg_running_.load();
    pool.stop_background_evictor();
    return running;
  }
  static void resume_background(MemoryLimitPool &pool) {
    pool.start_background_evictor();
  }
};
}  // namespace zvec::ailego

namespace zvec::core_interface {
namespace {
using StorageType = StorageOptions::StorageType;

class DiskAnnReadBudgetTest : public ::testing::TestWithParam<StorageType> {
 protected:
  static constexpr uint32_t kDimension = 64;
  static constexpr uint32_t kDocs = 512;
  static constexpr size_t kPoolBytes = 16UL << 20;
  static constexpr uint64_t kHighRss = uint64_t{1} << 40;

  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    previous_rss_ = pool.process_rss_limit();
    previous_capacity_ = ailego::PoolPayloadBudget::get_instance().capacity();
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(kPoolBytes));
    ASSERT_TRUE(pool.configure_process_rss_limit(0));
    const auto *test = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = test->name();
    std::replace(name.begin(), name.end(), '/', '_');
    directory_ = "diskann_read_budget_" + name;
    std::filesystem::remove_all(directory_);
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }

  void TearDown() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_TRUE(pool.configure_process_rss_limit(previous_rss_));
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
    EXPECT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(previous_capacity_));
    std::filesystem::remove_all(directory_);
  }

  template <typename Value>
  void build_index(const std::string &path, const BaseIndexParam &param,
                   std::vector<Value> *rows, unsigned seed,
                   uint32_t doc_count = kDocs) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    const auto dimension = static_cast<size_t>(param.dimension);
    rows->resize(doc_count * dimension);
    for (auto &value : *rows) value = static_cast<Value>(uniform(random));
    auto index = IndexFactory::CreateAndInitIndex(param);
    ASSERT_NE(nullptr, index);
    ASSERT_EQ(0, index->open(path, {StorageType::kMMAP, true}));
    for (uint32_t id = 0; id < doc_count; ++id) {
      ASSERT_EQ(
          0, index->add(VectorData{DenseVector{rows->data() + id * dimension}},
                        id));
    }
    ASSERT_EQ(0, index->train());
    ASSERT_EQ(0, index->close());
  }

  BaseIndexParam::Pointer parameters(bool fp16, uint32_t dimension = kDimension,
                                     uint32_t pq_chunks = 8) {
    return DiskAnnIndexParamBuilder()
        .with_metric_type(MetricType::kL2sq)
        .with_data_type(fp16 ? DataType::DT_FP16 : DataType::DT_FP32)
        .with_dimension(dimension)
        .with_max_degree(16)
        .with_list_size(48)
        .with_pq_chunk_num(pq_chunks)
        .build();
  }

  std::filesystem::path directory_;
  size_t previous_capacity_{0};
  uint64_t previous_rss_{0};
};

TEST_P(DiskAnnReadBudgetTest, SearchAndFetchWith4095ByteCache) {
  const auto path = (directory_ / "fp16.index").string();
  const auto param = parameters(true);
  std::vector<ailego::Float16> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 317));
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(4095));
  ASSERT_TRUE(ailego::MemoryLimitPool::get_instance()
                  .configure_process_rss_limit(kHighRss));

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 48;
  query->fetch_vector = true;
  const VectorData input{DenseVector{rows.data()}};
  SearchResult result;
  for (bool linear : {false, true}) {
    query->is_linear = linear;
    ASSERT_EQ(0, index->search(input, query, &result));
    ASSERT_EQ(1u, result.doc_list_.size());
    const auto id = result.doc_list_.front().key();
    if (linear) {
      EXPECT_EQ(0u, id);
    }
    ASSERT_LT(id, kDocs);
    const std::string expected(
        reinterpret_cast<const char *>(rows.data() + id * kDimension),
        kDimension * sizeof(ailego::Float16));
    VectorDataBuffer fetched;
    ASSERT_EQ(0, index->fetch(id, &fetched));
    EXPECT_EQ(expected, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
    EXPECT_EQ(expected, result.doc_list_.front().vector_string());
    EXPECT_EQ(0u, pool.stats().page_used);
  }
  query->is_linear = false;
  query->bf_pks = std::make_shared<std::vector<uint64_t>>(1, 7);
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(7u, result.doc_list_.front().key());
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, GraphPqBypassFitsSmallRssBudget) {
  if (GetParam() != StorageType::kBufferPool) GTEST_SKIP();
  constexpr uint32_t kGraphDimension = 32;
  constexpr uint32_t kGraphDocs = 4096;
  constexpr size_t kRssHeadroom = 64UL << 10;
  const auto path = (directory_ / "graph_pq.index").string();
  // Seven bytes per PQ code exercise cuts across page boundaries.
  const auto param = parameters(false, kGraphDimension, 7);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 317, kGraphDocs));

  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 32;
  query->list_size = 32;
  const std::array<float, kGraphDimension> values{};
  const VectorData input{DenseVector{values.data()}};
  SearchResult expected;
  {
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);
    ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
    ASSERT_EQ(0, index->search(input, query, &expected));
    ASSERT_EQ(query->topk, expected.doc_list_.size());
    ASSERT_EQ(0, index->close());
  }

  auto &payload = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, payload.init(4095));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
  auto &rss = ailego::MemoryLimitPool::get_instance();
  using Budget = ailego::QueueResidentSampleTestPeer;
  const bool background_running = Budget::pause_background(rss);
  const auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(rss.configure_process_rss_limit(0));
    if (background_running) Budget::resume_background(rss);
  });
  ASSERT_TRUE(Budget::configure_headroom(rss, kRssHeadroom));
  ASSERT_EQ(kRssHeadroom, Budget::available(rss));

  SearchResult actual;
  ASSERT_EQ(0, index->search(input, query, &actual));
  ASSERT_EQ(expected.doc_list_.size(), actual.doc_list_.size());
  for (size_t i = 0; i < actual.doc_list_.size(); ++i) {
    EXPECT_EQ(expected.doc_list_[i].key(), actual.doc_list_[i].key());
    EXPECT_EQ(expected.doc_list_[i].score(), actual.doc_list_[i].score());
  }
  EXPECT_EQ(0u, payload.stats().page_used);
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, PrimaryKeyFetchVectorFitsSmallRssBudget) {
  if (GetParam() != StorageType::kMMAP) GTEST_SKIP();
  constexpr uint32_t kWideDimension = 2048;
  constexpr uint64_t kKey = 7;
  constexpr size_t kRssHeadroom = 48UL << 10;
  const auto path = (directory_ / "wide_vector.index").string();
  const auto param = parameters(false, kWideDimension);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 317));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
  auto &rss = ailego::MemoryLimitPool::get_instance();
  using Budget = ailego::QueueResidentSampleTestPeer;
  const bool background_running = Budget::pause_background(rss);
  const auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(rss.configure_process_rss_limit(0));
    if (background_running) Budget::resume_background(rss);
  });
  ASSERT_TRUE(Budget::configure_headroom(rss, kRssHeadroom));
  ASSERT_EQ(kRssHeadroom, Budget::available(rss));

  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 1;
  query->fetch_vector = true;
  query->bf_pks = std::make_shared<std::vector<uint64_t>>(1, kKey);
  const std::array<float, kWideDimension> values{};
  const VectorData input{DenseVector{values.data()}};
  const std::string expected(
      reinterpret_cast<const char *>(rows.data() + kKey * kWideDimension),
      kWideDimension * sizeof(float));
  SearchResult result;
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(kKey, result.doc_list_.front().key());
  EXPECT_EQ(expected, result.doc_list_.front().vector_string());

  ASSERT_TRUE(Budget::configure_headroom(rss, 1024));
  EXPECT_EQ(core::IndexError_NoMemory, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(kKey, result.doc_list_.front().key());
  EXPECT_EQ(expected, result.doc_list_.front().vector_string());
  ASSERT_EQ(0, index->close());
  EXPECT_EQ(expected, result.doc_list_.front().vector_string());
}

TEST_P(DiskAnnReadBudgetTest, PrimaryKeySearchUsesCallerOutputBuffers) {
  const auto path = (directory_ / "converted_fp16.index").string();
  const auto param =
      DiskAnnIndexParamBuilder()
          .with_metric_type(MetricType::kL2sq)
          .with_data_type(DataType::DT_FP32)
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .with_dimension(kDimension)
          .with_max_degree(16)
          .with_list_size(48)
          .with_pq_chunk_num(8)
          .build();
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 317));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));

  constexpr uint32_t topk = 4096;
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = topk;
  query->list_size = 48;
  query->bf_pks = std::make_shared<std::vector<uint64_t>>(1, 7);
  std::vector<int64_t> ids(topk, -42);
  std::vector<float> scores(topk, -42.0f);
  // Refuse only an aligned DiskAnnReadVector<int64_t> of length topk.
  // The key lookup and its result should fit without a second ID buffer.
  refused_aligned_size = ids.size() * sizeof(int64_t) +
                         sizeof(ailego::ProcessRssBudget::MaterializationStamp);
  const auto restore =
      ailego::ScopeGuard::Make([] { refused_aligned_size = 0; });
  const VectorData input{DenseVector{rows.data() + 7 * kDimension}};
  ASSERT_EQ(
      0, index->search_internal_ids(input, query, ids.data(), scores.data()));
  EXPECT_EQ(7, ids.front());
  EXPECT_GE(scores.front(), 0.0f);
  EXPECT_LT(scores.front(), 1e-4f);
  EXPECT_EQ(-1, ids.back());
  EXPECT_TRUE(std::isnan(scores.back()));
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, PrimaryKeySearchGrowsHeapWithMatches) {
  const auto path = (directory_ / "large_topk.index").string();
  const auto param = parameters(false);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 317));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));

  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 4096;
  query->list_size = 48;
  query->bf_pks = std::make_shared<std::vector<uint64_t>>(1, 7);
  const VectorData input{DenseVector{rows.data() + 7 * kDimension}};
  SearchResult result;
  using HeapEntry = std::pair<core::diskann_id_t, core::VectorInfo>;
  refused_aligned_size = query->topk * sizeof(HeapEntry) +
                         sizeof(ailego::ProcessRssBudget::MaterializationStamp);
  const auto restore =
      ailego::ScopeGuard::Make([] { refused_aligned_size = 0; });
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(7u, result.doc_list_.front().key());

  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(7, &fetched));
  query->topk = 1;
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(7u, result.doc_list_.front().key());
  ASSERT_EQ(0, index->close());
}

class StartupFaultLogger final : public ailego::Logger {
 public:
  StartupFaultLogger(std::string_view format_prefix,
                     std::function<void()> fault)
      : format_prefix_(format_prefix), fault_(std::move(fault)) {}

  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }

  void log(int, const char *, int, const char *format, va_list) override {
    if (std::this_thread::get_id() == caller_ &&
        std::string_view(format).substr(0, format_prefix_.size()) ==
            format_prefix_) {
      fault_();
    }
  }

 private:
  const std::thread::id caller_{std::this_thread::get_id()};
  const std::string_view format_prefix_;
  const std::function<void()> fault_;
};

int WithStartupFault(std::string_view format_prefix,
                     const std::function<void()> &fault,
                     const std::function<int()> &run) {
  auto previous = ailego::LoggerBroker::Register(
      std::make_shared<StartupFaultLogger>(format_prefix, fault));
  int previous_level = ailego::Logger::LEVEL_DEBUG;
  while (previous_level < ailego::Logger::LEVEL_FATAL &&
         !ailego::LoggerBroker::IsLevelEnabled(previous_level)) {
    ++previous_level;
  }
  const auto restore = ailego::ScopeGuard::Make([&] {
    ailego::LoggerBroker::SetLevel(previous_level);
    ailego::LoggerBroker::Register(std::move(previous));
  });
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);
  return run();
}

constexpr std::string_view kMedoidReadLog =
    "Loading centroid data from medoid vector data";

TEST_P(DiskAnnReadBudgetTest, MedoidReadPreservesNoMemoryAndCanRetry) {
  if (GetParam() != StorageType::kBufferPool) GTEST_SKIP();
  const auto path = (directory_ / "startup.index").string();
  const auto param = parameters(false);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 77));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  auto &rss = ailego::MemoryLimitPool::get_instance();
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  const auto options = StorageOptions{GetParam(), false, true};
  bool tightened = false;
  const int status = WithStartupFault(
      kMedoidReadLog, [&] { tightened = rss.configure_process_rss_limit(1); },
      [&] { return index->open(path, options); });
  EXPECT_TRUE(tightened);
  EXPECT_EQ(core::IndexError_NoMemory, status);
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  ASSERT_EQ(0, index->open(path, options));
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, FailedSearcherLoadReleasesOwnedReadMemory) {
  if (GetParam() != StorageType::kMMAP) GTEST_SKIP();
  const auto path = (directory_ / "searcher_startup.index").string();
  const auto param = parameters(false);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 77));

  auto searcher = core::IndexFactory::CreateSearcher("DiskAnnSearcher");
  auto storage = core::IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(nullptr, searcher);
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, searcher->init(ailego::Params{}));
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  ASSERT_EQ(0, storage->open(path, false));

  auto &rss = ailego::MemoryLimitPool::get_instance();
  using Budget = ailego::QueueResidentSampleTestPeer;
  const bool background_running = Budget::pause_background(rss);
  const auto restore_budget = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(rss.configure_process_rss_limit(0));
    if (background_running) Budget::resume_background(rss);
  });
  // Give the index enough space for its PQ codes and key tables, but not
  // enough to admit the quantizer metadata.
  constexpr size_t kMeasurementLimit = 128UL << 20;
  constexpr size_t kReadAllowance = 64UL << 10;
  ASSERT_TRUE(Budget::configure_fixed_rss(rss, kMeasurementLimit));
  const size_t limit =
      kMeasurementLimit - Budget::available(rss) + kReadAllowance;
  ASSERT_TRUE(Budget::configure_fixed_rss(rss, limit));
  ASSERT_EQ(kReadAllowance, Budget::available(rss));
  bool reached_quantizer = false;
  EXPECT_EQ(core::IndexError_NoMemory,
            WithStartupFault(
                "Read quantizer meta buffer failed",
                [&] { reached_quantizer = true; },
                [&] { return searcher->load(storage, nullptr); }));
  ASSERT_TRUE(reached_quantizer);
  // Refresh the fixed sample so only live backing counts against the limit.
  ASSERT_TRUE(Budget::configure_fixed_rss(rss, limit));
  EXPECT_EQ(kReadAllowance, Budget::available(rss));

  ASSERT_TRUE(Budget::configure_fixed_rss(rss, kMeasurementLimit));
  ASSERT_EQ(0, searcher->load(storage, nullptr));
  EXPECT_EQ(kDocs, searcher->stats().loaded_count());
  ASSERT_EQ(0, searcher->unload());
}

TEST_P(DiskAnnReadBudgetTest, OptionalWarmupLogFailureDoesNotFailOpen) {
  if (GetParam() != StorageType::kBufferPool) GTEST_SKIP();
  const auto path = (directory_ / "warmup.index").string();
  const auto param = parameters(false);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 77));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  size_t attempts = 0;
  const auto options = StorageOptions{GetParam(), false, true};
  const int status = WithStartupFault(
      "DiskANN PQ preparation: status=",
      [&] {
        ++attempts;
        throw std::bad_alloc();
      },
      [&] { return index->open(path, options); });
  EXPECT_EQ(1u, attempts);
  ASSERT_EQ(0, status);
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 48;
  SearchResult result;
  ASSERT_EQ(
      0, index->search(VectorData{DenseVector{rows.data()}}, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(result.doc_list_.front().key(), &fetched));
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, InternalIdsPreserveNoMemoryFromAutoTrain) {
  if (GetParam() != StorageType::kBufferPool) GTEST_SKIP();
  const auto fine_param = FlatIndexParamBuilder()
                              .with_metric_type(MetricType::kL2sq)
                              .with_data_type(DataType::DT_FP32)
                              .with_dimension(kDimension)
                              .with_use_contiguous_memory(true)
                              .build();
  const auto fine_path = (directory_ / "flat.index").string();
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(fine_path, *fine_param, &rows, 77));
  auto fine = IndexFactory::CreateAndInitIndex(*fine_param);
  ASSERT_NE(nullptr, fine);
  ASSERT_EQ(0, fine->open(fine_path, {StorageType::kMMAP, false, true}));

  const auto param = parameters(false);
  auto &rss = ailego::MemoryLimitPool::get_instance();
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  for (bool refine : {false, true}) {
    SCOPED_TRACE(refine);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);
    const auto path =
        (directory_ / (refine ? "refined.index" : "coarse.index")).string();
    ASSERT_EQ(0, index->open(path, {GetParam(), true}));
    for (uint32_t id = 0; id < kDocs; ++id) {
      ASSERT_EQ(
          0, index->add(VectorData{DenseVector{rows.data() + id * kDimension}},
                        id));
    }

    auto query = std::make_shared<DiskAnnQueryParam>();
    query->topk = 1;
    query->list_size = 48;
    if (refine) {
      query->refiner_param = std::make_shared<RefinerParam>();
      query->refiner_param->scale_factor_ = 2;
      query->refiner_param->reference_index = fine;
    }
    const VectorData input{DenseVector{rows.data()}};
    int64_t id = -42;
    float score = -42.0f;
    bool tightened = false;
    const auto search = [&] {
      return index->search_internal_ids(input, query, &id, &score);
    };
    const int status = WithStartupFault(
        kMedoidReadLog, [&] { tightened = rss.configure_process_rss_limit(1); },
        search);
    EXPECT_TRUE(tightened);
    EXPECT_EQ(core::IndexError_NoMemory, status);
    EXPECT_EQ(-42, id);
    EXPECT_EQ(-42.0f, score);
    ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
    ASSERT_EQ(0, search());
    EXPECT_GE(id, 0);
    EXPECT_LT(id, kDocs);
    EXPECT_TRUE(std::isfinite(score));
    ASSERT_EQ(0, index->close());
  }
  ASSERT_EQ(0, fine->close());
}

TEST_P(DiskAnnReadBudgetTest, LargeVectorSearchReleasesIdleCandidateSlots) {
  const auto path = (directory_ / "candidate_slots.index").string();
  const auto param = parameters(false);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 77));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
  auto &rss = ailego::MemoryLimitPool::get_instance();
  using Budget = ailego::QueueResidentSampleTestPeer;
  // Keep real background RSS samples out of the fixed-footprint ledger.
  const bool background_running = Budget::pause_background(rss);
  const auto restore_budget = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(rss.configure_process_rss_limit(0));
    if (background_running) Budget::resume_background(rss);
  });
  constexpr size_t kLargeLimit = 64UL << 20;
  constexpr size_t kFetchAllowance = 16UL << 10;
  ASSERT_TRUE(Budget::configure_fixed_rss(rss, kLargeLimit));

  const VectorData input{DenseVector{rows.data()}};
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = kDocs;
  query->list_size = 48;
  query->is_linear = true;
  {
    SearchResult result;
    ASSERT_EQ(0, index->search(input, query, &result));
    ASSERT_EQ(kDocs, result.doc_list_.size());
  }
  // Reobserve the fixed footprint to compare only live budget charges.
  ASSERT_TRUE(Budget::configure_fixed_rss(rss, kLargeLimit));
  const size_t used_without_vectors = kLargeLimit - Budget::available(rss);
  query->fetch_vector = true;
  {
    SearchResult result;
    ASSERT_EQ(0, index->search(input, query, &result));
    ASSERT_EQ(kDocs, result.doc_list_.size());
    EXPECT_EQ(kDimension * sizeof(float),
              result.doc_list_.front().vector_string().size());
  }
  ASSERT_TRUE(Budget::configure_fixed_rss(
      rss, used_without_vectors + kFetchAllowance));
  VectorDataBuffer fetched;
  EXPECT_EQ(0, index->fetch(7, &fetched));
  EXPECT_EQ(0, index->fetch(8, &fetched));
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, GraphEvaluatesFilterOncePerExpandedNode) {
  const auto path = (directory_ / "filtered.index").string();
  const auto param = parameters(false);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 317));
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(4095));
  ASSERT_TRUE(
      ailego::MemoryLimitPool::get_instance().configure_process_rss_limit(
          kHighRss));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 8;
  query->list_size = 48;
  query->filter = std::make_shared<IndexFilter>();
  std::vector<unsigned> visits(kDocs);
  query->filter->set([&](uint64_t id) {
    EXPECT_LT(id, kDocs);
    if (id >= kDocs) return true;
    ++visits[id];
    return (id % 2) == 0;
  });

  SearchResult result;
  ASSERT_EQ(
      0, index->search(VectorData{DenseVector{rows.data()}}, query, &result));
  ASSERT_FALSE(result.doc_list_.empty());
  for (const auto &doc : result.doc_list_) {
    EXPECT_EQ(1u, doc.key() % 2);
  }
  EXPECT_EQ(1u, *std::max_element(visits.begin(), visits.end()));

  // A one-entry frontier also keeps visit order independent of I/O completion.
  query->topk = query->list_size = 1;
  std::vector<uint64_t> expanded, expanded_with_filter;
  query->filter->set([&](uint64_t id) {
    expanded.push_back(id);
    return false;
  });
  const VectorData input{DenseVector{rows.data()}};
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  query->filter->set([&](uint64_t id) {
    expanded_with_filter.push_back(id);
    return true;
  });
  ASSERT_EQ(0, index->search(input, query, &result));
  EXPECT_TRUE(result.doc_list_.empty());
  EXPECT_EQ(expanded, expanded_with_filter);
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, SameThreadSwitchRssRefusalAndRetry) {
  const auto param = parameters(false);
  const auto first_path = (directory_ / "first.index").string();
  const auto second_path = (directory_ / "second.index").string();
  std::vector<float> first_rows, second_rows;
  ASSERT_NO_FATAL_FAILURE(build_index(first_path, *param, &first_rows, 17));
  ASSERT_NO_FATAL_FAILURE(build_index(second_path, *param, &second_rows, 27));
  auto &rss = ailego::MemoryLimitPool::get_instance();
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));

  auto first = IndexFactory::CreateAndInitIndex(*param);
  auto second = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  ASSERT_EQ(0, first->open(first_path, {GetParam(), false, true}));
  ASSERT_EQ(0, second->open(second_path, {GetParam(), false, true}));

  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 2;
  query->list_size = 48;
  query->fetch_vector = true;
  const VectorData first_input{DenseVector{first_rows.data() + kDimension}};
  const VectorData second_input{DenseVector{second_rows.data() + 2 * kDimension}};

  SearchResult result;
  ASSERT_EQ(0, second->search(second_input, query, &result));
  ASSERT_EQ(2u, result.doc_list_.size());
  const auto second_key = result.doc_list_.front().key();
  ASSERT_EQ(0, first->search(first_input, query, &result));
  ASSERT_EQ(2u, result.doc_list_.size());
  const auto first_key = result.doc_list_.front().key();
  auto saved = result.doc_list_.front();
  core::IndexDocument assigned;
  assigned = saved;
  const auto *vector_data = saved.vector_string().data();
  EXPECT_EQ(vector_data, result.doc_list_.front().vector_string().data());
  const auto saved_bytes = saved.vector_string();
  ASSERT_TRUE(rss.configure_process_rss_limit(1));
  ASSERT_GT(rss.process_rss(), rss.process_rss_limit());

  EXPECT_EQ(core::IndexError_NoMemory,
            second->search(second_input, query, &result));
  ASSERT_EQ(2u, result.doc_list_.size());
  EXPECT_EQ(first_key, result.doc_list_.front().key());
  std::array<int64_t, 2> ids{-42, -42};
  std::array<float, 2> scores{-42.0f, -42.0f};
  EXPECT_EQ(core::IndexError_NoMemory,
            second->search_internal_ids(second_input, query, ids.data(), scores.data()));
  EXPECT_EQ(-42, ids.front());
  EXPECT_EQ(-42.0f, scores.front());
  VectorDataBuffer fetched;
  EXPECT_EQ(core::IndexError_NoMemory, second->fetch(2, &fetched));

  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  ASSERT_EQ(0, second->search(second_input, query, &result));
  ASSERT_EQ(2u, result.doc_list_.size());
  EXPECT_EQ(second_key, result.doc_list_.front().key());
  EXPECT_EQ(vector_data, saved.vector_string().data());
  EXPECT_EQ(saved_bytes, saved.vector_string());
  EXPECT_EQ(vector_data, assigned.vector_string().data());
  assigned = core::IndexDocument{7, 0.0f, 0, std::string("standalone")};
  EXPECT_EQ("standalone", assigned.vector_string());
  EXPECT_EQ(vector_data, saved.vector_string().data());
  ASSERT_EQ(0, second->search_internal_ids(second_input, query, ids.data(),
                                            scores.data()));
  EXPECT_EQ(static_cast<int64_t>(second_key), ids.front());
  ASSERT_EQ(0, second->fetch(2, &fetched));
  const auto previous = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
  ASSERT_FALSE(previous.empty());

  ASSERT_TRUE(rss.configure_process_rss_limit(1));
  EXPECT_EQ(core::IndexError_NoMemory, second->fetch(1, &fetched));
  EXPECT_EQ(previous, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  ASSERT_EQ(0, second->fetch(1, &fetched));
  EXPECT_NE(previous, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  ASSERT_EQ(0, second->close());
  ASSERT_TRUE(rss.configure_process_rss_limit(1));
  EXPECT_EQ(core::IndexError_NoMemory,
            second->open(second_path, {GetParam(), false, true}));
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  ASSERT_EQ(0, second->open(second_path, {GetParam(), false, true}));
  ASSERT_EQ(0, second->search(second_input, query, &result));
  ASSERT_EQ(2u, result.doc_list_.size());
  EXPECT_EQ(second_key, result.doc_list_.front().key());
  ASSERT_EQ(0, second->close());
  ASSERT_EQ(0, first->close());
}

TEST_P(DiskAnnReadBudgetTest, ProviderReportsKeyReadFailure) {
  const auto path = (directory_ / "provider.index").string();
  const auto param = parameters(false);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_index(path, *param, &rows, 67));
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
  auto provider = index->create_index_provider();
  ASSERT_NE(nullptr, provider);
  ASSERT_EQ(kDocs, provider->count());
  {
    auto iter = provider->create_iterator();
    ASSERT_NE(nullptr, iter);
    ASSERT_TRUE(iter->is_valid());
    ASSERT_EQ(0u, iter->key());
  }
  for (int i = 0; i < 64; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64, true);
  }
  auto &rss = ailego::MemoryLimitPool::get_instance();
  ASSERT_TRUE(rss.configure_process_rss_limit(1));
  ASSERT_GT(rss.process_rss(), rss.process_rss_limit());
  auto iter = provider->create_iterator();
  ASSERT_NE(nullptr, iter);
  if (GetParam() == StorageType::kBufferPool) {
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(core::IndexError_NoMemory, iter->status());
    iter->next();
    EXPECT_EQ(core::IndexError_NoMemory, iter->status());
  } else {
    ASSERT_TRUE(iter->is_valid());
    EXPECT_EQ(0u, iter->key());
  }
  ASSERT_TRUE(rss.configure_process_rss_limit(0));
  iter = provider->create_iterator();
  for (uint32_t key = 0; key < kDocs; ++key) {
    ASSERT_TRUE(iter->is_valid());
    EXPECT_EQ(key, iter->key());
    iter->next();
  }
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(0, iter->status());
  iter.reset();
  provider.reset();
  ASSERT_EQ(0, index->close());
}
TEST_P(DiskAnnReadBudgetTest, NestedInternalIdsStayIsolated) {
  const auto param = parameters(false);
  const auto first_path = (directory_ / "outer.index").string();
  const auto second_path = (directory_ / "inner.index").string();
  std::vector<float> outer, inner;
  ASSERT_NO_FATAL_FAILURE(build_index(first_path, *param, &outer, 27));
  ASSERT_NO_FATAL_FAILURE(build_index(second_path, *param, &inner, 31));
  auto first = IndexFactory::CreateAndInitIndex(*param);
  auto second = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(0, first->open(first_path, {GetParam(), false, true}));
  ASSERT_EQ(0, second->open(second_path, {GetParam(), false, true}));
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 256;
  const VectorData first_input{DenseVector{outer.data() + 5 * kDimension}};
  const VectorData second_input{DenseVector{inner.data() + 40 * kDimension}};
  int64_t expected = -1, expected_inner = -1;
  float expected_score = -1, expected_inner_score = -1;
  ASSERT_EQ(0, first->search_internal_ids(first_input, query, &expected, &expected_score));
  ASSERT_EQ(0, second->search_internal_ids(second_input, query, &expected_inner, &expected_inner_score));
  ASSERT_GE(expected, 0);
  ASSERT_GE(expected_inner, 0);
  ASSERT_NE(expected, expected_inner);
  auto outer_query = std::make_shared<DiskAnnQueryParam>(*query);
  outer_query->filter = std::make_shared<IndexFilter>();
  bool invoked = false;
  int nested_status = -2;
  int64_t nested = -1;
  float nested_score = -1;
  outer_query->filter->set([&](uint64_t) {
    if (!invoked) {
      invoked = true;
      nested_status = second->search_internal_ids(second_input, query, &nested, &nested_score);
    }
    return false;
  });
  int64_t result = -1;
  float score = -1;
  EXPECT_EQ(0, first->search_internal_ids(first_input, outer_query, &result, &score));
  EXPECT_TRUE(invoked);
  EXPECT_EQ(0, nested_status);
  EXPECT_EQ(expected_inner, nested);
  EXPECT_EQ(expected, result);
  EXPECT_EQ(expected_score, score);
  ASSERT_EQ(0, second->close());
  ASSERT_EQ(0, first->close());
}

INSTANTIATE_TEST_SUITE_P(Backends, DiskAnnReadBudgetTest,
                         ::testing::Values(StorageType::kMMAP,
                                           StorageType::kBufferPool));
}  // namespace
}  // namespace zvec::core_interface

#endif  // DISKANN_SUPPORTED
