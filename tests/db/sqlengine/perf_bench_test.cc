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

//! Micro benchmarks for the query paths affected by the size-optimization
//! work (third-party -Os for non-hot libraries, slimmed Arrow compute,
//! trimmed RocksDB features):
//!
//!   - forward filter   : column scan + expression evaluation (Arrow compute)
//!   - inverted search  : term/range/boolean filters through inverted indexes
//!   - FTS search       : tokenized match queries (cppjieba/snowball/utf8proc)
//!   - insert           : segment build throughput (RocksDB id map, Arrow IPC)
//!
//! The benchmarks reuse the recall-test fixtures, so results are comparable
//! against the same data layout as the functional tests.
//!
//! Disabled by default to keep CI fast. Run with:
//!
//!   ZVEC_RUN_PERF_BENCH=1 ./build/bin/perf_bench_test
//!
//! Tunables (environment variables):
//!   ZVEC_PERF_DOC_COUNT   documents per segment      (default 100000)
//!   ZVEC_PERF_ITERATIONS  measured runs per bench    (default 30)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/sqlengine/sqlengine.h"
#include "zvec/db/query_params.h"
#include "zvec/db/schema.h"
#include "recall_base.h"

namespace zvec::sqlengine {
namespace {

bool PerfEnabled() {
  const char *value = std::getenv("ZVEC_RUN_PERF_BENCH");
  return value != nullptr && std::string(value) == "1";
}

int64_t DocCount() {
  if (const char *value = std::getenv("ZVEC_PERF_DOC_COUNT")) {
    return std::atoll(value);
  }
  return 100000;
}

int Iterations() {
  if (const char *value = std::getenv("ZVEC_PERF_ITERATIONS")) {
    return std::atoi(value);
  }
  return 30;
}

// Executes fn once for warm-up, then measures repeated runs and prints a
// one-line [PERF] report with latency percentiles and throughput.
void RunBench(const std::string &name,
              const std::function<Result<DocPtrList>()> &fn) {
  auto result = fn();  // warm-up also validates the query
  if (!result.has_value()) {
    std::printf("[PERF] %-34s | ERROR: %s\n", name.c_str(),
                result.error().c_str());
    return;
  }
  const size_t rows = result.value().size();

  std::vector<double> samples_us;
  samples_us.reserve(Iterations());
  for (int i = 0; i < Iterations(); ++i) {
    const auto start = std::chrono::steady_clock::now();
    auto r = fn();
    const auto end = std::chrono::steady_clock::now();
    if (!r.has_value()) {
      std::printf("[PERF] %-34s | ERROR on run %d: %s\n", name.c_str(), i,
                  r.error().c_str());
      return;
    }
    samples_us.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  std::sort(samples_us.begin(), samples_us.end());
  const auto percentile = [&](double p) {
    const size_t idx = static_cast<size_t>(p * samples_us.size());
    return samples_us[std::min(idx, samples_us.size() - 1)];
  };
  const double sum_us =
      std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
  const double mean_us = sum_us / static_cast<double>(samples_us.size());

  std::printf(
      "[PERF] %-34s | rows=%6zu | median=%9.1fus mean=%9.1fus p90=%9.1fus "
      "p99=%9.1fus min=%9.1fus | qps=%8.1f\n",
      name.c_str(), rows, percentile(0.50), mean_us, percentile(0.90),
      percentile(0.99), samples_us.front(), 1e6 / mean_us);
  std::fflush(stdout);
}

}  // namespace

// ============================================================================
// Forward filter + inverted search benchmarks (recall_base schema)
// ============================================================================

class ScalarPerfBench : public RecallTest {
 protected:
  static void SetUpTestSuite() {
    if (!PerfEnabled()) {
      return;
    }
    FileHelper::RemoveDirectory(seg_path_);
    FileHelper::CreateDirectory(seg_path_);
    collection_schema_ = GetCollectionSchema();
    auto segment = create_segment();
    ASSERT_NE(segment, nullptr);

    const auto start = std::chrono::steady_clock::now();
    auto status = InsertDoc(segment, 0, static_cast<uint64_t>(DocCount()));
    const auto end = std::chrono::steady_clock::now();
    ASSERT_TRUE(status.ok()) << status.c_str();
    segments_.push_back(segment);

    const double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::printf(
        "[PERF] %-34s | docs=%6lld | total=%9.1fms | docs_per_sec=%9.0f\n",
        "insert_build_segment", static_cast<long long>(DocCount()), total_ms,
        static_cast<double>(DocCount()) / (total_ms / 1000.0));
    std::fflush(stdout);

    engine_ = SQLEngine::create(std::make_shared<Profiler>());
  }

  static void TearDownTestSuite() {
    engine_.reset();
    RecallTest::TearDownTestSuite();
  }

  // Builds a filter-only SearchQuery (no vector target) and returns a
  // runnable closure for RunBench.
  std::function<Result<DocPtrList>()> FilterQuery(const std::string &filter,
                                                  int topk = 2000) const {
    return [filter, topk]() -> Result<DocPtrList> {
      SearchQuery query;
      query.filter_ = filter;
      query.topk_ = topk;
      query.output_fields_ = {"id", "name", "age"};
      return engine_->execute(collection_schema_, query, segments_);
    };
  }

  static inline SQLEngine::Ptr engine_;
};

#define PERF_SKIP()                                                       \
  do {                                                                    \
    if (!PerfEnabled()) {                                                 \
      GTEST_SKIP() << "set ZVEC_RUN_PERF_BENCH=1 to run perf benchmarks"; \
    }                                                                     \
  } while (0)

TEST_F(ScalarPerfBench, ForwardFilterLowSelectivity) {
  PERF_SKIP();
  // age = doc_id % 100, so age < 5 matches 5% of the docs.
  RunBench("forward_filter_age_lt_5", FilterQuery("age < 5"));
}

TEST_F(ScalarPerfBench, ForwardFilterHalfSelectivity) {
  PERF_SKIP();
  RunBench("forward_filter_age_lt_50", FilterQuery("age < 50"));
}

TEST_F(ScalarPerfBench, ForwardFilterRange) {
  PERF_SKIP();
  RunBench("forward_filter_age_range", FilterQuery("age >= 25 and age <= 75"));
}

TEST_F(ScalarPerfBench, ForwardFilterString) {
  PERF_SKIP();
  // name = "user_" + doc_id % 100: full-scan string comparison.
  RunBench("forward_filter_name_eq", FilterQuery("name = 'user_5'"));
}

TEST_F(ScalarPerfBench, ForwardFilterDouble) {
  PERF_SKIP();
  RunBench("forward_filter_score_gt", FilterQuery("score > 50.0"));
}

TEST_F(ScalarPerfBench, InvertTerm) {
  PERF_SKIP();
  RunBench("invert_term_age_eq_5", FilterQuery("invert_age = 5"));
}

TEST_F(ScalarPerfBench, InvertRange) {
  PERF_SKIP();
  RunBench("invert_range_age_lt_10", FilterQuery("invert_age < 10"));
}

TEST_F(ScalarPerfBench, InvertStringTerm) {
  PERF_SKIP();
  RunBench("invert_term_name_eq", FilterQuery("invert_name = 'user_5'"));
}

TEST_F(ScalarPerfBench, InvertWideRange) {
  PERF_SKIP();
  const std::string filter = "invert_id >= " + std::to_string(DocCount() / 2);
  RunBench("invert_range_id_half", FilterQuery(filter));
}

TEST_F(ScalarPerfBench, InvertBooleanOr) {
  PERF_SKIP();
  RunBench("invert_or_two_terms",
           FilterQuery("invert_age = 5 or invert_age = 6"));
}

// Sanity-checks the hit counts that the benchmarks above rely on.
TEST_F(ScalarPerfBench, ExpectedHitCounts) {
  PERF_SKIP();
  const int64_t n = DocCount();
  const auto count_ids_where = [n](std::function<bool(uint64_t)> pred) {
    int64_t count = 0;
    for (uint64_t id = 0; id < static_cast<uint64_t>(n); ++id) {
      if (pred(id)) ++count;
    }
    return count;
  };

  auto result = FilterQuery("age < 5")();
  ASSERT_TRUE(result.has_value());
  const int64_t expected =
      count_ids_where([](uint64_t id) { return id % 100 < 5; });
  EXPECT_EQ(static_cast<int64_t>(result.value().size()),
            std::min<int64_t>(expected, 2000));

  result = FilterQuery("invert_age = 5")();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(static_cast<int64_t>(result.value().size()),
            std::min<int64_t>(n / 100, 2000));
}

// ============================================================================
// FTS benchmarks (dedicated schema with a tokenized content field)
// ============================================================================

class FtsPerfBench : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!PerfEnabled()) {
      return;
    }
    FileHelper::RemoveDirectory(seg_path_);
    FileHelper::CreateDirectory(seg_path_);
    build_schema();
    auto segment = create_segment();
    ASSERT_NE(segment, nullptr);

    const auto start = std::chrono::steady_clock::now();
    insert_docs(segment);
    const auto end = std::chrono::steady_clock::now();
    segments_.push_back(segment);

    const double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::printf(
        "[PERF] %-34s | docs=%6lld | total=%9.1fms | docs_per_sec=%9.0f\n",
        "fts_insert_build_segment", static_cast<long long>(DocCount()),
        total_ms, static_cast<double>(DocCount()) / (total_ms / 1000.0));
    std::fflush(stdout);

    engine_ = SQLEngine::create(std::make_shared<Profiler>());
  }

  static void TearDownTestSuite() {
    segments_.clear();
    engine_.reset();
    schema_.reset();
    FileHelper::RemoveDirectory(seg_path_);
  }

  std::function<Result<DocPtrList>()> FtsMatch(
      const std::string &match_string, const std::string &default_op = "",
      const std::string &filter = "", int topk = 100) const {
    return [match_string, default_op, filter, topk]() -> Result<DocPtrList> {
      SearchQuery query;
      query.topk_ = topk;
      query.target_.field_name_ = "content";
      FtsClause fts;
      fts.match_string_ = match_string;
      query.target_.clause_ = fts;
      if (!default_op.empty()) {
        auto fts_qp = std::make_shared<zvec::FtsQueryParams>();
        fts_qp->set_default_operator(default_op);
        query.target_.query_params_ = fts_qp;
      }
      if (!filter.empty()) {
        query.filter_ = filter;
      }
      return engine_->execute(schema_, query, segments_);
    };
  }

 private:
  static void build_schema() {
    auto fts_params = std::make_shared<FtsIndexParams>(
        "whitespace", std::vector<std::string>{"lowercase"}, "");
    auto invert_params = std::make_shared<InvertIndexParams>(true);
    schema_ = std::make_shared<CollectionSchema>(
        "fts_perf_bench",
        std::vector<FieldSchema::Ptr>{
            std::make_shared<FieldSchema>("content", DataType::STRING, false,
                                          fts_params),
            std::make_shared<FieldSchema>("tag", DataType::INT32, false,
                                          invert_params),
            // Dummy vector field required for filter parsing path in execute.
            std::make_shared<FieldSchema>(
                "vec", DataType::VECTOR_FP32, 4, false,
                std::make_shared<FlatIndexParams>(MetricType::L2)),
        });
  }

  static Segment::Ptr create_segment() {
    auto segment_meta = std::make_shared<SegmentMeta>();
    segment_meta->set_id(0);

    auto id_map = IDMap::CreateAndOpen("fts_perf_bench", seg_path_ + "/id_map",
                                       true, false);
    auto delete_store = std::make_shared<DeleteStore>("fts_perf_bench");

    Version v1;
    v1.set_schema(*schema_);
    std::string v_path = seg_path_ + "/manifest";
    FileHelper::CreateDirectory(v_path);
    auto vm = VersionManager::Create(v_path, v1);
    if (!vm.has_value()) {
      return nullptr;
    }

    BlockMeta mem_block;
    mem_block.id_ = 0;
    mem_block.type_ = BlockType::SCALAR;
    mem_block.min_doc_id_ = 0;
    mem_block.max_doc_id_ = 0;
    mem_block.doc_count_ = 0;
    segment_meta->set_writing_forward_block(mem_block);

    SegmentOptions options;
    options.read_only_ = false;
    options.enable_mmap_ = true;
    options.max_buffer_size_ = 256 * 1024;

    auto result = Segment::CreateAndOpen(seg_path_, *schema_, 0, 0, id_map,
                                         delete_store, vm.value(), options);
    if (!result) {
      return nullptr;
    }
    return result.value();
  }

  // Generates DocCount() documents with 8 tokens drawn from a 256-word
  // vocabulary (each word hits ~3% of docs), plus a 1% word ("mediumword")
  // and a 0.1% word ("rareword") for selectivity sweeps.
  static void insert_docs(const Segment::Ptr &segment) {
    for (int64_t doc_id = 0; doc_id < DocCount(); ++doc_id) {
      std::string content;
      for (int k = 0; k < 8; ++k) {
        if (k > 0) content += ' ';
        content += "w" + std::to_string((doc_id + k * 31) % 256);
      }
      if (doc_id % 100 == 0) {
        content += " mediumword";
      }
      if (doc_id % 1000 == 0) {
        content += " rareword";
      }

      Doc doc;
      doc.set_pk("pk_" + std::to_string(doc_id));
      doc.set_doc_id(static_cast<uint64_t>(doc_id));
      doc.set<std::string>("content", content);
      doc.set<int32_t>("tag", static_cast<int32_t>(doc_id % 2));
      auto status = segment->Insert(doc);
      if (!status.ok()) {
        std::printf("[PERF] fts insert failed at doc %lld: %s\n",
                    static_cast<long long>(doc_id), status.c_str());
        return;
      }
    }
  }

 protected:
  static inline std::string seg_path_ = "./fts_perf_bench_collection";
  static inline CollectionSchema::Ptr schema_;
  static inline std::vector<Segment::Ptr> segments_;
  static inline SQLEngine::Ptr engine_;
};

TEST_F(FtsPerfBench, RareTerm) {
  PERF_SKIP();
  RunBench("fts_match_rare_0.1pct", FtsMatch("rareword"));
}

TEST_F(FtsPerfBench, MediumTerm) {
  PERF_SKIP();
  RunBench("fts_match_medium_1pct", FtsMatch("mediumword"));
}

TEST_F(FtsPerfBench, CommonTerm) {
  PERF_SKIP();
  RunBench("fts_match_common_3pct", FtsMatch("w5"));
}

TEST_F(FtsPerfBench, MultiTermOr) {
  PERF_SKIP();
  RunBench("fts_match_3term_or", FtsMatch("w1 w2 w3"));
}

TEST_F(FtsPerfBench, MultiTermAnd) {
  PERF_SKIP();
  // Adjacent generated tokens differ by 31 (mod 256), so w1+w32 co-occur.
  RunBench("fts_match_2term_and", FtsMatch("w1 w32", "AND"));
}

TEST_F(FtsPerfBench, MatchWithInvertedFilter) {
  PERF_SKIP();
  // mediumword only appears in docs whose id % 100 == 0, all of which have
  // tag = 0, so filter on tag = 0 keeps every match.
  RunBench("fts_match_plus_filter", FtsMatch("mediumword", "", "tag = 0"));
}

}  // namespace zvec::sqlengine
