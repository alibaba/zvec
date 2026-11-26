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
#include <condition_variable>
#include <future>
#include <set>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "tests/test_util.h"
#include "hnsw_streamer.h"

namespace zvec::core {
namespace {

constexpr uint32_t kDimension = 16;
using Vector = std::array<float, kDimension>;

Vector MakeVector(float value) {
  Vector vector;
  vector.fill(value);
  return vector;
}

class HnswDuplicatesTest : public testing::TestWithParam<const char *> {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(64UL * 1024 * 1024);
    path_ = std::string("hnsw_duplicates_test_dir/") + GetParam() + "/" +
            testing::UnitTest::GetInstance()->current_test_info()->name();
    test_util::RemoveTestPath(path_);
    meta_.set_meta(IndexMeta::DT_FP32, kDimension);
    meta_.set_metric("SquaredEuclidean", 0, ailego::Params{});
    params_.set(PARAM_HNSW_STREAMER_DEDUPLICATE_VECTORS, true);
    params_.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 5U);
    params_.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 64U);
    params_.set(PARAM_HNSW_STREAMER_EF, 16U);
    params_.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);
    params_.set(PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT, 10000U);
    // Force several node chunks without requiring a large test dataset.
    params_.set(PARAM_HNSW_STREAMER_CHUNK_SIZE, 4096U);
  }

  void TearDown() override {
    context_.reset();
    streamer_.reset();
    storage_.reset();
    test_util::RemoveTestPath(path_);
  }

  void open(bool create = true, bool contiguous = false) {
    params_.set(PARAM_HNSW_STREAMER_USE_CONTIGUOUS_MEMORY, contiguous);
    storage_ = IndexFactory::CreateStorage(contiguous ? "MMapFileReadStorage"
                                                      : GetParam());
    ASSERT_NE(nullptr, storage_);
    ASSERT_EQ(0, storage_->init(ailego::Params{}));
    if (std::string(GetParam()) == "BufferStorage") {
      ASSERT_EQ(
          0, storage_->open(path_, IndexStorage::OpenOptions{create, false}));
    } else {
      ASSERT_EQ(0, storage_->open(path_, create));
    }
    streamer_ = IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_NE(nullptr, streamer_);
    ASSERT_EQ(0, streamer_->init(meta_, params_));
    ASSERT_EQ(0, streamer_->open(storage_, contiguous));
    context_ = streamer_->create_context();
    ASSERT_NE(nullptr, context_);
  }

  void add(uint64_t key, const Vector &vector) {
    ASSERT_EQ(0,
              streamer_->add_impl(key, vector.data(), query_meta_, context_));
  }

  void reopen(bool contiguous = false) {
    context_.reset();
    ASSERT_EQ(0, streamer_->close());
    streamer_.reset();
    ASSERT_EQ(0, storage_->close());
    storage_.reset();
    open(false, contiguous);
  }

  const HnswStreamerEntity &get_entity() const {
    return static_cast<const HnswStreamerEntity &>(
        static_cast<const HnswContext &>(*context_).get_entity());
  }

  void search_and_compare_scores(const Vector &query, uint32_t k) {
    context_->set_topk(k);
    ASSERT_EQ(0, streamer_->search_impl(query.data(), query_meta_, context_));
    const auto actual = context_->result();
    ASSERT_EQ(0,
              streamer_->search_bf_impl(query.data(), query_meta_, context_));
    const auto &expected = context_->result();
    ASSERT_EQ(expected.size(), actual.size());
    std::set<uint64_t> keys;
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_FLOAT_EQ(expected[i].score(), actual[i].score());
      EXPECT_TRUE(keys.insert(actual[i].key()).second);
    }
  }

  void build_concurrently(bool identical) {
    if (std::string(GetParam()) != "MMapFileStorage") return;
    storage_ = IndexFactory::CreateStorage(GetParam());
    ASSERT_EQ(0, storage_->init(ailego::Params{}));
    ASSERT_EQ(0, storage_->open(path_, true));
    IndexStreamer::Stats stats;
    HnswMmapStreamerEntity entity(stats);
    entity.set_vector_size(sizeof(Vector));
    entity.set_l0_neighbor_cnt(10);
    entity.set_upper_neighbor_cnt(5);
    entity.set_scaling_factor(5);
    entity.set_ef_construction(32);
    entity.set_deduplicate_vectors(true);
    ASSERT_EQ(0, entity.init(100));
    ASSERT_EQ(0, entity.open(storage_, 0, false));
    HnswAlgorithm<HnswMmapStreamerEntity> algorithm(entity);
    ASSERT_EQ(0, algorithm.init());
    auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
    ASSERT_NE(nullptr, metric);
    ASSERT_EQ(0, metric->init(meta_, ailego::Params{}));
    auto make_context = [&]() {
      auto ctx =
          std::make_unique<HnswContext>(kDimension, metric, entity.clone());
      EXPECT_EQ(0, ctx->init(HnswContext::kStreamerContext));
      return ctx;
    };
    node_id_t id;
    auto seed = MakeVector(0);
    ASSERT_EQ(0, entity.add_vector(0, 0, seed.data(), &id));
    auto seed_context = make_context();
    seed_context->reset_query(seed.data(), meta_);
    ASSERT_EQ(0, algorithm.add_node(0, 0, seed_context.get()));
    std::array<Vector, 2> vectors{MakeVector(10),
                                  MakeVector(identical ? 10 : 20)};
    for (uint32_t i = 0; i < 2; ++i) {
      ASSERT_EQ(0, entity.add_vector(0, i + 1, vectors[i].data(), &id));
    }
    auto first = make_context();
    auto second = make_context();
    std::mutex gate_mutex;
    std::condition_variable gate;
    unsigned entered = 0;
    bool overlapped = true;
    auto build = [&](uint32_t worker, HnswContext *ctx) {
      ctx->reset_query(vectors[worker].data(), meta_);
      // Limit both workers to the entry point so neither can discover the
      // other's equal vector after the first-distance rendezvous.
      if (identical) ctx->set_max_scan_num(1);
      bool first_distance = true;
      auto rendezvous = [&]() {
        if (!first_distance) return;
        first_distance = false;
        std::unique_lock<std::mutex> lock(gate_mutex);
        ++entered;
        gate.notify_all();
        // A global insertion lock prevents the second builder from reaching
        // distance evaluation, making this a deterministic regression check.
        if (!gate.wait_for(lock, std::chrono::seconds(2),
                           [&]() { return entered == 2; })) {
          overlapped = false;
        }
      };
      ctx->dist_calculator().update_distance(
          [&](const void *m, const void *q, size_t dim, float *out) {
            rendezvous();
            metric->distance()(m, q, dim, out);
          },
          [&](const void **m, const void *q, size_t num, size_t dim, float *out,
              const void **extra) {
            rendezvous();
            metric->batch_distance()(m, q, num, dim, out, extra);
          });
      return algorithm.add_node(worker + 1, 0, ctx);
    };
    auto a = std::async(std::launch::async, build, 0, first.get());
    auto b = std::async(std::launch::async, build, 1, second.get());
    EXPECT_EQ(0, a.get());
    EXPECT_EQ(0, b.get());
    EXPECT_TRUE(overlapped);
    if (identical) {
      for (node_id_t rep : {1U, 2U}) {
        HnswStreamerEntity::DuplicateLinks links;
        ASSERT_EQ(0, entity.get_duplicate_links(rep, 3, &links));
        EXPECT_EQ(rep, links.representative);
      }
      // Later duplicates can attach to either representative without a global
      // reservation or a second navigation insertion.
      ASSERT_EQ(0, entity.add_vector(0, 3, vectors[0].data(), &id));
      auto ctx = make_context();
      ctx->reset_query(vectors[0].data(), meta_);
      ctx->set_max_scan_num(1000);
      ASSERT_EQ(0, algorithm.add_node(3, 0, ctx.get()));
      HnswStreamerEntity::DuplicateLinks links;
      ASSERT_EQ(0, entity.get_duplicate_links(3, 4, &links));
      EXPECT_TRUE(links.representative == 1 || links.representative == 2);
      HnswStreamerEntity::DuplicateLinks head;
      ASSERT_EQ(0, entity.get_duplicate_links(links.representative, 4, &head));
      EXPECT_EQ(3U, head.next);
    }
    ASSERT_EQ(0, entity.close());
  }

  void check_candidate_gate(float distance, bool identical, level_t level,
                            bool expect_duplicate) {
    storage_ = IndexFactory::CreateStorage(GetParam());
    ASSERT_EQ(0, storage_->init(ailego::Params{}));
    ASSERT_EQ(0, storage_->open(path_, true));
    IndexStreamer::Stats stats;
    HnswMmapStreamerEntity entity(stats);
    entity.set_vector_size(sizeof(Vector));
    entity.set_l0_neighbor_cnt(10);
    entity.set_upper_neighbor_cnt(5);
    entity.set_scaling_factor(5);
    entity.set_ef_construction(32);
    entity.set_deduplicate_vectors(true);
    ASSERT_EQ(0, entity.init(100));
    ASSERT_EQ(0, entity.open(storage_, 0, false));
    HnswAlgorithm<HnswMmapStreamerEntity> algorithm(entity);
    ASSERT_EQ(0, algorithm.init());
    auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
    ASSERT_EQ(0, metric->init(meta_, ailego::Params{}));
    HnswContext ctx(kDimension, metric, entity.clone());
    ASSERT_EQ(0, ctx.init(HnswContext::kStreamerContext));
    const auto seed = MakeVector(0);
    const auto candidate = MakeVector(1);
    const auto input = MakeVector(identical ? 1.0f : 1.00001f);
    node_id_t id;
    ASSERT_EQ(0, entity.add_vector(1, 0, seed.data(), &id));
    ASSERT_EQ(0, entity.add_vector(1, 1, candidate.data(), &id));
    ASSERT_EQ(0, entity.add_vector(level, 2, input.data(), &id));
    // The matching candidate is not the entry point. A level-0 insertion
    // discovers it during upper-layer descent; level 1 uses the heap search.
    entity.update_ep_and_level(0, 1);
    ASSERT_EQ(0, entity.update_neighbors(1, 0, {{1, 16.0f}}));
    ASSERT_EQ(0, entity.update_neighbors(0, 0, {{1, 16.0f}}));
    ctx.reset_query(input.data(), meta_);
    ctx.set_max_scan_num(1000);
    auto score = [&](const void *vector) {
      return static_cast<const float *>(vector)[0] == 0 ? 16.0f : distance;
    };
    ctx.dist_calculator().update_distance(
        [&](const void *m, const void *, size_t, float *out) {
          *out = score(m);
        },
        [&](const void **m, const void *, size_t count, size_t, float *out,
            const void **) {
          for (size_t i = 0; i < count; ++i) out[i] = score(m[i]);
        });
    ASSERT_EQ(0, algorithm.add_node(2, level, &ctx));
    HnswStreamerEntity::DuplicateLinks links;
    ASSERT_EQ(0, entity.get_duplicate_links(2, 3, &links));
    EXPECT_EQ(expect_duplicate ? 1U : 2U, links.representative);
    if (expect_duplicate) {
      // Early attachment must not publish the leaf in either navigation layer.
      for (level_t l = 0; l <= 1; ++l) {
        const auto neighbors = entity.get_neighbors(l, 0);
        ASSERT_EQ(1U, neighbors.size());
        EXPECT_EQ(1U, neighbors[0]);
      }
      ASSERT_EQ(0, entity.get_duplicate_links(1, 3, &links));
      EXPECT_EQ(2U, links.next);
      EXPECT_EQ(0U, entity.get_neighbors(0, 1).size());
    }
    ASSERT_EQ(0, entity.close());
  }

  std::string path_;
  IndexMeta meta_;
  IndexQueryMeta query_meta_{IndexMeta::DT_FP32, kDimension};
  ailego::Params params_;
  IndexStorage::Pointer storage_;
  IndexStreamer::Pointer streamer_;
  IndexStreamer::Context::Pointer context_;
};

TEST_P(HnswDuplicatesTest, IdenticalVectorsRemainIndividuallySearchable) {
  ASSERT_NO_FATAL_FAILURE(open());
  const auto vector = MakeVector(1.0f);
  for (uint32_t id = 0; id < 256; ++id) add(1000 + id, vector);

  // The representative has no navigation neighbors; every leaf has exactly
  // one back edge. The reverse posting list is not bounded by M=5.
  EXPECT_EQ(0U, get_entity().get_neighbors(0, 0).size());
  for (uint32_t id = 1; id < 256; ++id) {
    const auto neighbors = get_entity().get_neighbors(0, id);
    ASSERT_EQ(1U, neighbors.size());
    EXPECT_EQ(0U, neighbors[0]);
  }
  search_and_compare_scores(vector, 40);
  context_->set_debug_mode(true);
  ASSERT_EQ(0, streamer_->search_impl(vector.data(), query_meta_, context_));
  EXPECT_LE(static_cast<HnswContext &>(*context_).get_scan_num(), 2U);

  // A representative rejected by the record filter must still expose the
  // allowed member, even when it is at the tail of a large posting list.
  context_->set_filter([](uint64_t key) { return key != 1001; });
  context_->set_topk(1);
  ASSERT_EQ(0, streamer_->search_impl(vector.data(), query_meta_, context_));
  ASSERT_EQ(1U, context_->result().size());
  EXPECT_EQ(1001U, context_->result()[0].key());

  std::vector<uint64_t> keys;
  std::vector<float> scores;
  ASSERT_EQ(0, streamer_->search_candidates_impl(vector.data(), query_meta_,
                                                 keys, context_, &scores));
  ASSERT_EQ(1U, keys.size());
  EXPECT_EQ(1001U, keys[0]);
  EXPECT_FLOAT_EQ(0, scores[0]);
}

TEST_P(HnswDuplicatesTest, DenseDuplicateGroupDoesNotReplaceNavigationEdges) {
  ASSERT_NO_FATAL_FAILURE(open());
  add(0, MakeVector(0));
  for (uint32_t id = 1; id < 512; ++id) add(id, MakeVector(0));
  for (uint32_t group = 1; group <= 24; ++group) {
    add(512 + group, MakeVector(group * 10.0f));
  }
  // Every edge in the representative graph targets another representative.
  for (node_id_t id : {0U, 512U, 520U, 535U}) {
    const auto neighbors = get_entity().get_neighbors(0, id);
    EXPECT_LE(neighbors.size(), 10U);
    for (uint32_t n = 0; n < neighbors.size(); ++n) {
      EXPECT_TRUE(neighbors[n] == 0 || neighbors[n] >= 512);
    }
  }
  search_and_compare_scores(MakeVector(240), 3);
  search_and_compare_scores(MakeVector(0), 32);
  ASSERT_EQ(0, streamer_->flush(42));
  ASSERT_NO_FATAL_FAILURE(reopen());
  add(2000, MakeVector(0));
  search_and_compare_scores(MakeVector(240), 3);
  context_->set_filter([](uint64_t key) { return key != 2000; });
  search_and_compare_scores(MakeVector(0), 1);
}

TEST_P(HnswDuplicatesTest, ExplicitIdsAndGapsSurviveReopen) {
  ASSERT_NO_FATAL_FAILURE(open());
  const auto vector = MakeVector(2);
  ASSERT_EQ(
      0, streamer_->add_with_id_impl(3, vector.data(), query_meta_, context_));
  ASSERT_EQ(
      0, streamer_->add_with_id_impl(9, vector.data(), query_meta_, context_));
  ASSERT_EQ(IndexError_Duplicate, streamer_->add_with_id_impl(
                                      3, vector.data(), query_meta_, context_));
  const auto other = MakeVector(20);
  ASSERT_EQ(
      0, streamer_->add_with_id_impl(15, other.data(), query_meta_, context_));
  {
    SCOPED_TRACE("initial explicit IDs");
    search_and_compare_scores(vector, 3);
  }
  ASSERT_NO_FATAL_FAILURE(reopen());
  ASSERT_EQ(
      0, streamer_->add_with_id_impl(5, vector.data(), query_meta_, context_));
  {
    SCOPED_TRACE("filled gap after reopen");
    search_and_compare_scores(vector, 4);
  }
  context_->set_filter([](uint64_t key) { return key != 5; });
  search_and_compare_scores(vector, 1);
}

TEST_P(HnswDuplicatesTest, NearDuplicatesAreNotMerged) {
  ASSERT_NO_FATAL_FAILURE(open());
  for (uint32_t id = 0; id < 32; ++id) {
    add(id, MakeVector(1.0f + id * 0.00001f));
  }
  const auto &entity = get_entity();
  for (node_id_t id = 0; id < 32; ++id) {
    HnswStreamerEntity::DuplicateLinks links;
    ASSERT_EQ(0, entity.get_duplicate_links(id, 32, &links));
    EXPECT_EQ(id, links.representative);
    EXPECT_EQ(kInvalidNodeId, links.next);
  }
  search_and_compare_scores(MakeVector(1.0001f), 5);
}

TEST_P(HnswDuplicatesTest, ConcurrentIdenticalInsertsRemainSearchable) {
  ASSERT_NO_FATAL_FAILURE(open());
  constexpr uint32_t workers = 4;
  constexpr uint32_t per_worker = 64;
  std::vector<IndexStreamer::Context::Pointer> contexts;
  for (uint32_t worker = 0; worker < workers; ++worker) {
    contexts.emplace_back(streamer_->create_context());
  }
  std::vector<std::future<int>> futures;
  for (uint32_t worker = 0; worker < workers; ++worker) {
    futures.emplace_back(std::async(std::launch::async, [&, worker]() {
      const auto vector = MakeVector(1);
      for (uint32_t i = 0; i < per_worker; ++i) {
        int ret = streamer_->add_impl(worker * per_worker + i, vector.data(),
                                      query_meta_, contexts[worker]);
        if (ret != 0) return ret;
      }
      return 0;
    }));
  }
  for (auto &future : futures) ASSERT_EQ(0, future.get());
  uint32_t representatives = 0;
  for (node_id_t id = 0; id < workers * per_worker; ++id) {
    HnswStreamerEntity::DuplicateLinks links;
    ASSERT_EQ(
        0, get_entity().get_duplicate_links(id, workers * per_worker, &links));
    if (links.representative == id) ++representatives;
  }
  EXPECT_GE(representatives, 1U);
  EXPECT_LE(representatives, workers);
  search_and_compare_scores(MakeVector(1), workers * per_worker);
}

TEST_P(HnswDuplicatesTest, PositiveDistanceCandidateIsComparedExactly) {
  if (std::string(GetParam()) != "MMapFileStorage") return;
  check_candidate_gate(0.5e-6f, true, 0, true);
}

TEST_P(HnswDuplicatesTest, HeapSearchAlsoChecksDuplicateCandidates) {
  if (std::string(GetParam()) != "MMapFileStorage") return;
  check_candidate_gate(0.5e-6f, true, 1, true);
}

TEST_P(HnswDuplicatesTest, ZeroDistanceDoesNotMergeDifferentBytes) {
  if (std::string(GetParam()) != "MMapFileStorage") return;
  check_candidate_gate(0.0f, false, 1, false);
}

TEST_P(HnswDuplicatesTest, ThresholdMissKeepsOrdinaryRepresentative) {
  if (std::string(GetParam()) != "MMapFileStorage") return;
  check_candidate_gate(1e-6f, true, 1, false);
}

TEST_P(HnswDuplicatesTest, ReopenRejectsWrongLayout) {
  ASSERT_NO_FATAL_FAILURE(open());
  add(1, MakeVector(1));
  add(2, MakeVector(1));
  context_.reset();
  ASSERT_EQ(0, streamer_->close());
  streamer_.reset();
  ASSERT_EQ(0, storage_->close());
  storage_.reset();
  params_.set(PARAM_HNSW_STREAMER_DEDUPLICATE_VECTORS, false);
  storage_ = IndexFactory::CreateStorage(GetParam());
  ASSERT_EQ(0, storage_->init(ailego::Params{}));
  if (std::string(GetParam()) == "BufferStorage") {
    ASSERT_EQ(0,
              storage_->open(path_, IndexStorage::OpenOptions{false, false}));
  } else {
    ASSERT_EQ(0, storage_->open(path_, false));
  }
  streamer_ = IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_EQ(0, streamer_->init(meta_, params_));
  EXPECT_EQ(IndexError_Mismatch, streamer_->open(storage_));
}

TEST_P(HnswDuplicatesTest, ConcurrentGroupsReduceNavigationDuplicates) {
  ASSERT_NO_FATAL_FAILURE(open());
  constexpr uint32_t workers = 4;
  constexpr uint32_t groups = 16;
  std::vector<IndexStreamer::Context::Pointer> contexts;
  for (uint32_t worker = 0; worker < workers; ++worker) {
    contexts.emplace_back(streamer_->create_context());
  }
  std::vector<std::future<int>> futures;
  for (uint32_t worker = 0; worker < workers; ++worker) {
    futures.emplace_back(std::async(std::launch::async, [&, worker]() {
      for (uint32_t i = 0; i < 64; ++i) {
        const auto vector = MakeVector((i + worker) % groups);
        int ret = streamer_->add_impl(worker * 64 + i, vector.data(),
                                      query_meta_, contexts[worker]);
        if (ret != 0) return ret;
      }
      return 0;
    }));
  }
  for (auto &future : futures) ASSERT_EQ(0, future.get());
  uint32_t representatives = 0;
  for (node_id_t id = 0; id < workers * 64; ++id) {
    HnswStreamerEntity::DuplicateLinks links;
    ASSERT_EQ(0, get_entity().get_duplicate_links(id, workers * 64, &links));
    if (links.representative == id) ++representatives;
  }
  EXPECT_GE(representatives, groups);
  EXPECT_LE(representatives, groups * workers);
  search_and_compare_scores(MakeVector(10), 25);
}

TEST_P(HnswDuplicatesTest, DistinctRepresentativesBuildConcurrently) {
  build_concurrently(false);
}

TEST_P(HnswDuplicatesTest, ConcurrentMissesCanPublishEqualRepresentatives) {
  build_concurrently(true);
}

TEST_P(HnswDuplicatesTest, Fp32QuantizerSharesOnlyIdenticalStoredRecords) {
  for (const char *metric : {"SquaredEuclidean", "Cosine", "InnerProduct"}) {
    SCOPED_TRACE(metric);
    IndexMeta raw_meta(IndexMeta::DT_FP32, kDimension);
    raw_meta.set_metric(metric, 0, ailego::Params{});
    auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
    ASSERT_NE(nullptr, quantizer);
    ASSERT_EQ(0, quantizer->init(raw_meta, ailego::Params{}));
    storage_ = IndexFactory::CreateStorage(GetParam());
    ASSERT_EQ(0, storage_->init(ailego::Params{}));
    ASSERT_EQ(0, storage_->open(path_ + metric, true));
    streamer_ = IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_EQ(0, streamer_->init(quantizer->meta(), params_, quantizer));
    ASSERT_EQ(0, streamer_->open(storage_));
    context_ = streamer_->create_context();
    ASSERT_NE(nullptr, context_);
    auto input = MakeVector(1);
    std::string code;
    IndexQueryMeta code_meta;
    ASSERT_EQ(
        0, quantizer->quantize(input.data(), query_meta_, &code, &code_meta));
    for (uint32_t i = 0; i < 64; ++i) {
      ASSERT_EQ(0, streamer_->add_impl(i, code.data(), code_meta, context_));
    }
    HnswStreamerEntity::DuplicateLinks links;
    ASSERT_EQ(0, get_entity().get_duplicate_links(1, 64, &links));
    EXPECT_EQ(0U, links.representative);
    context_->set_topk(20);
    ASSERT_EQ(0, streamer_->search_impl(code.data(), code_meta, context_));
    const auto actual = context_->result();
    ASSERT_EQ(0, streamer_->search_bf_impl(code.data(), code_meta, context_));
    ASSERT_EQ(20U, actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_FLOAT_EQ(context_->result()[i].score(), actual[i].score());
    }
    context_.reset();
    ASSERT_EQ(0, streamer_->close());
    streamer_.reset();
    ASSERT_EQ(0, storage_->close());
    storage_.reset();
    test_util::RemoveTestPath(path_ + metric);
  }
}

TEST_P(HnswDuplicatesTest, GroupByUsesAllRecordIdentities) {
  ASSERT_NO_FATAL_FAILURE(open());
  for (uint32_t id = 0; id < 64; ++id) add(id, MakeVector(1));
  context_->set_group_by([](uint64_t key) { return std::to_string(key % 4); });
  context_->set_group_params(4, 3);
  context_->set_topk(12);
  ASSERT_EQ(
      0, streamer_->search_impl(MakeVector(1).data(), query_meta_, context_));
  const auto &groups = context_->group_result();
  ASSERT_EQ(4U, groups.size());
  for (const auto &group : groups) {
    EXPECT_EQ(3U, group.docs().size());
  }
}

TEST_P(HnswDuplicatesTest, UnsupportedDistanceSpacesAreRejected) {
  params_.set(PARAM_HNSW_STREAMER_USE_EXTERNAL_VECTOR, true);
  auto streamer = IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(nullptr, streamer);
  EXPECT_EQ(IndexError_Unsupported, streamer->init(meta_, params_));
  params_.set(PARAM_HNSW_STREAMER_USE_EXTERNAL_VECTOR, false);
  meta_.set_quantizer("Int8Quantizer", 0, ailego::Params{});
  EXPECT_EQ(IndexError_Unsupported, streamer->init(meta_, params_));
}

TEST_P(HnswDuplicatesTest, DefaultLayoutDoesNotHaveDuplicateLinks) {
  params_.set(PARAM_HNSW_STREAMER_DEDUPLICATE_VECTORS, false);
  ASSERT_NO_FATAL_FAILURE(open());
  add(1, MakeVector(1));
  add(2, MakeVector(1));
  EXPECT_FALSE(get_entity().deduplicate_vectors());
  EXPECT_NE(0U, get_entity().get_neighbors(0, 0).size());
  search_and_compare_scores(MakeVector(1), 2);
}

TEST_P(HnswDuplicatesTest, ContiguousReadOnlyExpansion) {
  if (std::string(GetParam()) != "MMapFileStorage") return;
  ASSERT_NO_FATAL_FAILURE(open());
  for (uint32_t id = 0; id < 64; ++id) add(id, MakeVector(id / 16));
  ASSERT_NO_FATAL_FAILURE(reopen(true));
  search_and_compare_scores(MakeVector(2), 20);
}

INSTANTIATE_TEST_SUITE_P(StorageModes, HnswDuplicatesTest,
                         testing::Values("MMapFileStorage", "BufferStorage"));

}  // namespace
}  // namespace zvec::core
