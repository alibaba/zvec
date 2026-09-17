#include <future>
#include <string>
#include <type_traits>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/memory_helper.h>
#include <algorithm/hnsw/hnsw_algorithm.h>
#include <algorithm/hnsw/hnsw_context.h>
#include <algorithm/hnsw/hnsw_entity.h>
#include <algorithm/hnsw/hnsw_params.h>
#include <algorithm/hnsw/hnsw_streamer_entity.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_streamer.h>
#include "tests/test_util.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

constexpr size_t static dim = 16;

class HnswStreamerTest : public testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;
  void hybrid_scale(std::vector<float> &dense_value,
                    std::vector<float> &sparse_value, float alpha_scale);

  static std::string dir_;
  static std::shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string HnswStreamerTest::dir_("hnsw_streamer_buffer_test_dir/");
std::shared_ptr<IndexMeta> HnswStreamerTest::index_meta_ptr_;

void HnswStreamerTest::SetUp() {
  index_meta_ptr_.reset(new (std::nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());

  zvec::test_util::RemoveTestPath(dir_);
}

void HnswStreamerTest::TearDown() {
  zvec::test_util::RemoveTestPath(dir_);
}

TEST_F(HnswStreamerTest, MaxDegreeIsNeighborCountNotSerializedBytes) {
  IndexStreamer::Stats stats;
  HnswBufferPoolStreamerEntity entity(stats);
  entity.set_l0_neighbor_cnt(192);
  entity.set_upper_neighbor_cnt(96);

  EXPECT_EQ(192U, entity.max_degree(0));
  EXPECT_EQ(96U, entity.max_degree(1));
  EXPECT_EQ(sizeof(NeighborsHeader) + 192U * sizeof(node_id_t),
            entity.neighbors_size());
  EXPECT_EQ(sizeof(NeighborsHeader) + 96U * sizeof(node_id_t),
            entity.upper_neighbors_size());
}

TEST_F(HnswStreamerTest, BufferPruningUsesOriginalProviderVectors) {
  // Four fixed level-0 nodes make both pruning paths deterministic: the
  // existing triangle has full two-neighbor lists, and the new node must
  // prune its three candidates before updating the reverse links.
  class RecordingProvider
      : public MultiPassIndexProvider<IndexMeta::DataType::DT_FP32> {
   public:
    using Base = MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>;
    using Base::Base;

    int get_vector(uint64_t key,
                   IndexStorage::MemoryBlock &block) const override {
      requested_keys.push_back(key);
      return Base::get_vector(key, block);
    }

    mutable std::vector<uint64_t> requested_keys;
  };

  MemoryLimitPool::get_instance().init(64UL * 1024UL * 1024UL);
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(Params()));
  ASSERT_EQ(0, storage->open(dir_ + "ProviderPruning", true));

  IndexStreamer::Stats stats;
  auto entity = std::make_shared<HnswBufferPoolStreamerEntity>(stats);
  entity->set_vector_size(dim * sizeof(uint16_t));
  entity->set_l0_neighbor_cnt(2);
  entity->set_upper_neighbor_cnt(2);
  entity->set_prune_cnt(2);
  entity->set_scaling_factor(5);
  entity->set_ef_construction(4);
  entity->set_use_key_info_map(true);
  ASSERT_EQ(0, entity->init(4));
  ASSERT_EQ(0, entity->open(storage, 8UL * 1024UL * 1024UL, false));

  auto provider = std::make_shared<RecordingProvider>(dim);
  const float coordinates[] = {0.0f, 2.0f, -3.0f, 0.75f};
  NumericalVector<float> original(dim);
  NumericalVector<uint16_t> stored(dim);
  for (node_id_t id = 0; id < 4; ++id) {
    for (size_t d = 0; d < dim; ++d) {
      original[d] = coordinates[id];
      stored[d] = FloatHelper::ToFP16(original[d]);
    }
    // Deliberately use keys distinct from node IDs to cover provider lookup.
    ASSERT_TRUE(provider->emplace(100 + id, original));
    node_id_t inserted;
    ASSERT_EQ(0, entity->add_vector(0, 100 + id, stored.data(), &inserted));
    ASSERT_EQ(id, inserted);
  }
  ASSERT_EQ(0, entity->update_neighbors(0, 0, {{1, 0.0f}, {2, 0.0f}}));
  ASSERT_EQ(0, entity->update_neighbors(0, 1, {{0, 0.0f}, {2, 0.0f}}));
  ASSERT_EQ(0, entity->update_neighbors(0, 2, {{0, 0.0f}, {1, 0.0f}}));
  entity->update_ep_and_level(0, 0);

  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_NE(nullptr, metric);
  ASSERT_EQ(0, metric->init(*index_meta_ptr_, Params()));
  auto ctx = std::make_unique<HnswContext>(dim, metric, entity->clone());
  ASSERT_NE(nullptr, dynamic_cast<const HnswBufferPoolStreamerEntity *>(
                         &ctx->get_entity()));
  // Standalone contexts do not inherit the streamer's nonzero scan limits.
  ctx->set_min_scan_limit(4);
  ctx->set_max_scan_limit(4);
  ASSERT_EQ(0, ctx->init(HnswContext::kStreamerContext));
  ctx->bind_dist_space(metric->distance(), metric->batch_distance(), provider,
                       dim * sizeof(float), 0);
  ctx->reset_query_raw(original.data(), *index_meta_ptr_);
  HnswAlgorithm<HnswBufferPoolStreamerEntity> algorithm(*entity);
  ASSERT_EQ(0, algorithm.add_node(3, 0, ctx.get()));
  ASSERT_FALSE(ctx->dist_calculator().error());

  // All three forward candidates, then both full reverse lists, must be
  // fetched from the FP32 provider rather than read from FP16 index records.
  const std::vector<uint64_t> expected_prune_reads = {
      100, 101, 102, 100, 103, 101, 102, 101, 103, 100, 102};
  ASSERT_GE(provider->requested_keys.size(), expected_prune_reads.size());
  EXPECT_EQ(expected_prune_reads,
            std::vector<uint64_t>(
                provider->requested_keys.end() - expected_prune_reads.size(),
                provider->requested_keys.end()));

  const std::vector<std::vector<node_id_t>> expected_neighbors = {
      {3, 2}, {3}, {0, 1}, {0, 1}};
  for (node_id_t id = 0; id < 4; ++id) {
    SCOPED_TRACE(id);
    const auto neighbors = entity->get_neighbors(0, id);
    ASSERT_EQ(expected_neighbors[id].size(), neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i) {
      EXPECT_EQ(expected_neighbors[id][i], neighbors[i]);
    }
  }

  ctx.reset();
  ASSERT_EQ(0, entity->close());
  entity.reset();
  ASSERT_EQ(0, storage->close());
}

TEST_F(HnswStreamerTest, BufferSearchDispatchReusesContextAcrossVisitModes) {
  MemoryLimitPool::get_instance().init(2 * 1024UL * 1024UL * 1024UL);
  constexpr uint32_t kDocCount = 128;
  const std::string path = dir_ + "SearchDispatch";
  Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 64);
  params.set(PARAM_HNSW_STREAMER_EF, kDocCount);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);

  auto write_streamer = IndexFactory::CreateStreamer("HnswStreamer");
  auto write_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, write_streamer);
  ASSERT_NE(nullptr, write_storage);
  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, write_storage->init(Params()));
  ASSERT_EQ(0, write_storage->open(path, true));
  ASSERT_EQ(0, write_streamer->open(write_storage));
  auto write_ctx = write_streamer->create_context();
  ASSERT_NE(nullptr, write_ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  NumericalVector<float> vector(dim);
  for (uint32_t id = 0; id < kDocCount; ++id) {
    for (size_t j = 0; j < dim; ++j) {
      vector[j] = id;
    }
    ASSERT_EQ(0, write_streamer->add_impl(id, vector.data(), qmeta, write_ctx));
  }
  ASSERT_EQ(0, write_streamer->flush(0UL));
  write_ctx.reset();
  ASSERT_EQ(0, write_streamer->close());
  write_streamer.reset();
  ASSERT_EQ(0, write_storage->close());

  auto read_streamer = IndexFactory::CreateStreamer("HnswStreamer");
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_streamer);
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, read_storage->init(Params()));
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  auto ctx = read_streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  auto *hnsw_ctx = dynamic_cast<HnswContext *>(ctx.get());
  ASSERT_NE(nullptr, hnsw_ctx);
  ASSERT_NE(nullptr, dynamic_cast<const HnswBufferPoolStreamerEntity *>(
                         &hnsw_ctx->get_entity()));
  ctx->set_topk(3);
  for (size_t j = 0; j < dim; ++j) {
    vector[j] = 64.1f;
  }

  const bool expect_block =
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX2;
  for (const auto mode :
       {VisitFilter::ByteMap, VisitFilter::BitMap, VisitFilter::BloomFilter}) {
    SCOPED_TRACE(static_cast<int>(mode));
    ASSERT_EQ(0,
              hnsw_ctx->visit_filter().init(mode, kDocCount, kDocCount, 1e-9f));
    // Use the same context for pool -> filtered heap -> pool, ensuring both
    // query-boundary dispatch and result export retain the chosen backend.
    for (int phase = 0; phase < 3; ++phase) {
      SCOPED_TRACE(phase);
      const bool filtered = phase == 1;
      if (filtered) {
        ctx->set_filter([](uint64_t key) { return key == 64 || key == 65; });
      } else {
        ctx->set_filter(nullptr);
      }
      ASSERT_EQ(0, read_streamer->search_impl(vector.data(), qmeta, ctx));
      EXPECT_EQ(mode, hnsw_ctx->visit_filter().get_mode());
      hnsw_ctx->search_heap().dispatch([&](auto &active) {
        using Heap = std::decay_t<decltype(active)>;
        EXPECT_EQ(filtered, (std::is_same_v<Heap, TopkHeap>));
        EXPECT_EQ(!filtered && expect_block, (std::is_same_v<Heap, BlockHeap>));
        EXPECT_EQ(!filtered && !expect_block,
                  (std::is_same_v<Heap, LinearPool<float>>));
      });
      const std::vector<uint64_t> expected =
          filtered ? std::vector<uint64_t>{63, 66, 62}
                   : std::vector<uint64_t>{64, 65, 63};
      const auto &results = ctx->result();
      ASSERT_EQ(expected.size(), results.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], results[i].key());
      }
    }
  }
  ctx.reset();
  ASSERT_EQ(0, read_streamer->close());
  read_streamer.reset();
  ASSERT_EQ(0, read_storage->close());
}

TEST_F(HnswStreamerTest, TestHnswSearch) {
  MemoryLimitPool::get_instance().init(2 * 1024UL * 1024UL * 1024UL);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/HnswSearch", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 10000UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();
  storage->close();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/HnswSearch", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));

  auto pool = read_storage->vec_buffer_pool();
  ASSERT_TRUE(pool);
  const size_t page_size = kVectorPageSize;

  // The complete upper graph is a small, universal search hotset and should
  // be resident at high priority immediately after open.
  auto upper_chunk = read_storage->get("HnswT4S0");
  ASSERT_TRUE(upper_chunk);
  ASSERT_GT(upper_chunk->data_size(), 0U);
  const size_t upper_first = upper_chunk->data_offset() / page_size;
  const size_t upper_last =
      (upper_chunk->data_offset() + upper_chunk->data_size() - 1) / page_size;
  for (size_t page = upper_first; page <= upper_last; ++page) {
    EXPECT_TRUE(pool->is_page_resident(page));
    EXPECT_EQ(VecBufferPool::kHighPriority,
              pool->page_table_.eviction_priority(page));
  }

  // The entry node record (vector + key + L0 adjacency) is also shared by
  // every query and must retain high priority.
  auto header_chunk = read_storage->get("HnswT1S0");
  ASSERT_TRUE(header_chunk);
  HNSWHeader hnsw_header;
  ASSERT_EQ(sizeof(hnsw_header),
            header_chunk->fetch(0, &hnsw_header, sizeof(hnsw_header)));
  ASSERT_NE(kInvalidNodeId, hnsw_header.entry_point());
  auto first_node_chunk = read_storage->get("HnswT3S0");
  ASSERT_TRUE(first_node_chunk);
  ASSERT_GT(hnsw_header.graph.node_size, 0U);
  const size_t nodes_per_chunk =
      first_node_chunk->data_size() / hnsw_header.graph.node_size;
  ASSERT_GT(nodes_per_chunk, 0U);
  const size_t entry_chunk_id = hnsw_header.entry_point() / nodes_per_chunk;
  const size_t entry_local_id = hnsw_header.entry_point() % nodes_per_chunk;
  auto entry_chunk =
      read_storage->get("HnswT3S" + std::to_string(entry_chunk_id));
  ASSERT_TRUE(entry_chunk);
  const size_t entry_offset =
      entry_local_id * static_cast<size_t>(hnsw_header.graph.node_size);
  const size_t entry_first =
      (entry_chunk->data_offset() + entry_offset) / page_size;
  const size_t entry_last = (entry_chunk->data_offset() + entry_offset +
                             hnsw_header.graph.node_size - 1) /
                            page_size;
  for (size_t page = entry_first; page <= entry_last; ++page) {
    EXPECT_TRUE(pool->is_page_resident(page));
    EXPECT_EQ(VecBufferPool::kHighPriority,
              pool->page_table_.eviction_priority(page));
  }

  size_t topk = 3;
  auto provider = read_streamer->create_provider();

  // Query execution must not repeat range-prefetch priority scans after the
  // one-time open protection above. With no eviction pressure, the number of
  // normal-priority pages therefore remains unchanged.
  size_t normal_before = 0;
  for (size_t page = 0; page < pool->page_table_.entry_num(); ++page) {
    normal_before += pool->is_page_resident(page) &&
                     pool->page_table_.eviction_priority(page) ==
                         VecBufferPool::kNormalPriority;
  }
  NumericalVector<float> hotset_query(dim);
  for (size_t j = 0; j < dim; ++j) {
    hotset_query[j] = cnt / 2;
  }
  ctx->set_topk(topk);
  ASSERT_EQ(0, read_streamer->search_impl(hotset_query.data(), qmeta, ctx));
  size_t normal_after = 0;
  for (size_t page = 0; page < pool->page_table_.entry_num(); ++page) {
    normal_after += pool->is_page_resident(page) &&
                    pool->page_table_.eviction_priority(page) ==
                        VecBufferPool::kNormalPriority;
  }
  EXPECT_EQ(normal_after, normal_before);

  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(10, result[0].key());
  ASSERT_EQ(11, result[1].key());
  ASSERT_EQ(5, result[10].key());
  ASSERT_EQ(0, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());

  ElapsedTime elapsed_time;
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  read_streamer->close();
  read_streamer.reset();
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;
}

TEST_F(HnswStreamerTest, TestHnswSearchBuffer) {
  MemoryLimitPool::get_instance().init(2 * 1024UL * 1024UL * 1024UL);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/TestHnswSearchBuffer", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 10000UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();
  storage->close();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/TestHnswSearchBuffer", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  size_t topk = 3;
  auto provider = read_streamer->create_provider();
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(10, result[0].key());
  ASSERT_EQ(11, result[1].key());
  ASSERT_EQ(5, result[10].key());
  ASSERT_EQ(0, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());

  ElapsedTime elapsed_time;
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  read_streamer->close();
  read_streamer.reset();
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;
}

TEST_F(HnswStreamerTest, TestWideMConcurrentBuildBuffer) {
  MemoryLimitPool::get_instance().init(512UL * 1024UL * 1024UL);
  auto streamer = IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(nullptr, streamer);

  Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 96U);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 96U);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 128U);
  params.set(PARAM_HNSW_STREAMER_MAX_INDEX_SIZE, 128UL * 1024UL * 1024UL);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));

  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  Params storage_params;
  ASSERT_EQ(0, storage->init(storage_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/WideMConcurrentBuildBuffer", true));
  ASSERT_EQ(0, streamer->open(storage));

  constexpr size_t kThreadCount = 8;
  constexpr size_t kVectorsPerThread = 1000;
  auto add_vectors = [&streamer](size_t first, size_t vector_count) {
    auto context = streamer->create_context();
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    NumericalVector<float> vector(dim);
    size_t added = 0;
    for (size_t i = 0; i < vector_count; ++i) {
      const size_t key = first + i;
      for (size_t j = 0; j < dim; ++j) {
        const uint32_t bits =
            static_cast<uint32_t>(key * 2654435761ULL + j * 2246822519ULL);
        vector[j] = static_cast<float>(bits) / 4294967295.0f;
      }
      added += streamer->add_impl(key, vector.data(), query_meta, context) == 0;
    }
    return added;
  };

  std::vector<std::future<size_t>> workers;
  for (size_t thread = 0; thread < kThreadCount; ++thread) {
    workers.emplace_back(std::async(std::launch::async, add_vectors,
                                    thread * kVectorsPerThread,
                                    kVectorsPerThread));
  }
  for (auto &worker : workers) {
    EXPECT_EQ(kVectorsPerThread, worker.get());
  }

  auto provider = streamer->create_provider();
  ASSERT_NE(nullptr, provider);
  EXPECT_EQ(kThreadCount * kVectorsPerThread, provider->count());

  auto search_context = streamer->create_context();
  search_context->set_topk(1);
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  NumericalVector<float> query(dim);
  for (size_t key = 0; key < kThreadCount * kVectorsPerThread; key += 157) {
    for (size_t j = 0; j < dim; ++j) {
      const uint32_t bits =
          static_cast<uint32_t>(key * 2654435761ULL + j * 2246822519ULL);
      query[j] = static_cast<float>(bits) / 4294967295.0f;
    }
    ASSERT_EQ(0,
              streamer->search_impl(query.data(), query_meta, search_context));
    const auto &result = search_context->result();
    ASSERT_EQ(1U, result.size());
    EXPECT_EQ(key, result[0].key());
  }

  ASSERT_EQ(0, streamer->flush(0));
  ASSERT_EQ(0, streamer->close());
}

TEST_F(HnswStreamerTest, TestHnswSearchBufferMMap) {
  MemoryLimitPool::get_instance().init(2 * 1024UL * 1024UL * 1024UL);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/TestHnswSearchBufferMMap", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 10000UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();
  storage->close();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0,
            read_storage->open(dir_ + "Test/TestHnswSearchBufferMMap", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  size_t topk = 3;
  auto provider = read_streamer->create_provider();
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(10, result[0].key());
  ASSERT_EQ(11, result[1].key());
  ASSERT_EQ(5, result[10].key());
  ASSERT_EQ(0, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());

  ElapsedTime elapsed_time;
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  read_streamer->close();
  read_streamer.reset();
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;
}

TEST_F(HnswStreamerTest, TestHnswSearchMMap) {
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/HnswSearchMMap", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 10000UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();
  storage->close();

  ElapsedTime elapsed_time;
  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/HnswSearchMMap", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  size_t topk = 3;
  auto provider = read_streamer->create_provider();
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(10, result[0].key());
  ASSERT_EQ(11, result[1].key());
  ASSERT_EQ(5, result[10].key());
  ASSERT_EQ(0, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());

  read_streamer->close();
  read_streamer.reset();
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
