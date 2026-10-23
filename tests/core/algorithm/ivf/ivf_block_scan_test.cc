// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <atomic>
#include <limits>
#include <map>
#include <thread>
#include <gtest/gtest.h>
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_factory.h>
#include "ivf_dumper.h"
#include "ivf_entity.h"

namespace zvec::core {
namespace {
std::atomic<size_t> refinements{0};
// An exact first-coordinate L2 bound makes pruning independently testable on
// every architecture, including hosts without the RaBitQ SIMD implementation.
class TestBlockQuantizer : public turbo::Fp32Quantizer {
 public:
  size_t scan_block_size() const override {
    return 32 * sizeof(float);
  }
  int pack_scan_block(const void *rows, size_t n, size_t stride,
                      void *out) const override {
    std::memset(out, 0, scan_block_size());
    for (size_t i = 0; i < n; ++i)
      std::memcpy(static_cast<char *>(out) + i * sizeof(float),
                  static_cast<const char *>(rows) + i * stride, sizeof(float));
    return 0;
  }
  class Scanner : public turbo::BlockScanner {
   public:
    explicit Scanner(const turbo::DistanceImpl &d) : distance_(d) {}
    int estimate(const void *block, size_t n,
                 turbo::DistanceEstimate *out) const override {
      float query;
      std::memcpy(&query, distance_.query_storage().data(), sizeof(query));
      for (size_t i = 0; i < n; ++i) {
        float value;
        std::memcpy(&value,
                    static_cast<const char *>(block) + i * sizeof(float),
                    sizeof(value));
        float bound = (value - query) * (value - query);
        out[i] = {bound, bound};
      }
      return 0;
    }
    float refine(const void *row,
                 const turbo::DistanceEstimate &) const override {
      ++refinements;
      return distance_(row);
    }

   private:
    turbo::DistanceImpl distance_;
  };
  std::unique_ptr<turbo::BlockScanner> block_scanner(
      const turbo::DistanceImpl &d) const override {
    return std::make_unique<Scanner>(d);
  }
};
INDEX_FACTORY_REGISTER_QUANTIZER(TestBlockQuantizer);

constexpr size_t kDim = 73;
constexpr size_t kCount = 1057;

void WriteIndex(const std::string &path, size_t block_size, bool packed,
                bool malformed = false, bool deleted = false) {
  IndexMeta meta(IndexMeta::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  meta.set_major_order(IndexMeta::MO_ROW);
  meta.set_quantizer("TestBlockQuantizer", 0, ailego::Params{});
  auto quantizer = std::make_shared<TestBlockQuantizer>();
  ASSERT_EQ(0, quantizer->init(meta, ailego::Params{}));
  auto file = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, file->create(path));
  IVFDumper dumper(quantizer->meta(), file, 4, block_size);
  if (packed) dumper.enable_block_scan(quantizer);
  std::vector<float> row(kDim, 0);
  for (size_t i = 0; i < kCount; ++i) {
    row[0] = i;
    // Two empty lists and two different partial posting blocks.
    const uint64_t key = deleted && (i < 69 || i % 2 == 0) ? kInvalidKey : i;
    ASSERT_EQ(0, dumper.dump_inverted_vector(i < 37 ? 1 : 3, key, row.data()));
  }
  ASSERT_EQ(0, dumper.dump_inverted_vector_finished());
  ASSERT_EQ(0, dumper.dump_turbo_quantizer(quantizer));
  if (malformed) {
    uint32_t invalid = 0;
    ASSERT_EQ(sizeof(invalid), file->write(&invalid, sizeof(invalid)));
    ASSERT_EQ(0, file->append(IVF_TURBO_SCAN_SEG_ID, sizeof(invalid), 0, 0));
  }
  ASSERT_EQ(0, IndexHelper::SerializeToDumper(meta, file.get()));
  ASSERT_EQ(0, file->close());
}

TEST(IVFBlockScan, StorageFallbackFiltersGroupsAndQueryIsolation) {
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(
                   ailego::VecBufferPool::metadata_bytes_for_page_count(1) +
                   8 * ailego::kVectorPageSize));
  for (size_t block_size : {size_t(1), size_t(8), size_t(32)}) {
    for (bool packed : {false, true}) {
      const std::string path = "ivf_block_scan_test.index";
      WriteIndex(path, block_size, packed);
      for (const char *storage_type :
           {"FileReadStorage", "MMapFileReadStorage", "BufferReadStorage"}) {
        SCOPED_TRACE(testing::Message()
                     << block_size << ':' << packed << ':' << storage_type);
        const auto pool_before =
            ailego::MemoryLimitPool::get_instance().stats();
        auto storage = IndexFactory::CreateStorage(storage_type);
        ailego::Params storage_params;
        storage_params.set("proxima.buffer.read_storage.warmup_mode", "none");
        ASSERT_EQ(0, storage->init(storage_params));
        ASSERT_EQ(0, storage->open(path, false));
        if (std::strcmp(storage_type, "BufferReadStorage") == 0) {
          ASSERT_NE(nullptr, storage->vec_buffer_pool());
          ASSERT_TRUE(storage->vec_buffer_pool()->cache_enabled());
        }
        IVFEntity entity;
        ASSERT_EQ(0, entity.load(storage));
        std::vector<float> query(kDim, 0);
        IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDim);
        ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
        for (bool filtered : {false, true}) {
          ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
          IndexDocumentHeap heap(5);
          IndexContext::Stats stats;
          IndexFilter filter;
          filter.set(
              [filtered](uint64_t key) { return filtered && key % 2 == 0; });
          refinements = 0;
          if (filtered)
            ASSERT_EQ(0, entity.search(query.data(), filter, &heap, &stats));
          else
            ASSERT_EQ(0, entity.search(query.data(), &heap, &stats));
          ASSERT_EQ(5u, heap.size());
          for (const auto &doc : heap) {
            EXPECT_LT(doc.key(), filtered ? 10u : 5u);
            EXPECT_FLOAT_EQ(doc.key() * doc.key(), doc.score());
            if (filtered) EXPECT_EQ(1u, doc.key() % 2);
          }
          if (packed) {
            EXPECT_EQ(5u, refinements.load());
            EXPECT_GT(stats.dist_calced_count(), 0u);
          }
        }
        if (packed) {
          // An all-rejected window must not estimate or refine any candidates.
          ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
          IndexFilter reject_all;
          reject_all.set([](uint64_t) { return true; });
          IndexDocumentHeap heap(5);
          IndexContext::Stats stats;
          refinements = 0;
          ASSERT_EQ(0, entity.search(query.data(), reject_all, &heap, &stats));
          EXPECT_EQ(kCount, stats.filtered_count());
          EXPECT_EQ(0u, stats.dist_calced_count());
          EXPECT_EQ(0u, refinements.load());
          // Same-index A/B reference: identical top-k with pruning disabled.
          entity.set_packed_scan_pruning(false);
          ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
          ASSERT_EQ(0, entity.search(query.data(), &heap, &stats));
          EXPECT_EQ(kCount, refinements.load());
          EXPECT_EQ(kCount, stats.dist_calced_count());
          ASSERT_EQ(5u, heap.size());
          for (const auto &doc : heap) {
            EXPECT_LT(doc.key(), 5u);
            EXPECT_FLOAT_EQ(doc.key() * doc.key(), doc.score());
          }
          entity.set_packed_scan_pruning(true);
          ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
        }
        // A global top-k threshold would incorrectly suppress later groups.
        std::map<uint64_t, IndexDocumentHeap> groups;
        IVFEntity::CandidateVisitor visitor{
            [&](uint64_t key, float score) {
              groups.try_emplace(key % 4, 2).first->second.emplace(key, score);
            },
            [&](uint64_t key) {
              auto it = groups.find(key % 4);
              return it != groups.end() && it->second.full()
                         ? it->second.begin()->score()
                         : std::numeric_limits<float>::infinity();
            }};
        IndexDocumentHeap unused(1);
        IndexContext::Stats stats;
        ASSERT_EQ(0, entity.search(query.data(), &unused, &stats, visitor));
        ASSERT_EQ(4u, groups.size());
        for (const auto &entry : groups) {
          ASSERT_EQ(2u, entry.second.size());
          for (const auto &doc : entry.second) EXPECT_LT(doc.key(), 8u);
        }
        // Without a bound callback, grouped visitors must receive every row.
        size_t visited = 0;
        ASSERT_EQ(0, entity.search(query.data(), &unused, &stats,
                                   {[&](uint64_t, float) { ++visited; }, {}}));
        EXPECT_EQ(kCount, visited);
        // Clone concurrently and rebind twice, including a tail-block query.
        std::vector<std::thread> threads;
        for (size_t t = 0; t < 3; ++t) {
          auto clone = entity.clone();
          ASSERT_NE(nullptr, clone);
          threads.emplace_back([clone, t, qmeta] {
            for (size_t key : {t, kCount - 1 - t}) {
              std::vector<float> q(kDim, 0);
              q[0] = key;
              EXPECT_EQ(0, clone->bind_query(q.data(), qmeta));
              IndexDocumentHeap heap(1);
              IndexContext::Stats stats;
              EXPECT_EQ(0, clone->search(q.data(), &heap, &stats));
              ASSERT_EQ(1u, heap.size());
              EXPECT_EQ(key, heap.begin()->key());
              EXPECT_FLOAT_EQ(0, heap.begin()->score());
            }
          });
        }
        for (auto &thread : threads) thread.join();
        if (std::strcmp(storage_type, "BufferReadStorage") == 0) {
          // Background eviction may reclaim every page before this check.
          // Cumulative acquisitions still prove the cache path was exercised.
          const auto pool_after =
              ailego::MemoryLimitPool::get_instance().stats();
          EXPECT_GT(
              pool_after.alloc_from_freelist + pool_after.alloc_from_slab,
              pool_before.alloc_from_freelist + pool_before.alloc_from_slab);
        }
      }
      ailego::File::RemovePath(path);
    }
  }
}

TEST(IVFBlockScan, DeletedKeysInMappedUnfilteredScan) {
  const std::string path = "ivf_block_scan_deleted_test.index";
  // Includes an entirely deleted list, an entirely deleted block, live and
  // deleted lanes in the same block, and a partial final block.
  WriteIndex(path, 32, true, false, true);
  for (const char *type : {"FileReadStorage", "MMapFileReadStorage"}) {
    SCOPED_TRACE(type);
    auto storage = IndexFactory::CreateStorage(type);
    ASSERT_EQ(0, storage->open(path, false));
    IVFEntity entity;
    ASSERT_EQ(0, entity.load(storage));
    std::vector<float> query(kDim, 0);
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDim);
    for (bool pruning : {false, true}) {
      SCOPED_TRACE(pruning);
      entity.set_packed_scan_pruning(pruning);
      ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
      IndexDocumentHeap heap(5);
      IndexContext::Stats stats;
      refinements = 0;
      ASSERT_EQ(0, entity.search(query.data(), &heap, &stats));
      heap.sort();
      ASSERT_EQ(5u, heap.size());
      size_t rank = 0;
      for (const auto &doc : heap) {
        const uint64_t key = 69 + 2 * rank++;
        EXPECT_EQ(key, doc.key());
        EXPECT_FLOAT_EQ(key * key, doc.score());
      }
      EXPECT_EQ(988u, stats.dist_calced_count());
      EXPECT_EQ(pruning ? 5u : 494u, refinements.load());
    }
  }
  ailego::File::RemovePath(path);
}

TEST(IVFBlockScan, MappedPruningRetainsEqualAndUnorderedBounds) {
  const std::string path = "ivf_block_scan_bound_test.index";
  WriteIndex(path, 32, true);
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  IVFEntity entity;
  ASSERT_EQ(0, entity.load(storage));
  for (float value : {0.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    std::vector<float> query(kDim, 0);
    query[0] = value;
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDim);
    ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
    IndexDocumentHeap heap(1);
    if (value == 0) {
      heap.emplace(9999, 0.0f, 0);
    }
    IndexContext::Stats stats;
    refinements = 0;
    ASSERT_EQ(0, entity.search(query.data(), &heap, &stats));
    // A lower bound equal to the threshold must still be refined, including
    // infinity == infinity. An unordered comparison must not prune NaNs.
    EXPECT_EQ(value == 0 ? 1u : kCount, refinements.load());
  }
  storage->close();
  ailego::File::RemovePath(path);
}

TEST(IVFBlockScan, RejectsMalformedAuxiliarySegment) {
  const std::string path = "ivf_block_scan_bad.index";
  WriteIndex(path, 32, false, true);
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  ASSERT_EQ(0, storage->open(path, false));
  IVFEntity entity;
  EXPECT_EQ(IndexError_InvalidFormat, entity.load(storage));
  storage->close();
  ailego::File::RemovePath(path);
}
}  // namespace
}  // namespace zvec::core
