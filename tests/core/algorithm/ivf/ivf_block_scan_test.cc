// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <atomic>
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
                bool malformed = false) {
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
    ASSERT_EQ(0, dumper.dump_inverted_vector(i < 37 ? 1 : 3, i, row.data()));
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
            const auto &scan = entity.packed_scan_stats();
            EXPECT_EQ(5u, scan.refined_count);
            EXPECT_GT(scan.coarse_count, 0u);
            if (std::strcmp(storage_type, "MMapFileReadStorage") == 0) {
              EXPECT_EQ(0u, scan.key_reads);
              EXPECT_EQ(0u, scan.coarse_reads);
              EXPECT_EQ(0u, scan.row_reads);
              EXPECT_EQ(0u, scan.coarse_copy_bytes);
            } else {
              const size_t blocks1 = (37 + block_size - 1) / block_size;
              const size_t blocks3 =
                  (kCount - 37 + block_size - 1) / block_size;
              const size_t windows = (blocks1 + 31) / 32 + (blocks3 + 31) / 32;
              EXPECT_EQ(windows, scan.key_reads);
              EXPECT_LE(scan.coarse_reads, windows);
              EXPECT_GT(scan.coarse_reads, 0u);
              EXPECT_LT(scan.coarse_reads, blocks1 + blocks3);
              EXPECT_LE(scan.coarse_copy_bytes,
                        (blocks1 + blocks3) * 32 * sizeof(float));
            }
          }
        }
        if (packed) {
          // An all-rejected window must not fetch coarse codes or full rows.
          ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
          IndexFilter reject_all;
          reject_all.set([](uint64_t) { return true; });
          IndexDocumentHeap heap(5);
          IndexContext::Stats stats;
          ASSERT_EQ(0, entity.search(query.data(), reject_all, &heap, &stats));
          EXPECT_EQ(kCount, stats.filtered_count());
          EXPECT_EQ(0u, entity.packed_scan_stats().coarse_count);
          EXPECT_EQ(0u, entity.packed_scan_stats().coarse_reads);
          EXPECT_EQ(0u, entity.packed_scan_stats().row_reads);
          EXPECT_EQ(0u, entity.packed_scan_stats().refined_count);
          // Same-index A/B reference: identical top-k with pruning disabled.
          entity.set_packed_scan_pruning(false);
          ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
          ASSERT_EQ(0, entity.search(query.data(), &heap, &stats));
          EXPECT_EQ(kCount, entity.packed_scan_stats().refined_count);
          EXPECT_EQ(kCount, entity.packed_scan_stats().coarse_count);
          ASSERT_EQ(5u, heap.size());
          for (const auto &doc : heap) {
            EXPECT_LT(doc.key(), 5u);
            EXPECT_FLOAT_EQ(doc.key() * doc.key(), doc.score());
          }
          entity.set_packed_scan_pruning(true);
          ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
          EXPECT_EQ(0u, entity.packed_scan_stats().refined_count);
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
          EXPECT_EQ(0u, clone->packed_scan_stats().refined_count);
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
        if (std::strcmp(storage_type, "BufferReadStorage") == 0)
          EXPECT_GT(ailego::MemoryLimitPool::get_instance().stats().page_used,
                    0u);
      }
      ailego::File::RemovePath(path);
    }
  }
}

TEST(IVFBlockScan, SegmentTimingDoesNotChangeResultsOrLeakAcrossQueries) {
  const std::string path = "ivf_block_scan_timing_test.index";
  WriteIndex(path, 32, true);
  for (const char *type : {"FileReadStorage", "MMapFileReadStorage"}) {
    SCOPED_TRACE(type);
    auto storage = IndexFactory::CreateStorage(type);
    ASSERT_EQ(0, storage->open(path, false));
    IVFEntity entity;
    ASSERT_EQ(0, entity.load(storage));
    std::vector<float> query(kDim, 0);
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDim);
    std::map<uint64_t, float> reference;
    for (bool enabled : {false, true, false}) {
      entity.set_packed_scan_timing(enabled);
      ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
      IndexDocumentHeap heap(5);
      IndexContext::Stats stats;
      ASSERT_EQ(0, entity.search(query.data(), &heap, &stats));
      entity.finish_segment_timing();
      const auto scan = entity.packed_scan_stats();
      EXPECT_EQ(kCount, scan.coarse_count);
      EXPECT_EQ(5u, scan.refined_count);
      EXPECT_EQ(enabled, scan.prep_ns > 0);
      EXPECT_EQ(enabled, scan.coarse_ns > 0);
      EXPECT_EQ(enabled, scan.cand_ns > 0);
      EXPECT_EQ(enabled, scan.post_scan_ns > 0);
      std::map<uint64_t, float> actual;
      for (const auto &doc : heap) actual.emplace(doc.key(), doc.score());
      if (reference.empty()) reference = actual;
      EXPECT_EQ(reference, actual);
      // Publishing twice must not double count a query.
      entity.finish_segment_timing();
      EXPECT_EQ(scan.post_scan_ns, entity.packed_scan_stats().post_scan_ns);
    }
    entity.set_packed_scan_timing(true);
    ASSERT_EQ(0, entity.bind_query(query.data(), qmeta));
    IndexDocumentHeap heap(1);
    heap.emplace(0, 0.0f, 0);
    IndexContext::Stats stats;
    uint32_t scanned = 0;
    // Every candidate in list 3 is rejected by the existing best score.
    // FileReadStorage must count the post-scan work even on the early exit.
    ASSERT_EQ(0, entity.search(3, query.data(), &scanned, &heap, &stats));
    entity.finish_segment_timing();
    EXPECT_EQ(kCount - 37, scanned);
    EXPECT_EQ(0u, entity.packed_scan_stats().refined_count);
    EXPECT_GT(entity.packed_scan_stats().post_scan_ns, 0.0);
  }
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
