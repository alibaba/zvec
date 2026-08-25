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

#include "diskann_searcher.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <set>
#include <thread>
#include <unordered_set>
#include <ailego/math/distance.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_holder.h"
#include "diskann_index_provider.h"
#include "diskann_params.h"
#include "diskann_streamer.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

class DiskAnnCacheTestPeer {
 public:
  static std::shared_ptr<AlignedFileReader> reader(DiskAnnSearcher *searcher) {
    return searcher->diskann_indexer_->reader_;
  }

  static void set_reader(DiskAnnSearcher *searcher,
                         std::shared_ptr<AlignedFileReader> reader) {
    searcher->diskann_indexer_->reader_ = std::move(reader);
  }

  static int configure_cache(DiskAnnSearcher *searcher,
                             uint32_t cache_node_num) {
    return searcher->diskann_indexer_->configure_cache(cache_node_num);
  }

  static size_t coordinate_cache_size(const DiskAnnSearcher *searcher) {
    return searcher->diskann_indexer_->coord_cache_.size();
  }

  static size_t neighbor_cache_size(const DiskAnnSearcher *searcher) {
    return searcher->diskann_indexer_->neighbor_cache_.size();
  }
};

class DiskAnnProviderTestPeer {
 public:
  static DiskAnnContext *fetch_context(IndexProvider *provider) {
    auto *diskann_provider = dynamic_cast<DiskAnnIndexProvider *>(provider);
    return diskann_provider == nullptr
               ? nullptr
               : dynamic_cast<DiskAnnContext *>(
                     diskann_provider->fetch_context_.get());
  }

  static DiskAnnContext *iterator_context(IndexProvider::Iterator *iterator) {
    auto *diskann_iterator =
        dynamic_cast<DiskAnnIndexProvider::Iterator *>(iterator);
    return diskann_iterator == nullptr ? nullptr
                                       : dynamic_cast<DiskAnnContext *>(
                                             diskann_iterator->context_.get());
  }
};

}  // namespace core
}  // namespace zvec

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

constexpr size_t static dim = 64;

namespace {

class CountingAlignedFileReader final : public AlignedFileReader {
 public:
  explicit CountingAlignedFileReader(std::shared_ptr<AlignedFileReader> reader)
      : reader_(std::move(reader)) {}

  void open(const std::string &fname) override {
    reader_->open(fname);
  }

  void close() override {
    reader_->close();
  }

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false) override {
    requested_reads_ += read_reqs.size();
    return reader_->read(read_reqs, ctx, async);
  }

  int submit(PendingBatch &batch, std::vector<AlignedRead> &read_reqs,
             IOContext &ctx) override {
    return reader_->submit(batch, read_reqs, ctx);
  }

  int get_completed(PendingBatch &batch, IOContext &ctx, int min_completed,
                    std::vector<uint32_t> &completed_indices) override {
    return reader_->get_completed(batch, ctx, min_completed, completed_indices);
  }

  void release_io_ctx(IOContext &ctx) override {
    ++release_count_;
    reader_->release_io_ctx(ctx);
  }

  size_t requested_reads() const {
    return requested_reads_;
  }

  size_t release_count() const {
    return release_count_;
  }

 private:
  std::shared_ptr<AlignedFileReader> reader_;
  size_t requested_reads_{0};
  size_t release_count_{0};
};

class FailingAlignedFileReader final : public AlignedFileReader {
 public:
  explicit FailingAlignedFileReader(std::shared_ptr<AlignedFileReader> reader)
      : reader_(std::move(reader)) {}

  void open(const std::string &fname) override {
    reader_->open(fname);
  }

  void close() override {
    reader_->close();
  }

  int read(std::vector<AlignedRead> & /*read_reqs*/, IOContext & /*ctx*/,
           bool /*async*/ = false) override {
    return IndexError_ReadData;
  }

  int submit(PendingBatch & /*batch*/, std::vector<AlignedRead> & /*read_reqs*/,
             IOContext & /*ctx*/) override {
    return IndexError_ReadData;
  }

  int get_completed(PendingBatch & /*batch*/, IOContext & /*ctx*/,
                    int /*min_completed*/,
                    std::vector<uint32_t> & /*completed_indices*/) override {
    return IndexError_ReadData;
  }

  void release_io_ctx(IOContext &ctx) override {
    reader_->release_io_ctx(ctx);
  }

 private:
  std::shared_ptr<AlignedFileReader> reader_;
};

class CorruptibleDiskAnnSearcherEntity final : public DiskAnnSearcherEntity {
 public:
  void set_node_layout(uint64_t max_node_size, uint64_t node_per_sector) {
    meta_header_.max_node_size = max_node_size;
    meta_header_.node_per_sector = node_per_sector;
  }
};

size_t expected_fetch_buffer_size(const DiskAnnContext &context) {
  const auto &entity = context.get_entity();
  const uint64_t sector_num_per_node =
      entity.node_per_sector() > 0
          ? 1
          : DiskAnnUtil::div_round_up(entity.max_node_size(),
                                      DiskAnnUtil::kSectorSize);
  return static_cast<size_t>(sector_num_per_node) * DiskAnnUtil::kSectorSize;
}

}  // namespace

class DiskAnnSearcherTest : public testing::Test {
 protected:
  void SetUp(void) override;
  void TearDown(void) override;

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string DiskAnnSearcherTest::_dir("DiskAnnSearcherTest/");
shared_ptr<IndexMeta> DiskAnnSearcherTest::_index_meta_ptr;

void DiskAnnSearcherTest::SetUp(void) {
  LoggerBroker::SetLevel(Logger::LEVEL_INFO);

  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, Params());
}

void DiskAnnSearcherTest::TearDown(void) {
  std::filesystem::remove_all(_dir);
}

TEST_F(DiskAnnSearcherTest, TestGeneral) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // A fetch context now allocates only the sectors required for one node, so
  // the on-disk packing fields must be mutually consistent. Otherwise a
  // forged nodes-per-sector value can put the computed node offset beyond
  // that exact buffer even though the sector read itself fits.
  {
    auto malformed_storage = IndexFactory::CreateStorage("FileReadStorage");
    ASSERT_NE(malformed_storage, nullptr);
    ASSERT_EQ(0, malformed_storage->open(path, false));
    CorruptibleDiskAnnSearcherEntity malformed_entity;
    ASSERT_EQ(0, malformed_entity.load(*_index_meta_ptr, malformed_storage));
    const uint64_t max_node_size = malformed_entity.max_node_size();
    const uint64_t expected_node_per_sector =
        max_node_size <= DiskAnnUtil::kSectorSize
            ? DiskAnnUtil::kSectorSize / max_node_size
            : 0;
    malformed_entity.set_node_layout(max_node_size,
                                     expected_node_per_sector + 1);
    DiskAnnIndexer malformed_indexer(*_index_meta_ptr);
    EXPECT_EQ(IndexError_InvalidFormat,
              malformed_indexer.init(malformed_entity));

    // Keep the packing formula self-consistent, but remove the space required
    // for the declared adjacency list. This must be rejected independently of
    // the nodes-per-sector consistency check above.
    ASSERT_GT(malformed_entity.max_degree(), 0U);
    const uint64_t undersized_node =
        _index_meta_ptr->element_size() + sizeof(uint32_t);
    malformed_entity.set_node_layout(
        undersized_node, DiskAnnUtil::kSectorSize / undersized_node);
    DiskAnnIndexer undersized_indexer(*_index_meta_ptr);
    EXPECT_EQ(IndexError_InvalidFormat,
              undersized_indexer.init(malformed_entity));
  }

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);
  search_params.set("zvec.diskann.searcher.cache_node_num", 0);

  ASSERT_EQ(0, searcher->init(search_params));

#if defined(_WIN32) || defined(_WIN64)
  // Independent FileReadStorage segments can outlive the storage and keep
  // buffered handles open.  Windows DiskAnn must reject that configuration
  // before opening its unbuffered IOCP handles.
  auto independent_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(independent_storage, nullptr);
  Params independent_storage_params;
  independent_storage_params.set("proxima.file.read_storage.alone_file_handle",
                                 true);
  ASSERT_EQ(0, independent_storage->init(independent_storage_params));
  ASSERT_EQ(0, independent_storage->open(path, false));
  auto retained_independent_segment =
      independent_storage->get(DiskAnnEntity::kDiskAnnVectorSegmentId);
  ASSERT_NE(retained_independent_segment, nullptr);
  ASSERT_EQ(nullptr, independent_storage->file());
  EXPECT_EQ(IndexError_InvalidArgument,
            searcher->load(independent_storage, IndexMetric::Pointer()));
  retained_independent_segment.reset();
  independent_storage.reset();
#endif

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  auto retained_cached_file = storage->file();
  ASSERT_NE(retained_cached_file, nullptr);
  ASSERT_TRUE(retained_cached_file->is_valid());
  auto retained_segment = storage->get(DiskAnnEntity::kDiskAnnVectorSegmentId);
  ASSERT_NE(retained_segment, nullptr);
  std::weak_ptr<zvec::ailego::File> searcher_cached_file = retained_cached_file;
#if defined(_WIN32) || defined(_WIN64)
  // Keeping an ordinary buffered alias beside DiskAnn's unbuffered handles
  // causes a severe random-read regression on Windows. Reject the load without
  // invalidating either caller-owned alias. Releasing those aliases allows the
  // same, still-open storage to be retried.
  EXPECT_EQ(IndexError_InvalidArgument,
            searcher->load(storage, IndexMetric::Pointer()));
  EXPECT_TRUE(retained_cached_file->is_valid());
  uint8_t retained_file_byte = 0;
  EXPECT_EQ(1U, retained_segment->fetch(0, &retained_file_byte, 1));
  retained_cached_file.reset();
  retained_segment.reset();
  EXPECT_FALSE(searcher_cached_file.expired());
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  EXPECT_TRUE(searcher_cached_file.expired());
#else
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  // DiskAnn owns an independent descriptor. Loading must not close or enable
  // direct I/O on the File shared by caller-owned FileReadStorage segments.
  EXPECT_TRUE(retained_cached_file->is_valid());
  uint8_t retained_file_byte = 0;
  EXPECT_EQ(1U, retained_segment->fetch(0, &retained_file_byte, 1));
  retained_cached_file.reset();
  EXPECT_FALSE(searcher_cached_file.expired());
  retained_segment.reset();
  EXPECT_TRUE(searcher_cached_file.expired());
#endif
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();

  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 200;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  auto *diskann_searcher = dynamic_cast<DiskAnnSearcher *>(searcher.get());
  ASSERT_NE(diskann_searcher, nullptr);
  auto batch_counting_reader = std::make_shared<CountingAlignedFileReader>(
      DiskAnnCacheTestPeer::reader(diskann_searcher));
  DiskAnnCacheTestPeer::set_reader(diskann_searcher, batch_counting_reader);

  // A public count>1 call is one I/O lease. Its individual queries may issue
  // many batches, but the pooled context must be released exactly once when
  // the complete public operation returns.
  constexpr uint32_t kBatchQueryCount = 3;
  std::vector<float> batch_queries(kBatchQueryCount * dim);
  for (uint32_t query_index = 0; query_index < kBatchQueryCount;
       ++query_index) {
    std::fill(batch_queries.begin() + query_index * dim,
              batch_queries.begin() + (query_index + 1) * dim,
              static_cast<float>(query_index) + 0.1f);
  }

  size_t release_count = batch_counting_reader->release_count();
  ASSERT_EQ(0, searcher->search_impl(batch_queries.data(), qmeta,
                                     kBatchQueryCount, knnCtx));
  EXPECT_EQ(release_count + 1, batch_counting_reader->release_count());

  release_count = batch_counting_reader->release_count();
  ASSERT_EQ(0, searcher->search_bf_impl(batch_queries.data(), qmeta,
                                        kBatchQueryCount, linearCtx));
  EXPECT_EQ(release_count + 1, batch_counting_reader->release_count());

  std::vector<std::vector<uint64_t>> batch_p_keys(kBatchQueryCount,
                                                  {0, 1, 2, 3, 4, 5, 6, 7});
  release_count = batch_counting_reader->release_count();
  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
                   batch_queries.data(), batch_p_keys, qmeta, kBatchQueryCount,
                   linearByPKeysCtx));
  EXPECT_EQ(release_count + 1, batch_counting_reader->release_count());

  // do linear search test
  {
    float query[dim];
    for (size_t i = 0; i < dim; ++i) {
      query[i] = 3.1f;
    }
    ASSERT_EQ(0, searcher->search_bf_impl(query, qmeta, linearCtx));
    auto &linearResult = linearCtx->result();
    ASSERT_EQ(3UL, linearResult[0].key());
    ASSERT_EQ(4UL, linearResult[1].key());
    ASSERT_EQ(2UL, linearResult[2].key());
    ASSERT_EQ(5UL, linearResult[3].key());
    ASSERT_EQ(1UL, linearResult[4].key());
    ASSERT_EQ(6UL, linearResult[5].key());
    ASSERT_EQ(0UL, linearResult[6].key());
    ASSERT_EQ(7UL, linearResult[7].key());
    for (size_t i = 8; i < topk; ++i) {
      ASSERT_EQ(i, linearResult[i].key());
    }
  }

  // do linear search by p_keys test
  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0] = {8, 9, 10, 11, 3, 2, 1, 0};
  {
    float query[dim];
    for (size_t i = 0; i < dim; ++i) {
      query[i] = 3.1f;
    }

    ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(query, p_keys, qmeta,
                                                    linearByPKeysCtx));
    auto &linearByPKeysResult = linearByPKeysCtx->result();
    ASSERT_EQ(8, linearByPKeysResult.size());
    ASSERT_EQ(3UL, linearByPKeysResult[0].key());
    ASSERT_EQ(2UL, linearByPKeysResult[1].key());
    ASSERT_EQ(1UL, linearByPKeysResult[2].key());
    ASSERT_EQ(0UL, linearByPKeysResult[3].key());
    ASSERT_EQ(8UL, linearByPKeysResult[4].key());
    ASSERT_EQ(9UL, linearByPKeysResult[5].key());
    ASSERT_EQ(10UL, linearByPKeysResult[6].key());
    ASSERT_EQ(11UL, linearByPKeysResult[7].key());
  }

  size_t step = 500;
  for (size_t i = 0; i < doc_cnt; i += step) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));

    auto &knnResult = knnCtx->result();
    // TODO: check
    topk1Hits += i == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    for (size_t k = 0; k < topk; ++k) {
      totalCnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linearResult[j].key() == knnResult[k].key()) {
          totalHits++;
          break;
        }
      }
    }
  }

  float recall = totalHits * step * step * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / doc_cnt;

  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.80f);

  // A context created by the streamer must carry the streamer's magic so it
  // can be reused instead of being recreated on every search.
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  ASSERT_EQ(0, streamer->init(*_index_meta_ptr, search_params));

  auto streamer_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, streamer_storage->open(path, false));
  std::weak_ptr<zvec::ailego::File> streamer_cached_file =
      streamer_storage->file();
  ASSERT_FALSE(streamer_cached_file.expired());
  ASSERT_EQ(0, streamer->open(streamer_storage));
  EXPECT_TRUE(streamer_cached_file.expired());

  auto streamer_ctx = streamer->create_context();
  ASSERT_NE(streamer_ctx, nullptr);
  streamer_ctx->set_topk(topk);
  auto *original_ctx = streamer_ctx.get();

  ASSERT_EQ(0, streamer->search_impl(batch_queries.data(), qmeta,
                                     kBatchQueryCount, streamer_ctx));
  for (uint32_t i = 0; i < kBatchQueryCount; ++i) {
    EXPECT_FALSE(streamer_ctx->result(i).empty());
  }

  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, streamer_ctx));
  EXPECT_EQ(original_ctx, streamer_ctx.get());
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, streamer_ctx));
  EXPECT_EQ(original_ctx, streamer_ctx.get());

  // Concurrent fetches must return independent, correctly owned blocks.
  std::atomic<bool> fetch_ok{true};
  std::vector<std::thread> fetch_threads;
  for (uint32_t thread_id = 0; thread_id < 4; ++thread_id) {
    fetch_threads.emplace_back([&, thread_id]() {
      for (uint32_t i = thread_id; i < 40; i += 4) {
        IndexStorage::MemoryBlock block;
        if (streamer->get_vector_by_id(i, block) != 0 ||
            block.data() == nullptr ||
            *static_cast<const float *>(block.data()) !=
                static_cast<float>(i)) {
          fetch_ok.store(false);
          return;
        }
      }
    });
  }
  for (auto &thread : fetch_threads) {
    thread.join();
  }
  EXPECT_TRUE(fetch_ok.load());

  // Query metadata controls both the copy size and the batch stride, so a
  // mismatch must be rejected before either searcher touches the input.
  IndexQueryMeta wrong_type(IndexMeta::DataType::DT_FP16, dim);
  EXPECT_EQ(IndexError_Mismatch,
            searcher->search_impl(vec.data(), wrong_type, knnCtx));
  EXPECT_EQ(IndexError_Mismatch,
            streamer->search_impl(vec.data(), wrong_type, streamer_ctx));
  IndexQueryMeta wrong_dimension(IndexMeta::DataType::DT_FP32, dim - 1);
  EXPECT_EQ(IndexError_Mismatch,
            searcher->search_bf_impl(vec.data(), wrong_dimension, linearCtx));

  // Group parameters without a grouping callback are invalid instead of a
  // successful search with an empty result.
  auto invalid_group_ctx = searcher->create_context();
  ASSERT_NE(invalid_group_ctx, nullptr);
  invalid_group_ctx->set_group_params(2, 3);
  EXPECT_EQ(IndexError_InvalidArgument,
            searcher->search_impl(vec.data(), qmeta, invalid_group_ctx));

  // Closing/unloading releases the index and makes all query entry points
  // reject work until another index is loaded.
  ASSERT_EQ(0, streamer->close());
  EXPECT_EQ(nullptr, streamer->create_context());
  EXPECT_EQ(IndexError_NoReady,
            streamer->search_impl(vec.data(), qmeta, streamer_ctx));
  ASSERT_EQ(0, searcher->unload());
  EXPECT_EQ(nullptr, searcher->create_context());
  EXPECT_EQ(IndexError_NoReady,
            searcher->search_impl(vec.data(), qmeta, knnCtx));
}

TEST_F(DiskAnnSearcherTest, TestNodeCache) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestNodeCache";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  constexpr uint32_t kCacheNodes = 2 * DiskAnnUtil::kMaxSectorReadNum + 3;
  search_params.set("zvec.diskann.searcher.cache_node_num", kCacheNodes);
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  // Count all reads made by a second cache build. BFS-expanded nodes are
  // written directly into their final cache slots, so every selected node is
  // requested from the underlying reader at most once across BFS and the
  // final preload pass.
  auto *diskann_searcher = dynamic_cast<DiskAnnSearcher *>(searcher.get());
  ASSERT_NE(nullptr, diskann_searcher);
  auto counting_reader = std::make_shared<CountingAlignedFileReader>(
      DiskAnnCacheTestPeer::reader(diskann_searcher));
  DiskAnnCacheTestPeer::set_reader(diskann_searcher, counting_reader);
  ASSERT_EQ(
      0, DiskAnnCacheTestPeer::configure_cache(diskann_searcher, kCacheNodes));
  EXPECT_EQ(kCacheNodes,
            DiskAnnCacheTestPeer::coordinate_cache_size(diskann_searcher));
  EXPECT_EQ(kCacheNodes,
            DiskAnnCacheTestPeer::neighbor_cache_size(diskann_searcher));
  EXPECT_EQ(kCacheNodes, counting_reader->requested_reads());

  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();

  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 200;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  size_t step = 500;
  for (size_t i = 0; i < doc_cnt; i += step) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));

    auto &knnResult = knnCtx->result();
    // TODO: check
    topk1Hits += i == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    for (size_t k = 0; k < topk; ++k) {
      totalCnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linearResult[j].key() == knnResult[k].key()) {
          totalHits++;
          break;
        }
      }
    }
  }

  float recall = totalHits * step * step * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / doc_cnt;

  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.80f);
}

TEST_F(DiskAnnSearcherTest, TestFilter) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestFilter";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.cache_node_num", 32);
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();

  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  size_t key = 50;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = key + 0.1f;
  }

  // no filter
  {
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(50UL, linearResult[0].key());
    ASSERT_EQ(51UL, linearResult[1].key());
    ASSERT_EQ(49UL, linearResult[2].key());
  }

  // with filter
  {
    auto filterFunc = [](uint64_t key) {
      if (key == 50UL || key == 51UL || key == 49UL) {
        return true;
      }
      return false;
    };


    knnCtx->set_filter(filterFunc);
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
    std::unordered_set<uint64_t> knn_keys;
    for (const auto &result : knnResult) {
      ASSERT_TRUE(knn_keys.emplace(result.key()).second);
      EXPECT_NE(50UL, result.key());
      EXPECT_NE(51UL, result.key());
      EXPECT_NE(49UL, result.key());
    }

    linearCtx->set_filter(filterFunc);
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(52UL, linearResult[0].key());
    ASSERT_EQ(48UL, linearResult[1].key());
    ASSERT_EQ(53UL, linearResult[2].key());

    size_t hit_count = 0;
    for (const auto &result : linearResult) {
      hit_count += knn_keys.count(result.key());
    }
    const float recall = static_cast<float>(hit_count) / topk;
    EXPECT_GT(recall, 0.90f);
  }
}

TEST_F(DiskAnnSearcherTest, TestGroup) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestGroup";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t group_topk = 20;

  auto groupbyFunc = [](uint64_t key) {
    uint32_t group_id = key / 10 % 10;

    // std::cout << "key: " << key << ", group id: " << group_id << std::endl;

    return std::string("g_") + std::to_string(group_id);
  };

  size_t group_num = 5;

  ctx->set_group_params(group_num, group_topk);
  ctx->set_group_by(groupbyFunc);

  size_t query_value = doc_cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = query_value / 10 + 0.1f;
  }

  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));

  auto &group_result = ctx->group_result();
  ASSERT_EQ(group_num, group_result.size());

  std::set<std::string> seen_group_ids;
  for (uint32_t i = 0; i < group_result.size(); ++i) {
    const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();

    ASSERT_TRUE(seen_group_ids.insert(group_id).second);
    ASSERT_GT(result.size(), 0);
    ASSERT_LE(result.size(), group_topk);
    std::cout << "Group ID: " << group_id << std::endl;

    for (uint32_t j = 0; j < result.size(); ++j) {
      EXPECT_EQ(group_id, groupbyFunc(result[j].key()));
      std::cout << "\tKey: " << result[j].key() << std::fixed
                << std::setprecision(3) << ", Score: " << result[j].score()
                << std::endl;
    }
  }

  // Reusing a group context must not retain scores or documents from the
  // previous query.
  query_value = doc_cnt / 10;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = query_value / 10 + 0.1f;
  }
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  const auto &reused_group_result = ctx->group_result();
  ASSERT_EQ(group_num, reused_group_result.size());
  for (const auto &group : reused_group_result) {
    for (const auto &doc : group.docs()) {
      float delta = static_cast<float>(doc.key()) / 10.0f - vec[0];
      float expected_score = dim * delta * delta;
      EXPECT_NEAR(expected_score, doc.score(),
                  std::max(1e-3f, expected_score * 1e-4f));
    }
  }

  // Full linear group search must maintain a heap for every group while it
  // scans, rather than grouping only the global top-k afterward.
  auto linear_ctx = searcher->create_context();
  linear_ctx->set_group_params(group_num, group_topk);
  linear_ctx->set_group_by(groupbyFunc);
  ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linear_ctx));
  const auto &linear_group_result = linear_ctx->group_result();
  ASSERT_EQ(group_num, linear_group_result.size());
  for (const auto &group : linear_group_result) {
    ASSERT_EQ(group_topk, group.docs().size());
    for (const auto &doc : group.docs()) {
      EXPECT_EQ(group.group_id(), groupbyFunc(doc.key()));
    }
  }

  // do linear search by p_keys test
  auto groupbyFuncLinear = [](uint64_t key) {
    uint32_t group_id = key % 10;

    return std::string("g_") + std::to_string(group_id);
  };

  auto linear_pk_ctx = searcher->create_context();

  linear_pk_ctx->set_group_params(group_num, group_topk);
  linear_pk_ctx->set_group_by(groupbyFuncLinear);

  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0] = {4, 3, 2, 1, 5, 6, 7, 8, 9, 10};

  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                  linear_pk_ctx));
  auto &linear_by_pkeys_group_result = linear_pk_ctx->group_result();
  ASSERT_EQ(linear_by_pkeys_group_result.size(), group_num);

  for (uint32_t i = 0; i < linear_by_pkeys_group_result.size(); ++i) {
    const std::string &group_id = linear_by_pkeys_group_result[i].group_id();
    auto &result = linear_by_pkeys_group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    std::cout << "Group ID: " << group_id << std::endl;

    for (uint32_t j = 0; j < result.size(); ++j) {
      std::cout << "\tKey: " << result[j].key() << std::fixed
                << std::setprecision(3) << ", Score: " << result[j].score()
                << std::endl;
    }

    ASSERT_EQ(10 - i, result[0].key());
  }
}

TEST_F(DiskAnnSearcherTest, TestFetchVector) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  auto key_for_id = [](size_t id) { return 100000UL + id * 3; };
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(key_for_id(i), vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestFetchVector";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  size_t query_cnt = 20U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto *diskann_search_context =
      dynamic_cast<DiskAnnContext *>(linearCtx.get());
  ASSERT_NE(diskann_search_context, nullptr);
  EXPECT_EQ(static_cast<size_t>(DiskAnnUtil::kMaxSectorReadNum) *
                DiskAnnUtil::kSectorSize,
            diskann_search_context->sector_buffer_size());
  knnCtx->set_fetch_vector(true);

  for (size_t i = 0; i < doc_cnt; i += doc_cnt / 10) {
    std::string vec_value;
    ASSERT_EQ(0, searcher->get_vector(key_for_id(i), linearCtx, vec_value));

    ASSERT_GE(vec_value.size(), sizeof(float));
    float vector_value = 0.0f;
    std::memcpy(&vector_value, vec_value.data(), sizeof(vector_value));
    ASSERT_EQ(vector_value, i);
  }

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  NumericalVector<float> vec(dim);
  for (size_t i = 0; i < query_cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }

    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(key_for_id(i), linearResult[0].key());

    const auto &vector_string = knnResult[0].vector_string();
    ASSERT_GE(vector_string.size(), sizeof(float));
    // DiskAnn is approximate, so the first KNN result is not guaranteed to
    // be the exact query vector on every graph build. Verify that the fetched
    // payload belongs to the returned key instead.
    std::string expected_vector;
    ASSERT_EQ(0, searcher->get_vector(knnResult[0].key(), linearCtx,
                                      expected_vector));
    ASSERT_EQ(vector_string, expected_vector);
  }

  std::string missing_vector;
  EXPECT_EQ(IndexError_NoExist,
            searcher->get_vector(42, linearCtx, missing_vector));
  EXPECT_TRUE(missing_vector.empty());

  // A DiskAnn provider reads through the aligned index reader rather than a
  // FileReadStorage segment. It and its iterator therefore remain usable
  // after their source streamer is closed.
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  ASSERT_EQ(0, streamer->init(*_index_meta_ptr, search_params));
  auto streamer_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(streamer_storage, nullptr);
  ASSERT_EQ(0, streamer_storage->open(path, false));
  std::weak_ptr<zvec::ailego::File> provider_cached_file =
      streamer_storage->file();
  ASSERT_FALSE(provider_cached_file.expired());
  ASSERT_EQ(0, streamer->open(streamer_storage));
  ASSERT_TRUE(provider_cached_file.expired());

  auto provider = streamer->create_provider();
  ASSERT_NE(provider, nullptr);
  auto second_provider = streamer->create_provider();
  ASSERT_NE(second_provider, nullptr);
  EXPECT_TRUE(provider_cached_file.expired());
  EXPECT_EQ(doc_cnt, provider->count());
  EXPECT_EQ(dim, provider->dimension());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, provider->data_type());
  EXPECT_EQ(_index_meta_ptr->element_size(), provider->element_size());

  float provider_value = 0.0f;
  auto provider_iterator = provider->create_iterator();
  ASSERT_NE(provider_iterator, nullptr);
  ASSERT_TRUE(provider_iterator->is_valid());
  EXPECT_EQ(key_for_id(0), provider_iterator->key());
  EXPECT_EQ(nullptr, DiskAnnProviderTestPeer::fetch_context(provider.get()));
  EXPECT_EQ(nullptr,
            DiskAnnProviderTestPeer::iterator_context(provider_iterator.get()));
  float iterator_value = 0.0f;

  // Neither object has performed vector I/O yet. Their first lazy context and
  // aligned file handle must still be creatable after the streamer closes.
  ASSERT_EQ(0, streamer->close());

  const void *provider_vector = provider->get_vector(key_for_id(17));
  ASSERT_NE(provider_vector, nullptr);
  auto *provider_fetch_context =
      DiskAnnProviderTestPeer::fetch_context(provider.get());
  ASSERT_NE(provider_fetch_context, nullptr);
  EXPECT_EQ(expected_fetch_buffer_size(*provider_fetch_context),
            provider_fetch_context->sector_buffer_size());
  EXPECT_LT(provider_fetch_context->sector_buffer_size(),
            static_cast<size_t>(DiskAnnUtil::kMaxSectorReadNum) *
                DiskAnnUtil::kSectorSize);
  std::memcpy(&provider_value, provider_vector, sizeof(provider_value));
  EXPECT_EQ(17.0f, provider_value);

  // A returned pointer must not be overwritten by a concurrent fetch on a
  // different thread. Coordinate the calls so this deterministically catches
  // providers that share one result buffer globally.
  std::atomic<bool> first_fetch_ready{false};
  std::atomic<bool> second_fetch_done{false};
  std::atomic<bool> release_first_fetch{false};
  float first_thread_value = -1.0f;
  float second_thread_value = -1.0f;
  std::thread first_fetch([&]() {
    const void *value = provider->get_vector(key_for_id(31));
    first_fetch_ready.store(true, std::memory_order_release);
    while (!second_fetch_done.load(std::memory_order_acquire) &&
           !release_first_fetch.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (value != nullptr) {
      std::memcpy(&first_thread_value, value, sizeof(first_thread_value));
    }
  });
  std::thread second_fetch([&]() {
    while (!first_fetch_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    const void *value = provider->get_vector(key_for_id(47));
    if (value != nullptr) {
      std::memcpy(&second_thread_value, value, sizeof(second_thread_value));
    }
    second_fetch_done.store(true, std::memory_order_release);
  });

  constexpr auto kConcurrentFetchTimeout = std::chrono::seconds(5);
  const auto concurrent_fetch_deadline =
      std::chrono::steady_clock::now() + kConcurrentFetchTimeout;
  while (!second_fetch_done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < concurrent_fetch_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool concurrent_fetch_completed =
      second_fetch_done.load(std::memory_order_acquire);
  // Let the first worker exit before joining even when the second fetch is
  // starved. On Windows this releases its IOCP association, so a regression is
  // reported as a bounded test failure rather than hanging the whole suite.
  release_first_fetch.store(true, std::memory_order_release);
  first_fetch.join();
  second_fetch.join();
  EXPECT_TRUE(concurrent_fetch_completed);
  EXPECT_EQ(31.0f, first_thread_value);
  EXPECT_EQ(47.0f, second_thread_value);

  // Providers also need independent result bytes on the same thread. Keep the
  // first provider's pointer live while a second provider fetches a different
  // vector; a function-level TLS string shared by all providers overwrites it.
  const void *second_provider_vector =
      second_provider->get_vector(key_for_id(63));
  ASSERT_NE(second_provider_vector, nullptr);
  float second_provider_value = -1.0f;
  std::memcpy(&second_provider_value, second_provider_vector,
              sizeof(second_provider_value));
  EXPECT_EQ(63.0f, second_provider_value);

  float retained_provider_value = -1.0f;
  std::memcpy(&retained_provider_value, provider_vector,
              sizeof(retained_provider_value));
  EXPECT_EQ(17.0f, retained_provider_value);

  second_provider.reset();
  provider.reset();

  const void *iterator_vector = provider_iterator->data();
  ASSERT_NE(iterator_vector, nullptr);
  auto *iterator_fetch_context =
      DiskAnnProviderTestPeer::iterator_context(provider_iterator.get());
  ASSERT_NE(iterator_fetch_context, nullptr);
  EXPECT_EQ(expected_fetch_buffer_size(*iterator_fetch_context),
            iterator_fetch_context->sector_buffer_size());
  std::memcpy(&iterator_value, iterator_vector, sizeof(iterator_value));
  EXPECT_EQ(0.0f, iterator_value);
  provider_iterator->next();
  ASSERT_TRUE(provider_iterator->is_valid());
  EXPECT_EQ(key_for_id(1), provider_iterator->key());
  iterator_vector = provider_iterator->data();
  ASSERT_NE(iterator_vector, nullptr);
  std::memcpy(&iterator_value, iterator_vector, sizeof(iterator_value));
  EXPECT_EQ(1.0f, iterator_value);

  provider_iterator.reset();
  streamer.reset();

  // Cached nodes keep their coordinates and adjacency lists in separate
  // buffers. Fetching a cached vector must read the coordinate cache rather
  // than returning bytes from the neighbor cache.
  IndexSearcher::Pointer cached_searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(cached_searcher, nullptr);
  Params cached_search_params;
  cached_search_params.set("zvec.diskann.searcher.list_size", 500);
  cached_search_params.set("zvec.diskann.searcher.cache_node_num", doc_cnt);
  ASSERT_EQ(0, cached_searcher->init(cached_search_params));

  auto cached_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(cached_storage, nullptr);
  ASSERT_EQ(0, cached_storage->open(path, false));
  ASSERT_EQ(0, cached_searcher->load(cached_storage, IndexMetric::Pointer()));
  auto cached_ctx = cached_searcher->create_context();
  ASSERT_NE(cached_ctx, nullptr);

  for (size_t i = 0; i < doc_cnt; ++i) {
    std::string vec_value;
    ASSERT_EQ(
        0, cached_searcher->get_vector(key_for_id(i), cached_ctx, vec_value));
    ASSERT_GE(vec_value.size(), sizeof(float));
    float vector_value = 0.0f;
    std::memcpy(&vector_value, vec_value.data(), sizeof(vector_value));
    ASSERT_EQ(vector_value, i);
  }
  ASSERT_EQ(0, cached_searcher->unload());

  auto *diskann_searcher = dynamic_cast<DiskAnnSearcher *>(searcher.get());
  ASSERT_NE(diskann_searcher, nullptr);
  DiskAnnCacheTestPeer::set_reader(
      diskann_searcher, std::make_shared<FailingAlignedFileReader>(
                            DiskAnnCacheTestPeer::reader(diskann_searcher)));
  std::string vector_after_failure;
  EXPECT_EQ(IndexError_Runtime,
            searcher->get_vector(key_for_id(doc_cnt - 1), linearCtx,
                                 vector_after_failure));
  EXPECT_TRUE(vector_after_failure.empty());
}

TEST_F(DiskAnnSearcherTest, TestFp16Entrypoint) {
  IndexMeta fp16_meta(IndexMeta::DataType::DT_FP16, dim);
  fp16_meta.set_metric("SquaredEuclidean", 0, Params());

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>(dim);
  constexpr size_t doc_cnt = 2000;
  for (size_t i = 0; i < doc_cnt; ++i) {
    NumericalVector<Float16> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i) / 10.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 100);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 2);

  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);
  ASSERT_EQ(0, builder->init(fp16_meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  const string path = _dir + "/TestFp16Entrypoint";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(params));
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto ctx = searcher->create_context();
  ASSERT_NE(ctx, nullptr);
  ctx->set_topk(10);

  NumericalVector<Float16> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = 123.1f;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP16, dim);
  ASSERT_EQ(0, searcher->search_impl(query.data(), qmeta, ctx));
  ASSERT_EQ(10, ctx->result().size());

  // DiskAnn is approximate, so the exact top-1 key is not stable across graph
  // builds and platforms. Validate the FP16 search result contract instead.
  std::unordered_set<uint64_t> result_keys;
  for (size_t i = 0; i < ctx->result().size(); ++i) {
    const auto &result = ctx->result()[i];
    EXPECT_LT(result.key(), doc_cnt);
    EXPECT_TRUE(result_keys.emplace(result.key()).second);
    EXPECT_GE(result.score(), 0.0f);
    if (i > 0) {
      EXPECT_LE(ctx->result()[i - 1].score(), result.score());
    }
  }
}

TEST_F(DiskAnnSearcherTest, TestRnnSearch) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestRnnSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 0.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = results[topk / 2].score();
  ctx->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}
