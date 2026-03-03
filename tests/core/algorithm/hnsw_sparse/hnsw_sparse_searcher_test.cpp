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
#include "hnsw_sparse_searcher.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <future>
#include <iomanip>
#include <ailego/math/distance.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_framework.h"
#include "hnsw_sparse_params.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace std;
using namespace testing;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static sparse_dim_count = 16;

class HnswSparseSearcherTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);
  void generate_sparse_data(
      size_t cnt, uint32_t sparse_dim_count,
      std::vector<NumericalVector<uint32_t>> &sparse_indices_list,
      std::vector<NumericalVector<float>> &sparse_vec_list, bool norm);

  static std::string dir_;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string HnswSparseSearcherTest::dir_("HnswSparseSearcherTest/");
shared_ptr<IndexMeta> HnswSparseSearcherTest::_index_meta_ptr;

void HnswSparseSearcherTest::generate_sparse_data(
    size_t cnt, uint32_t sparse_dim_count,
    std::vector<NumericalVector<uint32_t>> &sparse_indices_list,
    std::vector<NumericalVector<float>> &sparse_vec_list, bool norm) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < cnt; ++i) {
    // prepare sparse
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_vec(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_vec[j] = dist(gen);
    }

    float norm;
    ailego::Norm2Matrix<float, 1>::Compute(sparse_vec.data(), sparse_dim_count,
                                           &norm);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_vec[j] = sparse_vec[j] / norm;
    }

    sparse_indices_list.push_back(sparse_indices);
    sparse_vec_list.push_back(sparse_vec);
  }
}

void HnswSparseSearcherTest::SetUp(void) {
  _index_meta_ptr.reset(new (nothrow) IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                                                IndexMeta::DataType::DT_FP32));
  _index_meta_ptr->set_metric("InnerProductSparse", 0, ailego::Params());
}

void HnswSparseSearcherTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(HnswSparseSearcherTest, TestGeneral) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_NE(streamer, nullptr);

  size_t sparse_dim_count = 32;

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGeneral.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // size_t cnt = 5000U;
  size_t cnt = 20000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    sparse_vec_list[i].data(), qmeta, ctx));
  }

  auto path = dir_ + "/TestGeneral";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher knn
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  size_t step = 50;
  for (size_t i = 0; i < cnt; i += step) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    auto t1 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, linearCtx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // ASSERT_EQ(topk, knnResult.size());
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
  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.95f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSparseSearcherTest, TestRnnSearch) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
  size_t doc_cnt = 1000UL;

  for (size_t i = 0; i < doc_cnt; ++i) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_velues);

    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/TestRnnSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 1.0;
  }

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  size_t topk = 50;
  ctx->set_topk(topk);

  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = -results[topk / 2].score();
  ctx->set_threshold(radius);

  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx));

  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(-results[k].score(), radius);
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(-results[topk - 1].score(), radius);
}

TEST_F(HnswSparseSearcherTest, TestClearAndReload) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
  size_t doc_cnt = 1000UL;

  for (size_t i = 0; i < doc_cnt; ++i) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_velues);

    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  params.set("proxima.hnsw.sparse_builder.thread_count", 3);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/TestClearAndReload";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.sparse_searcher.check_crc_enable", true);
  searcherParams.set("proxima.hnsw.sparse_searcher.max_scan_ratio",
                     1.1f);  // including upper layer
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!knnCtx);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 1.0;
  }

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  size_t topk = 100;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, knnCtx));
  ASSERT_EQ(0,
            searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, linearCtx));

  auto &knnResult = knnCtx->result();
  ASSERT_EQ(topk, knnResult.size());
  auto &linearResult = linearCtx->result();
  ASSERT_EQ(topk, linearResult.size());
  auto &stats = searcher->stats();
  ASSERT_EQ(doc_cnt, stats.loaded_count());
  // ASSERT_GT(stats.loaded_costtime(), 0UL);

  //! cleanup
  ASSERT_EQ(0, searcher->cleanup());
  ASSERT_EQ(nullptr, searcher->create_context());
  ASSERT_EQ(IndexError_Runtime,
            searcher->load(storage, IndexMetric::Pointer()));
  ASSERT_EQ(0UL, stats.loaded_count());

  ASSERT_EQ(0, searcher->init(searcherParams));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  linearCtx = searcher->create_context();
  knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!knnCtx);
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, knnCtx));
  ASSERT_EQ(0,
            searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, linearCtx));

  auto &knnResult1 = knnCtx->result();
  ASSERT_EQ(topk, knnResult1.size());
  auto &linearResult1 = linearCtx->result();
  ASSERT_EQ(topk, linearResult1.size());
  ASSERT_EQ(doc_cnt, stats.loaded_count());

  //! unload
  ASSERT_EQ(0, searcher->unload());
  ASSERT_EQ(nullptr, searcher->create_context());
  ASSERT_EQ(0UL, stats.loaded_count());
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  linearCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  linearCtx->set_topk(topk);

  ASSERT_EQ(0,
            searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, linearCtx));

  auto &linearResult2 = linearCtx->result();
  ASSERT_EQ(topk, linearResult2.size());
  ASSERT_EQ(doc_cnt, stats.loaded_count());
}

TEST_F(HnswSparseSearcherTest, TestFilter) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
  size_t doc_cnt = 100UL;
  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);

  for (size_t i = 0; i < doc_cnt; ++i) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      if (i <= 10) {
        sparse_velues[j] = i;
      } else {
        sparse_velues[j] = 10 - (i - 10) * 0.5;
      }
    }

    vec.add_sparses(sparse_indices, sparse_velues);

    ASSERT_TRUE(holder->emplace(i, vec));
    p_keys[0].push_back(i);
  }

  ailego::Params params;
  params.set("proxima.hnsw.sparse_builder.thread_count", 3);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/TestFilter";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.sparse_searcher.check_crc_enable", true);
  searcherParams.set("proxima.hnsw.sparse_searcher.max_scan_ratio", 1.0f);
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 10.1f;
  }

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  size_t topk = 10;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, knnCtx));
  ASSERT_EQ(0,
            searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, linearCtx));
  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
                   sparse_dim_count, sparse_indices.data(),
                   sparse_velues.data(), p_keys, qmeta, linearByPKeysCtx));

  auto filterFunc = [](uint64_t key) {
    if (key == 10UL || key == 11UL) {
      return true;
    }
    return false;
  };

  auto &knnResult = knnCtx->result();
  ASSERT_EQ(topk, knnResult.size());
  ASSERT_EQ(10UL, knnResult[0].key());
  ASSERT_EQ(11UL, knnResult[1].key());
  ASSERT_EQ(12UL, knnResult[2].key());

  auto &linearResult = linearCtx->result();
  ASSERT_EQ(topk, linearResult.size());
  ASSERT_EQ(10UL, linearResult[0].key());
  ASSERT_EQ(11UL, linearResult[1].key());
  ASSERT_EQ(12UL, linearResult[2].key());

  auto &linearByPKeysResult = linearByPKeysCtx->result();
  ASSERT_EQ(topk, linearByPKeysResult.size());
  ASSERT_EQ(10UL, linearByPKeysResult[0].key());
  ASSERT_EQ(11UL, linearByPKeysResult[1].key());
  ASSERT_EQ(12UL, linearByPKeysResult[2].key());

  knnCtx->set_filter(filterFunc);
  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, knnCtx));
  auto &knnResult1 = knnCtx->result();
  ASSERT_EQ(topk, knnResult1.size());
  ASSERT_EQ(12UL, knnResult1[0].key());
  ASSERT_EQ(9UL, knnResult1[1].key());
  ASSERT_EQ(13UL, knnResult1[2].key());

  linearCtx->set_filter(filterFunc);
  ASSERT_EQ(0,
            searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, linearCtx));
  auto &linearResult1 = linearCtx->result();
  ASSERT_EQ(topk, linearResult1.size());
  ASSERT_EQ(12UL, linearResult1[0].key());
  ASSERT_EQ(9UL, linearResult1[1].key());
  ASSERT_EQ(13UL, linearResult1[2].key());

  linearByPKeysCtx->set_filter(filterFunc);
  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
                   sparse_dim_count, sparse_indices.data(),
                   sparse_velues.data(), p_keys, qmeta, linearByPKeysCtx));
  auto &linearByPKeysResult1 = linearByPKeysCtx->result();
  ASSERT_EQ(topk, linearByPKeysResult1.size());
  ASSERT_EQ(12UL, linearByPKeysResult1[0].key());
  ASSERT_EQ(9UL, linearByPKeysResult1[1].key());
  ASSERT_EQ(13UL, linearByPKeysResult1[2].key());
}

TEST_F(HnswSparseSearcherTest, TestBatchQuery) {
  constexpr uint32_t sparse_dim_count = 8U;
  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params());
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
  size_t doc_cnt = 5000UL;

  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(2);
  p_keys[0].resize(doc_cnt);
  p_keys[1].resize(doc_cnt);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  float value_off = -(doc_cnt / 2.0);
  for (size_t i = 0; i < doc_cnt; ++i) {
    SparseVector<float> vec;

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;

      if (i <= 3 || i >= doc_cnt - 3) {
        sparse_velues[j] = 0;
      } else {
        sparse_velues[j] = i + value_off;
      }
    }

    vec.add_sparses(sparse_indices, sparse_velues);

    ASSERT_TRUE(holder->emplace(i, vec));

    p_keys[0][i] = i;
    p_keys[1][i] = i;
  }

  ailego::Params params;
  params.set("proxima.hnsw.sparse_builder.max_neighbor_count", 160);
  params.set("proxima.hnsw.sparse_builder.scaling_factor", 16);
  params.set("proxima.hnsw.sparse_builder.ef_construction", 10);
  params.set("proxima.hnsw.sparse_builder.thread_count", 1);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/TestBatchQuery";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.sparse_searcher.ef", 1000);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);
  linearCtx->set_debug_mode(true);
  linearByPKeysCtx->set_debug_mode(true);
  knnCtx->set_debug_mode(true);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  size_t topk = 200;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  // // do linear search test
  {
    uint32_t dims[] = {sparse_dim_count, sparse_dim_count};
    uint32_t indicies[] = {0, 20, 40, 60, 80, 100, 120, 140,
                           0, 20, 40, 60, 80, 100, 120, 140};
    float queries[] = {3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,
                       -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f};

    ASSERT_EQ(0, searcher->search_bf_impl(dims, indicies, queries, qmeta, 2,
                                          linearCtx));

    auto &linearResult0 = linearCtx->result(0);
    ASSERT_EQ(4996UL, linearResult0[0].key());
    ASSERT_EQ(4995UL, linearResult0[1].key());
    ASSERT_EQ(4994UL, linearResult0[2].key());
    ASSERT_EQ(4993UL, linearResult0[3].key());
    ASSERT_EQ(4992UL, linearResult0[4].key());
    ASSERT_EQ(4991UL, linearResult0[5].key());
    ASSERT_EQ(4990UL, linearResult0[6].key());
    ASSERT_EQ(4989UL, linearResult0[7].key());


    auto &linearResult1 = linearCtx->result(1);
    ASSERT_EQ(4UL, linearResult1[0].key());
    ASSERT_EQ(5UL, linearResult1[1].key());
    ASSERT_EQ(6UL, linearResult1[2].key());
    ASSERT_EQ(7UL, linearResult1[3].key());
    ASSERT_EQ(8UL, linearResult1[4].key());
    ASSERT_EQ(9UL, linearResult1[5].key());
    ASSERT_EQ(10UL, linearResult1[6].key());
    ASSERT_EQ(11UL, linearResult1[7].key());
  }

  // // do linear search by p_keys test
  {
    uint32_t dims[] = {sparse_dim_count, sparse_dim_count};
    uint32_t indicies[] = {0, 20, 40, 60, 80, 100, 120, 140,
                           0, 20, 40, 60, 80, 100, 120, 140};
    float queries[] = {3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,
                       -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f};

    ASSERT_EQ(
        0, searcher->search_bf_by_p_keys_impl(dims, indicies, queries, p_keys,
                                              qmeta, 2, linearByPKeysCtx));

    auto &bfResult0 = linearByPKeysCtx->result(0);
    ASSERT_EQ(4996UL, bfResult0[0].key());
    ASSERT_EQ(4995UL, bfResult0[1].key());
    ASSERT_EQ(4994UL, bfResult0[2].key());
    ASSERT_EQ(4993UL, bfResult0[3].key());
    ASSERT_EQ(4992UL, bfResult0[4].key());
    ASSERT_EQ(4991UL, bfResult0[5].key());
    ASSERT_EQ(4990UL, bfResult0[6].key());
    ASSERT_EQ(4989UL, bfResult0[7].key());

    auto &bfResult1 = linearByPKeysCtx->result(1);
    ASSERT_EQ(4UL, bfResult1[0].key());
    ASSERT_EQ(5UL, bfResult1[1].key());
    ASSERT_EQ(6UL, bfResult1[2].key());
    ASSERT_EQ(7UL, bfResult1[3].key());
    ASSERT_EQ(8UL, bfResult1[4].key());
    ASSERT_EQ(9UL, bfResult1[5].key());
    ASSERT_EQ(10UL, bfResult1[6].key());
    ASSERT_EQ(11UL, bfResult1[7].key());
  }

  // // do knn search test
  {
    uint32_t dims[] = {sparse_dim_count, sparse_dim_count};
    uint32_t indicies[] = {0, 20, 40, 60, 80, 100, 120, 140,
                           0, 20, 40, 60, 80, 100, 120, 140};
    float queries[] = {3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,  3.1f,
                       -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f, -7.1f};

    ASSERT_EQ(0,
              searcher->search_impl(dims, indicies, queries, qmeta, 2, knnCtx));

    auto &knnResult0 = knnCtx->result(0);
    ASSERT_EQ(4996UL, knnResult0[0].key());
    ASSERT_EQ(4995UL, knnResult0[1].key());
    ASSERT_EQ(4994UL, knnResult0[2].key());

    auto &knnResult1 = knnCtx->result(1);
    ASSERT_EQ(4UL, knnResult1[0].key());
    ASSERT_EQ(5UL, knnResult1[1].key());
    ASSERT_EQ(6UL, knnResult1[2].key());
  }
}

TEST_F(HnswSparseSearcherTest, TestStreamerDump) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestStreamerDump.index", true));
  ASSERT_EQ(0, streamer->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 10000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    sparse_vec_list[i].data(), qmeta, ctx));
  }

  auto path = dir_ + "/TestStreamerDump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher knn
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  size_t step = 50;

  for (size_t i = 0; i < cnt; i += step) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    auto t1 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, linearCtx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // ASSERT_EQ(topk, knnResult.size());
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
  float recall = totalHits * step * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.95f);
  EXPECT_GT(cost, 1.50f);
}

TEST_F(HnswSparseSearcherTest, TestSharedContext) {
  auto gen_holder = [](int start, size_t doc_cnt) {
    auto holder =
        make_shared<OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
    uint64_t key = start;

    for (size_t i = 0; i < doc_cnt; ++i) {
      SparseVector<float> vec;

      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = i;
      }

      vec.add_sparses(sparse_indices, sparse_velues);

      key += 3;

      holder->emplace(key, vec);
    }

    return holder;
  };
  auto gen_index = [&gen_holder](int start, size_t docs, std::string path) {
    auto holder = gen_holder(start, docs);
    IndexBuilder::Pointer builder =
        IndexFactory::CreateBuilder("HnswSparseBuilder");
    ailego::Params params;
    builder->init(*_index_meta_ptr, params);
    builder->train(holder);
    builder->build(holder);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    dumper->create(path);
    builder->dump(dumper);
    dumper->close();

    IndexSearcher::Pointer searcher =
        IndexFactory::CreateSearcher("HnswSparseSearcher");
    auto name = rand() % 2 ? "FileReadStorage" : "MMapFileReadStorage";
    auto storage = IndexFactory::CreateStorage(name);
    storage->open(path, false);
    params.set("proxima.hnsw.sparse_searcher.visit_bloomfilter_enable",
               rand() % 2);
    searcher->init(ailego::Params());
    searcher->load(storage, IndexMetric::Pointer());
    return searcher;
  };

  srand(ailego::Realtime::MilliSeconds());
  size_t docs1 = rand() % 500 + 100;
  size_t docs2 = rand() % 5000 + 100;
  size_t docs3 = rand() % 50000 + 100;
  auto path1 = dir_ + "/TestSharedContext.index1";
  auto path2 = dir_ + "/TestSharedContext.index2";
  auto path3 = dir_ + "/TestSharedContext.index3";
  auto searcher1 = gen_index(0, docs1, path1);
  auto searcher2 = gen_index(1, docs2, path2);
  auto searcher3 = gen_index(2, docs3, path3);

  srand(ailego::Realtime::MilliSeconds());
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  auto do_test = [&]() {
    IndexSearcher::Context::Pointer ctx;
    switch (rand() % 3) {
      case 0:
        ctx = searcher1->create_context();
        break;
      case 1:
        ctx = searcher2->create_context();
        break;
      case 2:
        ctx = searcher3->create_context();
        break;
    }
    ctx->set_topk(10);

    int ret = 0;
    for (int i = 0; i < 100; ++i) {
      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = -(i + 0.1f);
      }

      auto code = rand() % 6;
      switch (code) {
        case 0:
          ret = searcher1->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, ctx);
          break;
        case 1:
          ret = searcher2->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, ctx);
          break;
        case 2:
          ret = searcher3->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, ctx);
          break;
        case 3:
          ret =
              searcher1->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx);
          break;
        case 4:
          ret =
              searcher2->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx);
          break;
        case 5:
          ret =
              searcher3->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx);
          break;
      }

      EXPECT_EQ(0, ret);
      auto &results = ctx->result();
      EXPECT_EQ(10, results.size());
      for (int k = 0; k < 10; ++k) {
        // std::cout << "code: " << code << ", i: " << i << ", k: " << k
        //           << ", key: " << results[k].key()
        //           << ", score: " << results[k].score() << std::endl;

        EXPECT_EQ(code % 3, results[k].key() % 3);
      }
    }
  };
  auto t1 = std::async(std::launch::async, do_test);
  auto t2 = std::async(std::launch::async, do_test);
  t1.wait();
  t2.wait();

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  storage->init(ailego::Params());
  storage->open(dir_ + "/TestSharedContext.index4", true);
  streamer->init(*_index_meta_ptr, ailego::Params());
  streamer->open(storage);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 1.1f;
  }

  auto ctx1 = streamer->create_context();
  EXPECT_EQ(IndexError_Unsupported,
            searcher1->search_impl(sparse_dim_count, sparse_indices.data(),
                                   sparse_velues.data(), qmeta, ctx1));

  auto ctx2 = searcher1->create_context();
  EXPECT_EQ(IndexError_Unsupported,
            streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx2));
}

TEST_F(HnswSparseSearcherTest, TestProvider) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
  size_t doc_cnt = 5000UL;
  std::vector<key_t> keys(doc_cnt);
  srand(ailego::Realtime::MilliSeconds());
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < doc_cnt; ++i) {
      keys[i] = dt(mt);
    }
  } else {
    std::iota(keys.begin(), keys.end(), 0U);
    std::transform(keys.begin(), keys.end(), keys.begin(),
                   [&](key_t k) { return step * k; });
    if (rand_order) {
      uint32_t seed = ailego::Realtime::Seconds();
      std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    }
  }

  for (size_t i = 0; i < doc_cnt; ++i) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = keys[i];
    }

    vec.add_sparses(sparse_indices, sparse_velues);

    ASSERT_TRUE(holder->emplace(keys[i], vec));
  }

  ailego::Params params;
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/TestProvider";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.sparse_searcher.ef", 1);
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto provider = searcher->create_sparse_provider();
  for (size_t i = 0; i < keys.size(); ++i) {
    uint32_t sparse_count;
    std::string sparse_indices_buffer;
    std::string sparse_values_buffer;

    ASSERT_EQ(0, provider->get_sparse_vector(keys[i], &sparse_count,
                                             &sparse_indices_buffer,
                                             &sparse_values_buffer));

    const float *sparse_values_ptr =
        reinterpret_cast<const float *>(sparse_values_buffer.data());
    ASSERT_EQ(sparse_count, sparse_dim_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      ASSERT_FLOAT_EQ(sparse_values_ptr[j], keys[i]);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const uint32_t sparse_count = iter->sparse_count();
    ASSERT_EQ(sparse_count, sparse_dim_count);

    const float *d = reinterpret_cast<const float *>(iter->sparse_data());
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    cnt++;
    iter->next();
  }

  ASSERT_EQ(cnt, doc_cnt);
  ASSERT_EQ(_index_meta_ptr->data_type(), provider->data_type());
}

TEST_F(HnswSparseSearcherTest, TestRandomPaddingTopk) {
  std::mt19937 mt{};
  std::uniform_real_distribution<float> gen(0.0f, 1.0f);
  constexpr size_t static sparse_dim_count = 8;
  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<MultiPassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
  const size_t COUNT = 10000UL;

  for (size_t i = 0; i < COUNT; ++i) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_velues);

    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/TestRandomPadding";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  ailego::Params params;
  params.set("proxima.hnsw.sparse_searcher.force_padding_result_enable", true);
  params.set("proxima.hnsw.sparse_searcher.scan_ratio", 0.01f);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 1.0f;
  }

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  std::uniform_int_distribution<uint32_t> gen_int(1, COUNT);
  size_t topk = gen_int(mt);
  ctx->set_topk(topk);

  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx));

  auto &results = ctx->result();
  EXPECT_EQ(results.size(), topk);
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(results[i].key(), results[j].key());
    }
  }

  ctx->set_filter([](uint64_t key) { return true; });

  ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx));

  auto &results1 = ctx->result();
  EXPECT_EQ(results1.size(), 0);
}

TEST_F(HnswSparseSearcherTest, TestBruteForceSetupInContext) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>>();
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; ++i) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_velues);

    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  // params.set("proxima.hnsw.sparse_builder.max_neighbor_count", 16);
  params.set("proxima.hnsw.sparse_builder.scaling_factor", 16);
  params.set("proxima.hnsw.sparse_builder.ef_construction", 10);
  params.set("proxima.hnsw.sparse_builder.thread_count", 2);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/TestBruteForceSetupInContext";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.sparse_searcher.ef", 1);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  size_t topk = 200;
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  bool set_bf_threshold = false;
  bool use_update = false;

  size_t step = 50;
  for (size_t i = 0; i < doc_cnt; i += step) {
    auto linearCtx = searcher->create_context();
    auto knnCtx = searcher->create_context();

    ASSERT_TRUE(!!linearCtx);
    ASSERT_TRUE(!!knnCtx);

    linearCtx->set_topk(topk);
    knnCtx->set_topk(topk);

    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i + 0.1f;
    }

    auto t1 = ailego::Realtime::MicroSeconds();

    if (set_bf_threshold) {
      if (use_update) {
        ailego::Params searcherParamsExtra;

        searcherParamsExtra.set(
            "proxima.hnsw.sparse_searcher.brute_force_threshold", doc_cnt);
        knnCtx->update(searcherParamsExtra);
      } else {
        knnCtx->set_bruteforce_threshold(doc_cnt);
      }

      use_update = !use_update;
    }
    ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, linearCtx));
    // auto t3 = ailego::Realtime::MicroSeconds();

    if (set_bf_threshold) {
      linearTotalTime += t2 - t1;
    } else {
      knnTotalTime += t2 - t1;
    }

    set_bf_threshold = !set_bf_threshold;

    auto &knnResult = knnCtx->result();
    // TODO: check
    // ASSERT_EQ(topk, knnResult.size());
    topk1Hits += doc_cnt - 1 == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(doc_cnt - 1, linearResult[0].key());

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
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.90f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSparseSearcherTest, TestHalfFloatConverter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

  IndexMeta index_meta_raw(IndexMeta::MetaType::MT_SPARSE,
                           IndexMeta::DataType::DT_FP32);
  index_meta_raw.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("HalfFloatSparseConverter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  reformer->init(index_meta.reformer_params());

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestHalfFloatConverter.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // size_t cnt = 5000U;
  size_t cnt = 20000U;
  size_t sparse_dim_count = 32;

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    std::string new_vec;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(
                     sparse_dim_count, sparse_indices_list[i].data(),
                     sparse_vec_list[i].data(), qmeta, &new_vec, &new_meta));

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    new_vec.data(), new_meta, ctx));
  }

  auto path = dir_ + "/TestHalfFloatConverter";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher knn
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  size_t step = 50;
  for (size_t i = 0; i < cnt; i += step) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    std::string ovec;
    IndexQueryMeta new_qmeta;
    ASSERT_EQ(0,
              reformer->transform(sparse_dim_count, sparse_indices.data(),
                                  sparse_vec.data(), qmeta, &ovec, &new_qmeta));

    auto t1 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                       ovec.data(), new_qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       ovec.data(), new_qmeta, linearCtx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // ASSERT_EQ(topk, knnResult.size());
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
  float recall = totalHits * step * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.95f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSparseSearcherTest, TestQueryFilteringRatio) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_NE(streamer, nullptr);

  size_t sparse_dim_count = 32;

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestQueryFilteringRatio.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // size_t cnt = 5000U;
  size_t cnt = 20000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    sparse_vec_list[i].data(), qmeta, ctx));
  }

  auto path = dir_ + "/TestQueryFilteringRatio";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher knn
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_TRUE(searcher != nullptr);

  ailego::Params searcher_params;
  searcher_params.set(PARAM_HNSW_SPARSE_SEARCHER_QUERY_FILTERING_RATIO, 0.05);

  ASSERT_EQ(0, searcher->init(searcher_params));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  size_t topk = 20;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  size_t step = 100;
  for (size_t i = 0; i < cnt; i += step) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    auto t1 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, linearCtx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // ASSERT_EQ(topk, knnResult.size());
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
  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.95f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSparseSearcherTest, TestHalfFloatRevert) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE, true);

  IndexMeta index_meta_raw(IndexMeta::MetaType::MT_SPARSE,
                           IndexMeta::DataType::DT_FP32);
  index_meta_raw.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("HalfFloatSparseConverter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  reformer->init(index_meta.reformer_params());

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestHalfFloatRevert.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // size_t cnt = 5000U;
  size_t cnt = 20000U;
  size_t sparse_dim_count = 32;

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  IndexQueryMeta new_meta;
  for (size_t i = 0; i < cnt; i++) {
    std::string new_vec;
    ASSERT_EQ(0, reformer->transform(
                     sparse_dim_count, sparse_indices_list[i].data(),
                     sparse_vec_list[i].data(), qmeta, &new_vec, &new_meta));

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    new_vec.data(), new_meta, ctx));
  }

  const float epsilon = 1e-2;

  for (size_t i = 0; i < cnt; i++) {
    uint32_t sparse_count;
    std::string sparse_indices;
    std::string sparse_values;

    ASSERT_EQ(streamer->get_sparse_vector(i, &sparse_count, &sparse_indices,
                                          &sparse_values),
              0);
    ASSERT_EQ(sparse_count, sparse_dim_count);

    std::string sparse_values_out;
    sparse_values_out.resize(sparse_count * sizeof(float));

    ASSERT_EQ(reformer->revert(
                  sparse_count,
                  reinterpret_cast<const uint32_t *>(sparse_indices.data()),
                  sparse_values.data(), new_meta, &sparse_values_out),
              0);

    for (size_t j = 0; j < sparse_count; ++j) {
      float vector_value = *((float *)(sparse_values_out.data()) + j);
      EXPECT_NEAR(vector_value, sparse_vec_list[i][j], epsilon);
    }
  }

  auto path = dir_ + "/TestHalfFloatRevert";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher knn
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  for (size_t i = 0; i < cnt; i++) {
    uint32_t sparse_count;
    std::string sparse_indices;
    std::string sparse_values;

    ASSERT_EQ(searcher->get_sparse_vector(i, &sparse_count, &sparse_indices,
                                          &sparse_values),
              0);
    ASSERT_EQ(sparse_count, sparse_dim_count);

    std::string sparse_values_out;
    sparse_values_out.resize(sparse_count * sizeof(float));

    ASSERT_EQ(reformer->revert(
                  sparse_count,
                  reinterpret_cast<const uint32_t *>(sparse_indices.data()),
                  sparse_values.data(), new_meta, &sparse_values_out),
              0);

    for (size_t j = 0; j < sparse_count; ++j) {
      float vector_value = *((float *)(sparse_values_out.data()) + j);
      EXPECT_NEAR(vector_value, sparse_vec_list[i][j], epsilon);
    }
  }
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif