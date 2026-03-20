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
#include "hnsw_sparse_streamer.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <memory>
#include <ailego/math/norm_matrix.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>

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

class HnswSparseStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);
  void generate_sparse_data(
      size_t cnt, uint32_t sparse_dim_count,
      std::vector<NumericalVector<uint32_t>> &sparse_indices_list,
      std::vector<NumericalVector<float>> &sparse_vec_list, bool norm);

  static std::string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string HnswSparseStreamerTest::dir_("HnswSparseStreamerTest/");
shared_ptr<IndexMeta> HnswSparseStreamerTest::index_meta_ptr_;

void HnswSparseStreamerTest::generate_sparse_data(
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

void HnswSparseStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (nothrow) IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                                                IndexMeta::DataType::DT_FP32));
  index_meta_ptr_->set_metric("InnerProductSparse", 0, ailego::Params());

  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

void HnswSparseStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(HnswSparseStreamerTest, TestGeneral) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  size_t sparse_dim_count = 32;

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGeneral", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 20000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  auto linearCtx = streamer->create_context();
  ASSERT_TRUE(!!linearCtx);

  auto knnCtx = streamer->create_context();
  ASSERT_TRUE(!!knnCtx);

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

  // streamer->print_debug_info();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  for (size_t i = 0; i < cnt; i += 100) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    auto t1 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, linearCtx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    // std::cout << "i: " << i << std::endl;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
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
  float topk1Recall = topk1Hits * 100.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSparseStreamerTest, TestAddVector) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  size_t sparse_dim_count = 32;

  ailego::Params params;
  params.set("proxima.hnsw.sparse_streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw.sparse_streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw.sparse_streamer.scaling_factor", 5U);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestAddVector", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;
  size_t cnt = 1000UL;
  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    sparse_vec_list[i].data(), qmeta, ctx));
  }

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(HnswSparseStreamerTest, TestLinearSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set("proxima.hnsw.sparse_streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw.sparse_streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw.sparse_streamer.scaling_factor", 5U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);
  for (size_t i = 0; i < cnt; ++i) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = -1.0 * i - 1.0f;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                    sparse_velues.data(), qmeta, ctx));
  }

  size_t topk = 3;
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i + 1.0f;
    }

    ctx->set_topk(1U);
    ASSERT_EQ(0,
              streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(0, result1[0].key());

    ctx->set_topk(topk);
    ASSERT_EQ(0,
              streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(0, result2[0].key());
    ASSERT_EQ(1, result2[1].key());
    ASSERT_EQ(2, result2[2].key());
  }

  ctx->set_topk(100U);
  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 10.1f;
  }

  ASSERT_EQ(0, streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(0, result[0].key());
  ASSERT_EQ(1, result[1].key());
  ASSERT_EQ(10, result[10].key());
  ASSERT_EQ(20, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());
}

TEST_F(HnswSparseStreamerTest, TestLinearSearchByKeys) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set("proxima.hnsw.sparse_streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw.sparse_streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw.sparse_streamer.scaling_factor", 5U);
  params.set("proxima.hnsw.sparse_streamer.get_vector_enable", true);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearchByKeys.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);

  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0].resize(cnt);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);
  for (size_t i = 0; i < cnt; ++i) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = -1.0 * i - 1.0f;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                    sparse_velues.data(), qmeta, ctx));

    p_keys[0][i] = i;
  }

  size_t topk = 3;
  for (size_t i = 0; i < cnt; i += 1) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i + 1.0f;
    }
    ctx->set_topk(1U);
    ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                     sparse_dim_count, sparse_indices.data(),
                     sparse_velues.data(), p_keys, qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(0, result1[0].key());

    ctx->set_topk(topk);
    ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                     sparse_dim_count, sparse_indices.data(),
                     sparse_velues.data(), p_keys, qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(0, result2[0].key());
    ASSERT_EQ(1, result2[1].key());
    ASSERT_EQ(2, result2[2].key());
  }

  {
    ctx->set_topk(100U);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = 1.0f;
    }
    ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                     sparse_dim_count, sparse_indices.data(),
                     sparse_velues.data(), p_keys, qmeta, ctx));
    auto &result = ctx->result();
    ASSERT_EQ(100U, result.size());
    ASSERT_EQ(0, result[0].key());
    ASSERT_EQ(1, result[1].key());
    ASSERT_EQ(10, result[10].key());
    ASSERT_EQ(20, result[20].key());
    ASSERT_EQ(30, result[30].key());
    ASSERT_EQ(35, result[35].key());
    ASSERT_EQ(99, result[99].key());
  }

  {
    ctx->set_topk(100U);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = 10.0f;
    }

    p_keys[0] = {{cnt + 1, 10, 1, 15, cnt + 2}};
    ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                     sparse_dim_count, sparse_indices.data(),
                     sparse_velues.data(), p_keys, qmeta, ctx));
    auto &result = ctx->result();
    ASSERT_EQ(3U, result.size());
    ASSERT_EQ(1, result[0].key());
    ASSERT_EQ(10, result[1].key());
    ASSERT_EQ(15, result[2].key());
  }

  {
    ctx->set_topk(100U);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = 9.0f;
    }
    p_keys[0].clear();
    for (size_t j = 0; j < cnt; j += 10) {
      p_keys[0].push_back((uint64_t)j);
    }
    ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                     sparse_dim_count, sparse_indices.data(),
                     sparse_velues.data(), p_keys, qmeta, ctx));
    auto &result = ctx->result();
    ASSERT_EQ(100U, result.size());
    ASSERT_EQ(0, result[0].key());
    ASSERT_EQ(10, result[1].key());
    ASSERT_EQ(100, result[10].key());
    ASSERT_EQ(200, result[20].key());
    ASSERT_EQ(300, result[30].key());
    ASSERT_EQ(350, result[35].key());
    ASSERT_EQ(990, result[99].key());
  }
}

TEST_F(HnswSparseStreamerTest, TestOpenClose) {
  constexpr size_t static sparse_dim_count = 2048;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params());
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  auto storage1 = IndexFactory::CreateStorage("MMapFileStorage");
  auto storage2 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage1);
  ASSERT_NE(nullptr, storage2);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage1->init(stg_params));
  ASSERT_EQ(0, storage1->open(dir_ + "TessOpenAndClose1", true));
  ASSERT_EQ(0, storage2->init(stg_params));
  ASSERT_EQ(0, storage2->open(dir_ + "TessOpenAndClose2", true));
  ASSERT_EQ(0, streamer->init(meta, params));
  auto checkIter = [](size_t base, size_t total,
                      IndexStreamer::Pointer &streamer) {
    auto provider = streamer->create_sparse_provider();
    auto iter = provider->create_iterator();
    ASSERT_TRUE(!!iter);
    size_t cur = base;
    size_t cnt = 0;
    while (iter->is_valid()) {
      float *sparse_data = (float *)iter->sparse_data();
      ASSERT_EQ(cur, iter->key());
      for (size_t d = 0; d < sparse_dim_count; ++d) {
        ASSERT_FLOAT_EQ((float)cur, sparse_data[d]);
      }
      iter->next();
      cur += 2;
      cnt++;
    }
    ASSERT_EQ(cnt, total);
  };

  size_t testCnt = 200;
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < testCnt; i += 2) {
    float v1 = (float)i;
    ASSERT_EQ(0, streamer->open(storage1));
    auto ctx = streamer->create_context();
    ASSERT_TRUE(!!ctx);

    NumericalVector<uint32_t> sparse_indices1(sparse_dim_count);
    NumericalVector<float> sparse_velues1(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices1[j] = j * 20;
      sparse_velues1[j] = v1;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices1.data(),
                                    sparse_velues1.data(), qmeta, ctx));

    checkIter(0, i / 2 + 1, streamer);
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());

    float v2 = (float)(i + 1);
    NumericalVector<uint32_t> sparse_indices2(sparse_dim_count);
    NumericalVector<float> sparse_velues2(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices2[j] = j * 20;
      sparse_velues2[j] = v2;
    }

    ASSERT_EQ(0, streamer->open(storage2));
    ctx = streamer->create_context();
    ASSERT_TRUE(!!ctx);
    ASSERT_EQ(
        0, streamer->add_impl(i + 1, sparse_dim_count, sparse_indices2.data(),
                              sparse_velues2.data(), qmeta, ctx));
    checkIter(1, i / 2 + 1, streamer);
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());
  }

  IndexStreamer::Pointer streamer1 =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer1->init(meta, params));
  ASSERT_EQ(0, streamer1->open(storage1));

  IndexStreamer::Pointer streamer2 =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer2->init(meta, params));
  ASSERT_EQ(0, streamer2->open(storage2));

  checkIter(0, testCnt / 2, streamer1);
  checkIter(1, testCnt / 2, streamer2);
}

TEST_F(HnswSparseStreamerTest, TestCreateIterator) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_FILTER_SAME_KEY, true);
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestCreateIterator", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto checkIter = [](size_t total, IndexStreamer::Pointer &streamer) {
    auto provider = streamer->create_sparse_provider();
    auto iter = provider->create_iterator();
    ASSERT_TRUE(!!iter);
    size_t cur = 0;
    while (iter->is_valid()) {
      float *sparse_data = (float *)iter->sparse_data();
      ASSERT_EQ(cur, iter->key());
      for (size_t d = 0; d < sparse_dim_count; ++d) {
        ASSERT_FLOAT_EQ((float)cur, sparse_data[d]);
      }
      iter->next();
      cur++;
    }
    ASSERT_EQ(cur, total);
  };

  size_t cnt = 200;
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<uint32_t> sparse_indices1(sparse_dim_count);
    NumericalVector<float> sparse_velues1(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices1[j] = j * 20;
      sparse_velues1[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices1.data(),
                                    sparse_velues1.data(), qmeta, ctx));
    checkIter(i + 1, streamer);
  }

  streamer->flush(0UL);
  streamer->close();
  ASSERT_EQ(0, streamer->open(storage));
  checkIter(cnt, streamer);

  // check getVector
  auto provider = streamer->create_sparse_provider();
  for (size_t i = 0; i < cnt; i++) {
    uint32_t sparse_count;
    std::string sparse_indices_buffer;
    std::string sparse_values_buffer;

    ASSERT_EQ(
        0, provider->get_sparse_vector(i, &sparse_count, &sparse_indices_buffer,
                                       &sparse_values_buffer));

    const float *sparse_values_ptr =
        reinterpret_cast<const float *>(sparse_values_buffer.data());
    ASSERT_EQ(sparse_count, sparse_dim_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      ASSERT_FLOAT_EQ(sparse_values_ptr[j], i);
    }
  }
}

TEST_F(HnswSparseStreamerTest, TestNoInit) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  streamer->cleanup();
}


TEST_F(HnswSparseStreamerTest, TestForceFlush) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_FILTER_SAME_KEY, true);
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  stg_params.set("proxima.mmap_file.storage.copy_on_write", true);
  stg_params.set("proxima.mmap_file.storage.force_flush", true);
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestForceFlush", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto checkIter = [](size_t total, IndexStreamer::Pointer &streamer) {
    auto provider = streamer->create_sparse_provider();
    auto iter = provider->create_iterator();
    ASSERT_TRUE(!!iter);
    size_t cur = 0;
    while (iter->is_valid()) {
      ASSERT_EQ(cur, iter->key());
      const uint32_t sparse_count = iter->sparse_count();
      ASSERT_EQ(sparse_count, sparse_dim_count);

      const float *data = reinterpret_cast<const float *>(iter->sparse_data());
      for (size_t j = 0; j < sparse_dim_count; ++j) {
        ASSERT_FLOAT_EQ((float)cur, data[j]);
      }

      iter->next();
      cur++;
    }
    ASSERT_EQ(cur, total);
  };

  size_t cnt = 200;
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  auto ctx = streamer->create_context();

  for (size_t i = 0; i < cnt; ++i) {
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                    sparse_velues.data(), qmeta, ctx));
    checkIter(i + 1, streamer);
  }

  streamer->flush(0UL);
  streamer->close();
  storage->close();

  storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestForceFlush", true));
  ASSERT_EQ(0, streamer->open(storage));
  checkIter(cnt, streamer);

  // check getVector
  auto provider = streamer->create_sparse_provider();
  for (size_t i = 0; i < cnt; i++) {
    uint32_t sparse_count;
    std::string sparse_indices_buffer;
    std::string sparse_values_buffer;

    ASSERT_EQ(
        0, provider->get_sparse_vector(i, &sparse_count, &sparse_indices_buffer,
                                       &sparse_values_buffer));

    const float *sparse_values_ptr =
        reinterpret_cast<const float *>(sparse_values_buffer.data());
    ASSERT_EQ(sparse_count, sparse_dim_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      ASSERT_FLOAT_EQ(sparse_values_ptr[j], i);
    }
  }
}

TEST_F(HnswSparseStreamerTest, TestKnnMultiThread) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  constexpr size_t static sparse_dim_count = 32;
  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params());
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 128);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 64);
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_INDEX_SIZE, 30 * 1024 * 1024U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 32);
  params.set(PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessKnnMultiThread", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
    IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                         IndexMeta::DataType::DT_FP32);
    size_t succAdd = 0;
    auto ctx = streamer->create_context();
    for (size_t i = 0; i < addCnt; i++) {
      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = (float)i + baseKey;
      }

      succAdd += !streamer->add_impl(baseKey + i, sparse_dim_count,
                                     sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx);
    }
    streamer->flush(0UL);
    return succAdd;
  };

  auto t2 = std::async(std::launch::async, addVector, 1000, 1000);
  auto t3 = std::async(std::launch::async, addVector, 2000, 1000);
  auto t1 = std::async(std::launch::async, addVector, 0, 1000);
  ASSERT_EQ(1000U, t1.get());
  ASSERT_EQ(1000U, t2.get());
  ASSERT_EQ(1000U, t3.get());
  streamer->close();

  // checking data
  ASSERT_EQ(0, streamer->open(storage));
  auto provider = streamer->create_sparse_provider();
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);
  size_t total = 0;
  uint64_t min = 1000;
  uint64_t max = 0;
  while (iter->is_valid()) {
    const uint32_t sparse_count = iter->sparse_count();
    ASSERT_EQ(sparse_count, sparse_dim_count);

    const float *data = reinterpret_cast<const float *>(iter->sparse_data());
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      ASSERT_EQ((float)iter->key(), data[j]);
    }
    total++;
    min = std::min(min, iter->key());
    max = std::max(max, iter->key());
    iter->next();
  }

  ASSERT_EQ(3000, total);
  ASSERT_EQ(0, min);
  ASSERT_EQ(2999, max);

  // ====== multi thread search
  size_t topk = 10;
  size_t cnt = 3000;
  auto knnSearch = [&]() {
    auto linearCtx = streamer->create_context();
    auto linearByPkeysCtx = streamer->create_context();
    auto ctx = streamer->create_context();
    IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                         IndexMeta::DataType::DT_FP32);
    linearCtx->set_topk(topk);
    linearByPkeysCtx->set_topk(topk);
    ctx->set_topk(topk);
    size_t totalCnts = 0;
    size_t totalHits = 0;
    for (size_t i = 0; i < cnt; i += 1) {
      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = ((float)i + 1.1f);
      }

      ASSERT_EQ(0,
                streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, ctx));
      ASSERT_EQ(
          0, streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, linearCtx));
      std::vector<std::vector<uint64_t>> p_keys = {{cnt - 1, cnt - 2, cnt - 3}};
      ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                       sparse_dim_count, sparse_indices.data(),
                       sparse_velues.data(), p_keys, qmeta, linearByPkeysCtx));
      auto &r1 = ctx->result();
      ASSERT_EQ(topk, r1.size());
      auto &r2 = linearCtx->result();
      ASSERT_EQ(topk, r2.size());
      ASSERT_EQ(cnt - 1, r2[0].key());
      auto &r3 = linearByPkeysCtx->result();
      ASSERT_EQ(std::min(topk, p_keys[0].size()), r3.size());
#if 0
            printf("linear: %zd => %zd %zd %zd %zd %zd\n", i, r2[0].key,
                   r2[1].key, r2[2].key, r2[3].key, r2[4].key);
            printf("knn: %zd => %zd %zd %zd %zd %zd\n", i, r1[0].key, r1[1].key,
                   r1[2].key, r1[3].key, r1[4].key);
#endif
      for (size_t k = 0; k < topk; ++k) {
        totalCnts++;
        for (size_t j = 0; j < topk; ++j) {
          if (r2[j].key() == r1[k].key()) {
            totalHits++;
            break;
          }
        }
      }
    }
    printf("%f\n", totalHits * 1.0f / totalCnts);
    ASSERT_TRUE((totalHits * 1.0f / totalCnts) > 0.80f);
  };
  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  auto s3 = std::async(std::launch::async, knnSearch);
  s1.wait();
  s2.wait();
  s3.wait();
}

TEST_F(HnswSparseStreamerTest, TestKnnConcurrentAddAndSearch) {
  constexpr size_t static sparse_dim_count = 32;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;

  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  // meta.set_metric("InnerProductSparse", 0, ailego::Params());
  meta.set_metric("SquaredEuclideanSparse", 0, ailego::Params());
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 128);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 64);
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_INDEX_SIZE, 30 * 1024 * 1024U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_CHUNK_SIZE, 4096);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 32);
  params.set(PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessKnnConcurrentAddAndSearch", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
    IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                         IndexMeta::DataType::DT_FP32);
    size_t succAdd = 0;
    auto ctx = streamer->create_context();
    for (size_t i = 0; i < addCnt; i++) {
      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = (float)i + baseKey;
      }

      succAdd += !streamer->add_impl(baseKey + i, sparse_dim_count,
                                     sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx);
    }
    streamer->flush(0UL);
    return succAdd;
  };

  auto knnSearch = [&]() {
    size_t topk = 100;
    size_t cnt = 3000;
    auto linearCtx = streamer->create_context();
    auto linearByPkeysCtx = streamer->create_context();
    auto ctx = streamer->create_context();
    IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                         IndexMeta::DataType::DT_FP32);
    linearCtx->set_topk(topk);
    linearByPkeysCtx->set_topk(topk);
    ctx->set_topk(topk);
    size_t totalCnts = 0;
    size_t totalHits = 0;
    for (size_t i = 0; i < cnt; i += 1) {
      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = -((float)i + 1.1f);
      }

      ASSERT_EQ(0,
                streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, ctx));
      ASSERT_EQ(
          0, streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, linearCtx));
      std::vector<std::vector<uint64_t>> p_keys = {{0, 1, 2}};
      ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                       sparse_dim_count, sparse_indices.data(),
                       sparse_velues.data(), p_keys, qmeta, linearByPkeysCtx));
      auto &r1 = ctx->result();
      ASSERT_EQ(topk, r1.size());
      auto &r2 = linearCtx->result();
      ASSERT_EQ(topk, r2.size());
      ASSERT_EQ(0, r2[0].key());
      auto &r3 = linearByPkeysCtx->result();
      ASSERT_EQ(std::min(topk, p_keys[0].size()), r3.size());
#if 0
            printf("linear: %zd => %zd %zd %zd %zd %zd\n", i, r2[0].key,
                   r2[1].key, r2[2].key, r2[3].key, r2[4].key);
            printf("knn: %zd => %zd %zd %zd %zd %zd\n", i, r1[0].key, r1[1].key,
                   r1[2].key, r1[3].key, r1[4].key);
#endif
      for (size_t k = 0; k < topk; ++k) {
        totalCnts++;
        for (size_t j = 0; j < topk; ++j) {
          if (r2[j].key() == r1[k].key()) {
            totalHits++;
            break;
          }
        }
      }
    }
    printf("%f\n", totalHits * 1.0f / totalCnts);
    ASSERT_TRUE((totalHits * 1.0f / totalCnts) > 0.80f);
  };

  auto t0 = std::async(std::launch::async, addVector, 0, 1000);
  ASSERT_EQ(1000, t0.get());
  auto t1 = std::async(std::launch::async, addVector, 1000, 1000);
  auto t2 = std::async(std::launch::async, addVector, 2000, 1000);
  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  ASSERT_EQ(1000, t1.get());
  ASSERT_EQ(1000, t2.get());
  s1.wait();
  s2.wait();

  // checking data
  auto provider = streamer->create_sparse_provider();
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);
  size_t total = 0;
  uint64_t min = 1000;
  uint64_t max = 0;
  while (iter->is_valid()) {
    const uint32_t sparse_count = iter->sparse_count();
    ASSERT_EQ(sparse_count, sparse_dim_count);

    const float *data = reinterpret_cast<const float *>(iter->sparse_data());
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      ASSERT_FLOAT_EQ((float)iter->key(), data[j]);
    }
    total++;
    min = std::min(min, iter->key());
    max = std::max(max, iter->key());
    iter->next();
  }

  ASSERT_EQ(3000, total);
  ASSERT_EQ(0, min);
  ASSERT_EQ(2999, max);
}

TEST_F(HnswSparseStreamerTest, TestBfThreshold) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 16);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessBfThreshold", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 10000;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  ctx->set_topk(1U);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = (float)i + 1.0f;
    }

    streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                       sparse_velues.data(), qmeta, ctx);
  }
  streamer->flush(0UL);
  streamer->close();

  IndexStreamer::Pointer streamer1 =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_NE(streamer1, nullptr);
  auto params1 = params;
  params1.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, cnt - 1);
  ASSERT_EQ(0, streamer1->init(*index_meta_ptr_, params1));
  ASSERT_EQ(0, streamer1->open(storage));
  auto ctx1 = streamer1->create_context();

  IndexStreamer::Pointer streamer2 =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_NE(streamer2, nullptr);
  auto params2 = params;
  params2.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, cnt);
  ASSERT_EQ(0, streamer2->init(*index_meta_ptr_, params2));
  ASSERT_EQ(0, streamer2->open(storage));
  auto ctx2 = streamer2->create_context();

  // do searcher
  size_t cost1 = 0;
  size_t cost2 = 0;
  for (size_t i = 0; i < 100; ++i) {
    auto t1 = ailego::Monotime::MicroSeconds();
    ASSERT_EQ(0, streamer1->search_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx1));
    auto t2 = ailego::Monotime::MicroSeconds();
    ASSERT_EQ(0, streamer2->search_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx2));
    auto t3 = ailego::Monotime::MicroSeconds();
    cost1 += t2 - t1;
    cost2 += t3 - t2;
  }

  ASSERT_LT(cost1, cost2);

  ailego::Params update_params;
  update_params.set(PARAM_HNSW_SPARSE_STREAMER_VISIT_BLOOMFILTER_ENABLE, true);
  update_params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 50);
  ctx1->set_debug_mode(true);
  ctx1->update(update_params);
  ASSERT_EQ(0, streamer1->search_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, ctx1));
  LOG_DEBUG("%s", ctx1->debug_string().c_str());
}

TEST_F(HnswSparseStreamerTest, TestFilter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  // constexpr size_t static sparse_dim_count = 64;

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 500);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 1000);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestFilter", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 100UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  ctx->set_topk(10U);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = (float)i + 1.0f;
    }

    streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                       sparse_velues.data(), qmeta, ctx);
    p_keys[0].push_back(i);
  }

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = -100.1;
  }
  ASSERT_EQ(0, streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(10, results.size());
  ASSERT_EQ(0, results[0].key());
  ASSERT_EQ(1, results[1].key());
  ASSERT_EQ(2, results[2].key());

  auto filterFunc = [](uint64_t key) {
    if (key == 0UL || key == 3UL) {
      return true;
    }
    return false;
  };
  ctx->set_filter(filterFunc);

  // after set filter
  ASSERT_EQ(0, streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                     sparse_velues.data(), qmeta, ctx));
  auto &results1 = ctx->result();
  ASSERT_EQ(10, results1.size());
  ASSERT_EQ(1, results1[0].key());
  ASSERT_EQ(2, results1[1].key());
  ASSERT_EQ(4, results1[2].key());

  // linear
  ASSERT_EQ(0, streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx));
  auto &results2 = ctx->result();
  ASSERT_EQ(10, results2.size());
  ASSERT_EQ(1, results2[0].key());
  ASSERT_EQ(2, results2[1].key());
  ASSERT_EQ(4, results2[2].key());

  // linear by p_keys
  ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(
                   sparse_dim_count, sparse_indices.data(),
                   sparse_velues.data(), p_keys, qmeta, ctx));
  auto &results3 = ctx->result();
  ASSERT_EQ(10, results3.size());
  ASSERT_EQ(1, results3[0].key());
  ASSERT_EQ(2, results3[1].key());
  ASSERT_EQ(4, results3[2].key());
}

TEST_F(HnswSparseStreamerTest, TestMaxIndexSize) {
  constexpr size_t static sparse_dim_count = 128;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;

  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params());
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestMaxIndexSize", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t vsz0 = 0;
  size_t rss0 = 0;
  if (!ailego::MemoryHelper::SelfUsage(&vsz0, &rss0)) {
    // do not check if get mem usage failed
    return;
  }
  if (vsz0 > 1024 * 1024 * 1024 * 1024UL) {
    // asan mode
    return;
  }

  size_t writeCnt1 = 10000;
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  auto ctx = streamer->create_context();

  for (size_t i = 0; i < writeCnt1; ++i) {
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                    sparse_velues.data(), qmeta, ctx));
  }
  size_t vsz1 = 0;
  size_t rss1 = 0;
  ailego::MemoryHelper::SelfUsage(&vsz1, &rss1);
  size_t increment = rss1 - rss0;

  size_t total_write =
      writeCnt1 * sparse_dim_count * (sizeof(uint16_t) + sizeof(float)) +
      writeCnt1 * 32 + writeCnt1 * 100 * 4;

  ASSERT_GT(total_write, increment * 0.8f);
  ASSERT_LT(total_write, increment * 1.2f);

  LOG_DEBUG("total write: %zu, increment: %zu", total_write, increment);

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(HnswSparseStreamerTest, TestKnnCleanUp) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage1 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage1);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage1->init(stg_params));
  ASSERT_EQ(0, storage1->open(dir_ + "TestKnnCluenUp1", true));
  ailego::Params params;

  constexpr size_t static sparse_dim_count1 = 32;
  IndexMeta meta1(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta1.set_metric("InnerProductSparse", 0, ailego::Params());

  ASSERT_EQ(0, streamer->init(meta1, params));
  ASSERT_EQ(0, streamer->open(storage1));
  IndexQueryMeta qmeta1(IndexMeta::MetaType::MT_SPARSE,
                        IndexMeta::DataType::DT_FP32);
  auto ctx1 = streamer->create_context();

  NumericalVector<uint32_t> sparse_indices1(sparse_dim_count1);
  NumericalVector<float> sparse_velues1(sparse_dim_count1);

  for (size_t j = 0; j < sparse_dim_count1; ++j) {
    sparse_indices1[j] = j * 20;
    sparse_velues1[j] = 1.1f;
  }
  ASSERT_EQ(0, streamer->add_impl(1, sparse_dim_count, sparse_indices1.data(),
                                  sparse_velues1.data(), qmeta1, ctx1));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, streamer->cleanup());

  auto storage2 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage2);
  ASSERT_EQ(0, storage2->init(stg_params));
  ASSERT_EQ(0, storage2->open(dir_ + "TestKnnCluenUp2", true));

  constexpr size_t static sparse_dim_count2 = 64;
  IndexMeta meta2(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta2.set_metric("InnerProductSparse", 0, ailego::Params());

  ASSERT_EQ(0, streamer->init(meta2, params));
  ASSERT_EQ(0, streamer->open(storage2));
  IndexQueryMeta qmeta2(IndexMeta::MetaType::MT_SPARSE,
                        IndexMeta::DataType::DT_FP32);
  auto ctx2 = streamer->create_context();

  NumericalVector<uint32_t> sparse_indices2(sparse_dim_count2);
  NumericalVector<float> sparse_velues2(sparse_dim_count2);

  for (size_t j = 0; j < sparse_dim_count2; ++j) {
    sparse_indices2[j] = j * 20;
    sparse_velues2[j] = 1.1f;
  }

  ASSERT_EQ(0, streamer->add_impl(2, sparse_dim_count, sparse_indices1.data(),
                                  sparse_velues1.data(), qmeta2, ctx2));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, streamer->cleanup());
}

TEST_F(HnswSparseStreamerTest, TestIndexSizeQuota) {
  constexpr size_t static sparse_dim_count = 512;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestIndexSizeQuota", true));
  ailego::Params params;

  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params());
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_INDEX_SIZE, 2 * 1024 * 1024U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_CHUNK_SIZE, 100 * 1024U);
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t writeCnt1 = 850;
  int ret = 0;
  auto ctx = streamer->create_context();
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);

  for (size_t i = 0; i < writeCnt1; ++i) {
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    int iRet = streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx);
    if (iRet != 0) {
      ret = iRet;
    }
  }

  ASSERT_EQ(IndexError_IndexFull, ret);
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, streamer->cleanup());
}

TEST_F(HnswSparseStreamerTest, TestBloomFilter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestBloomFilter", true));
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 100);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_SPARSE_STREAMER_VISIT_BLOOMFILTER_ENABLE, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  ctx->set_topk(10U);
  size_t cnt = 5000;
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                       sparse_velues.data(), qmeta, ctx);

    if ((i + 1) % 10 == 0) {
      ASSERT_EQ(0,
                streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, ctx));
      auto &results = ctx->result();
      ASSERT_EQ(10, results.size());
    }
  }
}

TEST_F(HnswSparseStreamerTest, TestStreamerParams) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestStreamerParams", true));
  ailego::Params params;
  params.set("proxima.hnsw.sparse_streamer.docs_hard_limit", 5);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  auto ctx = streamer->create_context();

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 1.1f;
  }

  ASSERT_EQ(0, streamer->add_impl(1, sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(2, sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(3, sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(4, sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(5, sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx));

  ASSERT_EQ(IndexError_IndexFull,
            streamer->add_impl(6, sparse_dim_count, sparse_indices.data(),
                               sparse_velues.data(), qmeta, ctx));
}

TEST_F(HnswSparseStreamerTest, TestCheckStats) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  std::string path = dir_ + "/TestCheckStats.index";
  ASSERT_EQ(0, storage->open(path, true));
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 100);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_FILTER_SAME_KEY, true);
  params.set(PARAM_HNSW_SPARSE_STREAMER_CHUNK_SIZE, 512 * 1024U);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto &stats = streamer->stats();
  ASSERT_EQ(0U, stats.revision_id());
  ASSERT_EQ(0U, stats.loaded_count());
  ASSERT_EQ(0U, stats.added_count());
  ASSERT_EQ(0U, stats.discarded_count());
  // header chunk + meta chunk
  size_t init_size = ailego::MemoryHelper::PageSize() * 2;
  ASSERT_EQ(init_size, stats.index_size());
  ASSERT_EQ(0U, stats.dumped_size());
  ASSERT_EQ(0U, stats.check_point());
  auto createTime = stats.create_time();
  auto updateTime = stats.update_time();
  ASSERT_GT(createTime, 0UL);
  ASSERT_EQ(createTime, updateTime);

  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  size_t cnt = 3000;
  size_t size1 = stats.index_size();
  size_t size2 = 0;

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                    sparse_velues.data(), qmeta, ctx));

    ASSERT_EQ(i + 1, stats.added_count());
    if (i == 0UL) {
      size2 = stats.index_size();
    }
  }

  size_t size3 = stats.index_size();
  ASSERT_GT(size2, size1);
  ASSERT_GT(size3, size2);
  LOG_INFO("size1=%zu size2=%zu size3=%zu", size1, size2, size3);

  uint64_t checkPoint = 23423UL;
  streamer->flush(checkPoint);
  size_t size4 = stats.index_size();
  ASSERT_EQ(size3, size4);
  auto stats1 = streamer->stats();
  ASSERT_EQ(1U, stats1.revision_id());
  ASSERT_EQ(0U, stats1.loaded_count());
  ASSERT_EQ(cnt, stats1.added_count());
  ASSERT_EQ(0U, stats1.discarded_count());
  ASSERT_GT(stats1.index_size(), 0U);
  ASSERT_EQ(0U, stats1.dumped_size());
  ASSERT_EQ(checkPoint, stats1.check_point());
  auto createTime1 = stats1.create_time();
  auto updateTime1 = stats1.update_time();
  ASSERT_GE(updateTime1, createTime1);
  ASSERT_EQ(createTime, createTime1);
  streamer->close();

  ASSERT_EQ(0, streamer->open(storage));
  auto &stats2 = streamer->stats();
  ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);

  ASSERT_EQ(0,
            streamer->add_impl(10000UL, sparse_dim_count, sparse_indices.data(),
                               sparse_velues.data(), qmeta, ctx));

  ASSERT_EQ(2U, stats2.revision_id());
  ASSERT_EQ(cnt, stats2.loaded_count());
  ASSERT_EQ(1U, stats2.added_count());
  ASSERT_EQ(0U, stats2.discarded_count());
  ASSERT_GT(stats1.index_size(), 0);
  ASSERT_EQ(0U, stats2.dumped_size());
  ASSERT_EQ(checkPoint, stats2.check_point());
  auto createTime2 = stats2.create_time();
  auto updateTime2 = stats2.update_time();
  ASSERT_EQ(createTime2, createTime1);
  ASSERT_GE(updateTime2, updateTime1);

  sleep(1);
  streamer->flush(checkPoint + 1);

  ASSERT_NE(0, streamer->add_impl(0U, sparse_dim_count, sparse_indices.data(),
                                  sparse_velues.data(), qmeta, ctx));

  auto &stats3 = streamer->stats();
  ASSERT_EQ(2U, stats3.revision_id());
  ASSERT_EQ(cnt, stats3.loaded_count());
  ASSERT_EQ(1U, stats3.added_count());
  ASSERT_EQ(1U, stats3.discarded_count());
  ASSERT_EQ(stats2.index_size(), stats3.index_size());
  ASSERT_EQ(0U, stats3.dumped_size());
  ASSERT_EQ(checkPoint + 1, stats3.check_point());
  auto createTime3 = stats3.create_time();
  auto updateTime3 = stats3.update_time();
  ASSERT_EQ(createTime3, createTime1);
  ASSERT_GT(updateTime3, updateTime2);

  auto dpath = dir_ + "/dumpIndex";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(dpath));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  size_t doc_cnt = stats3.loaded_count() + stats3.added_count();
  struct stat st;
  ASSERT_EQ(3001UL, doc_cnt);
  ASSERT_EQ(0, stat(dpath.c_str(), &st));
  ASSERT_LT(st.st_size - stats3.dumped_size(), 8192);

  streamer->close();
}

TEST_F(HnswSparseStreamerTest, TestCheckDuplicateAndGetVector) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestCheckDuplicateAndGetVector", true));
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_FILTER_SAME_KEY, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);

  NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
  NumericalVector<float> sparse_velues(sparse_dim_count);

  for (size_t i = 0; i < 1000; i++) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                    sparse_velues.data(), qmeta, ctx));
  }

  for (size_t i = 0; i < 1000; i += 10) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    ASSERT_EQ(IndexError_Duplicate,
              streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                 sparse_velues.data(), qmeta, ctx));
  }

  // check getVector
  auto provider = streamer->create_sparse_provider();
  for (size_t i = 0; i < 1000; i++) {
    uint32_t sparse_count;
    std::string sparse_indices_buffer;
    std::string sparse_values_buffer;

    ASSERT_EQ(
        0, provider->get_sparse_vector(i, &sparse_count, &sparse_indices_buffer,
                                       &sparse_values_buffer));

    const float *sparse_values_ptr =
        reinterpret_cast<const float *>(sparse_values_buffer.data());
    ASSERT_EQ(sparse_count, sparse_dim_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      ASSERT_FLOAT_EQ(sparse_values_ptr[j], i);
    }
  }

  streamer->flush(0UL);
  streamer.reset();
}

class TestDumper : public IndexDumper {
  virtual int init(const ailego::Params &) {
    return 0;
  }
  virtual int cleanup(void) {
    return 0;
  }
  virtual int create(const std::string &path) {
    return 0;
  }
  virtual uint32_t magic(void) const {
    return 0;
  }
  virtual int close(void) {
    return 0;
  }
  virtual int append(const std::string &id, size_t data_size,
                     size_t padding_size, uint32_t crc) {
    usleep(100000);
    return 0;
  }
  virtual size_t write(const void *data, size_t len) {
    return len;
  }
};

TEST_F(HnswSparseStreamerTest, TestDumpIndexAndAdd) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestDumpIndexAndAdd", true));
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  ASSERT_NE(nullptr, ctx);
  int code = 0;
  std::mutex mutex;

  auto addVector = [&](int a, int b) {
    mutex.unlock();
    for (int i = a; i < b; i++) {
      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = (float)i;
      }

      int ret = streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                   sparse_velues.data(), qmeta, ctx);
      if (ret != 0) {
        code = ret;
        ASSERT_EQ(IndexError_Unsupported, code);
        i = i - 1;  // retry
        usleep(10000);
      }
    }
  };

  mutex.lock();
  addVector(0, 2000);
  mutex.lock();
  auto t2 = std::async(std::launch::async, addVector, 2000, 3000);
  auto path1 = dir_ + "/dumpIndex1";
  auto dumper1 = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper1, nullptr);
  ASSERT_EQ(0, dumper1->create(path1));
  mutex.lock();  // sync: wait addVector start and release lock
  auto test_dumper = std::make_shared<TestDumper>();
  ASSERT_EQ(0, streamer->dump(test_dumper));
  mutex.unlock();
  ASSERT_EQ(0, streamer->dump(dumper1));
  ASSERT_EQ(0, dumper1->close());
  t2.get();
  streamer->close();
  ASSERT_EQ(IndexError_Unsupported, code);

  // check dump index
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->init(ailego::Params()));
  ASSERT_EQ(0, container->open(path1, false));
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto iter = searcher->create_sparse_provider()->create_iterator();

  size_t docs = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const uint32_t sparse_count = iter->sparse_count();
    ASSERT_EQ(sparse_count, sparse_dim_count);

    const float *data = reinterpret_cast<const float *>(iter->sparse_data());
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      ASSERT_FLOAT_EQ((float)key, data[j]);
    }

    docs++;
    iter->next();
  }

  ASSERT_GE(docs, 2000U);

  // check streamer
  ASSERT_EQ(0, streamer->open(storage));
  iter = streamer->create_sparse_provider()->create_iterator();

  docs = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const uint32_t sparse_count = iter->sparse_count();
    ASSERT_EQ(sparse_count, sparse_dim_count);

    const float *data = reinterpret_cast<const float *>(iter->sparse_data());
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      ASSERT_FLOAT_EQ((float)key, data[j]);
    }

    docs++;
    iter->next();
  }

  ASSERT_EQ(docs, 3000U);
}

TEST_F(HnswSparseStreamerTest, TestProvider) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestProvider.index", true));
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);

  //! prepare data
  size_t docs = 10000UL;
  srand(ailego::Realtime::MilliSeconds());
  std::vector<key_t> keys(docs);
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < docs; ++i) {
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

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < keys.size(); i++) {
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = keys[i];
    }

    ASSERT_EQ(
        0, streamer->add_impl(keys[i], sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx));
  }

  auto path1 = dir_ + "/TestProvider";
  auto dumper1 = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper1, nullptr);
  ASSERT_EQ(0, dumper1->create(path1));
  ASSERT_EQ(0, streamer->dump(dumper1));
  ASSERT_EQ(0, dumper1->close());
  streamer->close();

  // check dump index
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSparseSearcher");
  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, container->init(ailego::Params()));
  ASSERT_EQ(0, container->open(path1, false));
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto iter = searcher->create_sparse_provider()->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();

    const uint32_t sparse_count = iter->sparse_count();
    ASSERT_EQ(sparse_count, sparse_dim_count);

    const float *data = reinterpret_cast<const float *>(iter->sparse_data());
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      ASSERT_FLOAT_EQ((float)key, data[j]);
    }

    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, docs);

  // check streamer
  ASSERT_EQ(0, streamer->open(storage));
  iter = streamer->create_sparse_provider()->create_iterator();
  cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();

    const uint32_t sparse_count = iter->sparse_count();
    ASSERT_EQ(sparse_count, sparse_dim_count);

    const float *data = reinterpret_cast<const float *>(iter->sparse_data());
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      ASSERT_FLOAT_EQ((float)key, data[j]);
    }

    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, docs);

  auto searcher_provider = searcher->create_sparse_provider();
  auto streamer_provider = streamer->create_sparse_provider();
  for (size_t i = 0; i < keys.size(); ++i) {
    {
      uint32_t sparse_count;
      std::string sparse_indices_buffer;
      std::string sparse_values_buffer;

      ASSERT_EQ(0, searcher_provider->get_sparse_vector(keys[i], &sparse_count,
                                                        &sparse_indices_buffer,
                                                        &sparse_values_buffer));

      const float *sparse_values_ptr =
          reinterpret_cast<const float *>(sparse_values_buffer.data());
      ASSERT_EQ(sparse_count, sparse_dim_count);
      for (size_t j = 0; j < sparse_count; ++j) {
        ASSERT_FLOAT_EQ(sparse_values_ptr[j], keys[i]);
      }
    }

    {
      uint32_t sparse_count;
      std::string sparse_indices_buffer;
      std::string sparse_values_buffer;
      ASSERT_EQ(0, streamer_provider->get_sparse_vector(keys[i], &sparse_count,
                                                        &sparse_indices_buffer,
                                                        &sparse_values_buffer));

      const float *sparse_values_ptr =
          reinterpret_cast<const float *>(sparse_values_buffer.data());
      ASSERT_EQ(sparse_count, sparse_dim_count);
      for (size_t j = 0; j < sparse_count; ++j) {
        ASSERT_FLOAT_EQ(sparse_values_ptr[j], keys[i]);
      }
    }
  }

  ASSERT_EQ(index_meta_ptr_->data_type(), streamer_provider->data_type());
}

TEST_F(HnswSparseStreamerTest, TestSharedContext) {
  auto create_streamer = [](std::string path) {
    IndexStreamer::Pointer streamer =
        IndexFactory::CreateStreamer("HnswSparseStreamer");
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ailego::Params stg_params;
    storage->init(stg_params);
    storage->open(path, true);
    ailego::Params params;
    streamer->init(*index_meta_ptr_, params);
    streamer->open(storage);
    return streamer;
  };
  auto streamer1 = create_streamer(dir_ + "TestSharedContext.index1");
  auto streamer2 = create_streamer(dir_ + "TestSharedContext.index2");
  auto streamer3 = create_streamer(dir_ + "TestSharedContext.index3");

  srand(ailego::Realtime::MilliSeconds());
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  auto do_test = [&](int start) {
    auto code = rand() % 3;
    IndexStreamer::Context::Pointer ctx;
    switch (code) {
      case 0:
        ctx = streamer1->create_context();
        break;
      case 1:
        ctx = streamer2->create_context();
        break;
      case 2:
        ctx = streamer3->create_context();
        break;
    };
    ctx->set_topk(1);
    uint64_t key1 = start + 0;
    uint64_t key2 = start + 1;
    uint64_t key3 = start + 2;

    NumericalVector<uint32_t> query_sparse_indices(sparse_dim_count);
    NumericalVector<float> query_sparse_velues(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      query_sparse_indices[j] = j * 20;
      query_sparse_velues[j] = 1.1f;
    }

    for (int i = 0; i < 1000; ++i) {
      NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
      NumericalVector<float> sparse_velues(sparse_dim_count);

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        sparse_indices[j] = j * 20;
        sparse_velues[j] = rand();
      }

      int ret = 0;
      auto code = rand() % 3;
      switch (code) {
        case 0:
          streamer1->add_impl(key1, sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx);
          key1 += 3;
          ret = streamer1->search_impl(sparse_dim_count,
                                       query_sparse_indices.data(),
                                       query_sparse_velues.data(), qmeta, ctx);
          break;
        case 1:
          streamer2->add_impl(key2, sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx);
          key2 += 3;
          streamer2->add_impl(key2, sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx);
          key2 += 3;
          ret = streamer2->search_impl(sparse_dim_count,
                                       query_sparse_indices.data(),
                                       query_sparse_velues.data(), qmeta, ctx);
          break;
        case 2:
          streamer3->add_impl(key3, sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx);
          key3 += 3;
          streamer3->add_impl(key3, sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx);
          key3 += 3;
          streamer3->add_impl(key3, sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx);
          key3 += 3;
          ret = streamer3->search_impl(sparse_dim_count,
                                       query_sparse_indices.data(),
                                       query_sparse_velues.data(), qmeta, ctx);
          break;
      }
      EXPECT_EQ(0, ret);
      auto &results = ctx->result();
      EXPECT_EQ(1, results.size());
      EXPECT_EQ(code, results[0].key() % 3);
    }
  };

  auto t1 = std::async(std::launch::async, do_test, 0);
  auto t2 = std::async(std::launch::async, do_test, 30000000);
  t1.wait();
  t2.wait();
}

TEST_F(HnswSparseStreamerTest, TestBruteForceSetupInContext) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  // params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0,
            storage->open(dir_ + "/TestBruteForceSetupInContext.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count, sparse_indices.data(),
                                    sparse_velues.data(), qmeta, ctx));
  }

  size_t topk = 20;
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  bool set_bf_threshold = false;
  bool use_update = false;

  size_t step = 50;
  for (size_t i = 0; i < cnt; i += step) {
    // for (size_t i = 0; i < cnt; i++) {
    auto linearCtx = streamer->create_context();
    auto knnCtx = streamer->create_context();

    ASSERT_TRUE(!!linearCtx);
    ASSERT_TRUE(!!knnCtx);

    linearCtx->set_topk(topk);
    knnCtx->set_topk(topk);

    NumericalVector<uint32_t> query_sparse_indices(sparse_dim_count);
    NumericalVector<float> query_sparse_velues(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      query_sparse_indices[j] = j * 20;
      query_sparse_velues[j] = i + 0.1f;
    }

    auto t1 = ailego::Realtime::MicroSeconds();

    if (set_bf_threshold) {
      if (use_update) {
        ailego::Params streamerParamsExtra;

        streamerParamsExtra.set(
            "proxima.hnsw.sparse_streamer.brute_force_threshold", cnt);
        knnCtx->update(streamerParamsExtra);
      } else {
        knnCtx->set_bruteforce_threshold(cnt);
      }

      use_update = !use_update;
    }
    ASSERT_EQ(
        0, streamer->search_impl(sparse_dim_count, query_sparse_indices.data(),
                                 query_sparse_velues.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, streamer->search_bf_impl(
                     sparse_dim_count, query_sparse_indices.data(),
                     query_sparse_velues.data(), qmeta, linearCtx));

    // auto t3 = ailego::Realtime::MicroSeconds();

    if (set_bf_threshold) {
      linearTotalTime += t2 - t1;
    } else {
      knnTotalTime += t2 - t1;
    }

    set_bf_threshold = !set_bf_threshold;

    auto &knnResult = knnCtx->result();
    // ASSERT_EQ(topk, knnResult.size());
    topk1Hits += cnt - 1 == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(cnt - 1, linearResult[0].key());

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

TEST_F(HnswSparseStreamerTest, TestQueryFilteringRatio) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

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
  params.set(PARAM_HNSW_SPARSE_STREAMER_QUERY_FILTERING_RATIO, 0.05);

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestQueryFilteringRatio", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 20000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  auto linearCtx = streamer->create_context();
  ASSERT_TRUE(!!linearCtx);

  auto knnCtx = streamer->create_context();
  ASSERT_TRUE(!!knnCtx);

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

  // streamer->print_debug_info();
  size_t topk = 200;
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

    ASSERT_EQ(0, streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, linearCtx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    // std::cout << "i: " << i << std::endl;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
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
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSparseStreamerTest, TestAddAndSearchWithID) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  size_t sparse_dim_count = 32;

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 20);
  params.set(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 5);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGeneral", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 20000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  auto linearCtx = streamer->create_context();
  ASSERT_TRUE(!!linearCtx);

  auto knnCtx = streamer->create_context();
  ASSERT_TRUE(!!knnCtx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);

  for (size_t i = 0; i < cnt; i += 4) {
    ASSERT_EQ(0, streamer->add_with_id_impl(
                     i, sparse_dim_count, sparse_indices_list[i].data(),
                     sparse_vec_list[i].data(), qmeta, ctx));
  }

  for (size_t i = 2; i < cnt; i += 4) {
    ASSERT_EQ(0, streamer->add_with_id_impl(
                     i, sparse_dim_count, sparse_indices_list[i].data(),
                     sparse_vec_list[i].data(), qmeta, ctx));
  }

  // streamer->print_debug_info();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  for (size_t i = 0; i < cnt / 100; i += 2) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    auto t1 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0,
              streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, linearCtx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    // std::cout << "i: " << i << std::endl;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
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

      auto func = [&](const IndexDocumentList &result) {
        for (size_t j = 0; j < topk / 10; ++j) {
          ASSERT_NE(result[j].key(), -1LLU);
          ASSERT_NE(result[j].index(), -1LLU);
          uint32_t sparse_count = 0;
          std::string sparse_indices_buffer;
          std::string sparse_values_buffer;
          ASSERT_EQ(0, streamer->get_sparse_vector_by_id(
                           result[j].index(), &sparse_count,
                           &sparse_indices_buffer, &sparse_values_buffer));
          ASSERT_EQ(sparse_dim_count, sparse_count);

          const auto &_sparse_indices = sparse_indices_list[result[j].index()];
          const auto &_sparse_vec = sparse_vec_list[result[j].index()];
          std::string original_sparse_values_buffer;
          original_sparse_values_buffer.resize(_sparse_vec.size() *
                                               sizeof(float));
          memcpy((char *)original_sparse_values_buffer.data(),
                 (char *)_sparse_vec.data(),
                 _sparse_vec.size() * sizeof(float));

          ASSERT_EQ(sparse_indices_buffer, _sparse_indices);

          ASSERT_EQ(sparse_values_buffer, original_sparse_values_buffer);
        }
      };

      func(linearResult);
      func(knnResult);
    }
  }
  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * 100.0f / (float(cnt) / 100);
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
  // EXPECT_GT(cost, 2.0f);
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif