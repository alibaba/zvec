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

#include "hnsw_rabitq_streamer.h"
#include <memory>
#include <gtest/gtest.h>
#include "zvec/ailego/container/params.h"
#include "zvec/core/framework/index_holder.h"
#include "zvec/core/framework/index_streamer.h"
#include "hnsw_rabitq_streamer.h"
#include "rabitq_converter.h"
#include "rabitq_reformer.h"

using namespace std;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static dim = 128;

class HnswRabitqStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string HnswRabitqStreamerTest::dir_("hnswRabitqStreamerTest");
shared_ptr<IndexMeta> HnswRabitqStreamerTest::index_meta_ptr_;

void HnswRabitqStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void HnswRabitqStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(HnswRabitqStreamerTest, TestBuildAndSearch) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, ailego::Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer;
  ASSERT_EQ(converter.to_reformer(&index_reformer), 0);
  auto reformer = std::dynamic_pointer_cast<RabitqReformer>(index_reformer);
  IndexStreamer::Pointer streamer =
      std::make_shared<HnswRabitqStreamer>(holder, reformer);

  ailego::Params params;
  params.set("proxima.hnsw_rabitq.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw_rabitq.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw_rabitq.streamer.scaling_factor", 5U);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/AddVector", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  for (auto it = holder->create_iterator(); it->is_valid(); it->next()) {
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    ASSERT_EQ(0,
              streamer->add_impl(it->key(), it->data(), query_meta, context));
  }
  streamer->flush(0UL);

  // Perform search verification
  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

  context->set_topk(10);
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));

  const auto &result = context->result(0);
  ASSERT_GT(result.size(), 0UL);
  ASSERT_LE(result.size(), 10UL);

  // reopen and load reformer from storage
  ASSERT_EQ(0, streamer->close());
  IndexStreamer::Pointer new_streamer =
      std::make_shared<HnswRabitqStreamer>(holder);
  ASSERT_EQ(0, new_streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, new_streamer->open(storage));
}

TEST_F(HnswRabitqStreamerTest, TestLinearSearch) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, ailego::Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer;
  ASSERT_EQ(converter.to_reformer(&index_reformer), 0);
  auto reformer = std::dynamic_pointer_cast<RabitqReformer>(index_reformer);
  IndexStreamer::Pointer streamer =
      std::make_shared<HnswRabitqStreamer>(holder, reformer);

  ailego::Params params;
  params.set("proxima.hnsw_rabitq.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw_rabitq.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw_rabitq.streamer.scaling_factor", 5U);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearch", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  for (auto it = holder->create_iterator(); it->is_valid(); it->next()) {
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    ASSERT_EQ(0,
              streamer->add_impl(it->key(), it->data(), query_meta, context));
  }

  // Test linear search with exact match
  size_t topk = 3;
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  NumericalVector<float> query_vec(dim);

  for (size_t i = 0; i < doc_cnt; i += 100) {
    for (size_t j = 0; j < dim; ++j) {
      query_vec[j] = static_cast<float>(i);
    }
    context->set_topk(1U);
    ASSERT_EQ(0,
              streamer->search_bf_impl(query_vec.data(), query_meta, context));
    auto &result1 = context->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(i, result1[0].key());

    // Test with slight offset
    for (size_t j = 0; j < dim; ++j) {
      query_vec[j] = static_cast<float>(i) + 0.1f;
    }
    context->set_topk(topk);
    ASSERT_EQ(0,
              streamer->search_bf_impl(query_vec.data(), query_meta, context));
    auto &result2 = context->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
  }
}

TEST_F(HnswRabitqStreamerTest, TestKnnSearch) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 2000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, ailego::Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer;
  ASSERT_EQ(converter.to_reformer(&index_reformer), 0);
  auto reformer = std::dynamic_pointer_cast<RabitqReformer>(index_reformer);
  IndexStreamer::Pointer streamer =
      std::make_shared<HnswRabitqStreamer>(holder, reformer);

  ailego::Params params;
  params.set("proxima.hnsw_rabitq.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw_rabitq.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw_rabitq.streamer.scaling_factor", 10U);
  params.set("proxima.hnsw_rabitq.streamer.efconstruction", 100U);
  params.set("proxima.hnsw_rabitq.streamer.ef", 50U);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestKnnSearch", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  for (auto it = holder->create_iterator(); it->is_valid(); it->next()) {
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    ASSERT_EQ(0,
              streamer->add_impl(it->key(), it->data(), query_meta, context));
  }

  // Compare KNN search with brute force search
  auto linear_ctx = streamer->create_context();
  auto knn_ctx = streamer->create_context();
  size_t topk = 50;
  linear_ctx->set_topk(topk);
  knn_ctx->set_topk(topk);

  int total_hits = 0;
  int total_cnts = 0;
  int topk1_hits = 0;
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  NumericalVector<float> query_vec(dim);

  for (size_t i = 0; i < doc_cnt; i += 100) {
    for (size_t j = 0; j < dim; ++j) {
      query_vec[j] = static_cast<float>(i) + 0.1f;
    }

    ASSERT_EQ(0,
              streamer->search_impl(query_vec.data(), query_meta, 1, knn_ctx));
    ASSERT_EQ(
        0, streamer->search_bf_impl(query_vec.data(), query_meta, linear_ctx));

    auto &knn_result = knn_ctx->result(0);
    ASSERT_EQ(topk, knn_result.size());
    topk1_hits += (i == knn_result[0].key());

    auto &linear_result = linear_ctx->result();
    ASSERT_EQ(topk, linear_result.size());
    ASSERT_EQ(i, linear_result[0].key());

    for (size_t k = 0; k < topk; ++k) {
      total_cnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linear_result[j].key() == knn_result[k].key()) {
          total_hits++;
          break;
        }
      }
    }
  }

  float recall = total_hits * 1.0f / total_cnts;
  float topk1_recall = topk1_hits * 100.0f / static_cast<float>(doc_cnt);
  EXPECT_GT(recall, 0.60f);
  // actual: no guarantee
  // TODO(jiliang.ljl): check if ok?
  EXPECT_GT(topk1_recall, 0.00f);
}

TEST_F(HnswRabitqStreamerTest, TestRandomData) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1500UL;

  // Add random vectors
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, ailego::Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer;
  ASSERT_EQ(converter.to_reformer(&index_reformer), 0);
  auto reformer = std::dynamic_pointer_cast<RabitqReformer>(index_reformer);
  IndexStreamer::Pointer streamer =
      std::make_shared<HnswRabitqStreamer>(holder, reformer);

  ailego::Params params;
  params.set("proxima.hnsw_rabitq.streamer.max_neighbor_count", 32U);
  params.set("proxima.hnsw_rabitq.streamer.upper_neighbor_count", 16U);
  params.set("proxima.hnsw_rabitq.streamer.scaling_factor", 20U);
  params.set("proxima.hnsw_rabitq.streamer.efconstruction", 200U);
  params.set("proxima.hnsw_rabitq.streamer.ef", 100U);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestRandomData", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  for (auto it = holder->create_iterator(); it->is_valid(); it->next()) {
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    ASSERT_EQ(0,
              streamer->add_impl(it->key(), it->data(), query_meta, context));
  }

  // Test with random queries
  auto linear_ctx = streamer->create_context();
  auto knn_ctx = streamer->create_context();
  size_t topk = 50;
  linear_ctx->set_topk(topk);
  knn_ctx->set_topk(topk);

  int total_hits = 0;
  int total_cnts = 0;
  int topk1_hits = 0;
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  NumericalVector<float> query_vec(dim);

  size_t query_cnt = 200;
  for (size_t i = 0; i < query_cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      query_vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    ASSERT_EQ(
        0, streamer->search_bf_impl(query_vec.data(), query_meta, linear_ctx));
    ASSERT_EQ(0,
              streamer->search_impl(query_vec.data(), query_meta, 1, knn_ctx));

    auto &knn_result = knn_ctx->result(0);
    ASSERT_EQ(topk, knn_result.size());

    auto &linear_result = linear_ctx->result();
    ASSERT_EQ(topk, linear_result.size());

    topk1_hits += (linear_result[0].key() == knn_result[0].key());

    for (size_t k = 0; k < topk; ++k) {
      total_cnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linear_result[j].key() == knn_result[k].key()) {
          total_hits++;
          break;
        }
      }
    }
  }

  float recall = total_hits * 1.0f / total_cnts;
  float topk1_recall = topk1_hits * 1.0f / query_cnt;
  EXPECT_GT(recall, 0.50f);
  EXPECT_GT(topk1_recall, 0.70f);
}

TEST_F(HnswRabitqStreamerTest, TestOpenClose) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 500UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, ailego::Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer;
  ASSERT_EQ(converter.to_reformer(&index_reformer), 0);
  auto reformer = std::dynamic_pointer_cast<RabitqReformer>(index_reformer);

  ailego::Params params;
  params.set("proxima.hnsw_rabitq.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw_rabitq.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw_rabitq.streamer.scaling_factor", 5U);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestOpenClose", true));

  IndexStreamer::Pointer streamer =
      std::make_shared<HnswRabitqStreamer>(holder, reformer);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

  // Add first half of vectors
  for (size_t i = 0; i < doc_cnt / 2; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), query_meta, context));
  }

  ASSERT_EQ(0, streamer->flush(0UL));
  ASSERT_EQ(0, streamer->close());

  // Reopen and add second half
  IndexStreamer::Pointer streamer2 =
      std::make_shared<HnswRabitqStreamer>(holder);
  ASSERT_EQ(0, streamer2->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer2->open(storage));

  auto context2 = streamer2->create_context();
  for (size_t i = doc_cnt / 2; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer2->add_impl(i, vec.data(), query_meta, context2));
  }

  ASSERT_EQ(0, streamer2->flush(0UL));

  // Verify search works after reopen
  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = 10.0f;
  }

  context2->set_topk(5);
  ASSERT_EQ(0,
            streamer2->search_impl(query_vec.data(), query_meta, 1, context2));
  const auto &result = context2->result(0);
  ASSERT_EQ(5UL, result.size());
  ASSERT_EQ(10UL, result[0].key());
}

TEST_F(HnswRabitqStreamerTest, TestCreateIterator) {
  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 300UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, ailego::Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<IndexReformer> index_reformer;
  ASSERT_EQ(converter.to_reformer(&index_reformer), 0);
  auto reformer = std::dynamic_pointer_cast<RabitqReformer>(index_reformer);
  IndexStreamer::Pointer streamer =
      std::make_shared<HnswRabitqStreamer>(holder, reformer);

  ailego::Params params;
  params.set("proxima.hnsw_rabitq.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw_rabitq.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw_rabitq.streamer.scaling_factor", 5U);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestCreateIterator", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), query_meta, context));
  }

  streamer->flush(0UL);

  // Test iterator
  auto provider = streamer->create_provider();
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);

  size_t count = 0;
  while (iter->is_valid()) {
    ASSERT_EQ(count, iter->key());
    // const float *data = (const float *)iter->data();
    // for (size_t j = 0; j < dim; ++j) {
    //   ASSERT_EQ(static_cast<float>(count), data[j]);
    // }
    iter->next();
    count++;
  }
  ASSERT_EQ(doc_cnt, count);

  // Test get_vector
  // for (size_t i = 0; i < doc_cnt; i++) {
  //   const float *data = (const float *)provider->get_vector(i);
  //   ASSERT_NE(data, nullptr);
  //   for (size_t j = 0; j < dim; ++j) {
  //     ASSERT_EQ(static_cast<float>(i), data[j]);
  //   }
  // }
}

TEST_F(HnswRabitqStreamerTest, TestDimensions) {
  std::vector<size_t> dimensions = {1,    2,    4,    8,    16,   32,   33,
                                    63,   64,   128,  256,  512,  1024, 2047,
                                    2048, 2049, 4095, 4096, 4097, 8192, 16384};
  size_t doc_cnt = 100;

  for (size_t test_dim : dimensions) {
    std::cout << "Testing dimension: " << test_dim << std::endl;

    IndexMeta index_meta(IndexMeta::DataType::DT_FP32, test_dim);
    index_meta.set_metric("SquaredEuclidean", 0, ailego::Params());

    auto holder =
        make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(
            test_dim);
    IndexStreamer::Pointer streamer =
        std::make_shared<HnswRabitqStreamer>(holder);

    ailego::Params params;
    params.set("proxima.hnsw_rabitq.streamer.max_neighbor_count", 16U);
    params.set("proxima.hnsw_rabitq.streamer.upper_neighbor_count", 8U);
    params.set("proxima.hnsw_rabitq.streamer.scaling_factor", 5U);
    params.set("proxima.hnsw_rabitq.general.dimension", test_dim);

    int ret = streamer->init(index_meta, params);

    // dimension <= 63 or >= 4096: init() should return -31
    if (test_dim <= 63 || test_dim >= 4096) {
      ASSERT_EQ(-31, ret) << "expected init to fail with -31, dim=" << test_dim;
      std::cout << "Dimension " << test_dim
                << " correctly rejected with ret=" << ret << std::endl;
      continue;
    }

    // Valid dimensions: verify full streaming build succeeds
    ASSERT_EQ(0, ret) << "init failed, dim=" << test_dim;

    for (size_t i = 0; i < doc_cnt; i++) {
      NumericalVector<float> vec(test_dim);
      for (size_t j = 0; j < test_dim; ++j) {
        vec[j] = static_cast<float>(i * test_dim + j) / 1000.0f;
      }
      ASSERT_TRUE(holder->emplace(i, std::move(vec))) << "dim=" << test_dim;
    }

    RabitqConverter converter;
    converter.init(index_meta, ailego::Params());
    ASSERT_EQ(0, converter.train(holder))
        << "converter train failed, dim=" << test_dim;
    std::shared_ptr<IndexReformer> index_reformer;
    ASSERT_EQ(0, converter.to_reformer(&index_reformer)) << "dim=" << test_dim;
    auto reformer = std::dynamic_pointer_cast<RabitqReformer>(index_reformer);

    // Recreate streamer with reformer
    streamer = std::make_shared<HnswRabitqStreamer>(holder, reformer);
    ASSERT_EQ(0, streamer->init(index_meta, params))
        << "init with reformer failed, dim=" << test_dim;

    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, storage);
    ailego::Params stg_params;
    ASSERT_EQ(0, storage->init(stg_params));
    std::string storage_path =
        dir_ + "/TestDimensions_" + std::to_string(test_dim);
    ASSERT_EQ(0, storage->open(storage_path, true))
        << "storage open failed, dim=" << test_dim;
    ASSERT_EQ(0, streamer->open(storage))
        << "streamer open failed, dim=" << test_dim;

    auto context = streamer->create_context();
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, test_dim);
    for (auto it = holder->create_iterator(); it->is_valid(); it->next()) {
      ASSERT_EQ(0,
                streamer->add_impl(it->key(), it->data(), query_meta, context))
          << "add failed, dim=" << test_dim << ", key=" << it->key();
    }
    ASSERT_EQ(0, streamer->flush(0UL)) << "flush failed, dim=" << test_dim;

    std::cout << "Dimension " << test_dim << " passed" << std::endl;
  }
}

}  // namespace core
}  // namespace zvec
