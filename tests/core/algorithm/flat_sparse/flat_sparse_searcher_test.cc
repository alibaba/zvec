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

#include <future>
#include <iostream>
#include <random>
#include <vector>
#include <ailego/math/norm2_matrix.h>
#include <ailego/utility/math_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_meta.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

constexpr size_t static sparse_dim_count = 16;

class FlatSparseSearcherTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);
  void generate_sparse_data(
      size_t cnt, uint32_t sparse_dim_count,
      std::vector<NumericalVector<uint32_t>> &sparse_indices_list,
      std::vector<NumericalVector<float>> &sparse_vec_list, bool norm);

  static std::string dir_;
  static std::shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string FlatSparseSearcherTest::dir_("searcher_test/");
std::shared_ptr<IndexMeta> FlatSparseSearcherTest::index_meta_ptr_;

void FlatSparseSearcherTest::generate_sparse_data(
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
    Norm2Matrix<float, 1>::Compute(sparse_vec.data(), sparse_dim_count, &norm);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_vec[j] = sparse_vec[j] / norm;
    }

    sparse_indices_list.push_back(sparse_indices);
    sparse_vec_list.push_back(sparse_vec);
  }
}


void FlatSparseSearcherTest::SetUp(void) {
  IndexLoggerBroker::SetLevel(2);

  index_meta_ptr_.reset(new IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                                      IndexMeta::DataType::DT_FP32));
  index_meta_ptr_->set_metric("InnerProductSparse", 0, Params());

  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

void FlatSparseSearcherTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(FlatSparseSearcherTest, TestGeneral) {
  // init storage
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage != nullptr);
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGeneral", true));


  // init streamer
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, Params());

  Params params;
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // generate sparse data
  size_t sparse_dim_count = 32;
  size_t cnt = 100U;
  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  // test add data
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    sparse_vec_list[i].data(), qmeta, ctx));
  }

  // test get data
  uint32_t sparse_count;
  std::string sparse_indices_buffer;
  std::string sparse_values_buffer;
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(
        0, streamer->get_sparse_vector(i, &sparse_count, &sparse_indices_buffer,
                                       &sparse_values_buffer));
    ASSERT_EQ(sparse_dim_count, sparse_count);
    const uint32_t *sparse_indices_ptr =
        reinterpret_cast<const uint32_t *>(sparse_indices_buffer.data());
    const float *sparse_values_ptr =
        reinterpret_cast<const float *>(sparse_values_buffer.data());
    for (size_t j = 0; j < sparse_count; ++j) {
      ASSERT_EQ(sparse_indices_ptr[j], sparse_indices_list[i][j]);
      ASSERT_FLOAT_EQ(sparse_values_ptr[j], sparse_vec_list[i][j]);
      // std::cout << "1: " << sparse_values_ptr[j]
      //           << " 2: " << sparse_vec_list[i][j] << std::endl;
    }

    // must clear ^_^
    sparse_indices_buffer.clear();
    sparse_values_buffer.clear();
  }

  // test dump
  auto path = dir_ + "/TestGeneral_dump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher get vector
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("FlatSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(read_storage != nullptr);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->init(Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  // test searcher get data
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(
        0, searcher->get_sparse_vector(i, &sparse_count, &sparse_indices_buffer,
                                       &sparse_values_buffer));
    ASSERT_EQ(sparse_dim_count, sparse_count);
    const uint32_t *sparse_indices_ptr =
        reinterpret_cast<const uint32_t *>(sparse_indices_buffer.data());
    const float *sparse_values_ptr =
        reinterpret_cast<const float *>(sparse_values_buffer.data());
    for (size_t j = 0; j < sparse_count; ++j) {
      ASSERT_EQ(sparse_indices_ptr[j], sparse_indices_list[i][j]);
      ASSERT_FLOAT_EQ(sparse_values_ptr[j], sparse_vec_list[i][j]);
      // std::cout << "1: " << sparse_values_ptr[j]
      //           << " 2: " << sparse_vec_list[i][j] << std::endl;
    }

    // must clear ^_^
    sparse_indices_buffer.clear();
    sparse_values_buffer.clear();
  }
}

TEST_F(FlatSparseSearcherTest, TestStreamerDump) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_NE(streamer, nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestStreamerDump.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 10000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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
      IndexFactory::CreateSearcher("FlatSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  for (size_t i = 0; i < cnt; i += 50) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    auto t1 = Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, knnCtx));

    auto t2 = Realtime::MicroSeconds();

    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_vec.data(), qmeta, linearCtx));

    auto t3 = Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    auto &linearResult = linearCtx->result();

    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(topk, knnResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    for (size_t k = 0; k < topk; ++k) {
      ASSERT_EQ(linearResult[k].key(), knnResult[k].key());
    }
  }

  printf("linear: %zu, knn: %zu\n", (size_t)linearTotalTime,
         (size_t)knnTotalTime);
}

TEST_F(FlatSparseSearcherTest, TestLoadClose) {
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage != nullptr);
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGeneral", true));


  // init streamer
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, Params());

  Params params;
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // generate sparse data
  size_t sparse_dim_count = 32;
  size_t cnt = 100U;
  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  // test add data
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, streamer->add_impl(i, sparse_dim_count,
                                    sparse_indices_list[i].data(),
                                    sparse_vec_list[i].data(), qmeta, ctx));
  }

  // test get data
  uint32_t sparse_count;
  std::string sparse_indices_buffer;
  std::string sparse_values_buffer;
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(
        0, streamer->get_sparse_vector(i, &sparse_count, &sparse_indices_buffer,
                                       &sparse_values_buffer));
    ASSERT_EQ(sparse_dim_count, sparse_count);
    const uint32_t *sparse_indices_ptr =
        reinterpret_cast<const uint32_t *>(sparse_indices_buffer.data());
    const float *sparse_values_ptr =
        reinterpret_cast<const float *>(sparse_values_buffer.data());
    for (size_t j = 0; j < sparse_count; ++j) {
      ASSERT_EQ(sparse_indices_ptr[j], sparse_indices_list[i][j]);
      ASSERT_FLOAT_EQ(sparse_values_ptr[j], sparse_vec_list[i][j]);
      // std::cout << "1: " << sparse_values_ptr[j]
      //           << " 2: " << sparse_vec_list[i][j] << std::endl;
    }

    // must clear ^_^
    sparse_indices_buffer.clear();
    sparse_values_buffer.clear();
  }

  // test dump
  auto path = dir_ + "/TestGeneral_dump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher get vector
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("FlatSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(read_storage != nullptr);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->init(Params()));

  uint32_t loop = 5;
  while (loop--) {
    std::cout << "loop: " << loop << std::endl;

    ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

    // test searcher get data
    for (size_t i = 0; i < cnt; i++) {
      ASSERT_EQ(0, searcher->get_sparse_vector(i, &sparse_count,
                                               &sparse_indices_buffer,
                                               &sparse_values_buffer));
      ASSERT_EQ(sparse_dim_count, sparse_count);
      const uint32_t *sparse_indices_ptr =
          reinterpret_cast<const uint32_t *>(sparse_indices_buffer.data());
      const float *sparse_values_ptr =
          reinterpret_cast<const float *>(sparse_values_buffer.data());
      for (size_t j = 0; j < sparse_count; ++j) {
        ASSERT_EQ(sparse_indices_ptr[j], sparse_indices_list[i][j]);
        ASSERT_FLOAT_EQ(sparse_values_ptr[j], sparse_vec_list[i][j]);
        // std::cout << "1: " << sparse_values_ptr[j]
        //           << " 2: " << sparse_vec_list[i][j] << std::endl;
      }

      // must clear ^_^
      sparse_indices_buffer.clear();
      sparse_values_buffer.clear();
    }

    ASSERT_EQ(searcher->unload(), 0);
  }
}

TEST_F(FlatSparseSearcherTest, TestSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);

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

  // test dump
  auto path = dir_ + "/TestGeneral_dump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher get vector
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("FlatSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(read_storage != nullptr);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->init(Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  size_t step = 50;
  for (size_t i = 0; i < cnt; i += step) {
    // std::cout << "search " << i << std::endl;
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i + 1.0f;
    }

    ctx->set_topk(1U);
    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(0, result1[0].key());
    // std::cout << result1[0].key() << " " << result1[0].score() << std::endl;

    ctx->set_topk(3U);
    ASSERT_EQ(0,
              searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                       sparse_velues.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(3UL, result2.size());
    for (size_t i = 0; i < 3UL; ++i) {
      // std::cout << result2[i].key() << " " << result2[i].score() <<
      // std::endl;
      ASSERT_EQ(i, result2[i].key());
    }
  }

  ctx->set_topk(100U);
  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 10.1f;
  }

  ASSERT_EQ(0, searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  for (size_t i = 0; i < 100; ++i) {
    ASSERT_EQ(i, result[i].key());
  }
}

TEST_F(FlatSparseSearcherTest, TestSearchPKeys) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearchByKeys.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);

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

  // test dump
  auto path = dir_ + "/TestGeneral_dump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher get vector
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("FlatSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(read_storage != nullptr);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->init(Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  size_t topk = 3;
  for (size_t i = 0; i < cnt; i += 50) {
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = i + 1.0f;
    }
    ctx->set_topk(1U);
    ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
                     sparse_dim_count, sparse_indices.data(),
                     sparse_velues.data(), p_keys, qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(0, result1[0].key());

    ctx->set_topk(topk);
    ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
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
    ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
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
    ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
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
    ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
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

TEST_F(FlatSparseSearcherTest, TestMultiThread) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  constexpr size_t static sparse_dim_count = 32;
  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, Params());
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessKnnMultiThread", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
    IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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

  std::set<uint64_t> keys;

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
    keys.insert(iter->key());
    iter->next();
  }

  ASSERT_EQ(3000, keys.size());

  ASSERT_EQ(3000, total);
  ASSERT_EQ(0, min);
  ASSERT_EQ(2999, max);

  // test dump
  auto path = dir_ + "/TestGeneral_dump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher get vector
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("FlatSparseSearcher");
  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(read_storage != nullptr);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->init(Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  // ====== multi thread search
  size_t topk = 10;
  size_t cnt = 3000;
  auto knnSearch = [&]() {
    auto linearCtx = searcher->create_context();
    auto linearByPkeysCtx = searcher->create_context();
    auto ctx = searcher->create_context();
    IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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
                searcher->search_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, ctx));
      ASSERT_EQ(
          0, searcher->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_velues.data(), qmeta, linearCtx));
      std::vector<std::vector<uint64_t>> p_keys = {{cnt - 1, cnt - 2, cnt - 3}};
      ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
                       sparse_dim_count, sparse_indices.data(),
                       sparse_velues.data(), p_keys, qmeta, linearByPkeysCtx));
      auto &r1 = ctx->result();
      ASSERT_EQ(topk, r1.size());
      // std::cout << "r1 top1: " << r1[0].key() << ", score: " << r1[0].score()
      //           << std::endl;
      ASSERT_EQ(cnt - 1, r1[0].key());
      auto &r2 = linearCtx->result();
      ASSERT_EQ(topk, r2.size());
      // std::cout << "r2 top1: " << r2[0].key() << ", score: " << r2[0].score()
      //           << std::endl;
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
    ASSERT_FLOAT_EQ(1.0f, totalHits * 1.0f / totalCnts);
  };

  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  auto s3 = std::async(std::launch::async, knnSearch);
  s1.wait();
  s2.wait();
  s3.wait();
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif