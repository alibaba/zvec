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
#include <string>
#include <vector>
#include <ailego/math/norm2_matrix.h>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/memory_helper.h>
#include <algorithm/flat_sparse/flat_sparse_utility.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_streamer.h>

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

constexpr static size_t sparse_dim_count = 16;

class FlatSparseStreamerTest : public testing::Test {
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

std::string FlatSparseStreamerTest::dir_("streamer_test/");
std::shared_ptr<IndexMeta> FlatSparseStreamerTest::index_meta_ptr_;

void FlatSparseStreamerTest::generate_sparse_data(
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


void FlatSparseStreamerTest::SetUp(void) {
  IndexLoggerBroker::SetLevel(2);

  index_meta_ptr_.reset(new IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                                      IndexMeta::DataType::DT_FP32));
  index_meta_ptr_->set_metric("InnerProductSparse", 0, Params());

  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

void FlatSparseStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(FlatSparseStreamerTest, TestGeneral) {
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
}

TEST_F(FlatSparseStreamerTest, TestLinearSearch) {
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

  for (size_t i = 0; i < cnt; i++) {
    // std::cout << "search " << i << std::endl;
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
    // std::cout << result1[0].key() << " " << result1[0].score() << std::endl;

    ctx->set_topk(3U);
    ASSERT_EQ(0,
              streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
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

  ASSERT_EQ(0, streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  for (size_t i = 0; i < 100; ++i) {
    ASSERT_EQ(i, result[i].key());
  }
}

TEST_F(FlatSparseStreamerTest, TestLinearSearchByKeys) {
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

TEST_F(FlatSparseStreamerTest, TestCreateIterator) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
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
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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

  streamer->flush(0UL);
  streamer->close();
  ASSERT_EQ(0, streamer->open(storage));
  checkIter(cnt, streamer);

  // check getVector
  provider = streamer->create_sparse_provider();
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

TEST_F(FlatSparseStreamerTest, TestOpenAndClose) {
  constexpr size_t static sparse_dim_count = 2048;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, Params());
  Params params;
  auto storage1 = IndexFactory::CreateStorage("MMapFileStorage");
  auto storage2 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage1);
  ASSERT_NE(nullptr, storage2);
  Params stg_params;
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
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer1->init(meta, params));
  ASSERT_EQ(0, streamer1->open(storage1));

  IndexStreamer::Pointer streamer2 =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer2->init(meta, params));
  ASSERT_EQ(0, streamer2->open(storage2));

  checkIter(0, testCnt / 2, streamer1);
  checkIter(1, testCnt / 2, streamer2);
}

TEST_F(FlatSparseStreamerTest, TestNoInit) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  streamer->cleanup();
}

TEST_F(FlatSparseStreamerTest, TestForceFlush) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
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
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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

TEST_F(FlatSparseStreamerTest, TestMultiThread) {
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
      ASSERT_FLOAT_EQ((float)iter->key(), data[j]);
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

  // ====== multi thread search
  size_t topk = 10;
  size_t cnt = 3000;
  auto knnSearch = [&]() {
    auto linearCtx = streamer->create_context();
    auto linearByPkeysCtx = streamer->create_context();
    auto ctx = streamer->create_context();
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
    ASSERT_FLOAT_EQ(1.0f, totalHits * 1.0f / totalCnts);
  };

  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  auto s3 = std::async(std::launch::async, knnSearch);
  s1.wait();
  s2.wait();
  s3.wait();
}

TEST_F(FlatSparseStreamerTest, TestConcurrentAddAndSearch) {
  constexpr size_t static sparse_dim_count = 32;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;

  IndexMeta meta(IndexMeta::MetaType::MT_SPARSE, IndexMeta::DataType::DT_FP32);
  // meta.set_metric("InnerProductSparse", 0, Params());
  meta.set_metric("SquaredEuclideanSparse", 0, Params());
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessConcurrentAddAndSearch", true));
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

  auto knnSearch = [&]() {
    size_t topk = 100;
    size_t cnt = 3000;
    auto linearCtx = streamer->create_context();
    auto linearByPkeysCtx = streamer->create_context();
    auto ctx = streamer->create_context();
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
    ASSERT_TRUE(totalHits * 1.0f / totalCnts > 0.8f);
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

TEST_F(FlatSparseStreamerTest, TestFilter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestFilter", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 100UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  ctx->set_topk(10U);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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
  // for (int i = 0; i < 10; i++) {
  //   std::cout << "i: " << results3[i].key() << std::endl;
  // }

  ASSERT_EQ(1, results3[0].key());
  ASSERT_EQ(2, results3[1].key());
  ASSERT_EQ(4, results3[2].key());
}

TEST_F(FlatSparseStreamerTest, TestProvider) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestProvider.index", true));
  Params params;
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);

  //! prepare data
  size_t docs = 10000UL;
  srand(Realtime::MilliSeconds());
  std::vector<uint64_t> keys(docs);
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (true) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < docs; ++i) {
      keys[i] = dt(mt);
    }
  } else {
    std::iota(keys.begin(), keys.end(), 0U);
    std::transform(keys.begin(), keys.end(), keys.begin(),
                   [&](uint64_t k) { return step * k; });
    if (rand_order) {
      uint32_t seed = Realtime::Seconds();
      std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    }
  }

  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
  for (size_t i = 0; i < keys.size(); i++) {
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_velues(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_velues[j] = keys[i];
    }

    for (size_t j = 0; j < sparse_dim_count; j++) {
      ASSERT_FLOAT_EQ(sparse_velues[j], keys[i]);
    }

    ASSERT_EQ(
        0, streamer->add_impl(keys[i], sparse_dim_count, sparse_indices.data(),
                              sparse_velues.data(), qmeta, ctx));

    // std::cout << "i: " << i << " key: " << keys[i] << std::endl;
  }

  {
    // check streamer
    auto iter = streamer->create_sparse_provider()->create_iterator();
    size_t cnt = 0;
    while (iter->is_valid()) {
      auto key = iter->key();

      const uint32_t sparse_count = iter->sparse_count();
      ASSERT_EQ(sparse_count, sparse_dim_count);

      const float *data = reinterpret_cast<const float *>(iter->sparse_data());

      // std::cout << "cnt: " << cnt << " key: " << key
      //           << ", gt_key: " << keys[cnt] << std::endl;

      // for (size_t j = 0; j < sparse_count; ++j) {
      //   std::cout << "j: " << j << " data: " << data[j] << std::endl;
      // }

      for (size_t j = 0; j < sparse_dim_count; ++j) {
        ASSERT_FLOAT_EQ((float)key, data[j]);
      }

      cnt++;
      iter->next();
    }
    ASSERT_EQ(cnt, docs);
  }

  // dump
  // auto path1 = dir_ + "/TestProvider";
  // auto dumper1 = IndexFactory::CreateDumper("FileDumper");
  // ASSERT_NE(dumper1, nullptr);
  // ASSERT_EQ(0, dumper1->create(path1));
  // ASSERT_EQ(0, streamer->dump(dumper1));
  // ASSERT_EQ(0, dumper1->close());
  // streamer->close();

  // // check dump index
  // IndexSparseSearcher::Pointer searcher =
  //     IndexFactory::CreateSparseSearcher("FlatSparseSearcher");
  // auto container = IndexFactory::CreateStorage("MMapFileContainer");
  // ASSERT_EQ(0, container->init(Params()));
  // ASSERT_EQ(0, container->load(path1));
  // ASSERT_NE(searcher, nullptr);
  // ASSERT_EQ(0, searcher->init(Params()));
  // ASSERT_EQ(0, searcher->load(container, IndexSparseMeasure::Pointer()));
  // auto iter = searcher->create_sparse_provider()->create_iterator();
  // size_t cnt = 0;
  // while (iter->is_valid()) {
  //   auto key = iter->key();

  //   const uint32_t sparse_count = iter->sparse_count();
  //   ASSERT_EQ(sparse_count, sparse_dim_count);

  //   const float *data = reinterpret_cast<const float *>(iter->sparse_data());
  //   for (size_t j = 0; j < sparse_dim_count; ++j) {
  //     ASSERT_FLOAT_EQ((float)key, data[j]);
  //   }

  //   cnt++;
  //   iter->next();
  // }
  // ASSERT_EQ(cnt, docs);

  // // check streamer
  // ASSERT_EQ(0, streamer->open(storage));
  // iter = streamer->create_sparse_provider()->create_iterator();
  // cnt = 0;
  // while (iter->is_valid()) {
  //   auto key = iter->key();

  //   const uint32_t sparse_count = iter->sparse_count();
  //   ASSERT_EQ(sparse_count, sparse_dim_count);

  //   const float *data = reinterpret_cast<const float *>(iter->sparse_data());
  //   for (size_t j = 0; j < sparse_dim_count; ++j) {
  //     ASSERT_FLOAT_EQ((float)key, data[j]);
  //   }

  //   cnt++;
  //   iter->next();
  // }
  // ASSERT_EQ(cnt, docs);

  // auto searcher_provider = searcher->create_sparse_provider();
  // auto streamer_provider = streamer->create_sparse_provider();
  // for (size_t i = 0; i < keys.size(); ++i) {
  //   {
  //     uint32_t sparse_count;
  //     std::string sparse_indices_buffer;
  //     std::string sparse_values_buffer;

  //     ASSERT_EQ(0, searcher_provider->get_sparse_vector(keys[i],
  //     &sparse_count,
  //                                                       &sparse_indices_buffer,
  //                                                       &sparse_values_buffer));

  //     const float *sparse_values_ptr =
  //         reinterpret_cast<const float *>(sparse_values_buffer.data());
  //     ASSERT_EQ(sparse_count, sparse_dim_count);
  //     for (size_t j = 0; j < sparse_count; ++j) {
  //       ASSERT_FLOAT_EQ(sparse_values_ptr[j], keys[i]);
  //     }
  //   }

  //   {
  //     uint32_t sparse_count;
  //     std::string sparse_indices_buffer;
  //     std::string sparse_values_buffer;
  //     ASSERT_EQ(0, streamer_provider->get_sparse_vector(keys[i],
  //     &sparse_count,
  //                                                       &sparse_indices_buffer,
  //                                                       &sparse_values_buffer));

  //     const float *sparse_values_ptr =
  //         reinterpret_cast<const float *>(sparse_values_buffer.data());
  //     ASSERT_EQ(sparse_count, sparse_dim_count);
  //     for (size_t j = 0; j < sparse_count; ++j) {
  //       ASSERT_FLOAT_EQ(sparse_values_ptr[j], keys[i]);
  //     }
  //   }
  // }

  // ASSERT_EQ(index_meta_ptr_->type(), streamer_provider->vector_type());
}

TEST_F(FlatSparseStreamerTest, TestParamsMaxDocCount) {
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
  uint32_t max_doc_count = 100U;
  params.set(PARAM_FLAT_SPARSE_STREAMER_MAX_DOC_CNT, max_doc_count);
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // generate sparse data
  size_t sparse_dim_count = 32;
  size_t cnt = max_doc_count * 2;
  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  // test add data
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    auto ret =
        streamer->add_impl(i, sparse_dim_count, sparse_indices_list[i].data(),
                           sparse_vec_list[i].data(), qmeta, ctx);
    if (i < max_doc_count) {
      ASSERT_EQ(0, ret);
    } else {
      ASSERT_EQ(IndexError_IndexFull, ret);
    }
  }

  // test get data
  uint32_t sparse_count;
  std::string sparse_indices_buffer;
  std::string sparse_values_buffer;
  for (size_t i = 0; i < cnt; i++) {
    auto ret = streamer->get_sparse_vector(
        i, &sparse_count, &sparse_indices_buffer, &sparse_values_buffer);
    if (i < max_doc_count) {
      ASSERT_EQ(ret, 0);
      ASSERT_EQ(0, streamer->get_sparse_vector(i, &sparse_count,
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
    } else {
      ASSERT_EQ(ret, IndexError_NoExist);
    }
  }
}

TEST_F(FlatSparseStreamerTest, TestParamsDataChunkSize) {
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
  uint32_t data_chunk_size = 1024 * 1024;
  uint32_t max_data_chunk_cnt = 1;
  params.set(PARAM_FLAT_SPARSE_STREAMER_DATA_CHUNK_SIZE, data_chunk_size);
  params.set(PARAM_FLAT_SPARSE_STREAMER_MAX_DATA_CHUNK_CNT, max_data_chunk_cnt);
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // generate sparse data
  size_t sparse_dim_count = 128;
  size_t cnt = 2000;
  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  // test add data
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
  uint32_t insert_fail_idx = -1U;
  for (size_t i = 0; i < cnt; i++) {
    auto ret =
        streamer->add_impl(i, sparse_dim_count, sparse_indices_list[i].data(),
                           sparse_vec_list[i].data(), qmeta, ctx);
    if (insert_fail_idx != -1U) {
      ASSERT_EQ(ret, IndexError_IndexFull);
    }
    if (ret != 0 && insert_fail_idx == -1U) {
      insert_fail_idx = i;
    }
  }

  // test get data
  uint32_t sparse_count;
  std::string sparse_indices_buffer;
  std::string sparse_values_buffer;
  for (size_t i = 0; i < cnt; i++) {
    auto ret = streamer->get_sparse_vector(
        i, &sparse_count, &sparse_indices_buffer, &sparse_values_buffer);
    if (i < insert_fail_idx) {
      ASSERT_EQ(ret, 0);
      ASSERT_EQ(0, streamer->get_sparse_vector(i, &sparse_count,
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
    } else {
      ASSERT_EQ(ret, IndexError_NoExist);
    }
  }
}

TEST_F(FlatSparseStreamerTest, TestSharedContext) {
  auto create_streamer = [](std::string path) {
    IndexStreamer::Pointer streamer =
        IndexFactory::CreateStreamer("FlatSparseStreamer");
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    Params stg_params;
    storage->init(stg_params);
    storage->open(path, true);
    Params params;
    streamer->init(*index_meta_ptr_, params);
    streamer->open(storage);
    return streamer;
  };
  auto streamer1 = create_streamer(dir_ + "TestSharedContext.index1");
  auto streamer2 = create_streamer(dir_ + "TestSharedContext.index2");
  auto streamer3 = create_streamer(dir_ + "TestSharedContext.index3");

  srand(Realtime::MilliSeconds());
  IndexQueryMeta qmeta(IndexMeta::DT_FP32);
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

TEST_F(FlatSparseStreamerTest, TestGroupBy) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearchGroup.index", true));
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

  auto groupbyFunc = [](uint64_t key) {
    uint32_t group_id = key / 10 % 10;
    // std::cout << "key: " << key << ", group id: " << group_id << std::endl;
    return std::string("g_") + std::to_string(group_id);
  };

  size_t group_topk = 200;
  size_t group_num = 5;
  ctx->set_group_params(group_num, group_topk);
  ctx->set_group_by(groupbyFunc);

  std::vector<std::string> expect_group_ids = {
      "g_0", "g_1", "g_2", "g_3", "g_4", "g_5", "g_6", "g_7", "g_8", "g_9"};

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 10.1f;
  }

  auto t1 = Monotime::MicroSeconds();
  ASSERT_EQ(0, streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx));
  auto t2 = Monotime::MicroSeconds();
  std::cout << "Search time: " << (t2 - t1) << " us" << std::endl;

  auto &group_result = ctx->group_result();
  ASSERT_EQ(group_num, group_result.size());
  for (uint32_t i = 0; i < group_result.size(); ++i) {
    const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();
    std::cout << "Group ID: " << group_id << std::endl;

    ASSERT_EQ(group_id, expect_group_ids[i]);

    ASSERT_GE(result.size(), group_topk);

    for (uint32_t j = 0; j < result.size(); ++j) {
      ASSERT_EQ(result[j].key() / 10 % 10, i);
      // std::cout << "\tKey: " << result[j].key() << std::fixed
      //           << std::setprecision(3) << ", Score: " << result[j].score()
      //           << std::endl;
    }
  }
}

TEST_F(FlatSparseStreamerTest, TestGroupByNotEnoughNum) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearchGroup.index", true));
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

  auto groupbyFunc = [](uint64_t key) {
    uint32_t group_id = key / 10 % 10;
    // std::cout << "key: " << key << ", group id: " << group_id << std::endl;
    return std::string("g_") + std::to_string(group_id);
  };

  size_t group_topk = 200;
  size_t group_num = 12;
  ctx->set_group_params(group_num, group_topk);
  ctx->set_group_by(groupbyFunc);

  std::vector<std::string> expect_group_ids = {
      "g_0", "g_1", "g_2", "g_3", "g_4", "g_5", "g_6", "g_7", "g_8", "g_9"};

  for (size_t j = 0; j < sparse_dim_count; ++j) {
    sparse_indices[j] = j * 20;
    sparse_velues[j] = 10.1f;
  }

  auto t1 = Monotime::MicroSeconds();
  ASSERT_EQ(0, streamer->search_bf_impl(sparse_dim_count, sparse_indices.data(),
                                        sparse_velues.data(), qmeta, ctx));
  auto t2 = Monotime::MicroSeconds();
  std::cout << "Search time: " << (t2 - t1) << " us" << std::endl;

  auto &group_result = ctx->group_result();
  ASSERT_EQ(10, group_result.size());
  for (uint32_t i = 0; i < group_result.size(); ++i) {
    const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();
    std::cout << "Group ID: " << group_id << std::endl;

    ASSERT_EQ(group_id, expect_group_ids[i]);

    ASSERT_GE(result.size(), group_topk);

    for (uint32_t j = 0; j < result.size(); ++j) {
      ASSERT_EQ(result[j].key() / 10 % 10, i);
      // std::cout << "\tKey: " << result[j].key() << std::fixed
      //           << std::setprecision(3) << ", Score: " << result[j].score()
      //           << std::endl;
    }
  }
}

TEST_F(FlatSparseStreamerTest, TestAddAndSearchWithID) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_NE(streamer, nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGroup.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto ctx = streamer->create_context();
  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  constexpr size_t cnt = 1000U;
  constexpr size_t sparse_dim_count = cnt;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i += 2) {
    // prepare sparse
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_vec(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j;
      sparse_vec[j] = (i == j ? 1.0f : 0.0f);
    }
    streamer->add_with_id_impl(i, sparse_dim_count, sparse_indices.data(),
                               sparse_vec.data(), qmeta, ctx);
  }
  for (size_t i = 1; i < cnt; i += 2) {
    // prepare sparse
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_vec(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j;
      sparse_vec[j] = (i == j ? 1.0f : 0.0f);
    }
    streamer->add_with_id_impl(i, sparse_dim_count, sparse_indices.data(),
                               sparse_vec.data(), qmeta, ctx);
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
    NumericalVector<uint32_t> query_sparse_indices(sparse_dim_count);
    NumericalVector<float> query_sparse_velues(sparse_dim_count);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      query_sparse_indices[j] = j;
      query_sparse_velues[j] = (i == j ? 1.1f : 0.0f);
    }
    auto t1 = Realtime::MicroSeconds();
    ASSERT_EQ(
        0, streamer->search_impl(sparse_dim_count, query_sparse_indices.data(),
                                 query_sparse_velues.data(), qmeta, knnCtx));
    auto t2 = Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(
                     sparse_dim_count, query_sparse_indices.data(),
                     query_sparse_velues.data(), qmeta, linearCtx));
    auto t3 = Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;
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
  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * 100.0f / cnt;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif