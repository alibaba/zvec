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

using namespace std;
using namespace testing;
using namespace zvec::ailego;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

namespace zvec {
namespace core {

class FlatSparseStreamerTest : public testing::Test {
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

std::string FlatSparseStreamerTest::dir_("FlatSparseStreamerTest/");
shared_ptr<IndexMeta> FlatSparseStreamerTest::index_meta_ptr_;

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
    ailego::Norm2Matrix<float, 1>::Compute(sparse_vec.data(), sparse_dim_count,
                                           &norm);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_vec[j] = sparse_vec[j] / norm;
    }

    sparse_indices_list.push_back(sparse_indices);
    sparse_vec_list.push_back(sparse_vec);
  }
}

void FlatSparseStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (nothrow) IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                                                IndexMeta::DataType::DT_FP32));
  index_meta_ptr_->set_metric("InnerProductSparse", 0, ailego::Params());

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
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  size_t sparse_dim_count = 32;

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params params;

  ailego::Params stg_params;
  auto write_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, write_storage->init(stg_params));
  ASSERT_EQ(0, write_storage->open(dir_ + "/Test/FlatSparseSearch", true));
  ASSERT_EQ(0, write_streamer->init(index_meta, params));
  ASSERT_EQ(0, write_streamer->open(write_storage));

  size_t cnt = 20000U;
  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, write_streamer->add_impl(
                     i, sparse_dim_count, sparse_indices_list[i].data(),
                     sparse_vec_list[i].data(), qmeta, ctx));
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "/Test/FlatSparseSearch", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));

  auto linearCtx = read_streamer->create_context();
  ASSERT_TRUE(!!linearCtx);

  auto knnCtx = read_streamer->create_context();
  ASSERT_TRUE(!!knnCtx);

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

    ASSERT_EQ(
        0, read_streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, read_streamer->search_bf_impl(
                     sparse_dim_count, sparse_indices.data(), sparse_vec.data(),
                     qmeta, linearCtx));

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
  // float cost = linearTotalTime * 1.0f / knnTotalTime;

  std::cout << "knnTotalTime=" << knnTotalTime
            << " linearTotalTime=" << linearTotalTime << std::endl;

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