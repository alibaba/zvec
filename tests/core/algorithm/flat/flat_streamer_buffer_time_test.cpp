#include <future>
#include <string>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/memory_helper.h>
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

constexpr size_t static dim = 128;

class FlatStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);
  void hybrid_scale(std::vector<float> &dense_value,
                    std::vector<float> &sparse_value, float alpha_scale);

  static std::string dir_;
  static std::shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string FlatStreamerTest::dir_("streamer_test/");
std::shared_ptr<IndexMeta> FlatStreamerTest::index_meta_ptr_;

void FlatStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (std::nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());

  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

void FlatStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(FlatStreamerTest, TestLinearSearchMMap) {
  BufferManager::Instance().init(50 * 1024 * 1024, 1);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/LinearSearchMMap", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t data_cnt = 300000UL, cnt = 500UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < data_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "/Test/LinearSearchMMap", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  size_t topk = 30;
  ElapsedTime elapsed_time;
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    // auto &result1 = ctx->result();
    // ASSERT_EQ(topk, result1.size());
    // ASSERT_EQ(i, result1[0].key());

    // for (size_t j = 0; j < dim; ++j) {
    //   vec[j] = i + 0.1f;
    // }
    // ctx->set_topk(topk);
    // ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    // auto &result2 = ctx->result();
    // ASSERT_EQ(topk, result2.size());
    // ASSERT_EQ(i, result2[0].key());
    // ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    // ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }
  cout << "Elapsed time: " << elapsed_time.micro_seconds() << " us" << endl;

  // ctx->set_topk(100U);
  // NumericalVector<float> vec(dim);
  // for (size_t j = 0; j < dim; ++j) {
  //   vec[j] = 10.1f;
  // }
  // ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, ctx));
  // auto &result = ctx->result();
  // ASSERT_EQ(100U, result.size());
  // ASSERT_EQ(10, result[0].key());
  // ASSERT_EQ(11, result[1].key());
  // ASSERT_EQ(5, result[10].key());
  // ASSERT_EQ(0, result[20].key());
  // ASSERT_EQ(30, result[30].key());
  // ASSERT_EQ(35, result[35].key());
  // ASSERT_EQ(99, result[99].key());

  read_streamer->close();
  read_streamer.reset();
  // cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif