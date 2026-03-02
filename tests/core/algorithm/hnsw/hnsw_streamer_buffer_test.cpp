#include <future>
#include <string>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/memory_helper.h>
#include <algorithm/hnsw/hnsw_params.h>
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

constexpr size_t static dim = 16;

class HnswStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);
  void hybrid_scale(std::vector<float> &dense_value,
                    std::vector<float> &sparse_value, float alpha_scale);

  static std::string dir_;
  static std::shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string HnswStreamerTest::dir_("streamer_test/");
std::shared_ptr<IndexMeta> HnswStreamerTest::index_meta_ptr_;

void HnswStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (std::nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());

  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

void HnswStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(HnswStreamerTest, TestHnswSearch) {
  BufferManager::Instance().init(10 * 1024 * 1024, 1);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/HnswSearch", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 10000UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
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
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "/Test/HnswSearch", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  size_t topk = 3;
  auto provider = read_streamer->create_provider();
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(10, result[0].key());
  ASSERT_EQ(11, result[1].key());
  ASSERT_EQ(5, result[10].key());
  ASSERT_EQ(0, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());

  ElapsedTime elapsed_time;
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  read_streamer->close();
  read_streamer.reset();
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;
}

TEST_F(HnswStreamerTest, TestHnswSearchMMap) {
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/HnswSearchMMap", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 10000UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();

  ElapsedTime elapsed_time;
  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "/Test/HnswSearchMMap", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  size_t topk = 3;
  auto provider = read_streamer->create_provider();
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, provider->get_vector(result1[0].key(), block));
    const float *data = (float *)block.data();
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(10, result[0].key());
  ASSERT_EQ(11, result[1].key());
  ASSERT_EQ(5, result[10].key());
  ASSERT_EQ(0, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());

  read_streamer->close();
  read_streamer.reset();
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif