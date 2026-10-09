#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_streamer.h>
#include "algorithm/flat/flat_streamer_entity.h"
#include "tests/test_util.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

constexpr size_t static dim = 16;

class FlatStreamerTest : public testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;
  void hybrid_scale(std::vector<float> &dense_value,
                    std::vector<float> &sparse_value, float alpha_scale);

  static std::string dir_;
  static std::shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string FlatStreamerTest::dir_("flat_streamer_buffer_test_dir/");
std::shared_ptr<IndexMeta> FlatStreamerTest::index_meta_ptr_;

void FlatStreamerTest::SetUp() {
  index_meta_ptr_.reset(new (std::nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());

  zvec::test_util::RemoveTestPath(dir_);
}

void FlatStreamerTest::TearDown() {
  zvec::test_util::RemoveTestPath(dir_);
}

TEST_F(FlatStreamerTest, TestLinearSearch) {
  MemoryLimitPool::get_instance().init(2 * 1024UL * 1024UL * 1024UL);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/LinearSearch", true));
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
  storage->close();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/LinearSearch", false));
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
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;

  read_streamer->close();
  read_streamer.reset();
}

TEST_F(FlatStreamerTest, TestLinearSearchBuffer) {
  MemoryLimitPool::get_instance().init(2 * 1024UL * 1024UL * 1024UL);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/LinearSearchBuffer", true));
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
  storage->close();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/LinearSearchBuffer", false));
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
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;

  read_streamer->close();
  read_streamer.reset();
}

TEST_F(FlatStreamerTest, TestLinearSearchBufferMMap) {
  MemoryLimitPool::get_instance().init(2 * 1024UL * 1024UL * 1024UL);
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/LinearSearchBuffer", true));
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
  storage->close();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/LinearSearchBuffer", false));
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
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;

  read_streamer->close();
  read_streamer.reset();
}


TEST_F(FlatStreamerTest, TestLinearSearchWithLRU) {
  MemoryLimitPool::get_instance().init(100 * 1024UL * 1024UL);
#ifdef __ANDROID__
  GTEST_SKIP()
      << "Skipped on Android: requires ~6GB memory/disk (emulator limit)";
#endif
  constexpr size_t static dim = 1600;
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  IndexMeta meta = IndexMeta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_EQ(0, write_streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/LinearSearchWithLRU", true));
  ASSERT_EQ(0, write_streamer->open(storage));

  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 50000UL;
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
  storage->close();


  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, read_streamer->init(meta, params));
  auto read_storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "/Test/LinearSearchWithLRU", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  size_t topk = 3;
  auto provider = read_streamer->create_provider();
  ElapsedTime elapsed_time;
  for (size_t i = 0; i < 10; i += 1) {
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
  cout << "Elapsed time: " << elapsed_time.milli_seconds() << " ms" << endl;

  read_streamer->close();
  read_streamer.reset();
}

TEST_F(FlatStreamerTest, TestLinearSearchMMap) {
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  Params params;
  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "Test/LinearSearchMMap", true));
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
  storage->close();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/LinearSearchMMap", false));
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
    for (size_t j = 0; j < dim; ++j) {
      const float *data = (float *)provider->get_vector(result1[0].key());
      EXPECT_FLOAT_EQ(data[j], i);
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

namespace {

struct FlatWriteFailure {
  size_t remaining_writes{0};
  size_t failure_count{0};
};

class FailingFlatSegment : public IndexStorage::Segment {
 public:
  FailingFlatSegment(Pointer segment, std::shared_ptr<FlatWriteFailure> failure)
      : segment_(std::move(segment)), failure_(std::move(failure)) {}

  size_t data_size() const override {
    return segment_->data_size();
  }
  uint32_t data_crc() const override {
    return segment_->data_crc();
  }
  size_t padding_size() const override {
    return segment_->padding_size();
  }
  size_t capacity() const override {
    return segment_->capacity();
  }
  size_t fetch(size_t offset, void *data, size_t length) const override {
    return segment_->fetch(offset, data, length);
  }
  size_t read(size_t offset, const void **data, size_t length) override {
    return segment_->read(offset, data, length);
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &block,
              size_t length) override {
    return segment_->read(offset, block, length);
  }
  size_t write(size_t offset, const void *data, size_t length) override {
    if (failure_->remaining_writes != 0 && --failure_->remaining_writes == 0) {
      ++failure_->failure_count;
      return 0;
    }
    return segment_->write(offset, data, length);
  }
  size_t resize(size_t size) override {
    return segment_->resize(size);
  }
  void update_data_crc(uint32_t crc) override {
    segment_->update_data_crc(crc);
  }
  Pointer clone() override {
    auto segment = segment_->clone();
    return segment ? std::make_shared<FailingFlatSegment>(segment, failure_)
                   : nullptr;
  }

 private:
  Pointer segment_;
  std::shared_ptr<FlatWriteFailure> failure_;
};

class FailingFlatStorage : public IndexStorage {
 public:
  explicit FailingFlatStorage(IndexStorage::Pointer storage)
      : storage_(std::move(storage)),
        failure_(std::make_shared<FlatWriteFailure>()) {}

  int init(const Params &params) override {
    return storage_->init(params);
  }
  int cleanup() override {
    return storage_->cleanup();
  }
  int open(const std::string &path, bool create) override {
    return storage_->open(path, create);
  }
  int flush() override {
    return storage_->flush();
  }
  int close() override {
    return storage_->close();
  }
  int append(const std::string &id, size_t size) override {
    return storage_->append(id, size);
  }
  void refresh(uint64_t checkpoint) override {
    storage_->refresh(checkpoint);
  }
  uint64_t check_point() const override {
    return storage_->check_point();
  }
  Segment::Pointer get(const std::string &id, int level = -1) override {
    auto segment = storage_->get(id, level);
    return segment ? std::make_shared<FailingFlatSegment>(segment, failure_)
                   : nullptr;
  }
  bool has(const std::string &id) const override {
    return storage_->has(id);
  }
  uint32_t magic() const override {
    return storage_->magic();
  }
  MemoryBlock::MemoryBlockType memory_block_type() const override {
    return storage_->memory_block_type();
  }
  void fail_write(size_t ordinal) {
    failure_->remaining_writes = ordinal;
  }
  size_t failure_count() const {
    return failure_->failure_count;
  }

 private:
  IndexStorage::Pointer storage_;
  std::shared_ptr<FlatWriteFailure> failure_;
};

class FlatStreamerWriteFailureTest
    : public testing::TestWithParam<std::tuple<uint32_t, size_t>> {
 protected:
  void SetUp() override {
    auto &pool = MemoryLimitPool::get_instance();
    previous_capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(8 * 1024UL * 1024UL));
    zvec::test_util::RemoveTestPath(path_);
    auto backing = IndexFactory::CreateStorage("BufferStorage");
    ASSERT_NE(nullptr, backing);
    storage_ = std::make_shared<FailingFlatStorage>(backing);
    ASSERT_EQ(0, storage_->init(Params()));
    ASSERT_EQ(0, storage_->open(path_ + "/index", true));
    IndexMeta meta(IndexMeta::DT_FP32, 4);
    meta.set_metric("SquaredEuclidean", 0, Params());
    entity_ = std::make_unique<FlatStreamerEntity>(stats_);
    *entity_->mutable_meta() = meta;
    entity_->set_block_vector_count(8);
    entity_->set_linear_list_count(1);
    entity_->set_segment_size(65536);
    ASSERT_EQ(0, entity_->open(storage_, meta));
  }

  void TearDown() override {
    if (entity_) {
      EXPECT_EQ(0, entity_->close());
      entity_.reset();
    }
    if (storage_) {
      EXPECT_EQ(0, storage_->close());
      storage_.reset();
    }
    zvec::test_util::RemoveTestPath(path_);
    auto &pool = MemoryLimitPool::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_capacity_));
  }

  std::string path_{"flat_streamer_write_failure_test_dir"};
  size_t previous_capacity_{0};
  IndexStreamer::Stats stats_;
  std::shared_ptr<FailingFlatStorage> storage_;
  std::unique_ptr<FlatStreamerEntity> entity_;
};

TEST_P(FlatStreamerWriteFailureTest, PropagatesShortWritesAndRetriesSameId) {
  const uint32_t id = std::get<0>(GetParam());
  const size_t failed_write = std::get<1>(GetParam());
  const std::vector<float> first{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> added{5.0f, 6.0f, 7.0f, 8.0f};
  const uint32_t bytes = static_cast<uint32_t>(first.size() * sizeof(float));
  ASSERT_EQ(0, entity_->add_vector_with_id(0, first.data(), bytes));
  ASSERT_EQ(1u, entity_->vector_count());
  ASSERT_EQ(1u, stats_.added_count());
  const size_t revision = stats_.revision_id();

  // The existing block has room: writes 1/2/3 update the vector, key and
  // block header. For id=3 these are writes for the first ID-hole filler.
  storage_->fail_write(failed_write);
  ASSERT_EQ(IndexError_WriteData,
            entity_->add_vector_with_id(id, added.data(), bytes));
  EXPECT_EQ(1u, storage_->failure_count());
  EXPECT_EQ(1u, entity_->vector_count());
  EXPECT_EQ(1u, stats_.added_count());
  EXPECT_EQ(revision, stats_.revision_id());
  {
    IndexStorage::MemoryBlock missing;
    EXPECT_NE(0, entity_->get_vector_by_key(id, missing));
  }

  ASSERT_EQ(0, entity_->add_vector_with_id(id, added.data(), bytes));
  EXPECT_EQ(id + 1u, entity_->vector_count());
  EXPECT_EQ(2u, stats_.added_count());
  for (uint32_t key : {0u, id}) {
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, entity_->get_vector_by_key(key, block));
    ASSERT_NE(nullptr, block.data());
    const auto *values = static_cast<const float *>(block.data());
    const auto &expected = key == 0 ? first : added;
    for (size_t column = 0; column < expected.size(); ++column) {
      EXPECT_FLOAT_EQ(expected[column], values[column]);
    }
  }
  for (uint32_t hole = 1; hole < id; ++hole) {
    EXPECT_EQ(kInvalidKey, entity_->key(hole));
  }
}

INSTANTIATE_TEST_SUITE_P(AppendAndIdHole, FlatStreamerWriteFailureTest,
                         testing::Combine(testing::Values(1u, 3u),
                                          testing::Values(1u, 2u, 3u)));

}  // namespace

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
