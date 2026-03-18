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

#include "flat_sparse/flat_sparse_builder.h"
#include <future>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

class FlatSparseBuilderTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string FlatSparseBuilderTest::_dir("FlatSparseBuilderTest");
shared_ptr<IndexMeta> FlatSparseBuilderTest::_index_meta_ptr;

void FlatSparseBuilderTest::SetUp(void) {
  _index_meta_ptr.reset(new (nothrow) IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                                                IndexMeta::DataType::DT_FP32));
  _index_meta_ptr->set_metric("InnerProductSparse", 0, Params());
}

void FlatSparseBuilderTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  system(cmdBuf);
}

TEST_F(FlatSparseBuilderTest, TestGeneral) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("FlatSparseBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = make_shared<OnePassIndexSparseHolder<IndexMeta::DT_FP32>>();
  uint32_t sparse_count = 4;
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_count);
    NumericalVector<float> sparse_values(sparse_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      sparse_indices[j] = 20 * j;
      sparse_values[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_values);

    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_EQ(stats.built_costtime(), 0UL);
  ASSERT_GT(stats.dumped_costtime(), 0UL);

  // cleanup and rebuild
  ASSERT_EQ(0, builder->cleanup());

  auto holder2 = make_shared<MultiPassIndexSparseHolder<IndexMeta::DT_FP32>>();
  size_t doc_cnt2 = 2000UL;
  for (size_t i = 0; i < doc_cnt2; i++) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_count);
    NumericalVector<float> sparse_values(sparse_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      sparse_indices[j] = 20 * j;
      sparse_values[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_values);

    ASSERT_TRUE(holder2->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder2));
  ASSERT_EQ(0, builder->build(holder2));
  auto dumper2 = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper2, nullptr);
  ASSERT_EQ(0, dumper2->create(path));
  ASSERT_EQ(0, builder->dump(dumper2));
  ASSERT_EQ(0, dumper2->close());

  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt2, stats.built_count());
  ASSERT_EQ(doc_cnt2, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_EQ(stats.built_costtime(), 0UL);
}

TEST_F(FlatSparseBuilderTest, TestIndexThreads) {
  IndexBuilder::Pointer builder1 =
      IndexFactory::CreateBuilder("FlatSparseBuilder");
  ASSERT_NE(builder1, nullptr);
  IndexBuilder::Pointer builder2 =
      IndexFactory::CreateBuilder("FlatSparseBuilder");
  ASSERT_NE(builder2, nullptr);

  auto holder = make_shared<MultiPassIndexSparseHolder<IndexMeta::DT_FP32>>();

  size_t doc_cnt = 1000UL;
  uint32_t sparse_count = 32;

  for (size_t i = 0; i < doc_cnt; i++) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_count);
    NumericalVector<float> sparse_values(sparse_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      sparse_indices[j] = 20 * j;
      sparse_values[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_values);

    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  std::srand(Realtime::MilliSeconds());
  auto threads =
      std::make_shared<SingleQueueIndexThreads>(std::rand() % 4, false);
  ASSERT_EQ(0, builder1->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder2->init(*_index_meta_ptr, params));

  auto build_index1 = [&]() {
    ASSERT_EQ(0, builder1->train(threads, holder));
    ASSERT_EQ(0, builder1->build(threads, holder));
  };
  auto build_index2 = [&]() {
    ASSERT_EQ(0, builder2->train(threads, holder));
    ASSERT_EQ(0, builder2->build(threads, holder));
  };

  auto t1 = std::async(std::launch::async, build_index1);
  auto t2 = std::async(std::launch::async, build_index2);
  t1.wait();
  t2.wait();


  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestIndexThreads";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder1->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder2->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats1 = builder1->stats();
  ASSERT_EQ(doc_cnt, stats1.built_count());
  auto &stats2 = builder2->stats();
  ASSERT_EQ(doc_cnt, stats2.built_count());
}

TEST_F(FlatSparseBuilderTest, TestHalfFloatConverter) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("FlatSparseBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = make_shared<OnePassIndexSparseHolder<IndexMeta::DT_FP32>>();
  uint32_t sparse_count = 4;
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_count);
    NumericalVector<float> sparse_values(sparse_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      sparse_indices[j] = 20 * j;
      sparse_values[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_values);

    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params converter_params;
  auto converter = IndexFactory::CreateConverter("HalfFloatSparseConverter");
  converter->init(*_index_meta_ptr, converter_params);

  IndexMeta index_meta = converter->meta();

  converter->transform(holder);

  auto converted_holder = converter->sparse_result();

  Params params;
  ASSERT_EQ(0, builder->init(index_meta, converter_params));

  ASSERT_EQ(0, builder->train(converted_holder));

  ASSERT_EQ(0, builder->build(converted_holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestHalFloatConverter";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_EQ(stats.built_costtime(), 0UL);
  //ASSERT_GT(stats.dumped_costtime(), 0UL);

  // cleanup and rebuild
  ASSERT_EQ(0, builder->cleanup());

  auto holder2 = make_shared<MultiPassIndexSparseHolder<IndexMeta::DT_FP32>>();
  size_t doc_cnt2 = 2000UL;
  for (size_t i = 0; i < doc_cnt2; i++) {
    SparseVector<float> vec;

    NumericalVector<uint32_t> sparse_indices(sparse_count);
    NumericalVector<float> sparse_values(sparse_count);
    for (size_t j = 0; j < sparse_count; ++j) {
      sparse_indices[j] = 20 * j;
      sparse_values[j] = i;
    }

    vec.add_sparses(sparse_indices, sparse_values);

    ASSERT_TRUE(holder2->emplace(i, vec));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder2));
  ASSERT_EQ(0, builder->build(holder2));
  auto dumper2 = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper2, nullptr);
  ASSERT_EQ(0, dumper2->create(path));
  ASSERT_EQ(0, builder->dump(dumper2));
  ASSERT_EQ(0, dumper2->close());

  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt2, stats.built_count());
  ASSERT_EQ(doc_cnt2, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_EQ(stats.built_costtime(), 0UL);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
