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

#include "diskann_builder.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <future>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_holder.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

constexpr size_t static dim = 64;

class DiskAnnBuilderTest : public testing::Test {
 protected:
  void SetUp(void) override;
  void TearDown(void) override;

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string DiskAnnBuilderTest::_dir("DiskAnnBuilderTest");
shared_ptr<IndexMeta> DiskAnnBuilderTest::_index_meta_ptr;

void DiskAnnBuilderTest::SetUp(void) {
  LoggerBroker::SetLevel(Logger::LEVEL_INFO);

  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, Params());
}

void DiskAnnBuilderTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  system(cmdBuf);
}

TEST_F(DiskAnnBuilderTest, TestGeneral) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 50);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

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
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);
}

// DiskAnn is now exposed implicitly: no caller ever invokes a
// ``LoadDiskAnnPlugin`` / ``IsLibAioAvailable`` API (those were removed from
// the public surface together with ``zvec.load_diskann_plugin()`` in Python).
// The only contract this test validates is the UX guarantee: once the DiskAnn
// module has been linked into the hosting binary (here, directly into the
// test via the ``core_knn_diskann`` target), its factory entries are
// registered automatically and the global ``IndexFactory`` can hand out a
// ``DiskAnnBuilder`` without any explicit setup step. On hosts missing
// libaio, DiskAnn would fail at the index-creation layer with a clear error
// while other index types (HNSW/IVF/Flat/Vamana) remain unaffected; that
// runtime branch lives in ``DiskAnnIndex::CreateAndInitStreamer`` and is
// covered by the higher-level interface tests.
TEST_F(DiskAnnBuilderTest, TestImplicitFactoryRegistration) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr)
      << "DiskAnnBuilder factory entry missing: DiskAnn must be available "
         "without any manual plugin load step.";

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr)
      << "DiskAnnStreamer factory entry missing: DiskAnn must be available "
         "without any manual plugin load step.";
}
