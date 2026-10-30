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
#include <cstring>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "core/metric/metric_params.h"
#include "tests/test_util.h"

using namespace zvec::core;
using namespace zvec::ailego;

class LegacyQuantizedIndexTest : public testing::Test {
 protected:
  const std::string path_ = "legacy_quantized_index_test/index";
  void TearDown() override {
    zvec::test_util::RemoveTestPath("legacy_quantized_index_test");
  }
};

TEST_F(LegacyQuantizedIndexTest, StoredInt4MomentsSurviveDumpLoadAndSearch) {
  // Frozen pre-rounding records, independent of the current RecordQuantizer.
  // Repeat [-8, 7] four times, followed by scale, bias, sum and squared sum.
  alignas(8) uint8_t records[2][20];
  const float tails[][4] = {{0.5f, 1.0f, -6.0f, 425.0f},
                            {0.25f, -1.0f, -2.0f, 421.0f}};
  for (size_t i = 0; i < 2; ++i) {
    std::memset(records[i], 0x78, 4);
    std::memcpy(records[i] + 4, tails[i], sizeof(tails[i]));
  }
  const double norm_a = 0.25 * 425 - 6 + 8;
  const double norm_b = 0.0625 * 421 + 1 + 8;
  const double cross_ab = 0.125 * 452 - 0.5 + 3 - 8;
  const double cross_bb = 0.0625 * 452 + 1 + 8;
  const float expected[] = {float(2 - 2 * cross_ab / std::max(norm_a, norm_b)),
                            float(2 - 2 * cross_bb / norm_b)};

  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    SCOPED_TRACE(order);
    Params params;
    params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME,
               "MipsSquaredEuclidean");
    IndexMeta meta(IndexMeta::DT_INT4, 40);
    meta.set_metric("QuantizedInteger", 0, params);
    meta.set_major_order(order);
    auto holder = std::make_shared<RandomAccessIndexHolder>(meta);
    for (uint64_t id = 0; id < 67; ++id) holder->emplace(id, records[id % 2]);
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_TRUE(builder);
    ASSERT_TRUE(dumper);
    ASSERT_EQ(0, builder->init(meta, Params()));
    ASSERT_EQ(0, dumper->create(path_));
    ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
    ASSERT_EQ(0, dumper->close());
    builder.reset();
    holder.reset();

    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
    ASSERT_TRUE(storage);
    ASSERT_TRUE(searcher);
    ASSERT_EQ(0, storage->init(Params()));
    ASSERT_EQ(0, storage->open(path_, false));
    ASSERT_EQ(0, searcher->init(Params()));
    ASSERT_EQ(0, searcher->load(storage, nullptr));
    const auto restored = searcher->meta();
    ASSERT_EQ("QuantizedInteger", restored.metric_name());
    for (uint64_t id = 0; id < 67; ++id) {
      const void *record = searcher->get_vector(id);
      ASSERT_NE(nullptr, record);
      EXPECT_EQ(0, std::memcmp(record, records[id % 2], sizeof(records[0])));
    }
    ASSERT_EQ(0, searcher->unload());
    // Flat normally switches MIPS to its IP query metric. Supply the restored
    // build metric explicitly to exercise persisted MIPS scores as well.
    auto metric = IndexFactory::CreateMetric(restored.metric_name());
    ASSERT_TRUE(metric);
    ASSERT_EQ(0, metric->init(restored, restored.metric_params()));
    ASSERT_EQ(0, searcher->load(storage, metric));
    metric.reset();
    auto context = searcher->create_context();
    ASSERT_TRUE(context);
    context->set_topk(67);
    ASSERT_EQ(0,
              searcher->search_impl(
                  records[1], IndexQueryMeta(IndexMeta::DT_INT4, 40), context));
    ASSERT_EQ(67U, context->result().size());
    for (const auto &document : context->result()) {
      ASSERT_LT(document.key(), 67U);
      EXPECT_NEAR(expected[document.key() % 2], document.score(), 1e-6f);
    }
    ASSERT_EQ(0, searcher->unload());
    ASSERT_EQ(0, storage->close());
  }
}
