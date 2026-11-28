// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/sqlengine/planner/fts_rank_batch.h"
#include <gtest/gtest.h>
#include "db/common/constants.h"

namespace zvec::sqlengine {
TEST(FtsRankBatchTest, SharedBackingSurvivesProducerAndOtherColumn) {
  arrow::ProxyMemoryPool pool(arrow::default_memory_pool());
  auto schema = arrow::schema({arrow::field(LOCAL_ROW_ID, arrow::int32()),
                               arrow::field(FIELD_SCORE, arrow::float32())});
  FtsIndexResults results({{7, 9.5f}, {3, 2.0f}, {11, 0.25f}});
  auto batch = MakeFtsRankBatch(results, 1, 8, schema, &pool);
  ASSERT_TRUE(batch.ok()) << batch.status();
  ASSERT_EQ(2, (*batch)->num_rows());
  auto rows = std::static_pointer_cast<arrow::Int32Array>((*batch)->column(0));
  auto scores =
      std::static_pointer_cast<arrow::FloatArray>((*batch)->column(1));
  EXPECT_EQ(3, rows->Value(0));
  EXPECT_FLOAT_EQ(0.25f, scores->Value(1));
  EXPECT_EQ(1, pool.num_allocations());
  batch.ValueOrDie().reset();
  rows.reset();
  EXPECT_GT(pool.bytes_allocated(), 0);
  EXPECT_FLOAT_EQ(2.0f, scores->Value(0));
  scores.reset();
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_TRUE(
      MakeFtsRankBatch(results, 3, 1, schema, &pool).status().IsInvalid());
  EXPECT_TRUE(
      MakeFtsRankBatch(results, 0, 0, schema, &pool).status().IsInvalid());
}
}  // namespace zvec::sqlengine
