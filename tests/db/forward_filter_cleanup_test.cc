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

#include <atomic>
#include <chrono>
#include <future>
#include <shared_mutex>
#include <gtest/gtest.h>
#include "db/sqlengine/planner/plan_info.h"
#include "db/sqlengine/planner/segment_node.h"

namespace zvec::test {
namespace {
using Batch = std::optional<arrow::compute::ExecBatch>;
struct PendingSource {
  arrow::Future<Batch> batch = arrow::Future<Batch>::Make();
  std::promise<void> entered;
  std::atomic<bool> started{false};

  sqlengine::PlanInfo::Ptr plan() {
    auto schema = arrow::schema({arrow::field("value", arrow::int32())});
    arrow::acero::Declaration declaration{
        "source", arrow::acero::SourceNodeOptions(schema, [this]() {
          if (!started.exchange(true)) {
            entered.set_value();
            return batch;
          }
          return arrow::Future<Batch>::MakeFinished(std::nullopt);
        })};
    return std::make_shared<sqlengine::PlanInfo>(std::move(declaration),
                                                 schema);
  }
};
}  // namespace

TEST(ForwardFilterCleanupTest, AbandonAndErrorDrainBeforeGuardRelease) {
  for (bool fail : {false, true}) {
    PendingSource source;
    std::shared_mutex schema_guard;
    std::promise<void> abandon;
    auto abandon_signal = abandon.get_future();
    std::promise<void> destroying;
    auto entered = source.entered.get_future();
    auto execution = std::async(std::launch::async, [&] {
      std::shared_lock<std::shared_mutex> guard(schema_guard);
      auto plan = source.plan();
      auto result = plan->execute_to_reader();
      EXPECT_TRUE(result.has_value());
      abandon_signal.wait();
      destroying.set_value();
      if (result) result.value().reset();
    });
    EXPECT_EQ(entered.wait_for(std::chrono::seconds(10)),
              std::future_status::ready);
    abandon.set_value();
    destroying.get_future().wait();
    auto maintenance = std::async(std::launch::async, [&] {
      std::unique_lock<std::shared_mutex> guard(schema_guard);
    });
    EXPECT_EQ(execution.wait_for(std::chrono::milliseconds(100)),
              std::future_status::timeout);
    EXPECT_EQ(maintenance.wait_for(std::chrono::milliseconds(100)),
              std::future_status::timeout);
    if (fail)
      source.batch.MarkFinished(
          arrow::Status::Invalid("injected source error"));
    else
      source.batch.MarkFinished(std::nullopt);
    execution.get();
    maintenance.get();
  }
}

TEST(ForwardFilterCleanupTest, SegmentReadErrorDrainsOtherStartedReaders) {
  ailego::ThreadPool pool(2, false);
  PendingSource first, second;
  auto a_started = first.entered.get_future();
  auto b_started = second.entered.get_future();
  auto node = std::make_shared<sqlengine::SegmentNode>(
      std::vector<sqlengine::PlanInfo::Ptr>{first.plan(), second.plan()},
      &pool);
  auto execution =
      std::async(std::launch::async, [&] { return node->gen()().result(); });
  EXPECT_EQ(a_started.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  EXPECT_EQ(b_started.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  second.batch.MarkFinished(
      arrow::Status::Invalid("injected last-segment error"));
  EXPECT_EQ(execution.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  first.batch.MarkFinished(std::nullopt);
  EXPECT_FALSE(execution.get().ok());
  // The node remains alive here: cleanup must have happened on error, not only
  // when the outer plan eventually destroys its captured generator.
}

TEST(ForwardFilterCleanupTest, PartialPreparationFailureAndNormalEof) {
  ailego::ThreadPool pool(2, false);
  auto schema = arrow::schema({arrow::field("value", arrow::int32())});
  auto good = std::make_shared<sqlengine::PlanInfo>(
      arrow::acero::Declaration{
          "source",
          arrow::acero::SourceNodeOptions(
              schema,
              [] { return arrow::Future<Batch>::MakeFinished(std::nullopt); })},
      schema);
  auto bad = std::make_shared<sqlengine::PlanInfo>(
      arrow::acero::Declaration{"missing_test_factory",
                                arrow::acero::ExecNodeOptions{}},
      schema);
  auto node = std::make_shared<sqlengine::SegmentNode>(
      std::vector<sqlengine::PlanInfo::Ptr>{good, bad}, &pool);
  EXPECT_FALSE(node->gen()().result().ok());
  auto reader = good->execute_to_reader();
  ASSERT_TRUE(reader.has_value());
  std::shared_ptr<arrow::RecordBatch> batch;
  EXPECT_TRUE(reader.value()->ReadNext(&batch).ok());
  EXPECT_EQ(batch, nullptr);
  EXPECT_TRUE(reader.value()->Close().ok());
  EXPECT_TRUE(reader.value()->Close().ok());
}
}  // namespace zvec::test
