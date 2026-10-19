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

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/compute/initialize.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include "db/sqlengine/planner/doc_filter.h"
#include "db/sqlengine/planner/segment_node.h"
#include "mock_segment.h"

namespace zvec::sqlengine {
namespace {

class QueryThreadingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    original_capacity_ = arrow::GetCpuThreadPoolCapacity();
    ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
    ASSERT_TRUE(arrow::compute::Initialize().ok());
  }

  void TearDown() override {
    EXPECT_TRUE(arrow::SetCpuThreadPoolCapacity(original_capacity_).ok());
  }

  template <typename T>
  void CheckCompletion(const arrow::Future<T> &future) {
    // A source callback must progress even when it occupies the only CPU
    // worker. Recover a regressed implementation so the test can report the
    // failure and release its readers instead of hanging the whole suite.
    const bool completed = future.Wait(2.0);
    if (!completed) {
      ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(2).ok());
    }
    ASSERT_TRUE(future.Wait(5.0));
    EXPECT_TRUE(completed)
        << "Nested query waited for another Arrow CPU worker";
  }

  std::shared_ptr<arrow::Table> MakeTable(int base, int rows = 32) {
    arrow::Int32Builder builder;
    for (int i = 0; i < rows; ++i) {
      EXPECT_TRUE(builder.Append(base + i).ok());
    }
    auto values = builder.Finish().ValueOrDie();
    return arrow::Table::Make(
        arrow::schema({arrow::field("value", arrow::int32())}), {values});
  }

 private:
  int original_capacity_{0};
};

TEST_F(QueryThreadingTest, ScalarReaderDoesNotWaitForBusyCpuPool) {
  auto entered = arrow::Future<>::Make();
  auto release = arrow::Future<>::Make();
  auto occupied =
      arrow::internal::GetCpuThreadPool()->Submit([entered, release]() mutable {
        entered.MarkFinished();
        release.Wait();
        return arrow::Status::OK();
      });
  ASSERT_TRUE(occupied.ok()) << occupied.status().ToString();
  EXPECT_TRUE(entered.Wait(2.0));

  auto table = MakeTable(0);
  auto caller = std::async(std::launch::async, [table]() {
    auto declaration = arrow::acero::Declaration::Sequence(
        {{"table_source", arrow::acero::TableSourceNodeOptions{table, 4}},
         {"filter", arrow::acero::FilterNodeOptions{arrow::compute::call(
                        "greater_equal", {arrow::compute::field_ref("value"),
                                          arrow::compute::literal(16)})}},
         {"fetch", arrow::acero::FetchNodeOptions{0, 3}}});
    PlanInfo plan(std::move(declaration), table->schema());
    auto reader = plan.execute_to_reader();
    if (!reader) {
      return arrow::Result<std::shared_ptr<arrow::Table>>(
          arrow::Status::ExecutionError(reader.error().c_str()));
    }
    return reader.value()->ToTable();
  });
  // Other queries can occupy the shared CPU pool. The synchronous caller
  // should drive its outer scalar plan instead of queueing behind them.
  const bool completed =
      caller.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  release.MarkFinished();
  EXPECT_TRUE(occupied->Wait(5.0));
  auto result = caller.get();
  EXPECT_TRUE(completed) << "Scalar reader waited for the shared CPU pool";
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  ASSERT_EQ(result.ValueOrDie()->num_rows(), 3);
  auto values = std::static_pointer_cast<arrow::Int32Array>(
      result.ValueOrDie()->column(0)->chunk(0));
  EXPECT_EQ(values->Value(0), 16);
}

TEST_F(QueryThreadingTest, SegmentBatchesCompleteOnTheOnlyCpuWorker) {
  for (bool filtered : {false, true}) {
    ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
    ailego::ThreadPool pool(2, false);
    std::vector<PlanInfo::Ptr> plans;
    auto schema = MakeTable(0)->schema();
    // The last segment is empty, so draining it must advance to earlier ones.
    for (int base : {0, 100, 200}) {
      arrow::acero::Declaration plan{
          "table_source", arrow::acero::TableSourceNodeOptions{
                              MakeTable(base, base == 200 ? 0 : 32), 4}};
      if (filtered) {
        plan = arrow::acero::Declaration{
            "filter",
            {std::move(plan)},
            arrow::acero::FilterNodeOptions{arrow::compute::call(
                "greater_equal", {arrow::compute::field_ref("value"),
                                  arrow::compute::literal(base + 16)})}};
      }
      plans.push_back(std::make_shared<PlanInfo>(std::move(plan), schema));
    }

    auto node = std::make_shared<SegmentNode>(std::move(plans), &pool);
    // Arrow resumes source generators on its CPU executor. Return the pending
    // future to release that worker, just as an Acero source does.
    auto submitted = arrow::internal::GetCpuThreadPool()->Submit(
        [gen = node->gen()]() { return arrow::CollectAsyncGenerator(gen); });
    ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
    auto future = submitted.MoveValueUnsafe().Then(
        [schema](const std::vector<std::optional<arrow::compute::ExecBatch>>
                     &batches) -> arrow::Result<std::vector<int32_t>> {
          std::vector<int32_t> values;
          for (const auto &batch : batches) {
            ARROW_ASSIGN_OR_RAISE(auto record_batch,
                                  batch->ToRecordBatch(schema));
            auto array = std::static_pointer_cast<arrow::Int32Array>(
                record_batch->column(0));
            for (int64_t i = 0; i < array->length(); ++i) {
              values.push_back(array->Value(i));
            }
          }
          return values;
        });
    CheckCompletion(future);
    ASSERT_TRUE(future.is_finished());
    ASSERT_TRUE(future.result().ok()) << future.status().ToString();

    std::vector<int32_t> expected;
    for (int base : {100, 0}) {
      for (int i = filtered ? 16 : 0; i < 32; ++i) {
        expected.push_back(base + i);
      }
    }
    EXPECT_EQ(*future.result(), expected);
  }
}

TEST_F(QueryThreadingTest, SegmentReadErrorReachesTheCaller) {
  auto schema = MakeTable(0)->schema();
  arrow::acero::Declaration failing_plan{
      "source",
      arrow::acero::SourceNodeOptions{
          schema, []() {
            return arrow::Future<std::optional<arrow::compute::ExecBatch>>::
                MakeFinished(
                    arrow::Status::Invalid("injected segment read error"));
          }}};
  arrow::acero::Declaration valid_plan{
      "table_source", arrow::acero::TableSourceNodeOptions{MakeTable(0), 4}};
  std::vector<PlanInfo::Ptr> plans{
      std::make_shared<PlanInfo>(std::move(failing_plan), schema),
      std::make_shared<PlanInfo>(std::move(valid_plan), schema)};
  ailego::ThreadPool pool(2, false);
  auto node = std::make_shared<SegmentNode>(std::move(plans), &pool);
  auto submitted = arrow::internal::GetCpuThreadPool()->Submit(
      [gen = node->gen()]() { return arrow::CollectAsyncGenerator(gen); });
  ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
  auto future = submitted.MoveValueUnsafe();
  CheckCompletion(future);
  ASSERT_TRUE(future.is_finished());
  EXPECT_FALSE(future.status().ok());
  EXPECT_NE(future.status().ToString().find("injected segment read error"),
            std::string::npos);
}

TEST_F(QueryThreadingTest, FirstBatchDoesNotWaitForOtherSegments) {
  auto schema = MakeTable(0)->schema();
  auto delayed =
      arrow::Future<std::optional<arrow::compute::ExecBatch>>::Make();
  arrow::acero::Declaration slow_plan{
      "source",
      arrow::acero::SourceNodeOptions{schema, [delayed]() { return delayed; }}};
  arrow::acero::Declaration fast_plan{
      "table_source", arrow::acero::TableSourceNodeOptions{MakeTable(100), 4}};
  std::vector<PlanInfo::Ptr> plans{
      std::make_shared<PlanInfo>(std::move(slow_plan), schema),
      std::make_shared<PlanInfo>(std::move(fast_plan), schema)};
  ailego::ThreadPool pool(2, false);
  auto node = std::make_shared<SegmentNode>(std::move(plans), &pool);
  // Segments are consumed backwards. The first result must be available
  // while an earlier segment's asynchronous source is still pending.
  auto submitted = arrow::internal::GetCpuThreadPool()->Submit(node->gen());
  ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
  auto future = submitted.MoveValueUnsafe();
  const bool completed = future.Wait(2.0);
  delayed.MarkFinished(std::nullopt);
  CheckCompletion(future);
  EXPECT_TRUE(completed) << "First batch waited for an unrelated segment";
  ASSERT_TRUE(future.result().ok()) << future.status().ToString();
  ASSERT_TRUE(future.result()->has_value());
  auto batch = future.result()->value().ToRecordBatch(schema).ValueOrDie();
  EXPECT_EQ(
      std::static_pointer_cast<arrow::Int32Array>(batch->column(0))->Value(0),
      100);
}

TEST_F(QueryThreadingTest, LimitCompletesWithoutWaitingForOtherSegments) {
  auto schema = MakeTable(0)->schema();
  auto delayed = arrow::Future<PlanInfo::Batch>::Make();
  arrow::acero::Declaration slow_plan{
      "source",
      arrow::acero::SourceNodeOptions{schema, [delayed]() { return delayed; }}};
  arrow::acero::Declaration fast_plan{
      "table_source", arrow::acero::TableSourceNodeOptions{MakeTable(100, 4)}};
  std::vector<PlanInfo::Ptr> plans{
      std::make_shared<PlanInfo>(std::move(slow_plan), schema),
      std::make_shared<PlanInfo>(std::move(fast_plan), schema)};
  ailego::ThreadPool pool(2, false);
  auto node = std::make_shared<SegmentNode>(std::move(plans), &pool, 4);
  auto plan = arrow::acero::Declaration::Sequence(
      {{"source",
        arrow::acero::SourceNodeOptions{schema, node->gen(),
                                        arrow::compute::Ordering::Implicit()}},
       {"fetch", arrow::acero::FetchNodeOptions{0, 4}}});
  auto future = arrow::acero::DeclarationToTableAsync(std::move(plan));
  const bool completed = future.Wait(2.0);
  delayed.MarkFinished(std::nullopt);
  CheckCompletion(future);
  EXPECT_TRUE(completed) << "LIMIT waited for an unrelated segment";
  ASSERT_TRUE(future.status().ok()) << future.status().ToString();
  EXPECT_EQ(future.result().ValueOrDie()->num_rows(), 4);
}

TEST_F(QueryThreadingTest, ReaderLimitDoesNotWaitForOtherSegments) {
  auto schema = MakeTable(0)->schema();
  auto delayed = arrow::Future<PlanInfo::Batch>::Make();
  arrow::acero::Declaration slow_plan{
      "source",
      arrow::acero::SourceNodeOptions{schema, [delayed]() { return delayed; }}};
  arrow::acero::Declaration fast_plan{
      "table_source", arrow::acero::TableSourceNodeOptions{MakeTable(100, 4)}};
  std::vector<PlanInfo::Ptr> plans{
      std::make_shared<PlanInfo>(std::move(slow_plan), schema),
      std::make_shared<PlanInfo>(std::move(fast_plan), schema)};
  ailego::ThreadPool pool(2, false);
  auto node = std::make_shared<SegmentNode>(std::move(plans), &pool, 1);
  auto caller = std::async(
      std::launch::async, [node = std::move(node), schema]() mutable {
        auto declaration = arrow::acero::Declaration::Sequence(
            {{"source",
              arrow::acero::SourceNodeOptions{
                  schema, node->gen(), arrow::compute::Ordering::Implicit()}},
             {"fetch", arrow::acero::FetchNodeOptions{0, 1}}});
        // The plan is the only owner of the segment node during destruction.
        node.reset();
        PlanInfo plan(std::move(declaration), schema);
        auto reader = plan.execute_to_reader();
        if (!reader) {
          return arrow::Result<std::shared_ptr<arrow::Table>>(
              arrow::Status::ExecutionError(reader.error().c_str()));
        }
        return reader.value()->ToTable();
      });
  // Reading and destroying the caller-driven outer plan must not wait for
  // an unrelated segment's pending source after LIMIT has been satisfied.
  const bool completed =
      caller.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  delayed.MarkFinished(std::nullopt);
  auto result = caller.get();
  EXPECT_TRUE(completed) << "Reader LIMIT waited for an unrelated segment";
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  ASSERT_EQ(result.ValueOrDie()->num_rows(), 1);
  auto values = std::static_pointer_cast<arrow::Int32Array>(
      result.ValueOrDie()->column(0)->chunk(0));
  EXPECT_EQ(values->Value(0), 100);
}

TEST_F(QueryThreadingTest, ForwardFilterCompletesOnTheOnlyCpuWorker) {
  auto table = MakeTable(0);
  arrow::acero::Declaration plan{
      "table_source", arrow::acero::TableSourceNodeOptions{table, 4}};
  plan = arrow::acero::Declaration{
      "project",
      {std::move(plan)},
      arrow::acero::ProjectNodeOptions{
          {arrow::compute::call("greater_equal",
                                {arrow::compute::field_ref("value"),
                                 arrow::compute::literal(16)})},
          {"matched"}}};
  auto filter = std::make_shared<DocFilter>(
      std::make_shared<MockSegment>(), std::make_shared<QueryInfo>(),
      std::make_unique<arrow::acero::Declaration>(std::move(plan)), nullptr);
  auto submitted = arrow::internal::GetCpuThreadPool()->Submit(
      [filter]() { return filter->compute_filter_async(); });
  ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
  auto future = submitted.MoveValueUnsafe();
  CheckCompletion(future);
  ASSERT_TRUE(future.is_finished());
  ASSERT_TRUE(future.status().ok()) << future.status().ToString();
  for (uint64_t i = 0; i < 32; ++i) {
    // MockSegment's deletion filter also excludes odd row IDs.
    EXPECT_EQ(filter->is_filtered(i), i < 16 || i % 2 == 1);
  }
}

TEST_F(QueryThreadingTest, ForwardBitmapPreservesNullsAndChunkOffsets) {
  for (int rows : {0, 17, 67}) {
    arrow::BooleanBuilder builder;
    ASSERT_TRUE(builder.Append(true).ok());
    for (int i = 0; i < rows; ++i) {
      ASSERT_TRUE(
          (i % 5 == 0 ? builder.AppendNull() : builder.Append(i % 3 == 0))
              .ok());
    }
    ASSERT_TRUE(builder.Append(false).ok());
    auto values = builder.Finish().ValueOrDie()->Slice(1, rows);
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("matched", arrow::boolean())}), {values});
    // Small batches exercise chunk boundaries and nonzero bitmap offsets;
    // zero rows must also produce a valid, empty filter.
    arrow::acero::Declaration plan{
        "table_source", arrow::acero::TableSourceNodeOptions{table, 3}};
    auto filter = std::make_shared<DocFilter>(
        std::make_shared<MockSegment>(), std::make_shared<QueryInfo>(),
        std::make_unique<arrow::acero::Declaration>(std::move(plan)), nullptr);
    auto submitted = arrow::internal::GetCpuThreadPool()->Submit(
        [filter]() { return filter->compute_filter_async(); });
    ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
    auto future = submitted.MoveValueUnsafe();
    CheckCompletion(future);
    ASSERT_TRUE(future.status().ok()) << future.status().ToString();
    for (int i = 0; i < rows; ++i) {
      // Null predicates reject a row, as does MockSegment's deletion filter.
      EXPECT_EQ(filter->is_filtered(i), i % 2 == 1 || i % 5 == 0 || i % 3 != 0);
    }
  }
}

TEST_F(QueryThreadingTest, PendingFilterReleasesTheCpuWorker) {
  auto gate = arrow::Future<PlanInfo::Batch>::Make();
  auto entered = arrow::Future<>::Make();
  auto schema = arrow::schema({arrow::field("matched", arrow::boolean())});
  arrow::acero::Declaration plan{
      "source",
      arrow::acero::SourceNodeOptions{schema, [gate, entered]() mutable {
                                        entered.MarkFinished();
                                        return gate;
                                      }}};
  auto filter = std::make_shared<DocFilter>(
      std::make_shared<MockSegment>(), std::make_shared<QueryInfo>(),
      std::make_unique<arrow::acero::Declaration>(std::move(plan)), nullptr);
  auto submitted = arrow::internal::GetCpuThreadPool()->Submit(
      [filter]() { return filter->compute_filter_async(); });
  ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
  auto future = submitted.MoveValueUnsafe();
  EXPECT_TRUE(entered.Wait(2.0));
  auto heartbeat = arrow::internal::GetCpuThreadPool()->Submit(
      []() { return arrow::Status::OK(); });
  ASSERT_TRUE(heartbeat.ok());
  const bool worker_available = heartbeat->Wait(2.0);
  EXPECT_FALSE(future.is_finished());
  gate.MarkFinished(std::nullopt);
  CheckCompletion(future);
  EXPECT_TRUE(worker_available) << "Pending filter occupied the CPU worker";
  EXPECT_TRUE(future.status().ok()) << future.status().ToString();
}

TEST_F(QueryThreadingTest, AbandonedPlanCancelsWithoutBlockingCpuWorker) {
  auto gate = arrow::Future<PlanInfo::Batch>::Make();
  auto entered = arrow::Future<>::Make();
  auto destroyed = arrow::Future<>::Make();
  PlanInfo::BatchGenerator generator;
  {
    // The source owns this token until the running plan is destroyed.
    auto token =
        std::shared_ptr<int>(new int(0), [destroyed](int *value) mutable {
          delete value;
          destroyed.MarkFinished();
        });
    auto schema = MakeTable(0)->schema();
    arrow::acero::Declaration plan{
        "source", arrow::acero::SourceNodeOptions{
                      schema, [gate, entered, token]() mutable {
                        entered.MarkFinished();
                        return gate;
                      }}};
    PlanInfo info(std::move(plan), schema);
    auto result = info.execute_to_generator();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    generator = result.MoveValueUnsafe();
  }
  EXPECT_TRUE(entered.Wait(2.0));
  auto submitted = arrow::internal::GetCpuThreadPool()->Submit(
      [gen = std::exchange(generator, {})]() mutable {
        // LIMIT can discard a child stream on an Arrow CPU worker.
        gen = {};
        return arrow::Status::OK();
      });
  ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
  auto future = submitted.MoveValueUnsafe();
  const bool cancelled = future.Wait(2.0);
  EXPECT_FALSE(destroyed.is_finished());
  gate.MarkFinished(std::nullopt);
  CheckCompletion(future);
  EXPECT_TRUE(cancelled) << "Cancellation blocked the CPU worker";
  EXPECT_TRUE(destroyed.Wait(2.0)) << "Abandoned plan retained its source";
}

TEST_F(QueryThreadingTest, ForwardFilterPropagatesErrors) {
  for (bool invalid_schema : {false, true}) {
    auto schema = MakeTable(0)->schema();
    arrow::acero::Declaration plan{
        "source", arrow::acero::SourceNodeOptions{
                      schema, []() -> arrow::Future<PlanInfo::Batch> {
                        return arrow::Status::Invalid("injected filter error");
                      }}};
    if (invalid_schema) {
      plan = arrow::acero::Declaration{
          "table_source", arrow::acero::TableSourceNodeOptions{MakeTable(0)}};
    }
    auto filter = std::make_shared<DocFilter>(
        std::make_shared<MockSegment>(), std::make_shared<QueryInfo>(),
        std::make_unique<arrow::acero::Declaration>(std::move(plan)), nullptr);
    auto submitted = arrow::internal::GetCpuThreadPool()->Submit(
        [filter]() { return filter->compute_filter_async(); });
    ASSERT_TRUE(submitted.ok()) << submitted.status().ToString();
    auto future = submitted.MoveValueUnsafe();
    CheckCompletion(future);
    EXPECT_FALSE(future.status().ok());
    EXPECT_NE(future.status().ToString().find(
                  invalid_schema ? "not boolean" : "injected filter error"),
              std::string::npos);
  }
}

}  // namespace
}  // namespace zvec::sqlengine
