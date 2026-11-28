// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <thread>
#include <vector>
#include <arrow/api.h>
#include <arrow/util/async_generator.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include "db/sqlengine/planner/plan_info.h"

namespace zvec::sqlengine {
namespace {

PlanInfo::Ptr MakePlan(std::weak_ptr<int> *owner, bool fail = false) {
  auto lease = std::make_shared<int>(1);
  *owner = lease;
  auto schema = arrow::schema({arrow::field("value", arrow::int32())});
  arrow::Int32Builder values;
  EXPECT_TRUE(values.Append(42).ok());
  auto batch =
      arrow::RecordBatch::Make(schema, 1, {values.Finish().ValueOrDie()});
  auto calls = std::make_shared<std::atomic<int>>(0);
  using Batch = std::optional<arrow::compute::ExecBatch>;
  arrow::AsyncGenerator<Batch> generator = [lease, calls, batch, fail] {
    if (fail)
      return arrow::Future<Batch>::MakeFinished(
          arrow::Status::IOError("injected source failure"));
    if (calls->fetch_add(1) == 0)
      return arrow::Future<Batch>::MakeFinished(
          arrow::compute::ExecBatch(*batch));
    return arrow::Future<Batch>::MakeFinished(std::nullopt);
  };
  return std::make_shared<PlanInfo>(
      arrow::acero::Declaration{"source",
                                arrow::acero::SourceNodeOptions{
                                    schema, std::move(generator),
                                    arrow::compute::Ordering::Implicit()}},
      schema);
}

TEST(PlanLifetimeTest, ConcurrentCompletionAndEarlyCloseReleaseSources) {
  const int capacity = arrow::GetCpuThreadPoolCapacity();
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&] {
      for (int iteration = 0; iteration < 512; ++iteration) {
        std::weak_ptr<int> source;
        auto plan = MakePlan(&source);
        auto result = plan->execute_to_reader();
        plan.reset();
        if (!result) {
          ++failures;
          continue;
        }
        auto reader = std::move(*result);
        std::shared_ptr<arrow::RecordBatch> batch;
        if (!reader->ReadNext(&batch).ok() || !batch ||
            batch->num_rows() != 1) {
          ++failures;
          continue;
        }
        if (iteration % 2 == 0) {
          if (!reader->ReadNext(&batch).ok() || batch) ++failures;
        } else if (!reader->Close().ok()) {
          ++failures;
        }
        reader.reset();
        // Completion may be publishing on another thread as the reader dies.
        // Sources must not become retained completion/plan ownership cycles.
        for (int attempt = 0; !source.expired() && attempt < 10000; ++attempt)
          std::this_thread::yield();
        if (!source.expired()) ++failures;
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0, failures.load());
  EXPECT_EQ(capacity, arrow::GetCpuThreadPoolCapacity());
}

// Drop the reader on the readiness edge, without first consuming EOF or
// retaining another finished future. This exercises the notification race.
TEST(PlanLifetimeTest, ReleaseImmediatelyAfterReadiness) {
  struct ProbeOptions : arrow::acero::ExecNodeOptions {
    std::shared_ptr<arrow::Table> table;
    std::shared_ptr<std::atomic<arrow::acero::ExecPlan *>> plan;
  };
  static const auto registered =
      arrow::acero::default_exec_factory_registry()->AddFactory(
          "zvec_test_capture_plan",
          [](arrow::acero::ExecPlan *plan,
             std::vector<arrow::acero::ExecNode *> inputs,
             const arrow::acero::ExecNodeOptions &base)
              -> arrow::Result<arrow::acero::ExecNode *> {
            const auto &options = static_cast<const ProbeOptions &>(base);
            options.plan->store(plan);
            return arrow::acero::MakeExecNode(
                "table_source", plan, std::move(inputs),
                arrow::acero::TableSourceNodeOptions{options.table});
          });
  ASSERT_TRUE(registered.ok()) << registered;
  arrow::Int32Builder values;
  ASSERT_TRUE(values.Append(42).ok());
  auto table =
      arrow::Table::Make(arrow::schema({arrow::field("value", arrow::int32())}),
                         {values.Finish().ValueOrDie()});
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&] {
      for (int iteration = 0; iteration < 256; ++iteration) {
        auto probe =
            std::make_shared<std::atomic<arrow::acero::ExecPlan *>>(nullptr);
        auto options = std::make_shared<ProbeOptions>();
        options->table = table;
        options->plan = probe;
        PlanInfo info(
            arrow::acero::Declaration{
                "zvec_test_capture_plan",
                std::static_pointer_cast<arrow::acero::ExecNodeOptions>(
                    options)},
            table->schema());
        auto reader = info.execute_to_reader();
        if (!reader) {
          ++failures;
          continue;
        }
        auto *plan = probe->load();
        if (!plan) {
          ++failures;
          continue;
        }
        // The construction wrapper must add no execution node (just
        // source/sink).
        if (plan->nodes().size() != 2) ++failures;
        while (!plan->finished().is_finished()) std::this_thread::yield();
        reader->reset();
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0, failures.load());
}

TEST(PlanLifetimeTest, SourceErrorReleasesPlanAndPropagates) {
  std::weak_ptr<int> source;
  {
    auto plan = MakePlan(&source, true);
    auto result = plan->execute_to_reader();
    ASSERT_TRUE(result) << result.error().message();
    plan.reset();
    std::shared_ptr<arrow::RecordBatch> batch;
    EXPECT_TRUE((*result)->ReadNext(&batch).IsIOError());
  }
  for (int attempt = 0; !source.expired() && attempt < 10000; ++attempt)
    std::this_thread::yield();
  EXPECT_TRUE(source.expired());
}

TEST(PlanLifetimeTest, FactoryFailureDoesNotRetainPlanOptions) {
  std::weak_ptr<int> source;
  {
    auto owner = std::make_shared<int>(1);
    source = owner;
    struct Options : arrow::acero::ExecNodeOptions {
      std::shared_ptr<int> owner;
    };
    auto options = std::make_shared<Options>();
    options->owner = std::move(owner);
    PlanInfo plan(
        arrow::acero::Declaration{
            "missing_zvec_test_factory",
            std::static_pointer_cast<arrow::acero::ExecNodeOptions>(options)},
        arrow::schema({}));
    auto reader = plan.execute_to_reader();
    ASSERT_TRUE(reader) << reader.error().message();
    std::shared_ptr<arrow::RecordBatch> batch;
    EXPECT_FALSE((*reader)->ReadNext(&batch).ok());
  }
  EXPECT_TRUE(source.expired());
}

}  // namespace
}  // namespace zvec::sqlengine
