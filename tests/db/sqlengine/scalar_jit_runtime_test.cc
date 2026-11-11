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

#include <gtest/gtest.h>
#include "db/sqlengine/scalar_jit/runtime.h"

#ifdef ZVEC_ENABLE_SCALAR_JIT
#include <atomic>
#include <future>
#include <thread>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gandiva/tree_expr_builder.h>
#include "db/sqlengine/scalar_jit/scalar_predicate_cache.h"
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace zvec::sqlengine::scalar_jit {
namespace {
using TB = gandiva::TreeExprBuilder;
using Clock = std::chrono::steady_clock;

template <typename T>
T Take(arrow::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return std::move(result).ValueUnsafe();
}
auto Schema() {
  return arrow::schema({arrow::field("value", arrow::int32())});
}
auto Condition(int32_t value = 5, const std::string &op = "greater_than") {
  return TB::MakeCondition(TB::MakeFunction(
      op, {TB::MakeField(Schema()->field(0)), TB::MakeLiteral(value)},
      arrow::boolean()));
}
auto Batch() {
  arrow::Int32Builder builder;
  if (!builder.AppendValues({2, 7, 12}).ok()) throw std::bad_alloc();
  return arrow::RecordBatch::Make(Schema(), 3, {Take(builder.Finish())});
}
void CheckArrowFallback() {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  auto expression =
      Take(arrow::compute::greater(arrow::compute::field_ref("value"),
                                   arrow::compute::literal(int32_t{5}))
               .Bind(*Schema()));
  auto value = Take(arrow::compute::ExecuteScalarExpression(
      expression, arrow::compute::ExecBatch(*Batch())));
  auto array =
      std::static_pointer_cast<arrow::BooleanArray>(value.make_array());
  EXPECT_FALSE(array->Value(0));
  EXPECT_TRUE(array->Value(1));
  EXPECT_TRUE(array->Value(2));
}

TEST(ScalarJitRuntime, InspectDoesNotInitializeLLVM) {
  const auto info = GetRuntimeSnapshot();
  EXPECT_TRUE(info.built);
  EXPECT_TRUE(info.enabled);
  EXPECT_FALSE(info.initialized);
}

TEST(ScalarJitRuntime, ParameterReuseAndIndependentOwnership) {
  ScalarPredicateCache cache;
  auto first = Take(cache.prepare(Schema(), Condition(5)));
  auto second = Take(cache.prepare(Schema(), Condition(10)));
  EXPECT_TRUE(second.cache_hit);
  EXPECT_EQ(first.compiled, second.compiled);
  auto one = Take(first.bind(Batch()));
  auto two = Take(second.bind(Batch()));
  std::static_pointer_cast<arrow::Int32Scalar>(first.parameters[0])->value = 99;
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i)
    threads.emplace_back([&] {
      for (int repeat = 0; repeat < 1000; ++repeat)
        if (!one->evaluate(1) || two->evaluate(1) || !two->evaluate(2))
          ++failures;
    });
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(cache.stats().compilations, 1u);
}

TEST(ScalarJitRuntime, SingleFlightAndPressureBypassWithoutWaiting) {
  auto code = Take(ScalarPredicate::Make(Schema(), Condition()));
  std::promise<void> started, release;
  auto released = release.get_future().share();
  ScalarPredicateCacheOptions options;
  options.max_compilations = 1;
  options.compiler = [&](auto, auto, auto, const auto &) {
    started.set_value();
    released.wait();
    return code;
  };
  ScalarPredicateCache cache(options);
  auto compiling = std::async(
      std::launch::async, [&] { return cache.prepare(Schema(), Condition()); });
  started.get_future().wait();
  const auto same = cache.prepare(Schema(), Condition(8));
  const auto other = cache.prepare(Schema(), Condition(8, "less_than"));
  EXPECT_EQ(FailureReason(same.status()), "compilation_busy");
  EXPECT_EQ(FailureReason(other.status()), "compilation_busy");
  EXPECT_EQ(cache.stats().in_flight_bypasses, 1u);
  EXPECT_EQ(cache.stats().admission_rejections, 1u);
  release.set_value();
  EXPECT_TRUE(compiling.get().ok());
  EXPECT_TRUE(Take(cache.prepare(Schema(), Condition(9))).cache_hit);
  EXPECT_EQ(cache.stats().negative_entries, 0u);
}

TEST(ScalarJitRuntime, NegativeBoundsBackoffAndExceptionRecovery) {
  auto now = Clock::time_point{};
  int calls = 0;
  ScalarPredicateCacheOptions options;
  options.negative_capacity = 2;
  options.now = [&] { return now; };
  options.compiler =
      [&](auto, auto, auto,
          const auto &) -> arrow::Result<std::shared_ptr<ScalarPredicate>> {
    ++calls;
    if (calls == 1) throw std::bad_alloc();
    return arrow::Status::IOError("Injected transient compiler failure");
  };
  ScalarPredicateCache cache(options);
  EXPECT_FALSE(cache.prepare(Schema(), Condition()).ok());
  EXPECT_EQ(FailureReason(cache.prepare(Schema(), Condition()).status()),
            "compile_backoff");
  EXPECT_EQ(calls, 1);
  now += std::chrono::seconds(1);
  EXPECT_FALSE(cache.prepare(Schema(), Condition()).ok());
  now += std::chrono::seconds(1);
  EXPECT_EQ(FailureReason(cache.prepare(Schema(), Condition()).status()),
            "compile_backoff");
  now += std::chrono::seconds(1);
  EXPECT_FALSE(cache.prepare(Schema(), Condition()).ok());
  EXPECT_EQ(calls, 3);
  for (const auto *op : {"equal", "less_than", "not_equal"})
    EXPECT_FALSE(cache.prepare(Schema(), Condition(2, op)).ok());
  EXPECT_EQ(cache.stats().negative_entries, 2u);
  EXPECT_EQ(cache.stats().active_compilations, 0u);
  EXPECT_GT(cache.stats().evictions, 0u);
  CheckArrowFallback();
}

TEST(ScalarJitRuntime, PermanentNegativeAndAdmissionAreDistinct) {
  int calls = 0;
  ScalarPredicateCacheOptions options;
  options.compiler =
      [&](auto, auto, auto,
          const auto &) -> arrow::Result<std::shared_ptr<ScalarPredicate>> {
    ++calls;
    return arrow::Status::Invalid("Injected malformed shape");
  };
  ScalarPredicateCache cache(options);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(FailureReason(cache.prepare(Schema(), Condition()).status()),
              "unsupported_expression");
  EXPECT_EQ(calls, 1);
  options.compiler =
      [](auto, auto, auto,
         const auto &) -> arrow::Result<std::shared_ptr<ScalarPredicate>> {
    return Unavailable("compilation_busy");
  };
  ScalarPredicateCache busy(options);
  EXPECT_FALSE(busy.prepare(Schema(), Condition()).ok());
  EXPECT_EQ(busy.stats().negative_entries, 0u);
  EXPECT_EQ(busy.stats().compilations, 0u);
}

TEST(ScalarJitRuntime, TransientRecoveryAndBindingFailureStayLocal) {
  auto now = Clock::time_point{};
  int calls = 0;
  ScalarPredicateCacheOptions options;
  options.now = [&] { return now; };
  options.compiler = [&](auto schema, auto condition, auto parameters,
                         const auto &compile)
      -> arrow::Result<std::shared_ptr<ScalarPredicate>> {
    if (++calls == 1) return arrow::Status::OutOfMemory("Injected failure");
    return ScalarPredicate::MakeParameterized(schema, condition, parameters,
                                              compile);
  };
  ScalarPredicateCache cache(options);
  EXPECT_FALSE(cache.prepare(Schema(), Condition()).ok());
  EXPECT_EQ(cache.stats().active_compilations, 0u);
  CheckArrowFallback();
  now += std::chrono::seconds(1);
  auto prepared = Take(cache.prepare(Schema(), Condition()));
  EXPECT_FALSE(prepared.bind(nullptr).ok());
  auto reused = Take(cache.prepare(Schema(), Condition(10)));
  EXPECT_TRUE(reused.cache_hit);
  EXPECT_EQ(reused.compiled, prepared.compiled);
  EXPECT_TRUE(Take(reused.bind(Batch()))->evaluate(2));
  EXPECT_EQ(cache.stats().negative_entries, 0u);
  EXPECT_EQ(cache.stats().failed_compilations, 1u);
  EXPECT_EQ(calls, 2);
}

TEST(ScalarJitRuntime, ProcessCompilationPermitsReleaseOnScopeExit) {
  EXPECT_EQ(GetRuntimeConfig().max_compilations, 2u);
  CompilePermit available;
  {
    CompilePermit first, second;
    ASSERT_TRUE(first.Acquire().ok());
    ASSERT_TRUE(second.Acquire().ok());
    EXPECT_EQ(FailureReason(available.Acquire()), "compilation_busy");
  }
  EXPECT_TRUE(available.Acquire().ok());
}

TEST(ScalarJitRuntime, ClockExceptionReleasesReservation) {
  int ticks = 0;
  ScalarPredicateCacheOptions options;
  options.now = [&] {
    if (++ticks == 2) throw std::runtime_error("Injected clock failure");
    return Clock::time_point{};
  };
  options.compiler =
      [](auto, auto, auto,
         const auto &) -> arrow::Result<std::shared_ptr<ScalarPredicate>> {
    throw std::bad_alloc();
  };
  ScalarPredicateCache cache(options);
  EXPECT_FALSE(cache.prepare(Schema(), Condition()).ok());
  EXPECT_EQ(cache.stats().active_compilations, 0u);
  EXPECT_EQ(cache.stats().negative_entries, 1u);
}

TEST(ScalarJitRuntime, LimitsBeforeCompilationAndParameterCopies) {
  for (int limit = 0; limit < 4; ++limit) {
    ScalarPredicateCacheOptions options;
    if (limit == 0) options.compile.limits.nodes = 2;
    if (limit == 1) options.compile.limits.depth = 1;
    if (limit == 2) options.compile.limits.key_bytes = 8;
    if (limit == 3) options.compile.limits.input_bytes = 1;
    int calls = 0;
    options.compiler =
        [&](auto, auto, auto,
            const auto &) -> arrow::Result<std::shared_ptr<ScalarPredicate>> {
      ++calls;
      throw std::bad_alloc();
    };
    ScalarPredicateCache cache(options);
    EXPECT_EQ(FailureReason(cache.prepare(Schema(), Condition()).status()),
              "expression_limit");
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(cache.stats().negative_entries, 0u);
    EXPECT_EQ(FailureReason(
                  ScalarPredicate::Make(Schema(), Condition(), options.compile)
                      .status()),
              "expression_limit");
  }
  auto field = arrow::field("text", arrow::utf8());
  auto condition = TB::MakeCondition(TB::MakeFunction(
      "equal",
      {TB::MakeField(field),
       TB::MakeStringLiteral(std::string(1024 * 1024 + 1, 'x'))},
      arrow::boolean()));
  EXPECT_EQ(
      FailureReason(
          ScalarPredicate::Make(arrow::schema({field}), condition).status()),
      "expression_limit");
}

TEST(ScalarJitRuntime, ResidentAndLiveMemoryFollowFinalOwner) {
  const auto baseline = GetRuntimeSnapshot().counters["live_programs"];
  std::shared_ptr<BoundScalarPredicate> binding;
  {
    ScalarPredicateCacheOptions options;
    options.capacity = 1;
    options.max_live_programs = 2;
    ScalarPredicateCache cache(options);
    binding = Take(Take(cache.prepare(Schema(), Condition())).bind(Batch()));
    {
      auto second = Take(cache.prepare(Schema(), Condition(3, "less_than")));
    }
    EXPECT_EQ(cache.stats().entries, 1u);
    EXPECT_EQ(cache.stats().live_programs, 2u);
    EXPECT_GT(cache.stats().live_memory.object_bytes,
              cache.stats().resident_memory.object_bytes);
    EXPECT_EQ(
        FailureReason(cache.prepare(Schema(), Condition(3, "equal")).status()),
        "live_program_limit");
    EXPECT_TRUE(binding->evaluate(1));
  }
  EXPECT_EQ(GetRuntimeSnapshot().counters["live_programs"], baseline + 1);
  EXPECT_TRUE(binding->evaluate(1));
  binding.reset();
  EXPECT_EQ(GetRuntimeSnapshot().counters["live_programs"], baseline);
}

TEST(ScalarJitRuntime, EvictedSlotRecoveryAndExplicitIRCapture) {
  ScalarPredicateCacheOptions options;
  options.capacity = 1;
  options.max_live_programs = 2;
  ScalarPredicateCache cache(options);
  auto first = Take(cache.prepare(Schema(), Condition()));
  EXPECT_TRUE(first.compiled->dump_ir().empty());
  auto second = Take(cache.prepare(Schema(), Condition(3, "less_than")));
  first.compiled.reset();
  EXPECT_TRUE(cache.prepare(Schema(), Condition(3, "equal")).ok());
  ScalarCompileOptions capture;
  capture.dump_ir = true;
  auto code = Take(ScalarPredicate::Make(Schema(), Condition(), capture));
  EXPECT_FALSE(code->dump_ir().empty());
  EXPECT_GT(code->memory_usage().ir_bytes, 0u);
  EXPECT_GT(code->memory_usage().object_bytes, 0u);
}

#ifndef _WIN32
TEST(ScalarJitRuntime, ForkGuardPrecedesBlockedCacheAndInitialization) {
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  ScalarPredicateCacheOptions options;
  options.compiler =
      [&](auto, auto, auto,
          const auto &) -> arrow::Result<std::shared_ptr<ScalarPredicate>> {
    auto status = InitializeBackend([&] {
      entered.set_value();
      released.wait();
      return arrow::Status::OK();
    });
    return status.ok() ? arrow::Status::OutOfMemory("Injected compiler failure")
                       : status;
  };
  ScalarPredicateCache cache(options);
  auto worker = std::async(
      std::launch::async, [&] { return cache.prepare(Schema(), Condition()); });
  entered.get_future().wait();
  const auto child = fork();
  if (child == 0) {
    alarm(5);
    const auto info = GetRuntimeSnapshot();
    bool called = false;
    auto status = InitializeBackend([&] {
      called = true;
      return arrow::Status::OK();
    });
    const auto prepared = cache.prepare(Schema(), Condition());
    _exit(info.reason == "forked_process" && !called &&
                  FailureReason(status) == "forked_process" &&
                  FailureReason(prepared.status()) == "forked_process"
              ? 0
              : 1);
  }
  release.set_value();
  EXPECT_FALSE(worker.get().ok());
  ASSERT_GT(child, 0);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
#endif

TEST(ScalarJitRuntimeDeathTest,
     InitializationFailureIsProcessLocalAndOOMRetries) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(([] {
                if (!InitializeBackend([] {
                       return arrow::Status::OutOfMemory("injected");
                     }).IsOutOfMemory())
                  std::_Exit(1);
                if (!UnavailableReason().empty()) std::_Exit(2);
                if (!InitializeBackend([] { return arrow::Status::OK(); }).ok())
                  std::_Exit(3);
                auto failure = InitializeBackend(
                    [] { return arrow::Status::IOError("injected"); });
                bool called = false;
                auto again = InitializeBackend([&] {
                  called = true;
                  return arrow::Status::OK();
                });
                if (called ||
                    FailureReason(failure) != "initialization_failed" ||
                    FailureReason(again) != "initialization_failed")
                  std::_Exit(4);
                CheckArrowFallback();
                std::_Exit(testing::Test::HasFailure() ? 5 : 0);
              }()),
              testing::ExitedWithCode(0), "");
}
}  // namespace
}  // namespace zvec::sqlengine::scalar_jit
#else
TEST(ScalarJitRuntime, BuildDisabledDiagnostics) {
  auto info = zvec::sqlengine::scalar_jit::GetRuntimeSnapshot();
  EXPECT_FALSE(info.built);
  EXPECT_FALSE(info.enabled);
  EXPECT_FALSE(info.initialized);
  EXPECT_EQ(info.reason, "build_disabled");
}
#endif
