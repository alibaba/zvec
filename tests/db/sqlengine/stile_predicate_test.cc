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
#include <future>
#include <limits>
#include <random>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include "db/sqlengine/stile/predicate.h"

namespace zvec::test {
namespace {
namespace cp = arrow::compute;
using stile::Predicate;
using stile::TableIndex;
using stile::Truth;

class StilePredicateTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(stile::Initialize().ok());
    arrow::Int32Builder integers;
    arrow::DoubleBuilder doubles;
    arrow::StringBuilder strings;
    arrow::BooleanBuilder booleans;
    for (int i = 0; i < 67; ++i) {
      ASSERT_TRUE(
          (i % 7 ? integers.Append(i - 31) : integers.AppendNull()).ok());
      ASSERT_TRUE(
          (i % 11 ? doubles.Append(i - 32.5) : doubles.AppendNull()).ok());
      ASSERT_TRUE((i % 5 ? strings.Append(i % 2 ? "alpha" : "beta")
                         : strings.AppendNull())
                      .ok());
      ASSERT_TRUE(
          (i % 3 ? booleans.Append(i % 2 != 0) : booleans.AppendNull()).ok());
    }
    arrow::ArrayVector arrays;
    for (arrow::ArrayBuilder *builder : std::vector<arrow::ArrayBuilder *>{
             &integers, &doubles, &strings, &booleans}) {
      auto array = builder->Finish();
      ASSERT_TRUE(array.ok());
      arrays.push_back(*array);
    }
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    for (size_t c = 0; c < arrays.size(); ++c) {
      // 非零 offset、空 chunk 及各列不同的分块，避免只验证连续数组。
      auto sliced = arrays[c]->Slice(1, 65);
      const int split = 2 + static_cast<int>(c) * 7;
      columns.push_back(
          std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
              sliced->Slice(0, 0), sliced->Slice(0, split),
              sliced->Slice(split, 0), sliced->Slice(split, 65 - split),
              sliced->Slice(65, 0)}));
    }
    table_ =
        arrow::Table::Make(arrow::schema({arrow::field("i", arrow::int32()),
                                          arrow::field("d", arrow::float64()),
                                          arrow::field("s", arrow::utf8()),
                                          arrow::field("b", arrow::boolean())}),
                           columns);
  }

  void expect_parity(const cp::Expression &expression,
                     const stile::CompileOptions &options = {}) {
    SCOPED_TRACE(expression.ToString());
    auto index = TableIndex::Make(table_, true);
    ASSERT_TRUE(index.ok()) << index.status().ToString();
    auto bound = expression.Bind(*table_->schema());
    ASSERT_TRUE(bound.ok()) << bound.status().ToString();
    auto compiled = Predicate::Compile(*bound, *index, options);
    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    for (int64_t row = 0; row < table_->num_rows(); ++row) {
      std::vector<arrow::Datum> scalars;
      for (const auto &column : table_->columns()) {
        auto scalar = column->GetScalar(row);
        ASSERT_TRUE(scalar.ok());
        scalars.emplace_back(*scalar);
      }
      auto result =
          cp::ExecuteScalarExpression(*bound, cp::ExecBatch(scalars, 1));
      ASSERT_TRUE(result.ok()) << result.status().ToString();
      const auto &value = result->scalar_as<arrow::BooleanScalar>();
      const auto expected = !value.is_valid ? Truth::kNull
                            : value.value   ? Truth::kTrue
                                            : Truth::kFalse;
      EXPECT_EQ((*compiled)->evaluate(row), expected) << row;
      EXPECT_EQ((*compiled)->matches(row), expected == Truth::kTrue) << row;
    }
    EXPECT_FALSE((*compiled)->matches(-1));
    EXPECT_FALSE((*compiled)->matches(table_->num_rows()));
    EXPECT_FALSE((*compiled)->matches(std::numeric_limits<int64_t>::max()));
  }

  std::shared_ptr<arrow::Table> table_;
};

TEST_F(StilePredicateTest, SeededThreeValuedDifferential) {
  const std::vector<cp::Expression> leaves{
      cp::field_ref("b"),
      cp::is_null(cp::field_ref("i")),
      cp::is_valid(cp::field_ref("s")),
      cp::call("less", {cp::field_ref("i"), cp::literal(int32_t{4})}),
      cp::call("greater_equal", {cp::field_ref("d"), cp::literal(-3.5)}),
      cp::call("equal", {cp::field_ref("s"), cp::literal("alpha")}),
      cp::literal(true),
      cp::literal(false),
      cp::literal(arrow::MakeNullScalar(arrow::boolean()))};
  const char *operations[]{"and", "or", "and_kleene", "or_kleene"};
  std::mt19937 random(0x5711e);
  for (int trial = 0; trial < 160; ++trial) {
    auto expression = leaves[random() % leaves.size()];
    for (int depth = 0; depth < 5; ++depth) {
      auto right = leaves[random() % leaves.size()];
      if (random() % 2) right = cp::call("invert", {right});
      expression = cp::call(operations[random() % 4], {expression, right});
    }
    ASSERT_NO_FATAL_FAILURE(expect_parity(expression));
  }
  // 融合优化开启与关闭时都必须保持相同语义。
  auto range = cp::call(
      "and_kleene",
      {cp::call("greater_equal",
                {cp::field_ref("i"), cp::literal(int32_t{-10})}),
       cp::call("less", {cp::field_ref("i"), cp::literal(int32_t{10})})});
  ASSERT_NO_FATAL_FAILURE(expect_parity(range));
  stile::CompileOptions unfused;
  unfused.max_fusion_pairs = 0;
  ASSERT_NO_FATAL_FAILURE(expect_parity(range, unfused));
}

TEST_F(StilePredicateTest, NanSignedZeroAndSetNullModes) {
  const uint64_t payload = UINT64_C(0x7ff8000000000042);
  double nan_payload;
  std::memcpy(&nan_payload, &payload, sizeof(payload));
  arrow::DoubleBuilder builder;
  ASSERT_TRUE(
      builder
          .AppendValues({-0.0, 0.0, -1.0, 1.0,
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN(), nan_payload})
          .ok());
  ASSERT_TRUE(builder.AppendNull().ok());
  auto values = builder.Finish();
  ASSERT_TRUE(values.ok());
  table_ =
      arrow::Table::Make(arrow::schema({arrow::field("d", arrow::float64())}),
                         arrow::ArrayVector{*values});
  for (double value : {-0.0, 0.0, nan_payload}) {
    for (const auto *op : {"equal", "not_equal", "less", "less_equal",
                           "greater", "greater_equal"}) {
      ASSERT_NO_FATAL_FAILURE(expect_parity(
          cp::call(op, {cp::field_ref("d"), cp::literal(value)})));
    }
  }
  arrow::DoubleBuilder members;
  ASSERT_TRUE(members.AppendValues({-0.0, nan_payload}).ok());
  ASSERT_TRUE(members.AppendNull().ok());
  auto set = members.Finish();
  ASSERT_TRUE(set.ok());
  for (auto mode :
       {cp::SetLookupOptions::MATCH, cp::SetLookupOptions::SKIP,
        cp::SetLookupOptions::EMIT_NULL, cp::SetLookupOptions::INCONCLUSIVE}) {
    auto expression = cp::call("is_in", {cp::field_ref("d")},
                               cp::SetLookupOptions(*set, mode));
    ASSERT_NO_FATAL_FAILURE(expect_parity(expression));
    ASSERT_NO_FATAL_FAILURE(expect_parity(cp::call("invert", {expression})));
  }
}

TEST_F(StilePredicateTest, LosslessCastsAndStringPatterns) {
  for (auto type : {arrow::int64(), arrow::float64()}) {
    auto cast =
        cp::call("cast", {cp::field_ref("i")}, cp::CastOptions::Safe(type));
    ASSERT_NO_FATAL_FAILURE(expect_parity(cp::call(
        "less", {cast, cp::literal(arrow::MakeScalar(type, 3).ValueOrDie())})));
  }
  for (const auto *pattern : {"a%", "%a", "%ph%", "%"}) {
    ASSERT_NO_FATAL_FAILURE(
        expect_parity(cp::call("match_like", {cp::field_ref("s")},
                               cp::MatchSubstringOptions(pattern))));
  }
  auto index = TableIndex::Make(table_);
  ASSERT_TRUE(index.ok());
  for (const auto &expression :
       {cp::call("less", {cp::call("cast", {cp::field_ref("i")},
                                   cp::CastOptions::Safe(arrow::float32())),
                          cp::literal(1.0f)}),
        cp::call("match_like", {cp::field_ref("s")},
                 cp::MatchSubstringOptions("alpha")),
        cp::call("match_like", {cp::field_ref("s")},
                 cp::MatchSubstringOptions("a_ph%")),
        cp::call("is_finite", {cp::field_ref("d")})}) {
    auto bound = expression.Bind(*table_->schema());
    ASSERT_TRUE(bound.ok());
    EXPECT_TRUE(Predicate::Compile(*bound, *index).status().IsNotImplemented());
  }
}

TEST_F(StilePredicateTest, BudgetErrorsRemainDistinctFromInvalidInput) {
  auto index = TableIndex::Make(table_);
  ASSERT_TRUE(index.ok());
  auto expression = cp::call(
      "and_kleene",
      {cp::call("less", {cp::field_ref("i"), cp::literal(int32_t{10})}),
       cp::is_valid(cp::field_ref("i"))});
  auto bound = expression.Bind(*table_->schema());
  ASSERT_TRUE(bound.ok());
  stile::CompileOptions options;
  options.max_nodes = 1;
  EXPECT_TRUE(
      Predicate::Compile(*bound, *index, options).status().IsCapacityError());
  options = {};
  options.max_operands = 1;
  EXPECT_TRUE(
      Predicate::Compile(*bound, *index, options).status().IsCapacityError());
  EXPECT_TRUE(Predicate::Compile(*bound, nullptr).status().IsInvalid());
  auto non_boolean = cp::field_ref("i").Bind(*table_->schema());
  ASSERT_TRUE(non_boolean.ok());
  EXPECT_TRUE(
      Predicate::Compile(*non_boolean, *index).status().IsNotImplemented());
  options = {};
  options.max_depth = -1;
  EXPECT_TRUE(Predicate::Compile(*bound, *index, options).status().IsInvalid());
  EXPECT_TRUE(Predicate::Compile(*bound, *index).ok());
}

TEST_F(StilePredicateTest,
       PredicateAloneOwnsBuffersDuringConcurrentEvaluation) {
  auto expression =
      cp::call("equal", {cp::field_ref("i"), cp::literal(int32_t{1})});
  auto bound = expression.Bind(*table_->schema());
  ASSERT_TRUE(bound.ok());
  auto index = TableIndex::Make(table_);
  ASSERT_TRUE(index.ok());
  auto result = Predicate::Compile(*bound, *index);
  ASSERT_TRUE(result.ok());
  auto predicate = std::move(result).ValueOrDie();
  const auto rows = table_->num_rows();
  std::weak_ptr<arrow::Buffer> buffer =
      table_->column(0)->chunk(1)->data()->buffers[1];
  *index = nullptr;
  table_.reset();
  EXPECT_FALSE(buffer.expired());
  std::vector<std::future<bool>> readers;
  for (int t = 0; t < 4; ++t) {
    readers.push_back(std::async(std::launch::async, [&] {
      for (int repeat = 0; repeat < 100; ++repeat)
        for (int64_t row = 0; row < rows; ++row)
          if (predicate->matches(row) != (row == 31)) return false;
      return true;
    }));
  }
  for (auto &reader : readers) EXPECT_TRUE(reader.get());
  predicate.reset();
  EXPECT_TRUE(buffer.expired());
}

}  // namespace
}  // namespace zvec::test
