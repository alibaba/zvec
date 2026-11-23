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

#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>
#include "db/sqlengine/stile/predicate.h"
#include "db/sqlengine/stile/value_index.h"

namespace zvec::test {
namespace {
namespace cp = arrow::compute;
using stile::Predicate;
using stile::TableIndex;
using stile::Truth;

void expect_batch_parity(const Predicate &predicate,
                         const std::vector<Truth> &expected) {
  predicate.evaluate_many(nullptr, 0, nullptr);
  EXPECT_EQ(predicate.match_many(nullptr, 0, nullptr), 0u);

  std::array<int64_t, 65> rows{};
  auto check_matches = [&](size_t count) {
    std::array<bool, 65> matches{};
    const auto matched =
        predicate.match_many(rows.data(), count, matches.data());
    size_t expected_count = 0;
    for (size_t i = 0; i < count; ++i) {
      const auto row = rows[i];
      const bool match = row >= 0 && row < predicate.num_rows() &&
                         expected[row] == Truth::kTrue;
      EXPECT_EQ(matches[i], match) << "count=" << count << " i=" << i;
      expected_count += match;
    }
    EXPECT_EQ(matched, expected_count) << "count=" << count;
  };

  // Exercise a partial block, a full block, and a tail after two full blocks.
  constexpr size_t counts[]{1, 31, 32, 33, 64, 65};
  if (!expected.empty()) {
    for (size_t i = 0; i < rows.size(); ++i) {
      rows[i] = static_cast<int64_t>((i * 17) % expected.size());
    }
    rows[3] = rows[2];
    rows[32] = rows[31];
    std::array<Truth, 65> values{};
    for (const auto count : counts) {
      predicate.evaluate_many(rows.data(), count, values.data());
      for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(values[i], expected[rows[i]])
            << "count=" << count << " i=" << i;
      }
      check_matches(count);
    }
  }

  // Only match_many accepts out-of-range rows. Mix them with valid rows on
  // either side of a block boundary and check the returned count as well.
  rows[0] = -1;
  rows[31] = predicate.num_rows();
  rows[32] = std::numeric_limits<int64_t>::max();
  check_matches(rows.size());
}

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
      // Exercise nonzero slice offsets, empty chunks, and different chunk
      // boundaries across columns.
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
    std::vector<Truth> arrow_results;
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
      arrow_results.push_back(expected);
      EXPECT_EQ((*compiled)->evaluate(row), expected) << row;
      EXPECT_EQ((*compiled)->matches(row), expected == Truth::kTrue) << row;
    }
    EXPECT_FALSE((*compiled)->matches(-1));
    EXPECT_FALSE((*compiled)->matches(table_->num_rows()));
    EXPECT_FALSE((*compiled)->matches(std::numeric_limits<int64_t>::max()));
    expect_batch_parity(**compiled, arrow_results);
  }

  void expect_set_parity(const std::shared_ptr<arrow::Array> &input,
                         const std::shared_ptr<arrow::Array> &members) {
    table_ =
        arrow::Table::Make(arrow::schema({arrow::field("v", input->type())}),
                           arrow::ArrayVector{input});
    const auto split = members->length() / 2;
    auto chunks = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
        members->Slice(0, split), members->Slice(0, 0), members->Slice(split)});
    for (const auto &set : {arrow::Datum(members), arrow::Datum(chunks)}) {
      for (auto mode : {cp::SetLookupOptions::MATCH, cp::SetLookupOptions::SKIP,
                        cp::SetLookupOptions::EMIT_NULL,
                        cp::SetLookupOptions::INCONCLUSIVE}) {
        auto in = cp::call("is_in", {cp::field_ref("v")},
                           cp::SetLookupOptions(set, mode));
        ASSERT_NO_FATAL_FAILURE(expect_parity(in));
        ASSERT_NO_FATAL_FAILURE(expect_parity(cp::not_(in)));
      }
    }
  }

  std::shared_ptr<arrow::Table> table_;
};

TEST_F(StilePredicateTest, EmptyTableRejectsEveryCandidate) {
  table_ = table_->Slice(0, 0);
  expect_parity(cp::literal(true));
}

TEST_F(StilePredicateTest, LocatesRowsInRegularAndIrregularSlicedChunks) {
  struct Layout {
    int stride;
    std::vector<int> lengths;
  };
  // In the irregular layout, several short chunks fall in the same bucket.
  for (const auto &layout :
       std::vector<Layout>{{63, {63, 63, 7}},
                           {64, {64, 64, 7}},
                           {0, {1, 1, 2, 1, 82, 1, 1, 1}}}) {
    SCOPED_TRACE(layout.stride);
    const int rows =
        std::accumulate(layout.lengths.begin(), layout.lengths.end(), 0);
    arrow::Int32Builder builder;
    // Skip the first value so the slices have nonzero offsets.
    for (int i = 0; i <= rows; ++i) {
      ASSERT_OK(i % 19 == 0 ? builder.AppendNull() : builder.Append(i - 1));
    }
    ASSERT_OK_AND_ASSIGN(auto data, builder.Finish());
    arrow::ArrayVector slices;
    int offset = 1;
    for (const int length : layout.lengths) {
      slices.push_back(data->Slice(offset, 0));
      slices.push_back(data->Slice(offset, length));
      offset += length;
    }
    auto chunks = std::make_shared<arrow::ChunkedArray>(std::move(slices));
    table_ = arrow::Table::Make(
        arrow::schema({arrow::field("v", data->type())}), {chunks});
    ASSERT_OK_AND_ASSIGN(auto index, TableIndex::Make(table_, true));
    const auto &column = *index->column(0);
    ASSERT_EQ(column.stride(), layout.stride);
    int row = 0;
    for (size_t chunk = 0; chunk < layout.lengths.size(); ++chunk) {
      EXPECT_EQ(column.chunk_begin(chunk), row);
      ASSERT_NO_FATAL_FAILURE(
          expect_parity(cp::equal(cp::field_ref("v"), cp::literal(row))));
      for (int i = 0; i < layout.lengths[chunk]; ++i, ++row) {
        int64_t local = -1;
        EXPECT_EQ(column.locate(row, &local), chunk) << row;
        EXPECT_EQ(local, i) << row;
      }
    }
  }
}

TEST_F(StilePredicateTest, LiteralOnLeftAndFusedRangeEndpointsMatchArrow) {
  const auto field = cp::field_ref("i");
  for (const auto *op : {"equal", "not_equal", "less", "less_equal", "greater",
                         "greater_equal"}) {
    ASSERT_NO_FATAL_FAILURE(
        expect_parity(cp::call(op, {cp::literal(int32_t{1}), field})));
  }
  stile::CompileOptions unfused;
  unfused.max_fusion_pairs = 0;
  for (const auto *lower : {"less", "less_equal"}) {
    for (const auto *upper : {"greater", "greater_equal"}) {
      for (const auto *op : {"and", "and_kleene"}) {
        auto range =
            cp::call(op, {cp::call(lower, {cp::literal(int32_t{-1}), field}),
                          cp::call(upper, {cp::literal(int32_t{1}), field})});
        ASSERT_NO_FATAL_FAILURE(expect_parity(range));
        ASSERT_NO_FATAL_FAILURE(expect_parity(range, unfused));
      }
    }
  }
}

TEST_F(StilePredicateTest, DenseSparseAndEmptyIntegerSetsMatchArrow) {
  arrow::Int32Builder numbers;
  ASSERT_OK(numbers.AppendValues({-1, 0, 1, 62, 63, 64, 65, 66, 126, 127, 128,
                                  std::numeric_limits<int32_t>::max()}));
  ASSERT_OK(numbers.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto input, numbers.Finish());
  // Exercise both sides of the 64-bit bitmap boundaries and the out-of-range
  // sentinel.
  ASSERT_OK(numbers.AppendValues({0, 63, 64, 65, 127, 127}));
  ASSERT_OK(numbers.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto dense, numbers.Finish());
  ASSERT_NO_FATAL_FAILURE(expect_set_parity(input, dense));
  ASSERT_NO_FATAL_FAILURE(expect_set_parity(input, dense->Slice(0, 0)));
  ASSERT_OK(numbers.AppendValues({0, std::numeric_limits<int32_t>::max()}));
  ASSERT_OK_AND_ASSIGN(auto sparse, numbers.Finish());
  ASSERT_NO_FATAL_FAILURE(expect_set_parity(input, sparse));
}

TEST_F(StilePredicateTest, CollidingIntegerSetsRespectProbeBudget) {
  // 乘法哈希的逆元构造合法但落入同一桶的 uint64 值。
  constexpr uint64_t inverse = UINT64_C(0xf1de83e19937733d);
  static_assert(inverse * UINT64_C(0x9e3779b97f4a7c15) == 1);
  arrow::UInt64Builder builder;
  for (uint64_t i = 0; i < 65; ++i) ASSERT_OK(builder.Append(i * inverse));
  ASSERT_OK_AND_ASSIGN(auto values, builder.Finish());
  ASSERT_OK(builder.Append(0));
  ASSERT_OK(builder.Append(inverse));
  ASSERT_OK(builder.Append(64 * inverse));
  ASSERT_OK(builder.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto input, builder.Finish());
  ASSERT_NO_FATAL_FAILURE(expect_set_parity(input, values->Slice(0, 64)));

  ASSERT_OK_AND_ASSIGN(auto index, TableIndex::Make(table_));
  const auto expression =
      cp::call("is_in", {cp::field_ref("v")}, cp::SetLookupOptions(values));
  ASSERT_OK_AND_ASSIGN(auto bound, expression.Bind(*table_->schema()));
  const auto compiled = Predicate::Compile(bound, index);
  ASSERT_TRUE(compiled.status().IsCapacityError())
      << compiled.status().ToString();
  EXPECT_NE(compiled.status().message().find("probe"), std::string::npos);
}

TEST(StileKeyIndexTest, CollisionLimitPreservesHitsDuplicatesAndWraparound) {
  constexpr uint64_t inverse = UINT64_C(0xf1de83e19937733d);
  // 128 个键分配 512 个桶；从末桶开始也必须正确处理环绕。
  for (uint64_t home : {uint64_t{0}, uint64_t{511}}) {
    stile::internal::KeyIndex index(128);
    const uint64_t base = home << 55;
    for (uint64_t i = 0; i < 64; ++i)
      EXPECT_EQ(index.insert((base + i) * inverse), static_cast<int32_t>(i));
    for (uint64_t i = 0; i < 64; ++i) {
      const auto key = (base + i) * inverse;
      EXPECT_EQ(index.find(key), static_cast<int32_t>(i));
      EXPECT_EQ(index.insert(key), static_cast<int32_t>(i));
    }
    index.finish();
    EXPECT_EQ(index.size(), 64);
    EXPECT_FALSE(index.contains((base + 64) * inverse));
    EXPECT_THROW(index.insert((base + 64) * inverse), std::length_error);
  }
}

TEST(StileKeyIndexTest, LongContiguousClusterPreservesDistinctHomeBuckets) {
  constexpr uint64_t inverse = UINT64_C(0xf1de83e19937733d);
  stile::internal::KeyIndex index(128);
  std::vector<uint64_t> keys;
  // 连续占用 127 个桶，但每个键距自己的起始桶不超过 63。
  for (uint64_t home = 0; home < 64; ++home) {
    for (uint64_t ordinal = 0; ordinal < (home == 63 ? 1u : 2u); ++ordinal) {
      const auto key = ((home << 55) + ordinal) * inverse;
      EXPECT_EQ(index.insert(key), static_cast<int32_t>(keys.size()));
      keys.push_back(key);
    }
  }
  index.finish();
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_EQ(index.find(keys[i]), static_cast<int32_t>(i));
    EXPECT_TRUE(index.contains(keys[i]));
  }
  EXPECT_EQ(index.find(2 * inverse), -1);
  EXPECT_FALSE(index.contains(2 * inverse));
}

TEST_F(StilePredicateTest, StringSetsPreserveEmbeddedNulAndEmptyValues) {
  const std::string embedded("a\0b", 3);
  arrow::StringBuilder words;
  ASSERT_OK(words.AppendValues({"", "a", embedded, "ab", "last"}));
  ASSERT_OK(words.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto input, words.Finish());
  ASSERT_OK(words.AppendValues({"", embedded, embedded}));
  ASSERT_OK(words.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto members, words.Finish());
  ASSERT_NO_FATAL_FAILURE(expect_set_parity(input, members));
}

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
  // Both fused and unfused range comparisons must agree with Arrow.
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
  for (bool nan_is_null : {false, true}) {
    ASSERT_NO_FATAL_FAILURE(
        expect_parity(cp::is_null(cp::field_ref("d"), nan_is_null)));
  }
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

TEST_F(StilePredicateTest, CastsRespectUnsignedAndFloatingPointLimits) {
  arrow::UInt32Builder values;
  ASSERT_OK(values.AppendValues(
      {0, 1u << 24, (1u << 24) + 1, std::numeric_limits<uint32_t>::max()}));
  ASSERT_OK(values.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto array, values.Finish());
  table_ = arrow::Table::Make(arrow::schema({arrow::field("u", array->type())}),
                              arrow::ArrayVector{array});
  auto cast = [](const std::shared_ptr<arrow::DataType> &target) {
    return cp::call("cast", {cp::field_ref("u")},
                    cp::CastOptions::Safe(target));
  };
  ASSERT_NO_FATAL_FAILURE(expect_parity(
      cp::greater(cast(arrow::int64()), cp::literal(int64_t{2147483647}))));
  ASSERT_NO_FATAL_FAILURE(expect_parity(
      cp::greater(cast(arrow::float64()), cp::literal(16777216.0))));
  ASSERT_OK_AND_ASSIGN(auto index, TableIndex::Make(table_));
  for (const auto &expr :
       {cp::greater(cast(arrow::int32()), cp::literal(int32_t{0})),
        cp::greater(cast(arrow::float32()), cp::literal(16777216.0f))}) {
    ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*table_->schema()));
    EXPECT_TRUE(Predicate::Compile(bound, index).status().IsNotImplemented())
        << expr.ToString();
  }
}

TEST_F(StilePredicateTest, Utf8LikeFastPathsAndEscapes) {
  const std::string cjk = "\xe4\xb8\xad";
  const std::string emoji = "\xf0\x9f\x8c\x8d";
  arrow::StringBuilder text;
  ASSERT_OK(text.AppendValues(
      {"", cjk, cjk + "x", "x" + cjk, "x" + emoji + "y", R"(a\b)", "ab%"}));
  ASSERT_OK(text.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto values, text.Finish());
  for (const auto &type : {arrow::utf8(), arrow::large_utf8()}) {
    ASSERT_OK_AND_ASSIGN(auto array, cp::Cast(values, type));
    table_ = arrow::Table::Make(arrow::schema({arrow::field("s", type)}),
                                arrow::ArrayVector{array.make_array()});
    for (const std::string &pattern : std::vector<std::string>{
             cjk + "%", "%" + cjk, "%" + emoji + "%", "%%", R"(%a\b%)"}) {
      ASSERT_NO_FATAL_FAILURE(
          expect_parity(cp::call("match_like", {cp::field_ref("s")},
                                 cp::MatchSubstringOptions(pattern))));
    }
    ASSERT_OK_AND_ASSIGN(auto index, TableIndex::Make(table_));
    for (const auto &options : {cp::MatchSubstringOptions(R"(ab\%)"),
                                cp::MatchSubstringOptions("a_c"),
                                cp::MatchSubstringOptions("x%", true)}) {
      auto expr = cp::call("match_like", {cp::field_ref("s")}, options);
      ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*table_->schema()));
      EXPECT_TRUE(Predicate::Compile(bound, index).status().IsNotImplemented())
          << expr.ToString();
    }
  }
}

TEST_F(StilePredicateTest, LongLikeSubstringsFallBackButAnchorsStaySupported) {
  // 按字节限制子串搜索；覆盖边界、重复前缀、NUL 和多字节 UTF-8。
  const std::string cjk = "\xe4\xb8\xad";
  const std::vector<std::string> literals{
      std::string(64, 'a'),
      std::string(65, 'a'),
      std::string(32, 'a') + "b" + std::string(32, 'a'),
      std::string(32, 'a') + '\0' + std::string(32, 'a'),
      std::string(62, 'a') + cjk,
      std::string(2048, 'a') + "b" + std::string(2048, 'a')};
  for (const auto &literal : literals) {
    SCOPED_TRACE(literal.size());
    arrow::StringBuilder builder;
    ASSERT_OK(builder.AppendValues({literal, "x" + literal + "y", literal + "y",
                                    "x" + literal,
                                    std::string(literal.size() * 2, 'a'), ""}));
    ASSERT_OK(builder.AppendNull());
    ASSERT_OK_AND_ASSIGN(auto values, builder.Finish());
    for (const auto &type : {arrow::utf8(), arrow::large_utf8(),
                             arrow::binary(), arrow::large_binary()}) {
      SCOPED_TRACE(type->ToString());
      ASSERT_OK_AND_ASSIGN(auto array, cp::Cast(values, type));
      table_ = arrow::Table::Make(arrow::schema({arrow::field("s", type)}),
                                  arrow::ArrayVector{array.make_array()});
      for (const auto &pattern : {literal + "%", "%" + literal}) {
        ASSERT_NO_FATAL_FAILURE(
            expect_parity(cp::call("match_like", {cp::field_ref("s")},
                                   cp::MatchSubstringOptions(pattern))));
      }
      auto expression =
          cp::call("match_like", {cp::field_ref("s")},
                   cp::MatchSubstringOptions("%" + literal + "%"));
      if (literal.size() <= 64) {
        ASSERT_NO_FATAL_FAILURE(expect_parity(expression));
        continue;
      }
      ASSERT_OK_AND_ASSIGN(auto index, TableIndex::Make(table_));
      ASSERT_OK_AND_ASSIGN(auto bound, expression.Bind(*table_->schema()));
      auto compiled = Predicate::Compile(bound, index);
      EXPECT_TRUE(compiled.status().IsNotImplemented())
          << compiled.status().ToString();
      ASSERT_OK_AND_ASSIGN(
          auto result, cp::ExecuteScalarExpression(
                           bound, cp::ExecBatch({array}, values->length())));
      const auto matches =
          std::static_pointer_cast<arrow::BooleanArray>(result.make_array());
      for (int row = 0; row < 4; ++row) EXPECT_TRUE(matches->Value(row));
      EXPECT_EQ(matches->Value(4), literal == std::string(literal.size(), 'a'));
      EXPECT_FALSE(matches->Value(5));
      EXPECT_TRUE(matches->IsNull(6));
    }
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

TEST_F(StilePredicateTest, SetAndLiteralBudgetsAreCumulative) {
  arrow::Int32Builder builder;
  ASSERT_OK(builder.AppendValues({1, 1, 2}));
  ASSERT_OK_AND_ASSIGN(auto members, builder.Finish());
  auto chunks = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
      members->Slice(0, 2), members->Slice(0, 0), members->Slice(2)});
  const auto in =
      cp::call("is_in", {cp::field_ref("i")}, cp::SetLookupOptions(chunks));
  const auto like = cp::call("match_like", {cp::field_ref("s")},
                             cp::MatchSubstringOptions("a%"));
  ASSERT_OK_AND_ASSIGN(auto index, TableIndex::Make(table_));
  auto check_limit = [&](const cp::Expression &expr,
                         size_t stile::CompileOptions::*budget,
                         size_t required) {
    SCOPED_TRACE(expr.ToString());
    ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*table_->schema()));
    stile::CompileOptions options;
    options.*budget = required;
    ASSERT_OK(Predicate::Compile(bound, index, options).status());
    options.*budget = required - 1;
    EXPECT_TRUE(
        Predicate::Compile(bound, index, options).status().IsCapacityError());
  };
  const auto twice = cp::or_(in, in);
  check_limit(twice, &stile::CompileOptions::max_set_values, 6);
  check_limit(twice, &stile::CompileOptions::max_set_chunks, 6);
  check_limit(cp::or_(like, like), &stile::CompileOptions::max_literal_bytes,
              4);

  // An unsupported sibling cannot hide a hard budget failure.
  const auto unsupported = cp::call("is_finite", {cp::field_ref("d")});
  for (const auto &expr :
       {cp::or_(unsupported, in), cp::or_(in, unsupported)}) {
    ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*table_->schema()));
    stile::CompileOptions options;
    options.max_set_chunks = 2;
    EXPECT_TRUE(
        Predicate::Compile(bound, index, options).status().IsCapacityError());
    options.max_set_chunks = 3;
    EXPECT_TRUE(
        Predicate::Compile(bound, index, options).status().IsNotImplemented());
  }
}

TEST_F(StilePredicateTest, SoftAndHardDepthLimitsAreDistinct) {
  ASSERT_OK_AND_ASSIGN(auto index, TableIndex::Make(table_));
  auto compile_at_depth = [&](int depth, int max_depth) -> arrow::Status {
    auto expression = cp::field_ref("b");
    for (int i = 0; i < depth; ++i)
      expression = cp::not_(std::move(expression));
    ARROW_ASSIGN_OR_RAISE(auto bound, expression.Bind(*table_->schema()));
    stile::CompileOptions options;
    options.max_depth = max_depth;
    return Predicate::Compile(bound, index, options).status();
  };
  ASSERT_OK(compile_at_depth(3, 3));
  EXPECT_TRUE(compile_at_depth(4, 3).IsNotImplemented());
  ASSERT_OK(compile_at_depth(256, 300));
  EXPECT_TRUE(compile_at_depth(257, 300).IsCapacityError());
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
