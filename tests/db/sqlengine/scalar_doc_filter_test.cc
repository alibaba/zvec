// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#ifdef ZVEC_ENABLE_SCALAR_JIT

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include "db/sqlengine/planner/scalar_doc_filter.h"

namespace zvec::sqlengine {
namespace {
namespace cp = arrow::compute;

class ScalarDocFilterTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const auto status = cp::Initialize();
    ASSERT_TRUE(status.ok()) << status.ToString();
  }
};

template <typename T>
T Unwrap(arrow::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return std::move(result).ValueOrDie();
}

void Ok(const arrow::Status &status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}

template <typename Builder, typename T>
std::shared_ptr<arrow::Array> Array(
    const std::vector<std::optional<T>> &values) {
  Builder builder;
  for (const auto &value : values) {
    if (value)
      Ok(builder.Append(*value));
    else
      Ok(builder.AppendNull());
  }
  return Unwrap(builder.Finish());
}

std::shared_ptr<arrow::Array> I32(
    const std::vector<std::optional<int32_t>> &values) {
  return Array<arrow::Int32Builder>(values);
}

std::shared_ptr<arrow::ChunkedArray> Chunks(
    const std::shared_ptr<arrow::Array> &array,
    const std::vector<int64_t> &sizes) {
  arrow::ArrayVector chunks;
  int64_t begin = 0;
  for (auto size : sizes) {
    if (size < 0 || begin + size > array->length())
      throw std::runtime_error("Bad test layout");
    chunks.push_back(array->Slice(begin, size));
    begin += size;
  }
  if (begin != array->length())
    throw std::runtime_error("Incomplete test layout");
  return std::make_shared<arrow::ChunkedArray>(std::move(chunks),
                                               array->type());
}

void Check(const ScalarDocFilter &filter, const std::vector<bool> &expected) {
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (size_t i = 0; i < expected.size(); ++i) {
      const auto row = expected.size() - i - 1;
      auto result = filter.evaluate(row);
      ASSERT_TRUE(result.has_value()) << "row=" << row;
      EXPECT_EQ(*result, expected[row]) << "row=" << row;
    }
  }
  EXPECT_FALSE(filter.evaluate(expected.size()).has_value());
  EXPECT_FALSE(
      filter.evaluate(std::numeric_limits<uint64_t>::max()).has_value());
}

using CheckedFilter =
    std::pair<std::unique_ptr<ScalarDocFilter>, std::vector<bool>>;

CheckedFilter CheckArrow(const cp::Expression &expression,
                         const std::shared_ptr<arrow::Table> &table) {
  SCOPED_TRACE(expression.ToString());
  auto bound = Unwrap(expression.Bind(*table->schema()));
  std::vector<bool> expected;
  arrow::TableBatchReader reader(*table);
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    Ok(reader.ReadNext(&batch));
    if (!batch) break;
    auto result =
        Unwrap(cp::ExecuteScalarExpression(bound, cp::ExecBatch(*batch)));
    if (result.is_scalar()) {
      const auto &scalar = result.scalar_as<arrow::BooleanScalar>();
      expected.insert(expected.end(), batch->num_rows(),
                      scalar.is_valid && scalar.value);
    } else {
      auto values =
          std::static_pointer_cast<arrow::BooleanArray>(result.make_array());
      for (int64_t row = 0; row < values->length(); ++row) {
        expected.push_back(values->IsValid(row) && values->Value(row));
      }
    }
  }
  auto filter = Unwrap(ScalarDocFilter::Make(bound, table));
  Check(*filter, expected);
  return {std::move(filter), std::move(expected)};
}

cp::Expression In(const cp::Expression &field,
                  const std::shared_ptr<arrow::Array> &values) {
  return cp::call(
      "is_in", {field},
      cp::SetLookupOptions(values, cp::SetLookupOptions::INCONCLUSIVE));
}

TEST_F(ScalarDocFilterTest, UniformAndIrregularChunkRoutingWithSlices) {
  const auto schema =
      arrow::schema({arrow::field("route_value", arrow::int32())});
  auto values = I32({-999, -999, -999, -1, 2, 0, 4, {}, 6, 1, 9})->Slice(3, 8);
  auto expression =
      Unwrap(cp::greater(cp::field_ref("route_value"), cp::literal(int32_t{2}))
                 .Bind(*schema));
  for (const std::vector<int64_t> sizes :
       {std::vector<int64_t>{3, 3, 2}, std::vector<int64_t>{2, 1, 4, 1},
        std::vector<int64_t>{1, 2, 5}, std::vector<int64_t>{8},
        std::vector<int64_t>{0, 2, 0, 1, 4, 1, 0}}) {
    auto table = arrow::Table::Make(schema, {Chunks(values, sizes)});
    auto filter = Unwrap(ScalarDocFilter::Make(expression, table));
    const auto nonempty = std::count_if(sizes.begin(), sizes.end(),
                                        [](int64_t n) { return n != 0; });
    EXPECT_EQ(filter->num_batches(), static_cast<size_t>(nonempty));
    table.reset();  // Bound record batches keep the backing arrays alive.
    Check(*filter, {false, false, false, true, false, true, false, true});
  }
}

TEST_F(ScalarDocFilterTest,
       DifferentlyChunkedColumnsAreAlignedWithoutChangingRows) {
  auto schema = arrow::schema({arrow::field("route_x", arrow::int32()),
                               arrow::field("route_y", arrow::int32()),
                               arrow::field("route_flag", arrow::boolean())});
  auto x = I32({-99, -99, -99, -1, 0, 1, 2, 3, {}, 5, 8, 9})->Slice(3, 9);
  auto y =
      I32({-99, -99, -99, -99, -99, 0, -1, 0, 3, 1, 2, {}, 7, 9})->Slice(5, 9);
  auto flags = Array<arrow::BooleanBuilder, bool>({false,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   true,
                                                   true,
                                                   {},
                                                   true,
                                                   false,
                                                   true,
                                                   true,
                                                   true,
                                                   true})
                   ->Slice(7, 9);
  auto table = arrow::Table::Make(
      schema,
      {Chunks(x, {3, 4, 2}), Chunks(y, {2, 3, 4}), Chunks(flags, {4, 5})});
  auto expression = Unwrap(
      cp::and_(cp::greater(cp::field_ref("route_x"), cp::field_ref("route_y")),
               cp::field_ref("route_flag"))
          .Bind(*schema));
  auto filter = Unwrap(ScalarDocFilter::Make(expression, table));
  // Union of chunk ends: 2,3,4,5,7,9. Each batch has aligned logical rows
  // even though every column has different physical offsets and chunk ends.
  EXPECT_EQ(filter->num_batches(), 6u);
  Check(*filter, {false, true, false, false, false, false, false, true, false});
  table.reset();
  x.reset();
  y.reset();
  flags.reset();
  Check(*filter, {false, true, false, false, false, false, false, true, false});
}

TEST_F(ScalarDocFilterTest, CachedProgramKeepsParametersAndTablesIndependent) {
  auto schema =
      arrow::schema({arrow::field("cached_route_value", arrow::int32())});
  auto make_expression = [&](int32_t threshold) {
    return Unwrap(
        cp::greater(cp::field_ref("cached_route_value"), cp::literal(threshold))
            .Bind(*schema));
  };
  auto first_table =
      arrow::Table::Make(schema, {Chunks(I32({1, 5, 9, {}}), {2, 2})});
  auto first = Unwrap(ScalarDocFilter::Make(make_expression(0), first_table));
  const auto after_first = ScalarDocFilter::CacheStats();
  auto second = Unwrap(ScalarDocFilter::Make(make_expression(8), first_table));
  auto other_table =
      arrow::Table::Make(schema, {Chunks(I32({10, -1, 3, 4}), {1, 3})});
  auto third = Unwrap(ScalarDocFilter::Make(make_expression(3), other_table));
  const auto after_reuse = ScalarDocFilter::CacheStats();
  EXPECT_EQ(after_reuse.compilations, after_first.compilations);
  EXPECT_GE(after_reuse.hits, after_first.hits + 2);
  first_table.reset();
  other_table.reset();
  Check(*first, {true, true, true, false});
  Check(*second, {false, false, true, false});
  Check(*third, {true, false, false, true});
}

TEST_F(ScalarDocFilterTest, EmptyMalformedAndUnsupportedSnapshotsReturnErrors) {
  auto schema = arrow::schema({arrow::field("route_bad", arrow::int32())});
  auto expression =
      Unwrap(cp::greater(cp::field_ref("route_bad"), cp::literal(int32_t{0}))
                 .Bind(*schema));
  EXPECT_FALSE(ScalarDocFilter::Make(expression, nullptr).ok());
  auto empty = arrow::Table::Make(schema, {Chunks(I32({}), {0})});
  EXPECT_FALSE(ScalarDocFilter::Make(expression, empty).ok());
  auto incomplete = arrow::Table::Make(schema, {Chunks(I32({1, 2}), {2})}, 3);
  EXPECT_FALSE(ScalarDocFilter::Make(expression, incomplete).ok());
  auto wrong_type = arrow::Table::Make(
      schema, {std::make_shared<arrow::ChunkedArray>(
                  Array<arrow::Int64Builder, int64_t>({1, 2}))});
  EXPECT_FALSE(ScalarDocFilter::Make(expression, wrong_type).ok());
  // Public Make must still validate every chunk, even though the internal
  // trusted-snapshot path binds/validates only the first RecordBatch. Matching
  // table types and lengths are insufficient for this malformed later chunk.
  auto short_values =
      arrow::Buffer::FromString(std::string(sizeof(int32_t), '\0'));
  auto malformed_chunk = arrow::MakeArray(
      arrow::ArrayData::Make(arrow::int32(), 2, {nullptr, short_values}, 0));
  auto malformed_later = arrow::Table::Make(
      schema, {std::make_shared<arrow::ChunkedArray>(
                  arrow::ArrayVector{I32({1, 2}), malformed_chunk})});
  ASSERT_FALSE(malformed_later->Validate().ok());
  auto malformed_result = ScalarDocFilter::Make(expression, malformed_later);
  ASSERT_FALSE(malformed_result.ok());
  EXPECT_TRUE(malformed_result.status().IsInvalid());
  auto valid = arrow::Table::Make(schema, {Chunks(I32({1, 2}), {2})});
  auto unsupported =
      Unwrap(cp::greater(cp::call("add", {cp::field_ref("route_bad"),
                                          cp::literal(int32_t{1})}),
                         cp::literal(int32_t{0}))
                 .Bind(*schema));
  auto rejected = ScalarDocFilter::Make(unsupported, valid);
  ASSERT_FALSE(rejected.ok());
  EXPECT_TRUE(rejected.status().IsNotImplemented());
}

TEST_F(ScalarDocFilterTest,
       IndependentChunkBuffersAndValidityRemainAliveAfterTableRelease) {
  std::weak_ptr<arrow::Buffer> later_values;
  std::weak_ptr<arrow::Buffer> later_validity;
  std::unique_ptr<ScalarDocFilter> filter;
  {
    auto schema = arrow::schema(
        {arrow::field("separate_chunk_lifetime", arrow::int32())});
    // Every chunk owns different buffers. Retaining only the prototype's
    // first RecordBatch cannot keep the later chunks alive.
    auto first = I32({-99, 1, 2})->Slice(1, 2);
    auto second = I32({-99, -99, 8, {}, 0})->Slice(2, 3);
    auto third = I32({-99, -99, -99, -1, 9, {}})->Slice(3, 3);
    // Force the first chunk's valid-all representation to have no bitmap.
    // Subsequent chunks do require their own validity buffers.
    auto first_data = first->data()->Copy();
    first_data->buffers[0].reset();
    first_data->null_count = 0;
    first = arrow::MakeArray(first_data);
    later_values = third->data()->buffers[1];
    later_validity = third->data()->buffers[0];
    ASSERT_FALSE(later_values.expired());
    ASSERT_FALSE(later_validity.expired());
    auto table = arrow::Table::Make(
        schema, {std::make_shared<arrow::ChunkedArray>(
                    arrow::ArrayVector{first, second, third})});
    auto expression =
        Unwrap(cp::greater(cp::field_ref("separate_chunk_lifetime"),
                           cp::literal(int32_t{1}))
                   .Bind(*schema));
    filter = Unwrap(ScalarDocFilter::Make(expression, table));
  }
  ASSERT_FALSE(later_values.expired());
  ASSERT_FALSE(later_validity.expired());
  Check(*filter, {false, true, true, false, false, false, true, false});
  filter.reset();
  EXPECT_TRUE(later_values.expired());
  EXPECT_TRUE(later_validity.expired());
}

TEST_F(ScalarDocFilterTest, ReferenceOrderDiffersFromSchemaWithUnusedColumns) {
  auto schema = arrow::schema({arrow::field("unused_int64", arrow::int64()),
                               arrow::field("ordered_x", arrow::int32()),
                               arrow::field("unused_bool", arrow::boolean()),
                               arrow::field("ordered_y", arrow::int32()),
                               arrow::field("ordered_flag", arrow::boolean())});
  auto unused = Array<arrow::Int64Builder, int64_t>(
      {9000, {}, -9000, 7000, {}, 5000, 8000, 10000});
  auto x = I32({-99, -99, -99, 0, 1, 2, {}, 4, 5, 6, 7})->Slice(3, 8);
  auto y =
      I32({-99, -99, -99, -99, -99, 1, 0, 3, 10, {}, 8, 6, 9})->Slice(5, 8);
  auto unused_bool = Array<arrow::BooleanBuilder, bool>(
      {false, {}, false, true, {}, false, false, true});
  auto flags = Array<arrow::BooleanBuilder, bool>({false,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   true,
                                                   true,
                                                   false,
                                                   true,
                                                   true,
                                                   true,
                                                   {},
                                                   true})
                   ->Slice(7, 8);
  auto table =
      arrow::Table::Make(schema, {Chunks(unused, {1, 7}), Chunks(x, {3, 5}),
                                  Chunks(unused_bool, {2, 6}),
                                  Chunks(y, {5, 3}), Chunks(flags, {4, 4})});
  // References arrive as schema indices 4,3,1, and fields 0,2 never appear.
  auto expression = Unwrap(cp::and_(cp::field_ref("ordered_flag"),
                                    cp::greater(cp::field_ref("ordered_y"),
                                                cp::field_ref("ordered_x")))
                               .Bind(*schema));
  auto filter = Unwrap(ScalarDocFilter::Make(expression, table));
  Check(*filter, {true, false, false, false, false, true, false, true});
}

TEST_F(ScalarDocFilterTest, LiteralOnlyPredicateHasNoReferencedDataBuffers) {
  auto schema =
      arrow::schema({arrow::field("unused_literal_input", arrow::int32())});
  auto table = arrow::Table::Make(
      schema, {Chunks(I32({1, {}, 2, 3, 4, 5, {}}), {1, 4, 2})});
  auto yes = Unwrap(
      ScalarDocFilter::Make(Unwrap(cp::literal(true).Bind(*schema)), table));
  auto no = Unwrap(
      ScalarDocFilter::Make(Unwrap(cp::literal(false).Bind(*schema)), table));
  auto unknown = Unwrap(ScalarDocFilter::Make(
      Unwrap(
          cp::literal(arrow::MakeNullScalar(arrow::boolean())).Bind(*schema)),
      table));
  table.reset();
  Check(*yes, std::vector<bool>(7, true));
  Check(*no, std::vector<bool>(7, false));
  Check(*unknown, std::vector<bool>(7, false));
}

TEST_F(ScalarDocFilterTest, TypedNullAndBooleanParametersAreBroadcastPerChunk) {
  auto schema =
      arrow::schema({arrow::field("parameter_chunk_value", arrow::int32())});
  auto table = arrow::Table::Make(
      schema, {Chunks(I32({0, 8, {}, 4, -1, 9}), {2, 1, 3})});
  auto make = [&](std::optional<int32_t> threshold, std::optional<bool> gate) {
    auto number = threshold
                      ? cp::literal(*threshold)
                      : cp::literal(arrow::MakeNullScalar(arrow::int32()));
    auto flag = gate ? cp::literal(*gate)
                     : cp::literal(arrow::MakeNullScalar(arrow::boolean()));
    auto expression = Unwrap(
        cp::or_(cp::greater(cp::field_ref("parameter_chunk_value"), number),
                flag)
            .Bind(*schema));
    return Unwrap(ScalarDocFilter::Make(expression, table));
  };
  auto false_gate = make(3, false);
  const auto after_first = ScalarDocFilter::CacheStats();
  auto true_with_null_threshold = make({}, true);
  auto false_with_null_threshold = make({}, false);
  auto null_gate = make(3, {});
  const auto after_reuse = ScalarDocFilter::CacheStats();
  EXPECT_EQ(after_reuse.compilations, after_first.compilations);
  EXPECT_GE(after_reuse.hits, after_first.hits + 3);
  table.reset();
  Check(*false_gate, {false, true, false, true, false, true});
  Check(*true_with_null_threshold, std::vector<bool>(6, true));
  Check(*false_with_null_threshold, std::vector<bool>(6, false));
  Check(*null_gate, {false, true, false, true, false, true});
}

TEST_F(ScalarDocFilterTest,
       ConcurrentCachedCodeWithIndependentParametersAndChunkLayouts) {
  constexpr int kQueries = 8;
  auto schema =
      arrow::schema({arrow::field("concurrent_chunk_value", arrow::int32())});
  std::vector<std::unique_ptr<ScalarDocFilter>> filters;
  std::vector<std::vector<bool>> expected;
  uint64_t first_compilations = 0;
  for (int query = 0; query < kQueries; ++query) {
    const int rows = 11 + query;
    const int32_t threshold = query + 2;
    std::vector<std::optional<int32_t>> values;
    std::vector<bool> truth;
    for (int row = 0; row < rows; ++row) {
      auto value = row % 4 == 1 ? std::optional<int32_t>{}
                                : std::optional<int32_t>{row * 3 - query * 2};
      values.push_back(value);
      truth.push_back(value && *value > threshold);
    }
    std::vector<int64_t> sizes;
    if (query % 3 == 0)
      sizes = {1, rows - 1};
    else if (query % 3 == 1)
      sizes = {3, 3, rows - 6};
    else
      sizes = {rows};
    auto table = arrow::Table::Make(schema, {Chunks(I32(values), sizes)});
    auto expression =
        Unwrap(cp::greater(cp::field_ref("concurrent_chunk_value"),
                           cp::literal(threshold))
                   .Bind(*schema));
    filters.push_back(Unwrap(ScalarDocFilter::Make(expression, table)));
    expected.push_back(std::move(truth));
    if (query == 0)
      first_compilations = ScalarDocFilter::CacheStats().compilations;
  }
  EXPECT_EQ(ScalarDocFilter::CacheStats().compilations, first_compilations);
  std::atomic<uint64_t> mismatches{0};
  std::vector<std::thread> threads;
  for (int thread = 0; thread < kQueries; ++thread) {
    threads.emplace_back([&, thread] {
      for (int repeat = 0; repeat < 1000; ++repeat) {
        const auto query = (thread + repeat) % kQueries;
        for (size_t row = 0; row < expected[query].size(); ++row) {
          auto actual = filters[query]->evaluate(row);
          if (!actual || *actual != expected[query][row]) ++mismatches;
        }
      }
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(mismatches.load(), 0u);
}

TEST_F(ScalarDocFilterTest, StringsWithIndependentChunksAndSlicedBuffers) {
  auto schema = arrow::schema({arrow::field("text_x", arrow::utf8()),
                               arrow::field("text_y", arrow::utf8())});
  const std::string nul("a\0b", 3);
  auto x = Array<arrow::StringBuilder, std::string>(
               {"pad", "pad", "", nul, "中文", {}, "x", "é", "last"})
               ->Slice(2);
  auto y = Array<arrow::StringBuilder, std::string>(
               {"pad", "", "a", "中", "x", {}, "é", "last"})
               ->Slice(1);
  auto table = arrow::Table::Make(
      schema, {Chunks(x, {0, 2, 1, 4}), Chunks(y, {1, 3, 0, 3})});
  auto expression =
      cp::or_(cp::less(cp::field_ref("text_x"), cp::field_ref("text_y")),
              cp::equal(cp::field_ref("text_x"), cp::literal(nul)));
  auto checked = CheckArrow(expression, table);
  table.reset();
  x.reset();
  y.reset();
  Check(*checked.first, checked.second);
}

TEST_F(ScalarDocFilterTest, LikeMatchesArrowEscapesUnicodeAndNullSemantics) {
  auto schema = arrow::schema({arrow::field("like_value", arrow::utf8())});
  auto values =
      Array<arrow::StringBuilder, std::string>({"padding",
                                                "",
                                                "a",
                                                "abc",
                                                "a_b",
                                                "a%b",
                                                "a\\b",
                                                "a\nb",
                                                "\n",
                                                "é",
                                                "中文",
                                                "x\ny",
                                                "trailing\\",
                                                "a.b",
                                                std::string("a\0b", 3),
                                                {}})
          ->Slice(1);
  auto table = arrow::Table::Make(schema, {Chunks(values, {2, 5, 1, 7})});
  const std::vector<std::string> patterns = {"",
                                             "%",
                                             "_",
                                             "__",
                                             "a%",
                                             "%b",
                                             "a_b",
                                             "a\\_b",
                                             "a\\%b",
                                             "a\\\\b",
                                             "a\\b",
                                             "%a\\b%",
                                             "a\\b%",
                                             "%a\\b",
                                             "trailing\\",
                                             "%\n%",
                                             "中_",
                                             "a.b",
                                             std::string("a\0b", 3)};
  uint64_t compilations = 0;
  for (size_t i = 0; i < patterns.size(); ++i) {
    auto expression = cp::call("match_like", {cp::field_ref("like_value")},
                               cp::MatchSubstringOptions(patterns[i]));
    CheckArrow(expression, table);
    CheckArrow(cp::not_(expression), table);
    // NULL must remain unknown inside a boolean expression, not become false
    // before NOT or Kleene OR applies.
    CheckArrow(cp::not_(cp::or_(expression, cp::literal(false))), table);
    if (i == 0) compilations = ScalarDocFilter::CacheStats().compilations;
    EXPECT_EQ(ScalarDocFilter::CacheStats().compilations, compilations);
  }
}

template <typename Builder, typename T>
void CheckNumericIn(const std::shared_ptr<arrow::DataType> &type) {
  SCOPED_TRACE(type->ToString());
  const T low = std::numeric_limits<T>::lowest();
  const T high = std::numeric_limits<T>::max();
  std::vector<std::optional<T>> values = {low, T{0}, T{1}, high, {}};
  std::vector<std::optional<T>> members = {T{0}, high, high};
  if constexpr (std::is_floating_point<T>::value) {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    using UInt = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    UInt payload;
    std::memcpy(&payload, &nan, sizeof(nan));
    payload ^= 1;
    T other_nan;
    std::memcpy(&other_nan, &payload, sizeof(other_nan));
    values.insert(values.end(), {T{-0.0f}, nan, other_nan,
                                 std::numeric_limits<T>::infinity()});
    members.push_back(nan);
  }
  auto schema = arrow::schema({arrow::field("in_" + type->ToString(), type)});
  auto field = cp::field_ref(schema->field(0)->name());
  auto array = Array<Builder>(values);
  auto table =
      arrow::Table::Make(schema, {Chunks(array, {1, array->length() - 1})});
  uint64_t compilations = 0;
  for (int variant = 0; variant < 4; ++variant) {
    if (variant == 1) members.push_back({});
    if (variant == 2) members = {T{1}};
    if (variant == 3) members.clear();
    auto expression = In(field, Array<Builder>(members));
    CheckArrow(expression, table);
    CheckArrow(cp::not_(expression), table);
    CheckArrow(cp::is_null(expression), table);
    if constexpr (std::is_same<T, int32_t>::value) {
      CheckArrow(cp::is_valid(expression), table);
    }
    if (variant == 0) compilations = ScalarDocFilter::CacheStats().compilations;
    EXPECT_EQ(ScalarDocFilter::CacheStats().compilations, compilations);
  }
}

TEST_F(ScalarDocFilterTest, InAndNotInMatchArrowForEveryNumericType) {
  CheckNumericIn<arrow::Int8Builder, int8_t>(arrow::int8());
  CheckNumericIn<arrow::UInt8Builder, uint8_t>(arrow::uint8());
  CheckNumericIn<arrow::Int16Builder, int16_t>(arrow::int16());
  CheckNumericIn<arrow::UInt16Builder, uint16_t>(arrow::uint16());
  CheckNumericIn<arrow::Int32Builder, int32_t>(arrow::int32());
  CheckNumericIn<arrow::UInt32Builder, uint32_t>(arrow::uint32());
  CheckNumericIn<arrow::Int64Builder, int64_t>(arrow::int64());
  CheckNumericIn<arrow::UInt64Builder, uint64_t>(arrow::uint64());
  CheckNumericIn<arrow::FloatBuilder, float>(arrow::float32());
  CheckNumericIn<arrow::DoubleBuilder, double>(arrow::float64());
}

TEST_F(ScalarDocFilterTest, StringAndLargeInSetsAreBoundQueryParameters) {
  const std::string nul("a\0b", 3);
  auto schema = arrow::schema({arrow::field("string_in", arrow::utf8())});
  auto table = arrow::Table::Make(
      schema, {Chunks(Array<arrow::StringBuilder, std::string>(
                          {"", "a", nul, "中文", "é", {}, "19999"}),
                      {2, 1, 4})});
  std::vector<std::optional<std::string>> members = {"", nul, "中文", "中文"};
  uint64_t compilations = 0;
  for (int variant = 0; variant < 4; ++variant) {
    if (variant == 1) members.push_back({});
    if (variant == 2) members = {"a", "é"};
    if (variant == 3) {
      members.clear();
      for (int value = 0; value < 4096; ++value)
        members.push_back(std::to_string(value));
    }
    auto expression =
        In(cp::field_ref("string_in"), Array<arrow::StringBuilder>(members));
    CheckArrow(expression, table);
    CheckArrow(cp::not_(expression), table);
    CheckArrow(cp::is_null(expression), table);
    CheckArrow(cp::is_valid(expression), table);
    if (variant == 0) compilations = ScalarDocFilter::CacheStats().compilations;
    EXPECT_EQ(ScalarDocFilter::CacheStats().compilations, compilations);
  }
}

TEST_F(ScalarDocFilterTest, OversizedHelpersRemainValidArrowExpressions) {
  auto schema = arrow::schema({arrow::field("text_limit", arrow::utf8())});
  auto values = Array<arrow::StringBuilder, std::string>({"a", "b"});
  auto table = arrow::Table::Make(schema, {Chunks(values, {2})});
  auto field = cp::field_ref("text_limit");
  std::vector<std::optional<std::string>> members(4097, "a");
  for (auto expression :
       {In(field, Array<arrow::StringBuilder>(members)),
        cp::call("match_like", {field},
                 cp::MatchSubstringOptions(std::string(1024 * 1024, 'a')))}) {
    auto bound = Unwrap(expression.Bind(*schema));
    auto filter = ScalarDocFilter::Make(bound, table);
    EXPECT_EQ(scalar_jit::FailureReason(filter.status()), "expression_limit");
    arrow::TableBatchReader reader(*table);
    auto batch = Unwrap(reader.Next());
    EXPECT_TRUE(cp::ExecuteScalarExpression(bound, cp::ExecBatch(*batch)).ok());
  }
}

TEST_F(ScalarDocFilterTest,
       CachedStringsPatternsAndSetsHaveIndependentBindings) {
  auto schema = arrow::schema({arrow::field("bound_text", arrow::utf8())});
  std::vector<CheckedFilter> queries;
  uint64_t compilations = 0;
  for (int query = 0; query < 8; ++query) {
    auto values = Array<arrow::StringBuilder, std::string>(
        {"", "a", "ab", "b", "bc", "中文", std::string("a\0b", 3), {}});
    auto table = arrow::Table::Make(
        schema, {Chunks(values, query % 2 ? std::vector<int64_t>{2, 1, 5}
                                          : std::vector<int64_t>{1, 4, 3})});
    auto field = cp::field_ref("bound_text");
    auto members = Array<arrow::StringBuilder, std::string>(
        query % 2
            ? std::vector<std::optional<std::string>>{"ab", "b"}
            : std::vector<std::optional<std::string>>{"a", "bc", "中文", {}});
    auto pattern = cp::call("match_like", {field},
                            cp::MatchSubstringOptions(query % 3 ? "%b%" : "_"));
    auto expression =
        cp::or_(cp::less(field, cp::literal(std::string(query, 'a'))),
                cp::and_(pattern, In(field, members)));
    queries.push_back(CheckArrow(expression, table));
    if (query == 0) compilations = ScalarDocFilter::CacheStats().compilations;
    EXPECT_EQ(ScalarDocFilter::CacheStats().compilations, compilations);
  }
  // Query inputs, patterns, and sets have left scope; each binding still owns
  // its parameters while all queries execute the same cached machine code.
  std::atomic<uint64_t> mismatches{0};
  std::vector<std::thread> threads;
  for (size_t thread = 0; thread < queries.size(); ++thread) {
    threads.emplace_back([&, thread] {
      for (size_t repeat = 0; repeat < 1000; ++repeat) {
        const auto &query = queries[(thread + repeat) % queries.size()];
        for (size_t row = 0; row < query.second.size(); ++row) {
          auto actual = query.first->evaluate(row);
          if (!actual || *actual != query.second[row]) ++mismatches;
        }
      }
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(mismatches.load(), 0u);
}

}  // namespace
}  // namespace zvec::sqlengine

#else

TEST(ScalarDocFilterTest, DisabledAtBuildTime) {
  GTEST_SKIP() << "ZVEC_ENABLE_SCALAR_JIT is disabled";
}

#endif  // ZVEC_ENABLE_SCALAR_JIT
