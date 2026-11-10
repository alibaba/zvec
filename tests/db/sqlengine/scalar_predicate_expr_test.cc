// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#ifdef ZVEC_ENABLE_SCALAR_JIT

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gandiva/projector.h>
#include <gandiva/tree_expr_builder.h>
#include "db/sqlengine/planner/ops/contain_op.h"
#include "db/sqlengine/planner/scalar_predicate_expr.h"
#include "db/sqlengine/scalar_jit/scalar_predicate.h"

namespace zvec::sqlengine {
namespace {
namespace cp = arrow::compute;

class ScalarPredicateExprTest : public testing::Test {
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

std::shared_ptr<arrow::Array> Bool(
    const std::vector<std::optional<bool>> &values) {
  return Array<arrow::BooleanBuilder>(values);
}

template <typename T>
bool Compare(const std::string &op, T x, T y) {
  if (op == "equal") return x == y;
  if (op == "not_equal") return x != y;
  if (op == "less") return x < y;
  if (op == "less_equal") return x <= y;
  if (op == "greater") return x > y;
  if (op == "greater_equal") return x >= y;
  throw std::runtime_error("Unknown test comparison");
}

void Verify(const cp::Expression &expression,
            const std::shared_ptr<arrow::RecordBatch> &batch,
            const std::vector<bool> &expected) {
  SCOPED_TRACE(expression.ToString());
  ASSERT_EQ(static_cast<int64_t>(expected.size()), batch->num_rows());
  auto bound = Unwrap(expression.Bind(*batch->schema()));
  auto condition = Unwrap(MakeScalarPredicateCondition(bound, batch->schema()));
  auto compiled =
      Unwrap(scalar_jit::ScalarPredicate::Make(batch->schema(), condition));
  auto jit = Unwrap(compiled->bind(batch));
  // Exercise the shared code generator's ordinary batch path as well.
  std::shared_ptr<gandiva::Projector> projector;
  Ok(gandiva::Projector::Make(
      batch->schema(),
      {gandiva::TreeExprBuilder::MakeExpression(
          condition->root(), arrow::field("result", arrow::boolean()))},
      &projector));
  arrow::ArrayVector projected;
  Ok(projector->Evaluate(*batch, arrow::default_memory_pool(), &projected));
  ASSERT_EQ(projected.size(), 1);
  const auto batch_result =
      std::static_pointer_cast<arrow::BooleanArray>(projected[0]);
  ASSERT_EQ(batch_result->length(), batch->num_rows());
  auto oracle =
      Unwrap(cp::ExecuteScalarExpression(bound, cp::ExecBatch(*batch)));
  std::shared_ptr<arrow::BooleanArray> array;
  if (oracle.is_array()) {
    array = std::static_pointer_cast<arrow::BooleanArray>(oracle.make_array());
  } else {
    ASSERT_TRUE(oracle.is_scalar());
  }
  // Repeat in reverse order to exercise resetting intermediate validity.
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (int64_t row = batch->num_rows() - 1; row >= 0; --row) {
      const bool arrow_valid =
          array ? array->IsValid(row)
                : oracle.scalar_as<arrow::BooleanScalar>().is_valid;
      const bool arrow_value =
          arrow_valid &&
          (array ? array->Value(row)
                 : oracle.scalar_as<arrow::BooleanScalar>().value);
      // Batch projection preserves NULL, while filtering maps NULL to false.
      EXPECT_EQ(batch_result->IsValid(row), arrow_valid) << "row=" << row;
      EXPECT_EQ(batch_result->IsValid(row) && batch_result->Value(row),
                arrow_value)
          << "row=" << row;
      EXPECT_EQ(arrow_value, expected[row]) << "row=" << row;
      EXPECT_EQ(jit->evaluate(row), arrow_value) << "row=" << row;
    }
  }
}

void Reject(const cp::Expression &expression,
            const std::shared_ptr<arrow::Schema> &schema) {
  SCOPED_TRACE(expression.ToString());
  auto bound = Unwrap(expression.Bind(*schema));
  auto converted = MakeScalarPredicateCondition(bound, schema);
  ASSERT_FALSE(converted.ok());
  EXPECT_TRUE(converted.status().IsNotImplemented())
      << converted.status().ToString();
}

TEST_F(ScalarPredicateExprTest, NumericComparisonMappingsAndNulls) {
  auto x = arrow::field("x", arrow::int32());
  const std::vector<std::optional<int32_t>> values = {
      std::numeric_limits<int32_t>::min(), -1, 0, 1, 2, {},
      std::numeric_limits<int32_t>::max()};
  auto batch = arrow::RecordBatch::Make(arrow::schema({x}), values.size(),
                                        {I32(values)});
  for (const std::string op : {"equal", "not_equal", "less", "less_equal",
                               "greater", "greater_equal"}) {
    std::vector<bool> expected;
    for (auto value : values)
      expected.push_back(value && Compare(op, *value, int32_t{1}));
    Verify(cp::call(op, {cp::field_ref("x"), cp::literal(int32_t{1})}), batch,
           expected);
  }
  Verify(cp::greater(cp::field_ref("x"),
                     cp::literal(arrow::MakeNullScalar(arrow::int32()))),
         batch, std::vector<bool>(values.size(), false));
}

template <typename Builder, typename T>
void VerifyNumericType(const std::shared_ptr<arrow::DataType> &type) {
  const T low = std::numeric_limits<T>::lowest();
  const T high = std::numeric_limits<T>::max();
  const T middle = std::is_unsigned<T>::value ? high / 2 : T{0};
  auto batch =
      arrow::RecordBatch::Make(arrow::schema({arrow::field("x", type)}), 4,
                               {Array<Builder, T>({low, middle, high, {}})});
  Verify(cp::greater(cp::field_ref("x"), cp::literal(middle)), batch,
         {low > middle, false, high > middle, false});
}

TEST_F(ScalarPredicateExprTest, FixedWidthTypesAndExtremes) {
  VerifyNumericType<arrow::Int8Builder, int8_t>(arrow::int8());
  VerifyNumericType<arrow::UInt8Builder, uint8_t>(arrow::uint8());
  VerifyNumericType<arrow::Int16Builder, int16_t>(arrow::int16());
  VerifyNumericType<arrow::UInt16Builder, uint16_t>(arrow::uint16());
  VerifyNumericType<arrow::UInt32Builder, uint32_t>(arrow::uint32());
  VerifyNumericType<arrow::Int64Builder, int64_t>(arrow::int64());
  VerifyNumericType<arrow::UInt64Builder, uint64_t>(arrow::uint64());
  VerifyNumericType<arrow::FloatBuilder, float>(arrow::float32());
  VerifyNumericType<arrow::DoubleBuilder, double>(arrow::float64());
}

TEST_F(ScalarPredicateExprTest, FloatingNaNInfinityAndNullComparisonSemantics) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::optional<double>> x = {-inf, -1., -0., 0., 1.,
                                          inf,  nan, {},  2.};
  std::vector<std::optional<double>> y = {0.,  nan, 0., -0., inf,
                                          inf, nan, 1., {}};
  auto schema = arrow::schema({arrow::field("x", arrow::float64()),
                               arrow::field("y", arrow::float64())});
  auto batch = arrow::RecordBatch::Make(
      schema, x.size(),
      {Array<arrow::DoubleBuilder>(x), Array<arrow::DoubleBuilder>(y)});
  for (const std::string op : {"equal", "not_equal", "less", "less_equal",
                               "greater", "greater_equal"}) {
    std::vector<bool> expected;
    for (size_t i = 0; i < x.size(); ++i)
      expected.push_back(x[i] && y[i] && Compare(op, *x[i], *y[i]));
    Verify(cp::call(op, {cp::field_ref("x"), cp::field_ref("y")}), batch,
           expected);
  }
  Verify(cp::is_null(cp::field_ref("x")), batch,
         {false, false, false, false, false, false, false, true, false});
  Verify(cp::is_valid(cp::field_ref("x")), batch,
         {true, true, true, true, true, true, true, false, true});
  Reject(cp::is_null(cp::field_ref("x"), true), schema);
}

std::optional<bool> And(std::optional<bool> x, std::optional<bool> y) {
  if ((x && !*x) || (y && !*y)) return false;
  if (!x || !y) return {};
  return true;
}

std::optional<bool> Or(std::optional<bool> x, std::optional<bool> y) {
  if ((x && *x) || (y && *y)) return true;
  if (!x || !y) return {};
  return false;
}

TEST_F(ScalarPredicateExprTest, BooleanThreeValuedLogicAndSlicedBitmaps) {
  std::vector<std::optional<bool>> x(3, false), y(5, false);
  std::vector<bool> ands, ors, invert_and, combined;
  for (std::optional<bool> a :
       {std::optional<bool>(false), std::optional<bool>(true),
        std::optional<bool>{}}) {
    for (std::optional<bool> b :
         {std::optional<bool>(false), std::optional<bool>(true),
          std::optional<bool>{}}) {
      x.push_back(a);
      y.push_back(b);
      auto both = And(a, b);
      auto either = Or(a, b);
      auto not_both = both ? std::optional<bool>(!*both) : std::nullopt;
      ands.push_back(both.value_or(false));
      ors.push_back(either.value_or(false));
      invert_and.push_back(not_both.value_or(false));
      combined.push_back(And(either, not_both).value_or(false));
    }
  }
  auto schema = arrow::schema({arrow::field("x", arrow::boolean()),
                               arrow::field("y", arrow::boolean())});
  auto batch = arrow::RecordBatch::Make(
      schema, 9, {Bool(x)->Slice(3, 9), Bool(y)->Slice(5, 9)});
  auto a = cp::field_ref("x"), b = cp::field_ref("y");
  Verify(cp::and_(a, b), batch, ands);
  Verify(cp::or_(a, b), batch, ors);
  Verify(cp::not_(cp::and_(a, b)), batch, invert_and);
  Verify(cp::and_(cp::or_(a, b), cp::not_(cp::and_(a, b))), batch, combined);
}

TEST_F(ScalarPredicateExprTest, BoundFieldPathsAndDistinctArrayOffsets) {
  auto schema = arrow::schema({arrow::field("unused", arrow::int32()),
                               arrow::field("a.b[0]|int32", arrow::int32()),
                               arrow::field("other", arrow::int32())});
  auto batch = arrow::RecordBatch::Make(
      schema, 4,
      {I32({123, 123, 123, 123}),
       I32({-99, -99, -99, 1, {}, 3, 9})->Slice(3, 4),
       I32({-99, -99, -99, -99, -99, 0, 5, 3, 10})->Slice(5, 4)});
  auto expression =
      cp::greater(cp::field_ref(arrow::FieldRef(1)), cp::field_ref("other"));
  Verify(expression, batch, {true, false, false, false});
  Verify(cp::greater(cp::field_ref("a.b[0]|int32"), cp::literal(int32_t{2})),
         batch, {false, false, true, true});
  auto bound =
      Unwrap(cp::greater(cp::field_ref("other"), cp::literal(int32_t{0}))
                 .Bind(*schema));
  auto wrong_schema =
      arrow::schema({schema->field(0), schema->field(2), schema->field(1)});
  EXPECT_FALSE(MakeScalarPredicateCondition(bound, wrong_schema).ok());
}

TEST_F(ScalarPredicateExprTest, BooleanEqualityPreservesNulls) {
  auto schema = arrow::schema({arrow::field("x", arrow::boolean()),
                               arrow::field("y", arrow::boolean())});
  auto batch =
      arrow::RecordBatch::Make(schema, 6,
                               {Bool({false, true, {}, false, true, {}}),
                                Bool({false, false, true, {}, true, {}})});
  Verify(cp::equal(cp::field_ref("x"), cp::field_ref("y")), batch,
         {true, false, false, false, true, false});
  Verify(cp::not_equal(cp::field_ref("x"), cp::field_ref("y")), batch,
         {false, true, false, false, false, false});
  Verify(cp::equal(cp::field_ref("x"), cp::literal(true)), batch,
         {false, true, false, false, true, false});
  Verify(cp::not_equal(cp::field_ref("x"), cp::literal(false)), batch,
         {false, true, false, false, true, false});
}

TEST_F(ScalarPredicateExprTest, StringComparisonsUseLengthsAndSliceOffsets) {
  const std::string nul("a\0b", 3);
  std::vector<std::optional<std::string>> x = {"",     "a", nul, "a",   "\n",
                                               "中文", "é", {},  "same"};
  std::vector<std::optional<std::string>> y = {"a",  "",  "a", nul, "\n",
                                               "中", "é", "x", {}};
  auto schema = arrow::schema(
      {arrow::field("x", arrow::utf8()), arrow::field("y", arrow::utf8())});
  auto padded_x = x, padded_y = y;
  padded_x.insert(padded_x.begin(), 3, "padding");
  padded_y.insert(padded_y.begin(), 5, "other padding");
  auto batch = arrow::RecordBatch::Make(
      schema, x.size(),
      {Array<arrow::StringBuilder>(padded_x)->Slice(3),
       Array<arrow::StringBuilder>(padded_y)->Slice(5)});
  for (const std::string op : {"equal", "not_equal", "less", "less_equal",
                               "greater", "greater_equal"}) {
    std::vector<bool> fields, literal;
    for (size_t row = 0; row < x.size(); ++row) {
      fields.push_back(x[row] && y[row] && Compare(op, *x[row], *y[row]));
      literal.push_back(x[row] && Compare(op, *x[row], nul));
    }
    Verify(cp::call(op, {cp::field_ref("x"), cp::field_ref("y")}), batch,
           fields);
    Verify(cp::call(op, {cp::field_ref("x"), cp::literal(nul)}), batch,
           literal);
  }
  Verify(cp::equal(cp::field_ref("x"),
                   cp::literal(arrow::MakeNullScalar(arrow::utf8()))),
         batch, std::vector<bool>(x.size(), false));
  Verify(cp::is_null(cp::field_ref("x")), batch,
         {false, false, false, false, false, false, false, true, false});
}

TEST_F(ScalarPredicateExprTest, StringParameterSnapshotsAndBroadcasts) {
  using TB = gandiva::TreeExprBuilder;
  const std::string first("a\0b", 3), second("a\0c", 3);
  auto field = arrow::field("text", arrow::utf8());
  auto parameter = arrow::field("query", arrow::utf8());
  auto schema = arrow::schema({field});
  auto values = Array<arrow::StringBuilder, std::string>(
      {"padding", "padding", first, second, "", {}, first, second, "", {}});
  auto batch = arrow::RecordBatch::Make(schema, 8, {values->Slice(2)});
  auto condition = TB::MakeCondition(TB::MakeFunction(
      "equal", {TB::MakeField(field), TB::MakeField(parameter)},
      arrow::boolean()));
  auto compiled = Unwrap(scalar_jit::ScalarPredicate::MakeParameterized(
      schema, condition, arrow::schema({parameter})));

  auto buffer =
      std::shared_ptr<arrow::Buffer>(Unwrap(arrow::AllocateBuffer(3)));
  std::memcpy(buffer->mutable_data(), first.data(), first.size());
  auto argument = std::make_shared<arrow::StringScalar>(buffer);
  auto original = Unwrap(compiled->bind(batch, {argument}));
  buffer->mutable_data()[2] = 'c';
  auto changed = Unwrap(compiled->bind(batch, {argument}));
  auto empty = Unwrap(compiled->bind(
      batch, {std::make_shared<arrow::StringScalar>(std::string{})}));
  auto null =
      Unwrap(compiled->bind(batch, {arrow::MakeNullScalar(arrow::utf8())}));

  // Release the caller's owners as well: every binding must remain independent
  // of both later buffer mutations and the lifetime of the input Scalar.
  argument.reset();
  buffer.reset();
  const std::vector<bool> original_expected = {true, false, false, false,
                                               true, false, false, false};
  const std::vector<bool> changed_expected = {false, true, false, false,
                                              false, true, false, false};
  const std::vector<bool> empty_expected = {false, false, true, false,
                                            false, false, true, false};
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (int64_t row = batch->num_rows() - 1; row >= 0; --row) {
      SCOPED_TRACE(row);
      EXPECT_EQ(original->evaluate(row), original_expected[row]);
      EXPECT_EQ(changed->evaluate(row), changed_expected[row]);
      EXPECT_EQ(empty->evaluate(row), empty_expected[row]);
      EXPECT_FALSE(null->evaluate(row));
    }
  }
}

TEST_F(ScalarPredicateExprTest, LiteralPredicates) {
  auto schema = arrow::schema({arrow::field("x", arrow::int32())});
  auto batch = arrow::RecordBatch::Make(schema, 3, {I32({0, {}, 1})});
  Verify(cp::literal(true), batch, {true, true, true});
  Verify(cp::literal(false), batch, {false, false, false});
  Verify(cp::literal(arrow::MakeNullScalar(arrow::boolean())), batch,
         {false, false, false});
}

TEST_F(ScalarPredicateExprTest, UnsupportedOperationsFallBack) {
  auto schema = arrow::schema({arrow::field("x", arrow::int32()),
                               arrow::field("y", arrow::int64()),
                               arrow::field("flag", arrow::boolean()),
                               arrow::field("text", arrow::utf8())});
  auto x = cp::field_ref("x");
  Reject(cp::greater(cp::call("add", {x, cp::literal(int32_t{1})}),
                     cp::literal(int32_t{0})),
         schema);
  Reject(
      cp::greater(cp::call("cast", {x}, cp::CastOptions::Safe(arrow::int64())),
                  cp::literal(int64_t{0})),
      schema);
  Reject(cp::greater(x, cp::field_ref("y")),
         schema);  // Bind inserts a field cast.
  Reject(cp::call("and", {cp::field_ref("flag"), cp::field_ref("flag")}),
         schema);
  Reject(
      cp::call("greater", {x, cp::literal(int32_t{0})}, cp::NullOptions(false)),
      schema);
  Reject(cp::is_null(cp::literal(I32({1, {}}))), schema);
  auto nested = arrow::schema({arrow::field(
      "parent", arrow::struct_({arrow::field("child", arrow::int32())}))});
  Reject(cp::greater(cp::field_ref(arrow::FieldRef("parent", "child")),
                     cp::literal(int32_t{0})),
         nested);
  EXPECT_FALSE(MakeScalarPredicateCondition(
                   cp::greater(x, cp::literal(int32_t{0})), schema)
                   .ok());
  EXPECT_FALSE(MakeScalarPredicateCondition(cp::Expression(), schema).ok());
  EXPECT_FALSE(MakeScalarPredicateCondition(cp::literal(true), nullptr).ok());
  EXPECT_FALSE(
      MakeScalarPredicateCondition(Unwrap(x.Bind(*schema)), schema).ok());
}

TEST_F(ScalarPredicateExprTest, ListPredicatesKeepTheArrowFallback) {
  auto schema =
      arrow::schema({arrow::field("items", arrow::list(arrow::int32()))});
  auto items = cp::field_ref("items");
  Reject(cp::greater(cp::call("list_value_length", {items}),
                     cp::literal(int32_t{1})),
         schema);
  Reject(cp::is_null(items), schema);
  if (!cp::GetFunctionRegistry()->GetFunction("contain_all").ok()) {
    Ok(ContainOp::register_op());
  }
  for (const std::string function : {"contain_all", "contain_any"}) {
    auto expression =
        cp::call(function, {items},
                 ContainOp::Options({I32({1, 2}), DataType::ARRAY_INT32}));
    Reject(expression, schema);
    Reject(cp::not_(expression), schema);
  }
}

}  // namespace
}  // namespace zvec::sqlengine

#else

TEST(ScalarPredicateExprTest, DisabledAtBuildTime) {
  GTEST_SKIP() << "ZVEC_ENABLE_SCALAR_JIT is disabled";
}

#endif  // ZVEC_ENABLE_SCALAR_JIT
