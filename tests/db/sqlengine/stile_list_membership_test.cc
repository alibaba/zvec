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

#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>
#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/ops/contain_op.h"
#include "db/sqlengine/stile/predicate.h"

namespace zvec::test {
namespace {
namespace cp = arrow::compute;
using sqlengine::ContainOp;
using stile::ListMembership;
using stile::Truth;

class StileListMembershipTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_OK(stile::Initialize());
    static const auto status = ContainOp::register_op();
    ASSERT_OK(status);
  }

  // The zvec Arrow kernels compare the physical values in null slots. For the
  // skip policy, supply the same lists and set with null elements removed.
  void expect_arrow_parity(DataType data_type,
                           const std::shared_ptr<arrow::Array> &lists,
                           const std::shared_ptr<arrow::Array> &members,
                           ListMembership::Nulls nulls,
                           const std::shared_ptr<arrow::Array> &arrow_lists,
                           const std::shared_ptr<arrow::Array> &arrow_members) {
    auto schema = arrow::schema({arrow::field("items", lists->type())});
    ASSERT_OK_AND_ASSIGN(auto index, stile::TableIndex::Make(arrow::Table::Make(
                                         schema, arrow::ArrayVector{lists})));
    for (const auto mode :
         {ListMembership::Mode::kAny, ListMembership::Mode::kAll}) {
      const auto &name = mode == ListMembership::Mode::kAll
                             ? sqlengine::kContainAll
                             : sqlengine::kContainAny;
      SCOPED_TRACE(name);
      SCOPED_TRACE(members->length());
      auto bind = [&](const std::shared_ptr<arrow::Array> &set) {
        return cp::call(name, {cp::field_ref("items")},
                        ContainOp::Options({set, data_type}))
            .Bind(*schema);
      };
      ASSERT_OK_AND_ASSIGN(auto reference, bind(arrow_members));
      ASSERT_OK_AND_ASSIGN(
          auto result,
          cp::ExecuteScalarExpression(
              reference, cp::ExecBatch({arrow_lists}, arrow_lists->length())));
      auto expected =
          std::static_pointer_cast<arrow::BooleanArray>(result.make_array());
      ASSERT_EQ(expected->length(), lists->length());

      stile::FunctionRegistry registry;
      ASSERT_OK(registry.add_list_membership(
          name, [mode, nulls](const cp::Expression::Call &call) {
            const auto &options =
                static_cast<const ContainOp::Options &>(*call.options);
            return ListMembership{mode, nulls, options.args.value_set};
          }));
      ASSERT_OK_AND_ASSIGN(auto bound, bind(members));
      EXPECT_TRUE(
          stile::Predicate::Compile(bound, index).status().IsNotImplemented());
      stile::CompileOptions options;
      options.functions = &registry;
      ASSERT_OK_AND_ASSIGN(auto predicate,
                           stile::Predicate::Compile(bound, index, options));
      std::vector<int64_t> rows(lists->length());
      std::iota(rows.begin(), rows.end(), 0);
      std::vector<Truth> batch(rows.size());
      predicate->evaluate_many(rows.data(), rows.size(), batch.data());
      for (const auto row : rows) {
        const Truth truth = expected->IsNull(row)  ? Truth::kNull
                            : expected->Value(row) ? Truth::kTrue
                                                   : Truth::kFalse;
        EXPECT_EQ(predicate->evaluate(row), truth) << row;
        EXPECT_EQ(predicate->matches(row), truth == Truth::kTrue) << row;
        EXPECT_EQ(batch[row], truth) << row;
      }
    }
  }
};

TEST_F(StileListMembershipTest, NullListsElementsAndDuplicateMembers) {
  auto make_lists = [](bool skip_null_elements)
      -> arrow::Result<std::shared_ptr<arrow::Array>> {
    auto values = std::make_shared<arrow::Int32Builder>();
    arrow::ListBuilder lists(arrow::default_memory_pool(), values);
    ARROW_RETURN_NOT_OK(lists.AppendNull());
    const std::vector<std::vector<std::optional<int32_t>>> rows{
        {}, {std::nullopt}, {0}, {1, 1}, {std::nullopt, 1}};
    for (const auto &row : rows) {
      ARROW_RETURN_NOT_OK(lists.Append());
      for (const auto &value : row) {
        if (value) {
          ARROW_RETURN_NOT_OK(values->Append(*value));
        } else if (!skip_null_elements) {
          ARROW_RETURN_NOT_OK(values->AppendNull());
        }
      }
    }
    return lists.Finish();
  };
  ASSERT_OK_AND_ASSIGN(auto lists, make_lists(false));
  ASSERT_OK_AND_ASSIGN(auto valid_elements, make_lists(true));
  arrow::Int32Builder values;
  ASSERT_OK(values.AppendValues({0, 1, 1}));
  ASSERT_OK(values.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto set, values.Finish());
  for (const auto &members :
       {set->Slice(0, 0), set->Slice(3), set->Slice(0, 1), set->Slice(1)}) {
    SCOPED_TRACE(members->ToString());
    ASSERT_OK_AND_ASSIGN(auto valid_members, cp::DropNull(*members));
    ASSERT_NO_FATAL_FAILURE(expect_arrow_parity(
        DataType::ARRAY_INT32, lists, members,
        ListMembership::Nulls::kStoredValue, lists, members));
    ASSERT_NO_FATAL_FAILURE(expect_arrow_parity(
        DataType::ARRAY_INT32, lists, members, ListMembership::Nulls::kSkip,
        valid_elements, valid_members));
  }
}

TEST_F(StileListMembershipTest, AllMembersAcrossBitMaskBoundaries) {
  std::vector<int32_t> sequence(257);
  std::iota(sequence.begin(), sequence.end(), 0);
  auto values = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  ASSERT_OK(lists.Append());
  ASSERT_OK(values->AppendValues(sequence));
  // Keep the length identical so a missing bit cannot pass a length-only test.
  ASSERT_OK(lists.Append());
  auto missing = sequence;
  missing[63] = 62;
  ASSERT_OK(values->AppendValues(missing));
  ASSERT_OK_AND_ASSIGN(auto input, lists.Finish());
  ASSERT_OK(values->AppendValues(sequence));
  ASSERT_OK_AND_ASSIGN(auto full_set, values->Finish());
  for (const int length : {64, 65, 256, 257}) {
    const auto members = full_set->Slice(0, length);
    ASSERT_NO_FATAL_FAILURE(
        expect_arrow_parity(DataType::ARRAY_INT32, input, members,
                            ListMembership::Nulls::kSkip, input, members));
  }
}

TEST_F(StileListMembershipTest, SignedZerosMatchAndNaNDoesNot) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  auto values = std::make_shared<arrow::DoubleBuilder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  for (double value : {-0.0, 0.0, nan, 1.0}) {
    ASSERT_OK(lists.Append());
    ASSERT_OK(values->Append(value));
  }
  ASSERT_OK_AND_ASSIGN(auto input, lists.Finish());
  ASSERT_OK(values->AppendValues({0.0, nan}));
  ASSERT_OK_AND_ASSIGN(auto set, values->Finish());
  for (const auto &members : {set->Slice(0, 1), set->Slice(1), set}) {
    ASSERT_NO_FATAL_FAILURE(
        expect_arrow_parity(DataType::ARRAY_DOUBLE, input, members,
                            ListMembership::Nulls::kSkip, input, members));
  }
}

}  // namespace
}  // namespace zvec::test
