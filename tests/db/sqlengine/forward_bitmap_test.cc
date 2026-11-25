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
#include <functional>
#include <future>
#include <thread>
#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/array/util.h>
#include "db/sqlengine/planner/doc_filter.h"
#include "db/sqlengine/planner/ops/contain_op.h"
#include "mock_segment.h"

namespace zvec::test {
namespace cp = arrow::compute;
namespace ac = arrow::acero;

namespace {
constexpr int64_t kBatchRows = 256 * 1024;

class BitmapSegment : public MockSegment {
 public:
  explicit BitmapSegment(TablePtr input) : table(std::move(input)) {
    for (const auto &field : table->schema()->fields()) {
      DataType type = DataType::INT32;
      if (field->type()->id() == arrow::Type::BOOL) type = DataType::BOOL;
      if (field->type()->id() == arrow::Type::STRING) type = DataType::STRING;
      if (field->type()->id() == arrow::Type::LIST)
        type = DataType::ARRAY_INT32;
      fields.add_field(
          std::make_shared<FieldSchema>(field->name(), type, true));
    }
  }

  arrow::Result<ForwardReadState> prepare_forward_read(
      const std::vector<std::string> &columns, bool) override {
    if (!prepare_status.ok()) return prepare_status;
    std::vector<int> indices;
    for (const auto &name : columns) {
      const int index = table->schema()->GetFieldIndex(name);
      if (index < 0) return arrow::Status::Invalid("Unknown column: ", name);
      indices.push_back(index);
    }
    ARROW_ASSIGN_OR_RAISE(auto projected, table->SelectColumns(indices));
    return ForwardReadState{
        captured_schema ? captured_schema : projected->schema(),
        static_cast<uint64_t>(table->num_rows()),
        make_view ? make_view(projected) : nullptr, nullptr};
  }

  RecordBatchReaderPtr scan(const std::vector<std::string> &) const override {
    ++scans;
    if (reader_factory) return reader_factory();
    auto reader = std::make_shared<arrow::TableBatchReader>(table);
    reader->set_chunksize(2 * kBatchRows + 17);
    return reader;
  }

  const IndexFilter::Ptr get_filter() override {
    return nullptr;
  }

  TablePtr table;
  CollectionSchema fields{"bitmap"};
  std::function<ForwardReadView::Ptr(TablePtr)> make_view;
  std::function<RecordBatchReaderPtr()> reader_factory;
  std::shared_ptr<arrow::Schema> captured_schema;
  arrow::Status prepare_status;
  mutable std::atomic<int> scans{0};
};

class FailingReader : public arrow::RecordBatchReader {
 public:
  explicit FailingReader(std::shared_ptr<arrow::RecordBatch> batch)
      : batch_(std::move(batch)) {}
  std::shared_ptr<arrow::Schema> schema() const override {
    return batch_->schema();
  }
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
    if (read_) return arrow::Status::IOError("injected reader failure");
    read_ = true;
    *out = batch_;
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<arrow::RecordBatch> batch_;
  bool read_{false};
};

class EmptyReader : public arrow::RecordBatchReader {
 public:
  std::shared_ptr<arrow::Schema> schema() const override {
    return nullptr;
  }
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
    out->reset();
    return arrow::Status::OK();
  }
};

struct ThreadProbeState : cp::KernelState {
  std::thread::id owner{std::this_thread::get_id()};
};

arrow::Status register_thread_probe() {
  auto function = std::make_shared<cp::ScalarFunction>(
      "bitmap_thread_probe", cp::Arity::Unary(),
      cp::FunctionDoc{"检查 bitmap 执行线程", "", {"value"}}, nullptr, false);
  cp::ScalarKernel kernel(
      {arrow::int32()}, arrow::boolean(),
      [](cp::KernelContext *context, const cp::ExecSpan &batch,
         cp::ExecResult *out) -> arrow::Status {
        const auto *state = static_cast<ThreadProbeState *>(context->state());
        if (state->owner != std::this_thread::get_id() ||
            context->exec_context()->executor() != nullptr) {
          return arrow::Status::Invalid("bitmap escaped its calling thread");
        }
        ARROW_ASSIGN_OR_RAISE(
            auto values,
            arrow::MakeArrayFromScalar(arrow::BooleanScalar(true), batch.length,
                                       context->memory_pool()));
        out->value = values->data();
        return arrow::Status::OK();
      },
      [](cp::KernelContext *, const cp::KernelInitArgs &)
          -> arrow::Result<std::unique_ptr<cp::KernelState>> {
        return std::make_unique<ThreadProbeState>();
      });
  kernel.mem_allocation = cp::MemAllocation::NO_PREALLOCATE;
  kernel.null_handling = cp::NullHandling::OUTPUT_NOT_NULL;
  ARROW_RETURN_NOT_OK(function->AddKernel(std::move(kernel)));
  return cp::GetFunctionRegistry()->AddFunction(std::move(function));
}

arrow::Result<TablePtr> input_table(int64_t rows, bool split = false) {
  arrow::Int32Builder integers;
  arrow::BooleanBuilder flags;
  // 保留前缀，使输入数组带非零 offset，包括位图的非字节对齐 offset。
  for (int64_t i = 0; i < rows + 3; ++i) {
    ARROW_RETURN_NOT_OK(i % 11 ? integers.Append(static_cast<int32_t>(i))
                               : integers.AppendNull());
    ARROW_RETURN_NOT_OK(i % 7 ? flags.Append(i % 2 == 0) : flags.AppendNull());
  }
  ARROW_ASSIGN_OR_RAISE(auto ints, integers.Finish());
  ARROW_ASSIGN_OR_RAISE(auto bools, flags.Finish());
  auto chunks = [&](std::shared_ptr<arrow::Array> array, int64_t boundary) {
    array = array->Slice(3, rows);
    arrow::ArrayVector parts{array};
    if (split) {
      parts = {array->Slice(0, 0), array->Slice(0, boundary),
               array->Slice(boundary, 0), array->Slice(boundary),
               array->Slice(rows, 0)};
    }
    return std::make_shared<arrow::ChunkedArray>(std::move(parts));
  };
  return arrow::Table::Make(
      arrow::schema({arrow::field("id", arrow::int32()),
                     arrow::field("flag", arrow::boolean())}),
      {chunks(ints, rows / 3), chunks(bools, rows / 2)}, rows);
}
}  // namespace

class ForwardBitmapTest : public testing::TestWithParam<bool> {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(cp::Initialize().ok());
    ASSERT_TRUE(sqlengine::ContainOp::register_op().ok());
    ASSERT_TRUE(register_thread_probe().ok());
  }

  std::shared_ptr<BitmapSegment> segment(TablePtr table) {
    auto result = std::make_shared<BitmapSegment>(std::move(table));
    if (GetParam()) {
      result->make_view = [](TablePtr input) {
        return ForwardReadView::Make(std::move(input));
      };
    }
    return result;
  }

  static std::unique_ptr<sqlengine::DocFilter> filter(
      const std::shared_ptr<BitmapSegment> &segment,
      const cp::Expression &expression, bool no_fields = false) {
    auto info = std::make_shared<sqlengine::QueryInfo>();
    if (!no_fields) {
      for (const auto &field : segment->table->schema()->fields()) {
        info->add_forward_filter_schema_ptr(
            field->name(), segment->fields.get_field(field->name()));
      }
    }
    return std::make_unique<sqlengine::DocFilter>(
        segment, info, std::make_unique<cp::Expression>(expression), nullptr);
  }

  static std::shared_ptr<arrow::ChunkedArray> bitmap(
      const sqlengine::DocFilter &filter) {
    return filter.forward_bitmap_;
  }

  static arrow::Result<std::shared_ptr<arrow::ChunkedArray>> build(
      const sqlengine::DocFilter &filter) {
    return filter.build_forward_bitmap();
  }

  void expect_acero_parity(const TablePtr &table,
                           const cp::Expression &expression) {
    SCOPED_TRACE(expression.ToString());
    auto input = segment(table);
    auto candidate = filter(input, expression);
    auto status = candidate->compute_filter();
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_NE(bitmap(*candidate), nullptr);
    ASSERT_TRUE(bitmap(*candidate)->ValidateFull().ok());
    EXPECT_EQ(bitmap(*candidate)->length(), table->num_rows());
    EXPECT_EQ(input->scans.load(), GetParam() ? 0 : 1);
    // 对照替换前的 reader -> Acero project 路径，不共用生产 helper。
    ac::Declaration source{
        "record_batch_reader_source",
        ac::RecordBatchReaderSourceNodeOptions(
            std::make_shared<arrow::TableBatchReader>(table))};
    ac::Declaration project{"project",
                            {std::move(source)},
                            ac::ProjectNodeOptions({expression}, {"matched"})};
    auto reference = ac::DeclarationToTable(std::move(project), false);
    ASSERT_TRUE(reference.ok()) << reference.status().ToString();
    ASSERT_TRUE(bitmap(*candidate)->Equals(*reference.ValueOrDie()->column(0)));
    for (int64_t row = 0; row < table->num_rows(); ++row) {
      auto scalar = reference.ValueOrDie()->column(0)->GetScalar(row);
      ASSERT_TRUE(scalar.ok());
      const auto &value = static_cast<const arrow::BooleanScalar &>(**scalar);
      ASSERT_EQ(candidate->is_filtered(row), !value.is_valid || !value.value)
          << row;
    }
  }
};

TEST_P(ForwardBitmapTest, NullableBooleanAndScalarPredicatesMatchAcero) {
  auto table = input_table(67, true).ValueOrDie();
  arrow::Int32Builder members;
  ASSERT_TRUE(members.AppendValues({3, 8, 12}).ok());
  ASSERT_TRUE(members.AppendNull().ok());
  auto values = members.Finish().ValueOrDie();
  const auto in =
      cp::call("is_in", {cp::field_ref("id")}, cp::SetLookupOptions(values));
  const auto positive =
      cp::call("greater", {cp::field_ref("id"), cp::literal(int32_t{17})});
  const std::vector<cp::Expression> expressions{
      cp::literal(true),
      cp::literal(false),
      cp::literal(arrow::MakeNullScalar(arrow::boolean())),
      cp::field_ref("flag"),
      cp::call("invert", {cp::field_ref("flag")}),
      cp::call("is_null", {cp::field_ref("id")}),
      in,
      cp::call("invert", {in}),
      positive,
      cp::call("and_kleene", {positive, cp::field_ref("flag")}),
      cp::call("or_kleene", {positive, cp::field_ref("flag")}),
      cp::call("less", {cp::call("cast", {cp::field_ref("id")},
                                 cp::CastOptions::Safe(arrow::int64())),
                        cp::literal(int64_t{12})})};
  for (const auto &expression : expressions) {
    ASSERT_NO_FATAL_FAILURE(expect_acero_parity(table, expression));
  }
}

TEST_P(ForwardBitmapTest, LikeAndArrayPredicatesMatchAcero) {
  arrow::StringBuilder strings;
  auto elements = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), elements);
  for (int i = 0; i < 32; ++i) {
    ASSERT_TRUE((i % 5 ? strings.Append(i % 2 ? "prefix_你好" : "other")
                       : strings.AppendNull())
                    .ok());
    if (i % 5 == 0) {
      ASSERT_TRUE(lists.AppendNull().ok());
    } else {
      ASSERT_TRUE(lists.Append().ok());
      for (int j = 0; j < i % 4; ++j) ASSERT_TRUE(elements->Append(j).ok());
    }
  }
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("text", arrow::utf8()),
                     arrow::field("items", arrow::list(arrow::int32()))}),
      arrow::ArrayVector{strings.Finish().ValueOrDie(),
                         lists.Finish().ValueOrDie()});
  arrow::Int32Builder members;
  ASSERT_TRUE(members.AppendValues({0, 1}).ok());
  const auto values = members.Finish().ValueOrDie();
  sqlengine::ContainOp::Options options({values, DataType::ARRAY_INT32});
  const auto any = cp::call("contain_any", {cp::field_ref("items")}, options);
  const std::vector<cp::Expression> expressions{
      cp::call("match_like", {cp::field_ref("text")},
               cp::MatchSubstringOptions("prefix_%")),
      any, cp::call("invert", {any}),
      cp::call("contain_all", {cp::field_ref("items")}, options),
      cp::call("equal",
               {cp::call("list_value_length", {cp::field_ref("items")}),
                cp::literal(int32_t{0})})};
  for (const auto &expression : expressions) {
    ASSERT_NO_FATAL_FAILURE(expect_acero_parity(table, expression));
  }
}

TEST_P(ForwardBitmapTest, BatchBoundariesAndIndependentSlicedChunks) {
  for (int64_t rows : {int64_t{0}, int64_t{1}, kBatchRows - 1, kBatchRows,
                       kBatchRows + 1, 2 * kBatchRows + 17}) {
    for (bool split : {false, true}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " split=" << split);
      auto table = input_table(rows, split).ValueOrDie();
      ASSERT_NO_FATAL_FAILURE(expect_acero_parity(
          table, cp::call("or_kleene",
                          {cp::field_ref("flag"),
                           cp::call("less", {cp::field_ref("id"),
                                             cp::literal(int32_t{262147})})})));
    }
  }
}

TEST_P(ForwardBitmapTest, LiteralWithoutFieldsAndEmptyInput) {
  for (int64_t rows : {int64_t{0}, int64_t{1}, kBatchRows + 1}) {
    auto input = segment(input_table(rows).ValueOrDie());
    for (const auto &value : {std::make_shared<arrow::BooleanScalar>(true),
                              std::make_shared<arrow::BooleanScalar>(false),
                              std::make_shared<arrow::BooleanScalar>()}) {
      auto candidate = filter(input, cp::literal(value), true);
      auto status = candidate->compute_filter();
      ASSERT_TRUE(status.ok()) << status.message();
      ASSERT_EQ(bitmap(*candidate)->length(), rows);
      EXPECT_EQ(input->scans.load(), 0);
      for (int64_t row = 0; row < rows; ++row) {
        ASSERT_EQ(candidate->is_filtered(row),
                  !value->is_valid || !value->value);
      }
    }
  }
}

TEST_P(ForwardBitmapTest, DiagnosticsPreserveStageAndArrowErrorCode) {
  auto input = segment(input_table(9).ValueOrDie());
  auto candidate = filter(input, cp::literal(true));
  input->prepare_status = arrow::Status::IOError("injected prepare failure");
  auto prepared = build(*candidate);
  ASSERT_FALSE(prepared.ok());
  EXPECT_TRUE(prepared.status().IsIOError());
  EXPECT_NE(prepared.status().message().find("prepare failed: segment[0]"),
            std::string::npos);
  EXPECT_EQ(input->scans.load(), 0);
  input->prepare_status = arrow::Status::OK();
  for (const auto &expression :
       {cp::field_ref("missing"), cp::field_ref("id")}) {
    auto invalid = build(*filter(input, expression));
    ASSERT_FALSE(invalid.ok());
    EXPECT_NE(invalid.status().message().find("bind failed: segment[0]"),
              std::string::npos);
  }
  // 安全转换只能在执行时发现溢出，不能被绑定阶段误当成成功。
  input->table = input_table(300).ValueOrDie();
  auto overflow = filter(
      input,
      cp::call("greater", {cp::call("cast", {cp::field_ref("id")},
                                    cp::CastOptions::Safe(arrow::int8())),
                           cp::literal(int8_t{0})}));
  auto failed = build(*overflow);
  ASSERT_FALSE(failed.ok());
  EXPECT_TRUE(failed.status().IsInvalid());
  EXPECT_NE(failed.status().message().find("evaluate failed: segment[0]"),
            std::string::npos);
}

TEST_P(ForwardBitmapTest, ReaderFailureDoesNotPublishPartialOrStaleBitmap) {
  auto input = segment(input_table(9).ValueOrDie());
  input->make_view = nullptr;
  auto candidate = filter(input, cp::literal(true));
  ASSERT_TRUE(candidate->compute_filter().ok());
  ASSERT_NE(bitmap(*candidate), nullptr);
  auto batch = arrow::TableBatchReader(input->table).Next().ValueOrDie();
  input->reader_factory = [batch] {
    return std::make_shared<FailingReader>(batch);
  };
  auto status = candidate->compute_filter();
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(bitmap(*candidate), nullptr);
  EXPECT_NE(status.message().find("scan failed: segment[0]"),
            std::string::npos);
  EXPECT_NE(status.message().find("injected reader failure"),
            std::string::npos);
  auto raw = build(*candidate);
  EXPECT_TRUE(raw.status().IsIOError());
  input->reader_factory = nullptr;
  ASSERT_TRUE(candidate->compute_filter().ok());
  EXPECT_EQ(bitmap(*candidate)->length(), 9);
}

TEST_P(ForwardBitmapTest, ScannerSchemaAndMissingReaderDiagnostics) {
  auto input = segment(input_table(9).ValueOrDie());
  input->make_view = nullptr;
  // scanner 的列顺序与捕获的 schema 不同，必须重新按 scanner schema 绑定。
  input->captured_schema =
      arrow::schema({arrow::field("flag", arrow::boolean()),
                     arrow::field("id", arrow::int32())});
  auto candidate = filter(input, cp::field_ref("flag"));
  ASSERT_TRUE(candidate->compute_filter().ok());
  EXPECT_TRUE(bitmap(*candidate)->Equals(*input->table->column(1)));
  input->reader_factory = [] { return RecordBatchReaderPtr{}; };
  auto missing = build(*candidate);
  ASSERT_FALSE(missing.ok());
  EXPECT_NE(missing.status().message().find("Missing forward filter scan"),
            std::string::npos);
  input->reader_factory = [] { return std::make_shared<EmptyReader>(); };
  auto schema_missing = build(*candidate);
  ASSERT_FALSE(schema_missing.ok());
  EXPECT_NE(schema_missing.status().message().find("scan schema"),
            std::string::npos);
  input->table = input->table->Slice(0, 0);
  ASSERT_TRUE(candidate->compute_filter().ok());
  EXPECT_EQ(bitmap(*candidate)->length(), 0);
}

TEST_P(ForwardBitmapTest, ConcurrentEvaluationStaysOnEachCallingThread) {
  auto input = segment(input_table(kBatchRows + 1, true).ValueOrDie());
  std::atomic<bool> start{false};
  std::vector<std::future<bool>> readers;
  for (int i = 0; i < 8; ++i) {
    readers.push_back(std::async(std::launch::async, [&, input] {
      while (!start.load()) std::this_thread::yield();
      for (int repeat = 0; repeat < 8; ++repeat) {
        auto candidate = filter(
            input, cp::call("bitmap_thread_probe", {cp::field_ref("id")}));
        if (!candidate->compute_filter().ok() ||
            bitmap(*candidate)->length() != input->table->num_rows() ||
            candidate->is_filtered(0) || candidate->is_filtered(kBatchRows)) {
          return false;
        }
      }
      return true;
    }));
  }
  start = true;
  for (auto &reader : readers) EXPECT_TRUE(reader.get());
}

INSTANTIATE_TEST_SUITE_P(ViewAndScanner, ForwardBitmapTest, testing::Bool());
}  // namespace zvec::test
