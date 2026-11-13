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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include <zvec/db/config.h>
#include "db/common/file_helper.h"
#include "db/common/query_read_context.h"
#include "db/sqlengine/planner/doc_filter.h"
#include "index/utils/utils.h"

namespace zvec::test {
namespace cp = arrow::compute;

class ForwardFilterReadTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
    ailego::MemoryLimitPool::get_instance().init(MIN_MEMORY_LIMIT_BYTES);
    ASSERT_TRUE(cp::Initialize().ok());
    FileHelper::RemoveDirectory(path_);
    FileHelper::CreateDirectory(path_);
    schema_ = std::make_shared<CollectionSchema>("forward_reader");
    const std::vector<std::pair<std::string, DataType>> fields = {
        {"i32", DataType::INT32},        {"i64", DataType::INT64},
        {"u32", DataType::UINT32},       {"u64", DataType::UINT64},
        {"f32", DataType::FLOAT},        {"f64", DataType::DOUBLE},
        {"text", DataType::STRING},      {"flag", DataType::BOOL},
        {"items", DataType::ARRAY_INT32}};
    for (const auto &[name, type] : fields) {
      schema_->add_field(std::make_shared<FieldSchema>(name, type, true));
    }
    id_map_ = IDMap::CreateAndOpen(
        "forward_reader", FileHelper::MakeFilePath(path_, FileID::ID_FILE, 0),
        true, false);
    ASSERT_NE(id_map_, nullptr);
    deletes_ = std::make_shared<DeleteStore>("forward_reader");
    Version version;
    version.set_schema(*schema_);
    version.set_enable_mmap(GetParam());
    auto vm = VersionManager::Create(path_, version);
    ASSERT_TRUE(vm.has_value()) << vm.error().message();
    versions_ = vm.value();
    options_.enable_mmap_ = GetParam();
    options_.max_buffer_size_ = 16 * 1024;
  }

  void TearDown() override {
    segment_.reset();
    versions_.reset();
    deletes_.reset();
    id_map_.reset();
    FileHelper::RemoveDirectory(path_);
  }

  QueryReadContext context(bool read_only = true) const {
    // This fixture substitutes for the collection-owned guarded scope. It never
    // mutates/closes a segment during view preparation or query execution.
    return QueryReadContext(read_only, {segment_});
  }

  void create(int count = 2053) {
    auto created = Segment::CreateAndOpen(path_, *schema_, 0, 10000, id_map_,
                                          deletes_, versions_, options_);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    segment_ = created.value();
    for (int i = 0; i < count; ++i) {
      auto doc = i % 11 == 0 ? TestHelper::CreateDocNull(i, *schema_)
                             : TestHelper::CreateDoc(i, *schema_);
      if (i % 11 != 0) {
        doc.set<int32_t>("i32",
                         i % 2 ? std::numeric_limits<int32_t>::min() : i);
        doc.set<int64_t>("i64",
                         i % 2 ? std::numeric_limits<int64_t>::min() : i);
        doc.set<uint32_t>("u32", std::numeric_limits<uint32_t>::max() - i);
        doc.set<uint64_t>("u64", std::numeric_limits<uint64_t>::max() - i);
        doc.set<float>("f32",
                       i % 2 ? -0.0f : std::numeric_limits<float>::max());
        doc.set<double>("f64",
                        i % 2 ? -0.0 : std::numeric_limits<double>::max());
      }
      ASSERT_TRUE(segment_->insert(doc).ok());
    }
  }

  void reopen_persisted(bool dump = true) {
    if (dump) {
      ASSERT_TRUE(segment_->dump().ok());
      segment_->remove_writing_forward_block();
    }
    auto meta = *segment_->meta();
    segment_.reset();
    auto options = options_;
    options.read_only_ = true;
    auto opened = Segment::Open(path_, *schema_, meta, id_map_, deletes_,
                                versions_, options);
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    segment_ = opened.value();
  }

  std::unique_ptr<sqlengine::DocFilter> filter(
      const std::vector<std::string> &columns, cp::Expression expression,
      QueryReadContext ctx = {}) {
    auto info = std::make_shared<sqlengine::QueryInfo>();
    for (const auto &column : columns) {
      info->add_forward_filter_schema_ptr(column, schema_->get_field(column));
    }
    return std::make_unique<sqlengine::DocFilter>(
        segment_, info, nullptr, std::make_unique<cp::Expression>(expression),
        ctx);
  }

  static ForwardReadView::Ptr make_owned_view(TablePtr table) {
    return ForwardReadView::make(std::move(table));
  }

  const std::string path_{"./forward_filter_read_test_collection"};
  CollectionSchema::Ptr schema_;
  IDMap::Ptr id_map_;
  DeleteStore::Ptr deletes_;
  VersionManager::Ptr versions_;
  SegmentOptions options_{};
  Segment::Ptr segment_;
};

TEST_P(ForwardFilterReadTest, DefaultDenyMutableSealedAndForeignSegment) {
  create(24);
  ASSERT_NE(segment_, nullptr);
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"i32"}, context()),
            nullptr);
  EXPECT_FALSE(QueryReadContext{}.allows_forward_view(segment_.get()));
  EXPECT_FALSE(context(false).allows_forward_view(segment_.get()));
  EXPECT_FALSE(context().allows_forward_view(nullptr));
  ASSERT_TRUE(segment_->flush().ok());
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"i32"}, context()),
            nullptr);
  ASSERT_TRUE(segment_->dump().ok());
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"i32"}, context()),
            nullptr);
  segment_->remove_writing_forward_block();
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"i32"}, context()),
            nullptr);
  const auto foreign_context = context();
  reopen_persisted(false);
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"i32"}, foreign_context),
            nullptr);
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"i32"}), nullptr);
  // Optimize can open read-only segments in a writable collection: still
  // denied.
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"i32"}, context(false)),
            nullptr);
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {}, context()), nullptr);
  EXPECT_EQ(Segment::make_forward_read_view(segment_, {"missing"}, context()),
            nullptr);
  EXPECT_EQ(Segment::make_forward_read_view(nullptr, {"i32"}, context()),
            nullptr);
  EXPECT_EQ(bool(Segment::make_forward_read_view(segment_, {"i32"}, context())),
            GetParam());
}

TEST_P(ForwardFilterReadTest, ReorderedProjectionBoundariesAndOwnedView) {
  create();
  reopen_persisted();
  ASSERT_FALSE(segment_->has_identity_doc_ids());
  ASSERT_GT(segment_->meta()->persisted_blocks().size(), 1u);
  auto columns = schema_->forward_field_names();
  std::reverse(columns.begin(), columns.end());
  auto view = Segment::make_forward_read_view(segment_, columns, context());
  ASSERT_EQ(bool(view), GetParam());
  if (view) {
    EXPECT_EQ(view->num_rows(), 2053);
    EXPECT_TRUE(view->table()->schema()->Equals(
        *segment_->fetch(columns, std::vector<int>{})->schema()));
    EXPECT_EQ(view->read_row(2053), nullptr);
    EXPECT_EQ(view->read_row(UINT64_MAX), nullptr);
  }
  for (int row = 0; row < 2053; ++row) {
    auto baseline = segment_->fetch(columns, row);
    ASSERT_NE(baseline, nullptr) << row;
    ASSERT_EQ(baseline->values.size(), columns.size());
    if (view) {
      auto projected = view->read_row(row);
      ASSERT_NE(projected, nullptr) << row;
      EXPECT_EQ(projected->length, 1);
      ASSERT_EQ(projected->values.size(), columns.size());
      for (size_t c = 0; c < columns.size(); ++c) {
        ASSERT_TRUE(projected->values[c].is_scalar());
        EXPECT_TRUE(baseline->values[c].Equals(projected->values[c]))
            << row << ": " << columns[c];
      }
    }
  }
  EXPECT_EQ(segment_->fetch(columns, 2053), nullptr);
  auto weak = std::weak_ptr<Segment>(segment_);
  segment_.reset();
  EXPECT_TRUE(weak.expired());
  if (view) EXPECT_NE(view->read_row(1024), nullptr);
}

TEST_P(ForwardFilterReadTest, ArrowPredicateParityAndGlobalIdDeletes) {
  create();
  // The delete filter maps local row 7 to global ID 10007, not global ID 7.
  ASSERT_TRUE(segment_->Delete(uint64_t{10007}).ok());
  reopen_persisted();
  const auto columns = schema_->forward_field_names();
  arrow::Int32Builder members;
  ASSERT_TRUE(
      members.AppendValues({0, 2, std::numeric_limits<int32_t>::min()}).ok());
  std::shared_ptr<arrow::Array> member_array;
  ASSERT_TRUE(members.Finish(&member_array).ok());
  auto in = cp::call("is_in", {cp::field_ref("i32")},
                     cp::SetLookupOptions(member_array));
  const std::vector<cp::Expression> expressions = {
      in,
      cp::call("invert", {in}),
      cp::call("less", {cp::field_ref("i32"), cp::literal(int32_t{0})}),
      cp::call("less", {cp::field_ref("i64"), cp::literal(int64_t{0})}),
      cp::call("greater", {cp::field_ref("u32"), cp::literal(uint32_t{1})}),
      cp::call("greater", {cp::field_ref("u64"), cp::literal(uint64_t{1})}),
      cp::call("equal", {cp::field_ref("f32"), cp::literal(-0.0f)}),
      cp::call("greater", {cp::field_ref("f64"), cp::literal(0.0)}),
      cp::call("equal", {cp::field_ref("text"), cp::literal("value_10")}),
      cp::field_ref("flag"),
      cp::call("is_null", {cp::field_ref("i32")}),
      cp::call("greater",
               {cp::call("list_value_length", {cp::field_ref("items")}),
                cp::literal(int32_t{0})}),
      cp::call("less", {cp::call("cast", {cp::field_ref("i32")},
                                 cp::CastOptions::Safe(arrow::int64())),
                        cp::literal(int64_t{0})}),
      cp::call("and_kleene", {cp::call("less", {cp::field_ref("i32"),
                                                cp::literal(int32_t{0})}),
                              cp::call("invert", {cp::field_ref("flag")})})};
  for (const auto &expr : expressions) {
    SCOPED_TRACE(expr.ToString());
    auto locked = filter(columns, expr);
    auto candidate = filter(columns, expr, context());
    auto status = locked->compute_filter();
    ASSERT_TRUE(status.ok()) << status.message();
    status = candidate->compute_filter();
    ASSERT_TRUE(status.ok()) << status.message();
    for (int row = 0; row < 2053; ++row) {
      EXPECT_EQ(locked->is_filtered(row), candidate->is_filtered(row)) << row;
    }
    EXPECT_TRUE(candidate->is_filtered(7));
  }
  auto negative = filter({"i32"}, expressions[2], context());
  ASSERT_TRUE(negative->compute_filter().ok());
  EXPECT_FALSE(negative->is_filtered(1));
  EXPECT_TRUE(negative->is_filtered(2));
  EXPECT_TRUE(negative->is_filtered(0));
  EXPECT_TRUE(negative->is_filtered(7));
}

TEST_P(ForwardFilterReadTest, IndependentColumnChunkBoundaries) {
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_TRUE(segment_->dump().ok());
  segment_->remove_writing_forward_block();
  auto meta = *segment_->meta();
  segment_.reset();
  // A separately persisted column has different record-batch/row-group
  // boundaries from the original wide scalar blocks, as after schema DDL.
  arrow::Int32Builder builder;
  for (int i = 0; i < 2053; ++i) {
    ASSERT_TRUE((i % 13 ? builder.Append(i) : builder.AppendNull()).ok());
  }
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  auto arrow_schema = arrow::schema({arrow::field("other", arrow::int32())});
  const auto file =
      FileHelper::MakeForwardBlockPath(path_, 0, 100000, !GetParam());
  auto output = arrow::io::FileOutputStream::Open(file).ValueOrDie();
  if (GetParam()) {
    auto writer = arrow::ipc::MakeFileWriter(output, arrow_schema).ValueOrDie();
    int row = 0;
    while (row < 2053) {
      int size = std::min(row % 2 ? 101 : 7, 2053 - row);
      auto batch = arrow::RecordBatch::Make(arrow_schema, size,
                                            {array->Slice(row, size)});
      ASSERT_TRUE(writer->WriteRecordBatch(*batch).ok());
      row += size;
    }
    ASSERT_TRUE(writer->Close().ok());
  } else {
    auto table = arrow::Table::Make(
        arrow_schema, {std::make_shared<arrow::ChunkedArray>(array)});
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           output, 101)
                    .ok());
  }
  ASSERT_TRUE(output->Close().ok());
  meta.add_persisted_block(
      {100000, BlockType::SCALAR, 10000, 12052, 2053, {"other"}});
  schema_->add_field(
      std::make_shared<FieldSchema>("other", DataType::INT32, true));
  auto options = options_;
  options.read_only_ = true;
  auto opened = Segment::Open(path_, *schema_, meta, id_map_, deletes_,
                              versions_, options);
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  segment_ = opened.value();
  const std::vector<std::string> columns{"other", "i32", "text"};
  auto view = Segment::make_forward_read_view(segment_, columns, context());
  ASSERT_EQ(bool(view), GetParam());
  if (view) {
    EXPECT_EQ(view->num_rows(), 2053);
    EXPECT_NE(view->table()->column(0)->num_chunks(),
              view->table()->column(1)->num_chunks());
  }
  for (int row = 0; row < 2053; ++row) {
    auto a = segment_->fetch(columns, row);
    ASSERT_NE(a, nullptr);
    ASSERT_EQ(a->values.size(), columns.size());
    if (view) {
      auto projected = view->read_row(row);
      ASSERT_NE(projected, nullptr) << row;
      ASSERT_EQ(projected->values.size(), a->values.size());
      EXPECT_EQ(projected->length, 1);
      for (size_t c = 0; c < a->values.size(); ++c) {
        ASSERT_TRUE(projected->values[c].is_scalar());
        EXPECT_TRUE(a->values[c].Equals(projected->values[c]))
            << row << ": " << c;
      }
    }
    const auto &value = a->values[0].scalar_as<arrow::Int32Scalar>();
    EXPECT_EQ(value.is_valid, row % 13 != 0);
    if (value.is_valid) EXPECT_EQ(value.value, row);
  }
}

TEST_P(ForwardFilterReadTest, ViewOwnsBuffersAfterSegmentClose) {
  if (!GetParam())
    GTEST_SKIP() << "Parquet view ownership is not supported yet";
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  const auto columns = schema_->forward_field_names();
  auto view = Segment::make_forward_read_view(segment_, columns, context());
  ASSERT_NE(view, nullptr);
  // Keep serialized expectations, not scalars/arrays that could accidentally
  // retain the buffers whose ownership this test exercises.
  std::vector<std::vector<std::string>> expected(2053);
  for (int row = 0; row < 2053; ++row) {
    auto batch = segment_->fetch(columns, row);
    ASSERT_NE(batch, nullptr);
    for (const auto &value : batch->values)
      expected[row].push_back(value.ToString());
  }
  auto weak = std::weak_ptr<Segment>(segment_);
  // SegmentImpl's destructor closes stores and releases their Arrow owners.
  segment_.reset();
  ASSERT_TRUE(weak.expired());
  for (int row = 0; row < 2053; ++row) {
    auto batch = view->read_row(row);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->length, 1);
    ASSERT_EQ(batch->values.size(), expected[row].size());
    for (size_t c = 0; c < batch->values.size(); ++c)
      EXPECT_EQ(batch->values[c].ToString(), expected[row][c])
          << row << ": " << c;
  }
}

TEST_P(ForwardFilterReadTest, ViewLocatesIndependentEmptyAndSlicedChunks) {
  arrow::Int32Builder integers;
  arrow::StringBuilder strings;
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE((i == 4 ? integers.AppendNull() : integers.Append(i)).ok());
    ASSERT_TRUE((i == 5 ? strings.AppendNull()
                        : strings.Append("text_" + std::to_string(i)))
                    .ok());
  }
  std::shared_ptr<arrow::Array> integer_array, string_array;
  ASSERT_TRUE(integers.Finish(&integer_array).ok());
  ASSERT_TRUE(strings.Finish(&string_array).ok());
  auto sliced = integer_array->Slice(2, 6);
  auto text = string_array->Slice(2, 6);
  ASSERT_EQ(sliced->offset(), 2);
  ASSERT_EQ(text->offset(), 2);
  auto schema = arrow::schema(
      {arrow::field("i", arrow::int32()), arrow::field("s", arrow::utf8())});
  auto table = arrow::Table::Make(
      schema,
      {std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
           sliced->Slice(0, 0), sliced->Slice(0, 2), sliced->Slice(2, 0),
           sliced->Slice(2, 4), sliced->Slice(6, 0)}),
       std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
           text->Slice(0, 1), text->Slice(1, 0), text->Slice(1, 4),
           text->Slice(5, 1), text->Slice(6, 0)})});
  auto view = make_owned_view(table);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->num_rows(), 6);
  const int chunks[2][6] = {{1, 1, 3, 3, 3, 3}, {0, 2, 2, 2, 2, 3}};
  const int offsets[2][6] = {{0, 1, 0, 1, 2, 3}, {0, 0, 1, 2, 3, 0}};
  for (int row = 0; row < 6; ++row) {
    auto batch = view->read_row(row);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->length, 1);
    ASSERT_EQ(batch->values.size(), 2u);
    for (size_t c = 0; c < 2; ++c) {
      auto location = view->locate(c, row);
      ASSERT_TRUE(location.has_value());
      EXPECT_EQ(location->chunk_index, chunks[c][row]);
      EXPECT_EQ(location->offset, offsets[c][row]);
      ASSERT_TRUE(batch->values[c].is_scalar());
      auto expected = (c == 0 ? sliced : text)->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      EXPECT_TRUE(batch->values[c].Equals(arrow::Datum(expected.ValueOrDie())));
    }
  }
  EXPECT_FALSE(view->locate(2, 0).has_value());
  EXPECT_FALSE(view->locate(0, 6).has_value());
  EXPECT_FALSE(view->locate(0, UINT64_MAX).has_value());
  EXPECT_EQ(view->read_row(6), nullptr);
  EXPECT_EQ(view->read_row(UINT64_MAX), nullptr);

  EXPECT_EQ(make_owned_view(nullptr), nullptr);
  EXPECT_EQ(make_owned_view(
                arrow::Table::Make(arrow::schema({}), arrow::ArrayVector{}, 0)),
            nullptr);
  EXPECT_EQ(make_owned_view(arrow::Table::Make(
                schema, arrow::ArrayVector{sliced, text->Slice(0, 5)})),
            nullptr);
  EXPECT_EQ(make_owned_view(
                arrow::Table::Make(schema, arrow::ArrayVector{text, sliced})),
            nullptr);
  auto empty = make_owned_view(table->Slice(0, 0));
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(empty->num_rows(), 0);
  EXPECT_EQ(empty->read_row(0), nullptr);
  EXPECT_FALSE(empty->locate(0, 0).has_value());
}

TEST_P(ForwardFilterReadTest, BindFailureAndUnfilteredDeletionRemainUnchanged) {
  create(24);
  ASSERT_TRUE(segment_->Delete(uint64_t{10007}).ok());
  reopen_persisted();
  auto invalid = filter({"i32"}, cp::field_ref("missing"), context());
  EXPECT_FALSE(invalid->compute_filter().ok());
  auto info = std::make_shared<sqlengine::QueryInfo>();
  sqlengine::DocFilter unfiltered(segment_, info, nullptr, nullptr, context());
  ASSERT_TRUE(unfiltered.compute_filter().ok());
  EXPECT_FALSE(unfiltered.is_filtered(1));
  EXPECT_TRUE(unfiltered.is_filtered(7));
}

INSTANTIATE_TEST_SUITE_P(PersistedStores, ForwardFilterReadTest,
                         testing::Bool());
}  // namespace zvec::test
