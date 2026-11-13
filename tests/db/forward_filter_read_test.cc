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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include <zvec/db/config.h>
#include "db/common/file_helper.h"
#include "db/sqlengine/planner/doc_filter.h"
#include "index/utils/utils.h"

namespace zvec::test {
namespace cp = arrow::compute;

namespace {
class CaptureBarrier final : public ailego::Logger {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  void log(int, const char *, int, const char *format, va_list) override {
    if (!std::strstr(format, "Forward filter capture:")) return;
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [&] { return released_; });
  }
  bool wait_entered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(10),
                        [&] { return entered_; });
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_{false}, released_{false};
};
}  // namespace

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

  ForwardReadView::Ptr capture(const std::vector<std::string> &columns) {
    auto result = Segment::make_forward_read_view(segment_, columns);
    EXPECT_TRUE(result.ok()) << result.status().ToString();
    return result.ok() ? result.MoveValueUnsafe() : nullptr;
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
      const std::vector<std::string> &columns, cp::Expression expression) {
    auto info = std::make_shared<sqlengine::QueryInfo>();
    for (const auto &column : columns) {
      info->add_forward_filter_schema_ptr(column, schema_->get_field(column));
    }
    return std::make_unique<sqlengine::DocFilter>(
        segment_, info, nullptr, std::make_unique<cp::Expression>(expression));
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

TEST_P(ForwardFilterReadTest, MutableSealedAndDirectCapture) {
  create(24);
  ASSERT_NE(capture({"i32"}), nullptr);
  ASSERT_TRUE(segment_->flush().ok());
  ASSERT_NE(capture({"i32"}), nullptr);
  ASSERT_TRUE(segment_->dump().ok());
  ASSERT_NE(capture({"i32"}), nullptr);
  segment_->remove_writing_forward_block();
  ASSERT_NE(capture({"i32"}), nullptr);
  reopen_persisted(false);
  ASSERT_NE(capture({"i32"}), nullptr);
  auto constant = capture({});
  ASSERT_NE(constant, nullptr);
  EXPECT_EQ(constant->num_rows(), 24);
  EXPECT_EQ(constant->read_row(0)->values.size(), 0u);
  EXPECT_FALSE(Segment::make_forward_read_view(segment_, {"missing"}).ok());
  EXPECT_FALSE(Segment::make_forward_read_view(nullptr, {"i32"}).ok());
}

TEST_P(ForwardFilterReadTest, MemoryTailAppendFlushAndRetirement) {
  options_.max_buffer_size_ = 1024 * 1024;
  create(97);
  auto columns = schema_->forward_field_names();
  auto view = capture(columns);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->num_rows(), 97);
  ASSERT_GT(view->table()->column(0)->num_chunks(), 1);
  std::vector<std::string> expected;
  for (int row = 0; row < 97; ++row) {
    expected.push_back(view->read_row(row)->ToString());
  }
  // Seed an unrelated tombstone so the existing live delete filter is active.
  ASSERT_TRUE(segment_->Delete(uint64_t{10096}).ok());
  auto candidate = filter({"i32"}, cp::literal(true));
  auto constant = filter({}, cp::literal(true));
  ASSERT_TRUE(candidate->compute_filter().ok());
  ASSERT_TRUE(constant->compute_filter().ok());
  auto added = TestHelper::CreateDoc(97, *schema_);
  ASSERT_TRUE(segment_->insert(added).ok());
  EXPECT_TRUE(candidate->is_filtered(97));
  EXPECT_TRUE(candidate->is_filtered(UINT64_MAX));
  EXPECT_TRUE(constant->is_filtered(97));
  EXPECT_FALSE(candidate->is_filtered(1));
  ASSERT_TRUE(segment_->Delete(uint64_t{10001}).ok());
  EXPECT_TRUE(candidate->is_filtered(1));
  ASSERT_TRUE(segment_->flush().ok());
  EXPECT_EQ(capture(columns)->num_rows(), 98);
  for (int row = 0; row < 97; ++row) {
    EXPECT_EQ(view->read_row(row)->ToString(), expected[row]);
  }
  candidate.reset();
  constant.reset();
  // Keep only sliced arrays, not the view/table/store wrapper.
  auto slice = view->table()->Slice(1, 95);
  view.reset();
  segment_.reset();
  auto retained = make_owned_view(slice);
  ASSERT_NE(retained, nullptr);
  for (int row = 0; row < 95; ++row) {
    EXPECT_EQ(retained->read_row(row)->ToString(), expected[row + 1]);
  }
}

TEST_P(ForwardFilterReadTest, CaptureRacesAppendAndPublicFlush) {
  options_.max_buffer_size_ = 1024 * 1024;
  create(24);
  std::atomic<bool> start{false};
  auto writer = std::async(std::launch::async, [&] {
    while (!start.load()) std::this_thread::yield();
    for (int i = 24; i < 124; ++i) {
      auto doc = TestHelper::CreateDoc(i, *schema_);
      if (!segment_->insert(doc).ok()) return false;
      if (i % 10 == 0 && !segment_->flush().ok()) return false;
    }
    return true;
  });
  std::vector<ForwardReadView::Ptr> views;
  start = true;
  for (int i = 0; i < 30; ++i) {
    auto view = capture({GLOBAL_DOC_ID, "text", "items"});
    ASSERT_NE(view, nullptr);
    views.push_back(std::move(view));
  }
  ASSERT_TRUE(writer.get());
  segment_.reset();
  for (const auto &view : views) {
    for (int64_t row = 0; row < view->num_rows(); ++row) {
      auto batch = view->read_row(row);
      ASSERT_NE(batch, nullptr);
      EXPECT_EQ(batch->values[0].scalar_as<arrow::UInt64Scalar>().value,
                10000u + row);
    }
  }
}

TEST_P(ForwardFilterReadTest, RetirementBetweenCaptureAndPersistedDecode) {
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, "text", "items"};
  std::vector<std::string> expected;
  for (int row = 0; row < 2053; ++row) {
    auto batch = segment_->fetch(columns, row);
    ASSERT_NE(batch, nullptr);
    expected.push_back(batch->ToString());
  }
  auto barrier = std::make_shared<CaptureBarrier>();
  auto previous = ailego::LoggerBroker::Register(barrier);
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_DEBUG);
  auto pending = std::async(std::launch::async, [&] {
    return Segment::make_forward_read_view(segment_, columns);
  });
  EXPECT_TRUE(barrier->wait_entered());
  // Explicit retirement clears the segment's store owners even though the
  // capture still retains the segment. shared_ptr<Segment> alone is not enough.
  auto retiring =
      std::async(std::launch::async, [&] { return segment_->destroy(); });
  EXPECT_EQ(retiring.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  barrier->release();
  auto retired = retiring.get();
  auto captured = pending.get();
  ailego::LoggerBroker::Register(previous);
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
  ASSERT_TRUE(retired.ok()) << retired.message();
  ASSERT_TRUE(captured.ok()) << captured.status().ToString();
  // A new capture cannot use the retired stores; only the earlier retained
  // publication may complete. This distinguishes ownership from live access.
  EXPECT_FALSE(Segment::make_forward_read_view(segment_, columns).ok());
  auto view = captured.MoveValueUnsafe();
  segment_.reset();
  for (int row = 0; row < 2053; ++row) {
    auto batch = view->read_row(row);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->ToString(), expected[row]);
  }
}

TEST_P(ForwardFilterReadTest,
       ConcurrentWideMultiSegmentProjectionsReleaseMemory) {
  constexpr int kSegments = 3, kRows = 257, kColumns = 6, kReaders = 4;
  std::vector<std::string> columns{GLOBAL_DOC_ID};
  for (int c = 0; c < kColumns; ++c) {
    auto name = "wide_" + std::to_string(c);
    columns.push_back(name);
    schema_->add_field(std::make_shared<FieldSchema>(name, DataType::STRING));
  }
  auto value = [](int s, int row, int c) {
    return std::to_string(s) + ":" + std::to_string(row) + ":" +
           std::string(1024, static_cast<char>('a' + c));
  };
  options_.max_buffer_size_ = 1024 * 1024;
  std::vector<Segment::Ptr> segments;
  for (int s = 0; s < kSegments; ++s) {
    auto created =
        Segment::CreateAndOpen(path_, *schema_, s, 10000 + s * kRows, id_map_,
                               deletes_, versions_, options_);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    auto segment = created.value();
    for (int row = 0; row < kRows; ++row) {
      Doc doc;
      doc.set_pk("wide_" + std::to_string(s) + "_" + std::to_string(row));
      for (int c = 0; c < kColumns; ++c)
        doc.set<std::string>(columns[c + 1], value(s, row, c));
      ASSERT_TRUE(segment->insert(doc).ok());
    }
    ASSERT_TRUE(segment->dump().ok());
    segment->remove_writing_forward_block();
    segments.push_back(std::move(segment));
  }
  using Views = std::vector<ForwardReadView::Ptr>;
  const auto before = arrow::default_memory_pool()->bytes_allocated();
  std::vector<std::future<arrow::Result<Views>>> pending;
  for (int reader = 0; reader < kReaders; ++reader) {
    pending.push_back(
        std::async(std::launch::async, [&]() -> arrow::Result<Views> {
          Views views;
          for (const auto &segment : segments) {
            ARROW_ASSIGN_OR_RAISE(
                auto view, Segment::make_forward_read_view(segment, columns));
            views.push_back(std::move(view));
          }
          return views;
        }));
  }
  std::vector<Views> held;
  for (auto &future : pending) {
    auto captured = future.get();
    ASSERT_TRUE(captured.ok()) << captured.status().ToString();
    held.push_back(captured.MoveValueUnsafe());
  }
  if (!GetParam()) {
    // Parquet projections are independently owned and outside the cache budget.
    EXPECT_GE(arrow::default_memory_pool()->bytes_allocated() - before,
              int64_t{kSegments} * kRows * kColumns * 1024 * kReaders);
  }
  for (const auto &views : held) {
    for (int s = 0; s < kSegments; ++s) {
      ASSERT_TRUE(views[s]->table()->ValidateFull().ok());
      for (int row : {0, 128, kRows - 1}) {
        auto batch = views[s]->read_row(row);
        ASSERT_NE(batch, nullptr);
        EXPECT_EQ(batch->values[0].scalar_as<arrow::UInt64Scalar>().value,
                  uint64_t{10000} + s * kRows + row);
        for (int c = 0; c < kColumns; ++c)
          EXPECT_EQ(batch->values[c + 1].scalar()->ToString(),
                    value(s, row, c));
      }
    }
  }
  held.clear();
  EXPECT_EQ(arrow::default_memory_pool()->bytes_allocated(), before);
}

TEST_P(ForwardFilterReadTest, EmptySegmentAndMutableTail) {
  options_.max_buffer_size_ = 16 * 1024 * 1024;
  create(0);
  auto empty = capture({"i32"});
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(empty->num_rows(), 0);
  auto candidate = filter({"i32"}, cp::literal(true));
  ASSERT_TRUE(candidate->compute_filter().ok());
  auto doc = TestHelper::CreateDoc(1, *schema_);
  ASSERT_TRUE(segment_->insert(doc).ok());
  EXPECT_TRUE(candidate->is_filtered(0));
  auto tail = capture({"i32", "text", "items"});
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->num_rows(), 1);
  auto expected = tail->read_row(0)->ToString();
  ASSERT_TRUE(segment_->flush().ok());
  candidate.reset();
  segment_.reset();
  EXPECT_EQ(tail->read_row(0)->ToString(), expected);
}

TEST_P(ForwardFilterReadTest, ReorderedProjectionBoundariesAndOwnedView) {
  create();
  reopen_persisted();
  ASSERT_FALSE(segment_->has_identity_doc_ids());
  ASSERT_GT(segment_->meta()->persisted_blocks().size(), 1u);
  auto columns = schema_->forward_field_names();
  std::reverse(columns.begin(), columns.end());
  auto view = capture(columns);
  ASSERT_NE(view, nullptr);
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
    auto candidate = filter(columns, expr);
    auto status = candidate->compute_filter();
    ASSERT_TRUE(status.ok()) << status.message();
    auto bound =
        expr.Bind(*segment_->fetch(columns, std::vector<int>{})->schema());
    ASSERT_TRUE(bound.ok());
    for (int row = 0; row < 2053; ++row) {
      auto batch = segment_->fetch(columns, row);
      ASSERT_NE(batch, nullptr);
      auto result = cp::ExecuteScalarExpression(*bound, *batch);
      ASSERT_TRUE(result.ok());
      const auto &value = result->scalar_as<arrow::BooleanScalar>();
      EXPECT_EQ(row == 7 || !value.is_valid || !value.value,
                candidate->is_filtered(row))
          << row;
    }
    EXPECT_TRUE(candidate->is_filtered(7));
  }
  auto negative = filter({"i32"}, expressions[2]);
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
  auto view = capture(columns);
  ASSERT_NE(view, nullptr);
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
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  const auto columns = schema_->forward_field_names();
  auto view = capture(columns);
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
  auto sliced = view->table()->Slice(1, 2051);
  view.reset();
  view = make_owned_view(sliced);
  sliced.reset();
  // No store, original table, view, or unsliced array remains as a hidden
  // owner.
  for (int row = 1; row < 2052; ++row) {
    auto batch = view->read_row(row - 1);
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
  EXPECT_NE(make_owned_view(
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
  auto invalid = filter({"i32"}, cp::field_ref("missing"));
  EXPECT_FALSE(invalid->compute_filter().ok());
  EXPECT_TRUE(invalid->is_filtered(1));
  auto missing_info = std::make_shared<sqlengine::QueryInfo>();
  // Simulate analyzed metadata whose column cannot be captured by the segment.
  missing_info->add_forward_filter_schema_ptr("missing",
                                              schema_->get_field("i32"));
  auto missing = std::make_unique<sqlengine::DocFilter>(
      segment_, missing_info, nullptr,
      std::make_unique<cp::Expression>(cp::literal(true)));
  auto status = missing->compute_filter();
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("Capture forward view failed"),
            std::string::npos);
  EXPECT_TRUE(missing->is_filtered(1));
  auto info = std::make_shared<sqlengine::QueryInfo>();
  sqlengine::DocFilter unfiltered(segment_, info, nullptr, nullptr);
  ASSERT_TRUE(unfiltered.compute_filter().ok());
  EXPECT_FALSE(unfiltered.is_filtered(1));
  EXPECT_TRUE(unfiltered.is_filtered(7));
}

INSTANTIATE_TEST_SUITE_P(PersistedStores, ForwardFilterReadTest,
                         testing::Bool());
}  // namespace zvec::test
