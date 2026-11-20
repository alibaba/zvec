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
#include <functional>
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
    if (!std::strstr(format, "Forward read captured:")) return;
    std::unique_lock<std::mutex> lock(mutex_);
    ++entered_;
    cv_.notify_all();
    cv_.wait(lock, [&] { return released_; });
  }
  bool wait_entered(size_t count = 1) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(10),
                        [&] { return entered_ >= count; });
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t entered_{0};
  bool released_{false};
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
    auto result = segment_->prepare_forward_read(columns, false);
    EXPECT_TRUE(result.ok()) << result.status().ToString();
    return result.ok() ? std::move(result->view) : nullptr;
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

  static bool uses_view(const sqlengine::DocFilter &filter) {
    return filter.forward_view_ != nullptr;
  }

  static bool uses_sealed_deletion(const sqlengine::DocFilter &filter) {
    return filter.sealed_deletion_ != nullptr;
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
    return ForwardReadView::Make(std::move(table));
  }

  // The view's physical columns as a table (virtual columns excluded).
  static TablePtr view_table(const ForwardReadView &view) {
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    for (int i = 0; i < view.num_columns(); ++i) {
      columns.push_back(view.column(i)->data());
    }
    return arrow::Table::Make(view.schema(), columns, view.num_rows());
  }

  // Reads a non-null value through the raw spans, as a compiled evaluator
  // would, without touching any Arrow object.
  static std::string raw_value(const ForwardColumn &column, uint64_t row) {
    const auto location = column.locate(row);
    EXPECT_TRUE(location.has_value());
    if (!location) return {};
    const auto &span = column.spans().at(location->chunk_index);
    const int64_t i = span.offset + location->offset;
    if (span.validity && !arrow::bit_util::GetBit(span.validity, i)) {
      return "null";
    }
    switch (column.layout()) {
      case ForwardColumn::Layout::kBitmap:
        return arrow::bit_util::GetBit(span.values, i) ? "true" : "false";
      case ForwardColumn::Layout::kBinary: {
        const auto *offsets = reinterpret_cast<const int32_t *>(span.offsets);
        return std::string(
            reinterpret_cast<const char *>(span.values) + offsets[i],
            offsets[i + 1] - offsets[i]);
      }
      case ForwardColumn::Layout::kLargeBinary: {
        const auto *offsets = reinterpret_cast<const int64_t *>(span.offsets);
        return std::string(
            reinterpret_cast<const char *>(span.values) + offsets[i],
            offsets[i + 1] - offsets[i]);
      }
      case ForwardColumn::Layout::kFixedWidth:
        if (column.type()->id() == arrow::Type::INT32) {
          return std::to_string(
              reinterpret_cast<const int32_t *>(span.values)[i]);
        }
        if (column.type()->id() == arrow::Type::INT64) {
          return std::to_string(
              reinterpret_cast<const int64_t *>(span.values)[i]);
        }
        break;
      default:
        break;
    }
    ADD_FAILURE() << "unsupported layout in test";
    return {};
  }

  const std::string path_{"./forward_filter_read_test_collection"};
  CollectionSchema::Ptr schema_;
  IDMap::Ptr id_map_;
  DeleteStore::Ptr deletes_;
  VersionManager::Ptr versions_;
  SegmentOptions options_{};
  Segment::Ptr segment_;
};

class IpcForwardFilterReadTest : public ForwardFilterReadTest {};

TEST_P(IpcForwardFilterReadTest, ConstantPredicateView) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto constant = capture({});
  ASSERT_NE(constant, nullptr);
  EXPECT_EQ(constant->num_rows(), 24);
  EXPECT_EQ(constant->read_row(0)->values.size(), 0u);
}

TEST_P(IpcForwardFilterReadTest, SealedViewIsReusedAcrossQueriesAndDeletes) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  // Activate the existing live deletion filter before preparing queries.
  ASSERT_TRUE(segment_->Delete(uint64_t{10023}).ok());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto first = capture({"i32"});
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(capture({"i32"})->column(0), first->column(0));
  auto equals = filter({"i32"}, cp::call("equal", {cp::field_ref("i32"),
                                                   cp::literal(int32_t{2})}));
  auto greater = filter(
      {"i32"},
      cp::call("greater", {cp::field_ref("i32"), cp::literal(int32_t{2})}));
  ASSERT_TRUE(equals->compute_filter().ok());
  ASSERT_TRUE(greater->compute_filter().ok());
  EXPECT_FALSE(equals->is_filtered(2));
  EXPECT_TRUE(greater->is_filtered(2));
  EXPECT_FALSE(greater->is_filtered(4));
  EXPECT_EQ(capture({"i32"})->column(0), first->column(0));

  // Sealed scalar data does not change when the live delete mask changes.
  // Collection writes update this shared mask through the writing segment.
  deletes_->mark_deleted(uint64_t{10002});
  EXPECT_TRUE(equals->is_filtered(2));
  EXPECT_EQ(capture({"i32"})->column(0), first->column(0));
  EXPECT_EQ(first->read_row(2)->values[0].scalar_as<arrow::Int32Scalar>().value,
            2);
}

TEST_P(IpcForwardFilterReadTest, ColumnsAreSharedAcrossProjections) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto first = capture({"text", "i32"});
  ASSERT_NE(first, nullptr);
  auto text = std::weak_ptr<const ForwardColumn>(first->column(0));
  auto i32 = std::weak_ptr<const ForwardColumn>(first->column(1));
  first.reset();
  // The segment, not the query, owns built columns.
  EXPECT_FALSE(text.expired());
  EXPECT_FALSE(i32.expired());
  // Another order, a subset and a superset compose the same columns; only
  // the genuinely new column is built.
  auto second = capture({"i32", "text"});
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->column(0), i32.lock());
  EXPECT_EQ(second->column(1), text.lock());
  auto subset = capture({"i32"});
  ASSERT_NE(subset, nullptr);
  EXPECT_EQ(subset->column(0), i32.lock());
  auto superset = capture({"i64", "text", "i32"});
  ASSERT_NE(superset, nullptr);
  EXPECT_EQ(superset->column(1), text.lock());
  EXPECT_EQ(superset->column(2), i32.lock());
  EXPECT_EQ(capture({"i64"})->column(0), superset->column(0));

  // Each view keeps its own column order and schema.
  auto expected = segment_->fetch({"i32", "text"}, 2);
  auto row = second->read_row(2);
  ASSERT_NE(row, nullptr);
  ASSERT_NE(expected, nullptr);
  EXPECT_EQ(row->ToString(), expected->ToString());
  EXPECT_EQ(second->schema()->field(0)->name(), "i32");
  EXPECT_EQ(superset->schema()->field(0)->name(), "i64");

  auto constant = capture({});
  ASSERT_NE(constant, nullptr);
  EXPECT_EQ(constant->num_columns(), 0);
  // A failed projection leaves published columns available.
  EXPECT_FALSE(segment_->prepare_forward_read({"i32", "missing"}, false).ok());
  EXPECT_EQ(capture({"i32"})->column(0), i32.lock());
  EXPECT_EQ(second->read_row(2)->ToString(), expected->ToString());
}

TEST_P(IpcForwardFilterReadTest, RowIdIsVirtualAndCopiesNoData) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  const auto bytes = arrow::default_memory_pool()->bytes_allocated();
  auto view = capture({LOCAL_ROW_ID, "i32"});
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(arrow::default_memory_pool()->bytes_allocated(), bytes);
  EXPECT_EQ(view->column(0)->layout(), ForwardColumn::Layout::kRowId);
  EXPECT_EQ(view->column(0)->data(), nullptr);
  EXPECT_EQ(capture({LOCAL_ROW_ID})->column(0), view->column(0));
  EXPECT_FALSE(view->locate(0, 5).has_value());
  for (uint64_t row : {0u, 5u, 23u}) {
    auto batch = view->read_row(row);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->values[0].scalar_as<arrow::UInt64Scalar>().value, row);
    auto expected = segment_->fetch({LOCAL_ROW_ID, "i32"}, row);
    ASSERT_NE(expected, nullptr);
    EXPECT_TRUE(batch->values[1].Equals(expected->values[1]));
  }
  EXPECT_EQ(view->read_row(24), nullptr);
}

TEST_P(IpcForwardFilterReadTest, RawSpansMatchArrowValues) {
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto view = capture({"i32", "i64", "flag", "text", "items"});
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->column(0)->layout(), ForwardColumn::Layout::kFixedWidth);
  EXPECT_EQ(view->column(0)->byte_width(), 4);
  EXPECT_EQ(view->column(1)->byte_width(), 8);
  EXPECT_EQ(view->column(2)->layout(), ForwardColumn::Layout::kBitmap);
  EXPECT_EQ(view->column(3)->layout(), ForwardColumn::Layout::kBinary);
  // Nested values stay Arrow-only: chunks are located, but no raw spans.
  EXPECT_EQ(view->column(4)->layout(), ForwardColumn::Layout::kOpaque);
  EXPECT_TRUE(view->column(4)->spans().empty());
  EXPECT_FALSE(view->column(4)->arrays().empty());
  for (int c = 0; c < 4; ++c) {
    const auto &column = *view->column(c);
    ASSERT_EQ(column.spans().size(), column.arrays().size());
    ASSERT_EQ(column.spans().size(), column.chunk_ends().size());
    EXPECT_EQ(column.chunk_ends().back(), view->num_rows());
  }
  for (uint64_t row = 0; row < static_cast<uint64_t>(view->num_rows()); ++row) {
    for (int c = 0; c < 4; ++c) {
      auto scalar = view->column(c)->scalar(row);
      ASSERT_TRUE(scalar.ok());
      const auto expected =
          (*scalar)->is_valid ? (*scalar)->ToString() : "null";
      ASSERT_EQ(raw_value(*view->column(c), row), expected)
          << "row " << row << " column " << c;
    }
  }
}

TEST_P(IpcForwardFilterReadTest, IncompleteBlockFallsBackToLockedFetch) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_TRUE(segment_->dump().ok());
  segment_->remove_writing_forward_block();
  auto good = std::make_shared<SegmentMeta>(*segment_->meta());
  auto bad = std::make_shared<SegmentMeta>(*good);
  auto &blocks = bad->persisted_blocks();
  const auto block = std::find_if(
      blocks.begin(), blocks.end(),
      [](const BlockMeta &b) { return b.type() == BlockType::SCALAR; });
  ASSERT_NE(block, blocks.end());
  block->set_doc_count(block->doc_count() + 1);
  segment_.reset();
  auto options = options_;
  options.read_only_ = true;
  // The read state is an optimization; it never makes a segment unopenable.
  auto opened = Segment::Open(path_, *schema_, *bad, id_map_, deletes_,
                              versions_, options);
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  segment_ = opened.value();
  auto state = segment_->prepare_forward_read({"i32"}, false);
  ASSERT_TRUE(state.ok()) << state.status().ToString();
  EXPECT_EQ(state->view, nullptr);

  // A reload whose state cannot be prepared still succeeds, and a later
  // reload with consistent metadata restores the sealed view.
  const std::unordered_map<std::string, VectorColumnIndexer::Ptr> none;
  ASSERT_TRUE(segment_->reload_vector_index(*schema_, good, none, none).ok());
  EXPECT_NE(capture({"i32"}), nullptr);
  ASSERT_TRUE(segment_->reload_scalar_index(*schema_, bad, nullptr).ok());
  state = segment_->prepare_forward_read({"i32"}, false);
  ASSERT_TRUE(state.ok()) << state.status().ToString();
  EXPECT_EQ(state->view, nullptr);
  ASSERT_TRUE(segment_->reload_fts_index(*schema_, good, nullptr).ok());
  EXPECT_NE(capture({"i32"}), nullptr);
}

TEST_P(IpcForwardFilterReadTest, FailedDdlKeepsThePublishedState) {
#ifdef _WIN32
  GTEST_SKIP() << "Windows cannot remove a mapped block file";
#endif
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  ASSERT_TRUE(segment_
                  ->add_column(std::make_shared<FieldSchema>(
                                   "extra", DataType::INT32, true),
                               "7", AddColumnOptions())
                  .ok());
  // Make drop_column fail after it has started changing the segment: the
  // single-column block file of "extra" can no longer be removed.
  const auto &blocks = segment_->meta()->persisted_blocks();
  const auto block =
      std::find_if(blocks.begin(), blocks.end(), [](const BlockMeta &b) {
        return b.type() == BlockType::SCALAR && b.columns().size() == 1 &&
               b.columns()[0] == "extra";
      });
  ASSERT_NE(block, blocks.end());
  ASSERT_TRUE(FileHelper::RemoveFile(
      FileHelper::MakeForwardBlockPath(path_, segment_->id(), block->id())));
  EXPECT_FALSE(segment_->drop_column("extra").ok());
  // Columns unaffected by the failed DDL keep the sealed fast path.
  auto view = capture({"i32"});
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->read_row(2)->values[0].scalar_as<arrow::Int32Scalar>().value,
            2);
}

TEST_P(IpcForwardFilterReadTest,
       InvalidColumnDoesNotBlockIndependentProjections) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_TRUE(segment_->dump().ok());
  segment_->remove_writing_forward_block();
  // Reopen an INT32 column with incompatible schema metadata.
  ASSERT_TRUE(schema_
                  ->alter_field("i32", std::make_shared<FieldSchema>(
                                           "i32", DataType::INT64, true))
                  .ok());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted(false));
  EXPECT_FALSE(segment_->prepare_forward_read({"i32"}, false).ok());
  auto view = capture({"text", GLOBAL_DOC_ID});
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->read_row(2)->values[1].scalar_as<arrow::UInt64Scalar>().value,
            10002u);
}

TEST_P(IpcForwardFilterReadTest, ConcurrentProjectionsSharePublishedColumns) {
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto existing = capture({"u64"});
  ASSERT_NE(existing, nullptr);
  const std::vector<std::vector<std::string>> projections{
      {"i32", "text"}, {"text", "i32"}, {"i32", "i64"}};
  auto barrier = std::make_shared<CaptureBarrier>();
  auto previous = ailego::LoggerBroker::Register(barrier);
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_DEBUG);
  constexpr int kReaders = 8;
  std::vector<std::future<arrow::Result<Segment::ForwardReadState>>> pending;
  for (int i = 0; i < kReaders; ++i) {
    pending.push_back(std::async(std::launch::async, [&, i] {
      return segment_->prepare_forward_read(projections[i % projections.size()],
                                            false);
    }));
  }
  // Pause all readers after capturing the state, before projecting columns.
  EXPECT_TRUE(barrier->wait_entered(kReaders));
  barrier->release();
  std::vector<arrow::Result<Segment::ForwardReadState>> states;
  for (auto &reader : pending) {
    states.push_back(reader.get());
  }
  ailego::LoggerBroker::Register(previous);
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
  // Every projection, including the first, uses the columns published at open.
  auto later = capture({"i32", "text", "i64"});
  ASSERT_NE(later, nullptr);
  EXPECT_EQ(capture({"u64"})->column(0), existing->column(0));
  for (int i = 0; i < kReaders; ++i) {
    const auto &state = states[i];
    ASSERT_TRUE(state.ok()) << state.status().ToString();
    ASSERT_NE(state->view, nullptr);
    const auto &projection = projections[i % projections.size()];
    ASSERT_EQ(state->view->num_columns(), projection.size());
    for (size_t c = 0; c < projection.size(); ++c) {
      const int index = later->schema()->GetFieldIndex(projection[c]);
      EXPECT_EQ(state->view->column(c), later->column(index));
      EXPECT_EQ(state->schema->field(c)->name(), projection[c]);
    }
    EXPECT_EQ(state->num_rows, 2053u);
    auto expected = segment_->fetch(projection, 2);
    auto row = state->view->read_row(2);
    ASSERT_NE(expected, nullptr);
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->ToString(), expected->ToString());
  }
}

TEST_P(IpcForwardFilterReadTest, DdlReleasesColumnsAndRetainsOldReaders) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto first = capture({"i32"});
  ASSERT_NE(first, nullptr);
  auto weak = std::weak_ptr<const ForwardColumn>(first->column(0));
  ASSERT_TRUE(segment_
                  ->alter_column("i32",
                                 std::make_shared<FieldSchema>(
                                     "i32", DataType::INT64, true),
                                 AlterColumnOptions())
                  .ok());
  // A reader that captured before the DDL keeps its own owned buffers.
  EXPECT_EQ(first->schema()->field(0)->type()->id(), arrow::Type::INT32);
  EXPECT_EQ(first->read_row(2)->values[0].scalar_as<arrow::Int32Scalar>().value,
            2);
  first.reset();
  // DDL replaced the segment's state before the next capture.
  EXPECT_TRUE(weak.expired());
  auto replacement = capture({"i32"});
  ASSERT_NE(replacement, nullptr);
  EXPECT_EQ(replacement->schema()->field(0)->type()->id(), arrow::Type::INT64);
  EXPECT_EQ(
      replacement->read_row(2)->values[0].scalar_as<arrow::Int64Scalar>().value,
      2);
  EXPECT_EQ(capture({"i32"})->column(0), replacement->column(0));

  // Dropping the column removes its single-column block file. No segment
  // entry may keep that file mapped; only this explicit reader does.
  auto dropped = std::weak_ptr<const ForwardColumn>(replacement->column(0));
  auto text = std::weak_ptr<const ForwardColumn>(capture({"text"})->column(0));
  ASSERT_TRUE(segment_->drop_column("i32").ok());
  EXPECT_FALSE(segment_->prepare_forward_read({"i32"}, false).ok());
  EXPECT_TRUE(text.expired());  // Unrelated columns are released as well.
  ASSERT_TRUE(segment_
                  ->add_column(std::make_shared<FieldSchema>(
                                   "i32", DataType::INT32, true),
                               "42", AddColumnOptions())
                  .ok());
  auto added = capture({"i32"});
  ASSERT_NE(added, nullptr);
  EXPECT_EQ(added->read_row(2)->values[0].scalar_as<arrow::Int32Scalar>().value,
            42);
  EXPECT_EQ(capture({"i32"})->column(0), added->column(0));
  EXPECT_EQ(
      replacement->read_row(2)->values[0].scalar_as<arrow::Int64Scalar>().value,
      2);  // Earlier queries keep their original owned buffers.
  replacement.reset();
  EXPECT_TRUE(dropped.expired());
}

TEST_P(IpcForwardFilterReadTest, LifecycleChangesReplacePublishedColumns) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  const std::unordered_map<std::string, VectorColumnIndexer::Ptr> none;
  const std::vector<std::pair<std::string, std::function<Status()>>>
      invalidations = {
          {"reload_vector_index",
           [&] {
             return segment_->reload_vector_index(*schema_, segment_->meta(),
                                                  none, none);
           }},
          {"reload_scalar_index",
           [&] {
             return segment_->reload_scalar_index(*schema_, segment_->meta(),
                                                  nullptr);
           }},
          {"reload_fts_index",
           [&] {
             return segment_->reload_fts_index(*schema_, segment_->meta(),
                                               nullptr);
           }},
          {"add_column",
           [&] {
             return segment_->add_column(
                 std::make_shared<FieldSchema>("extra", DataType::INT32, true),
                 "7", AddColumnOptions());
           }},
      };
  for (const auto &[name, invalidate] : invalidations) {
    SCOPED_TRACE(name);
    std::weak_ptr<const ForwardColumn> weak;
    {
      auto view = capture({"i32"});
      ASSERT_NE(view, nullptr);
      weak = view->column(0);
    }
    ASSERT_FALSE(weak.expired());  // Only the segment's catalog holds it.
    auto status = invalidate();
    ASSERT_TRUE(status.ok()) << status.message();
    // The writer replaced the state before returning.
    EXPECT_TRUE(weak.expired());
    auto view = capture({"i32"});
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(
        view->read_row(2)->values[0].scalar_as<arrow::Int32Scalar>().value, 2);
  }
}

TEST_P(IpcForwardFilterReadTest,
       DdlBetweenCaptureAndProjectionKeepsCapturedSchema) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto barrier = std::make_shared<CaptureBarrier>();
  auto previous = ailego::LoggerBroker::Register(barrier);
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_DEBUG);
  auto pending = std::async(std::launch::async, [&] {
    return segment_->prepare_forward_read({"i32"}, false);
  });
  EXPECT_TRUE(barrier->wait_entered());
  auto changing = std::async(std::launch::async, [&] {
    return segment_->alter_column(
        "i32", std::make_shared<FieldSchema>("i32", DataType::INT64, true),
        AlterColumnOptions());
  });
  EXPECT_EQ(changing.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  barrier->release();
  auto changed = changing.get();
  auto old = pending.get();
  ailego::LoggerBroker::Register(previous);
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
  ASSERT_TRUE(changed.ok()) << changed.message();
  ASSERT_TRUE(old.ok()) << old.status().ToString();
  ASSERT_NE(old->view, nullptr);
  EXPECT_EQ(old->view->schema()->field(0)->type()->id(), arrow::Type::INT32);
  auto weak = std::weak_ptr<const ForwardColumn>(old->view->column(0));
  old->view.reset();
  EXPECT_TRUE(weak.expired());  // Only the overlapping query retained it.
  auto current = capture({"i32"});
  ASSERT_NE(current, nullptr);
  EXPECT_EQ(current->schema()->field(0)->type()->id(), arrow::Type::INT64);
  EXPECT_EQ(capture({"i32"})->column(0), current->column(0));
}

TEST_P(IpcForwardFilterReadTest,
       CloseReleasesColumnsWithoutInvalidatingReaders) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto view = capture({"i32"});
  ASSERT_NE(view, nullptr);
  auto weak = std::weak_ptr<const ForwardColumn>(view->column(0));
  ASSERT_TRUE(segment_->destroy().ok());
  EXPECT_FALSE(segment_->prepare_forward_read({"i32"}, false).ok());
  EXPECT_EQ(view->read_row(2)->values[0].scalar_as<arrow::Int32Scalar>().value,
            2);
  view.reset();
  EXPECT_TRUE(weak.expired());
}

TEST_P(ForwardFilterReadTest, HybridEligibilityAcrossFlushSealAndReopen) {
  options_.max_buffer_size_ = 100 * 1024 * 1024;
  ASSERT_NO_FATAL_FAILURE(create(24));
  auto verify = [&](bool expected_view) {
    const std::vector<std::string> columns{"text", "i32", "items"};
    const auto bytes = arrow::default_memory_pool()->bytes_allocated();
    auto state = segment_->prepare_forward_read(columns, false);
    ASSERT_TRUE(state.ok()) << state.status().ToString();
    EXPECT_EQ(state->view != nullptr, expected_view);
    EXPECT_EQ(state->num_rows, 24u);
    EXPECT_TRUE(state->schema->Equals(
        *segment_->fetch(columns, std::vector<int>{})->schema()));
    // Hybrid preparation never copies the mutable tail or decodes Parquet.
    EXPECT_EQ(arrow::default_memory_pool()->bytes_allocated(), bytes);
    auto candidate = filter({"i32"}, cp::literal(true));
    ASSERT_TRUE(candidate->compute_filter().ok());
    EXPECT_EQ(uses_view(*candidate), expected_view);
    EXPECT_FALSE(candidate->is_filtered(2));
    // Only an owning view has a boundary; it holds rows 0..23.
    if (expected_view) EXPECT_TRUE(candidate->is_filtered(24));
  };
  verify(false);
  ASSERT_TRUE(segment_->flush().ok());
  verify(false);  // No live tail, but the writing block can grow again.
  ASSERT_TRUE(segment_->dump().ok());
  verify(false);  // Publication has not yet removed the writing block.
  segment_->remove_writing_forward_block();
  verify(GetParam());
  ASSERT_NO_FATAL_FAILURE(reopen_persisted(false));
  verify(GetParam());  // Reopened segments do not rely on an in-memory flag.
  EXPECT_FALSE(segment_->prepare_forward_read({"missing"}, false).ok());
}

TEST_P(ForwardFilterReadTest, SealingPublishesOwningViews) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  EXPECT_EQ(capture({"i32"}), nullptr);
  ASSERT_TRUE(segment_->dump().ok());
  EXPECT_EQ(capture({"i32"}), nullptr);  // The writing block still exists.
  segment_->remove_writing_forward_block();
  EXPECT_FALSE(segment_->meta()->has_writing_forward_block());
  auto view = capture({"i32"});
  EXPECT_EQ(view != nullptr, GetParam());
  if (view) {
    EXPECT_EQ(
        view->read_row(2)->values[0].scalar_as<arrow::Int32Scalar>().value, 2);
  }
}

TEST_P(ForwardFilterReadTest, LockedDecisionSurvivesSealingAndReadsLive) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  auto expression = cp::call("greater_equal",
                             {cp::field_ref("i32"), cp::literal(int32_t{0})});
  auto candidate = filter({"i32"}, expression);
  auto constant = filter({}, cp::literal(true));
  ASSERT_TRUE(candidate->compute_filter().ok());
  ASSERT_TRUE(constant->compute_filter().ok());
  EXPECT_FALSE(uses_view(*candidate));
  auto added = TestHelper::CreateDoc(24, *schema_);
  ASSERT_TRUE(segment_->insert(added).ok());
  ASSERT_TRUE(segment_->flush().ok());
  ASSERT_TRUE(segment_->dump().ok());
  segment_->remove_writing_forward_block();
  EXPECT_FALSE(uses_view(*candidate));
  EXPECT_FALSE(candidate->is_filtered(2));
  EXPECT_TRUE(candidate->is_filtered(1));
  EXPECT_TRUE(candidate->is_filtered(0));
  // Locked fetch reads the live segment: the appended row is evaluated.
  EXPECT_FALSE(candidate->is_filtered(24));
  EXPECT_FALSE(constant->is_filtered(2));
  EXPECT_FALSE(constant->is_filtered(24));
  auto next = filter({"i32"}, expression);
  ASSERT_TRUE(next->compute_filter().ok());
  EXPECT_EQ(uses_view(*next), GetParam());
  EXPECT_FALSE(next->is_filtered(24));
}

TEST_P(ForwardFilterReadTest, HybridPreparationRacesAppendAndFlush) {
  ASSERT_NO_FATAL_FAILURE(create(24));
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
  start = true;
  for (int i = 0; i < 100; ++i) {
    auto candidate = filter(
        {"i32"},
        cp::call("equal", {cp::field_ref("i32"), cp::literal(int32_t{2})}));
    ASSERT_TRUE(candidate->compute_filter().ok());
    EXPECT_FALSE(uses_view(*candidate));
    EXPECT_FALSE(candidate->is_filtered(2));
    EXPECT_TRUE(candidate->is_filtered(1));
  }
  EXPECT_TRUE(writer.get());
}

TEST_P(ForwardFilterReadTest, FailedReprepareClearsPreviousReadState) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  ASSERT_NO_FATAL_FAILURE(reopen_persisted());
  auto candidate = filter({"i32"}, cp::literal(true));
  ASSERT_TRUE(candidate->compute_filter().ok());
  EXPECT_EQ(uses_view(*candidate), GetParam());
  ASSERT_TRUE(segment_->destroy().ok());
  EXPECT_FALSE(candidate->compute_filter().ok());
  EXPECT_FALSE(uses_view(*candidate));
  EXPECT_TRUE(candidate->is_filtered(2));
}

TEST_P(ForwardFilterReadTest, MutableLockedReadsSeeAppendsAndDeletes) {
  options_.max_buffer_size_ = 1024 * 1024;
  ASSERT_NO_FATAL_FAILURE(create(97));
  // Seed an unrelated tombstone so the existing live delete filter is active.
  ASSERT_TRUE(segment_->Delete(uint64_t{10096}).ok());
  auto candidate = filter({"i32"}, cp::literal(true));
  auto constant = filter({}, cp::literal(true));
  ASSERT_TRUE(candidate->compute_filter().ok());
  ASSERT_TRUE(constant->compute_filter().ok());
  // A writing segment keeps the locked deletion lookup.
  EXPECT_FALSE(uses_sealed_deletion(*candidate));
  auto added = TestHelper::CreateDoc(97, *schema_);
  ASSERT_TRUE(segment_->insert(added).ok());
  EXPECT_FALSE(candidate->is_filtered(97));
  EXPECT_FALSE(constant->is_filtered(97));
  EXPECT_FALSE(candidate->is_filtered(1));
  ASSERT_TRUE(segment_->Delete(uint64_t{10001}).ok());
  EXPECT_TRUE(candidate->is_filtered(1));
  ASSERT_TRUE(segment_->flush().ok());
  EXPECT_FALSE(uses_view(*candidate));
  EXPECT_FALSE(candidate->is_filtered(2));
  EXPECT_TRUE(candidate->is_filtered(1));
  EXPECT_FALSE(candidate->is_filtered(97));
  auto next = filter({"i32"}, cp::literal(true));
  ASSERT_TRUE(next->compute_filter().ok());
  EXPECT_FALSE(uses_view(*next));
  EXPECT_FALSE(next->is_filtered(97));
}

TEST_P(IpcForwardFilterReadTest, RetirementBetweenCaptureAndProjection) {
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
    return segment_->prepare_forward_read(columns, false);
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
  EXPECT_FALSE(segment_->prepare_forward_read(columns, false).ok());
  auto view = std::move(captured->view);
  ASSERT_NE(view, nullptr);
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
                auto state, segment->prepare_forward_read(columns, false));
            views.push_back(std::move(state.view));
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
  EXPECT_EQ(arrow::default_memory_pool()->bytes_allocated(), before);
  for (const auto &views : held) {
    for (int s = 0; s < kSegments; ++s) {
      ASSERT_EQ(views[s] != nullptr, GetParam());
      if (!views[s]) continue;
      ASSERT_TRUE(view_table(*views[s])->ValidateFull().ok());
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
  ASSERT_NO_FATAL_FAILURE(create(0));
  auto empty = segment_->prepare_forward_read({"i32"}, false);
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty->view, nullptr);
  EXPECT_EQ(empty->num_rows, 0u);
  auto candidate = filter({"i32"}, cp::literal(true));
  ASSERT_TRUE(candidate->compute_filter().ok());
  EXPECT_FALSE(uses_view(*candidate));
  auto doc = TestHelper::CreateDoc(1, *schema_);
  ASSERT_TRUE(segment_->insert(doc).ok());
  EXPECT_FALSE(candidate->is_filtered(0));
  auto next = filter({"i32"}, cp::literal(true));
  ASSERT_TRUE(next->compute_filter().ok());
  EXPECT_FALSE(uses_view(*next));
  EXPECT_FALSE(next->is_filtered(0));
  ASSERT_TRUE(segment_->flush().ok());
  EXPECT_FALSE(candidate->is_filtered(0));
  EXPECT_FALSE(next->is_filtered(0));
  EXPECT_FALSE(uses_view(*next));
}

TEST_P(ForwardFilterReadTest, ReorderedProjectionBoundariesAndOwnedView) {
  create();
  reopen_persisted();
  ASSERT_FALSE(segment_->has_identity_doc_ids());
  ASSERT_GT(segment_->meta()->persisted_blocks().size(), 1u);
  auto columns = schema_->forward_field_names();
  std::reverse(columns.begin(), columns.end());
  auto view = capture(columns);
  ASSERT_EQ(view != nullptr, GetParam());
  if (view) {
    EXPECT_EQ(view->num_rows(), 2053);
    EXPECT_TRUE(view->schema()->Equals(
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
  ASSERT_EQ(view != nullptr, GetParam());
  if (view) {
    EXPECT_EQ(view->num_rows(), 2053);
    EXPECT_NE(view->column(0)->data()->num_chunks(),
              view->column(1)->data()->num_chunks());
    // Irregular added-column batches need a search; no stride shortcut.
    EXPECT_EQ(view->column(0)->stride(), 0);
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

TEST_P(IpcForwardFilterReadTest, ViewOwnsBuffersAfterSegmentClose) {
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
  auto sliced = view_table(*view)->Slice(1, 2051);
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

TEST_P(IpcForwardFilterReadTest, ViewLocatesIndependentEmptyAndSlicedChunks) {
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
  // Chunk indices count non-empty chunks only (parallel to spans/arrays).
  const int chunks[2][6] = {{0, 0, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 2}};
  const int offsets[2][6] = {{0, 1, 0, 1, 2, 3}, {0, 0, 1, 2, 3, 0}};
  ASSERT_EQ(view->column(0)->spans().size(), 2u);
  ASSERT_EQ(view->column(1)->spans().size(), 3u);
  // Spans keep Arrow's array offset: slicing never copies or re-bases.
  EXPECT_EQ(view->column(0)->spans()[0].offset, 2);
  EXPECT_EQ(view->column(0)->spans()[1].offset, 4);
  EXPECT_EQ(view->column(1)->spans()[1].offset, 3);
  EXPECT_EQ(view->column(0)->stride(), 0);  // 2 then 4 rows: irregular.
  for (int row = 0; row < 6; ++row) {
    auto expected = sliced->GetScalar(row).ValueOrDie();
    EXPECT_EQ(raw_value(*view->column(0), row),
              expected->is_valid ? expected->ToString() : "null");
    expected = text->GetScalar(row).ValueOrDie();
    EXPECT_EQ(raw_value(*view->column(1), row),
              expected->is_valid ? expected->ToString() : "null");
  }
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

TEST_P(IpcForwardFilterReadTest, UniformChunksLocateByStride) {
  arrow::Int64Builder builder;
  for (int64_t i = 0; i < 8; ++i) ASSERT_TRUE(builder.Append(i * 10).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  auto schema = arrow::schema({arrow::field("v", arrow::int64())});
  auto uniform = make_owned_view(arrow::Table::Make(
      schema, {std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
                  array->Slice(0, 3), array->Slice(3, 0), array->Slice(3, 3),
                  array->Slice(6, 2)})}));
  ASSERT_NE(uniform, nullptr);
  const auto &column = *uniform->column(0);
  EXPECT_EQ(column.stride(), 3);  // Empty chunks do not break uniformity.
  for (uint64_t row = 0; row < 8; ++row) {
    auto location = column.locate(row);
    ASSERT_TRUE(location.has_value());
    EXPECT_EQ(location->chunk_index, static_cast<int>(row / 3));
    EXPECT_EQ(location->offset, static_cast<int64_t>(row % 3));
    EXPECT_EQ(raw_value(column, row), std::to_string(row * 10));
  }
  EXPECT_FALSE(column.locate(8).has_value());
  // A larger last chunk cannot be located by division.
  auto irregular = make_owned_view(arrow::Table::Make(
      schema, {std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
                  array->Slice(0, 3), array->Slice(3, 5)})}));
  ASSERT_NE(irregular, nullptr);
  EXPECT_EQ(irregular->column(0)->stride(), 0);
  EXPECT_EQ(irregular->column(0)->locate(7)->chunk_index, 1);
  EXPECT_EQ(irregular->column(0)->locate(7)->offset, 4);
}

// A literal predicate projects no columns. Both access paths evaluate it;
// on main, preparing it failed the whole query.
TEST_P(ForwardFilterReadTest, LiteralPredicatesAreEvaluatedOnBothPaths) {
  ASSERT_NO_FATAL_FAILURE(create(24));
  for (bool sealed : {false, true}) {
    SCOPED_TRACE(sealed ? "sealed" : "writing");
    if (sealed) ASSERT_NO_FATAL_FAILURE(reopen_persisted());
    auto never = filter({}, cp::literal(false));
    auto always = filter({}, cp::literal(true));
    ASSERT_TRUE(never->compute_filter().ok());
    ASSERT_TRUE(always->compute_filter().ok());
    EXPECT_EQ(uses_view(*always), sealed && GetParam());
    for (uint64_t row = 0; row < 24; ++row) {
      EXPECT_TRUE(never->is_filtered(row)) << row;
      EXPECT_FALSE(always->is_filtered(row)) << row;
    }
  }
}

TEST_P(ForwardFilterReadTest, SealedDeletionMatchesSegmentFilter) {
  ASSERT_NO_FATAL_FAILURE(create());
  for (uint64_t doc_id : {10000u, 10007u, 10100u, 11024u, 12052u}) {
    ASSERT_TRUE(segment_->Delete(doc_id).ok());
  }
  // First the dumped segment object a writable collection keeps using, then
  // the same data reopened from disk.
  ASSERT_TRUE(segment_->dump().ok());
  segment_->remove_writing_forward_block();
  uint64_t next_delete = 10003;
  for (bool reopened : {false, true}) {
    SCOPED_TRACE(reopened ? "reopened" : "dumped");
    if (reopened) ASSERT_NO_FATAL_FAILURE(reopen_persisted(false));
    const auto live = segment_->get_filter();
    ASSERT_NE(live, nullptr);
    auto filtered = filter({"i32"}, cp::literal(true));
    auto info = std::make_shared<sqlengine::QueryInfo>();
    sqlengine::DocFilter unfiltered(segment_, info, nullptr, nullptr);
    ASSERT_TRUE(filtered->compute_filter().ok());
    ASSERT_TRUE(unfiltered.compute_filter().ok());
    // Sealed IPC replaces the locked segment lookup; Parquet keeps it.
    EXPECT_EQ(uses_sealed_deletion(*filtered), GetParam());
    EXPECT_EQ(uses_sealed_deletion(unfiltered), GetParam());
    auto state = segment_->prepare_forward_read({}, true);
    ASSERT_TRUE(state.ok());
    EXPECT_EQ(state->deletion_filter != nullptr, GetParam());
    EXPECT_EQ(state->deletion_filter,
              segment_->prepare_forward_read({}, true)->deletion_filter);
    EXPECT_EQ(segment_->prepare_forward_read({}, false)->deletion_filter,
              nullptr);  // Only built on request.
    auto expect_parity = [&] {
      for (uint64_t row = 0; row < 2053; ++row) {
        const bool deleted = live->is_filtered(row);
        ASSERT_EQ(filtered->is_filtered(row), deleted) << row;
        ASSERT_EQ(unfiltered.is_filtered(row), deleted) << row;
        if (state->deletion_filter) {
          ASSERT_EQ(state->deletion_filter->is_filtered(row), deleted) << row;
        }
      }
      // Rows outside the segment are not deleted rows, as with get_filter().
      EXPECT_EQ(unfiltered.is_filtered(2053), live->is_filtered(2053));
      EXPECT_FALSE(unfiltered.is_filtered(UINT64_MAX));
    };
    expect_parity();
    EXPECT_TRUE(unfiltered.is_filtered(0));
    EXPECT_TRUE(unfiltered.is_filtered(2052));
    EXPECT_FALSE(unfiltered.is_filtered(1));
    // The bitmap stays live: deletes after preparation are still observed.
    const auto row = next_delete - 10000;
    EXPECT_FALSE(unfiltered.is_filtered(row));
    deletes_->mark_deleted(next_delete++);
    EXPECT_TRUE(filtered->is_filtered(row));
    EXPECT_TRUE(unfiltered.is_filtered(row));
    expect_parity();
  }
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
  EXPECT_NE(status.message().find("Prepare forward read failed"),
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
INSTANTIATE_TEST_SUITE_P(SealedIpc, IpcForwardFilterReadTest,
                         testing::Values(true));
}  // namespace zvec::test
