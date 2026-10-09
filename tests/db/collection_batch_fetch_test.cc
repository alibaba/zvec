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
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include "db/index/storage/parquet_memory_pool.h"

namespace zvec {
namespace {

class CollectionBatchFetchTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(128ULL * 1024 * 1024);
    ailego::FileHelper::RemoveDirectory(path_.c_str());
    options_.enable_mmap_ = GetParam();
  }

  void TearDown() override {
    collection_.reset();
    ailego::FileHelper::RemoveDirectory(path_.c_str());
  }

  void create(size_t count) {
    CollectionSchema schema("batch_fetch");
    schema.set_max_doc_count_per_segment(1000);
    for (const auto &[name, type] :
         std::vector<std::pair<std::string, DataType>>{
             {"payload", DataType::STRING},
             {"category", DataType::INT64},
             {"enabled", DataType::BOOL},
             {"numbers", DataType::ARRAY_INT32},
             {"labels", DataType::ARRAY_STRING},
             {"binary", DataType::BINARY}}) {
      ASSERT_TRUE(
          schema.add_field(std::make_shared<FieldSchema>(name, type)).ok());
    }
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "optional", DataType::FLOAT, true))
                    .ok());
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "emb", DataType::VECTOR_FP32, 4, false,
                        std::make_shared<FlatIndexParams>(MetricType::L2)))
                    .ok());
    auto result = Collection::CreateAndOpen(path_, schema, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    collection_ = std::move(result.value());
    std::vector<Doc> docs;
    for (size_t i = 0; i < count; ++i) {
      docs.push_back(make_doc(i));
      if (docs.size() == 1024) {
        ASSERT_NO_FATAL_FAILURE(
            check_write(collection_->insert(docs), docs.size()));
        docs.clear();
      }
    }
    if (!docs.empty()) {
      ASSERT_NO_FATAL_FAILURE(
          check_write(collection_->insert(docs), docs.size()));
    }
  }

  static std::string pk(size_t row) {
    return "row-" + std::to_string(row);
  }

  static Doc make_doc(size_t row) {
    Doc doc;
    doc.set_pk(pk(row));
    doc.set("payload", std::string(128, static_cast<char>('a' + row % 26)));
    doc.set("category", static_cast<int64_t>(row));
    doc.set("enabled", row % 2 == 0);
    doc.set("numbers", std::vector<int32_t>{static_cast<int32_t>(row), -3});
    doc.set("labels", std::vector<std::string>{pk(row), "retained"});
    doc.set("binary", std::string("a\0b", 3));
    if (row % 3 != 0) {
      doc.set("optional", static_cast<float>(row));
    }
    doc.set("emb",
            std::vector<float>{static_cast<float>(row), 1.0F, 2.0F, 3.0F});
    return doc;
  }

  static void check_write(const Result<WriteResults> &result, size_t count) {
    ASSERT_TRUE(result.has_value()) << result.error().message();
    ASSERT_EQ(count, result->size());
    for (const auto &status : *result) {
      ASSERT_TRUE(status.ok()) << status.message();
    }
  }

  void reopen() {
    collection_.reset();
    auto result = Collection::Open(path_, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    collection_ = std::move(result.value());
  }

  void compare_single_fetch(
      const std::vector<std::string> &ids,
      const std::optional<std::vector<std::string>> &fields = std::nullopt,
      bool include_vector = true) {
    auto result = collection_->fetch(ids, fields, include_vector);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    for (const auto &id : ids) {
      auto single = collection_->fetch({id}, fields, include_vector);
      ASSERT_TRUE(single.has_value()) << single.error().message();
      ASSERT_EQ(1U, single->size());
      ASSERT_EQ(1U, result->count(id));
      const auto &expected = single->at(id);
      const auto &actual = result->at(id);
      if (!expected) {
        EXPECT_EQ(nullptr, actual) << id;
        continue;
      }
      ASSERT_NE(nullptr, actual) << id;
      EXPECT_EQ(expected->doc_id(), actual->doc_id()) << id;
      EXPECT_EQ(*expected, *actual) << id;
    }
  }

  std::string path_ = "collection_batch_fetch_test";
  CollectionOptions options_;
  Collection::Ptr collection_;
};

TEST_P(CollectionBatchFetchTest, PreservesValuesAcrossSegmentsAndMicrobatches) {
  constexpr size_t kRows = 1103;
  ASSERT_NO_FATAL_FAILURE(create(kRows));
  std::vector<std::string> ids;
  for (size_t i = 0; i < kRows; ++i) {
    ids.push_back(pk(kRows - i - 1));
  }
  ids.insert(ids.begin() + 255, "missing");
  ids.insert(ids.begin() + 256, pk(7));
  ids.push_back(pk(7));
  ids.push_back("missing");

  auto before = collection_->fetch(ids);
  ASSERT_TRUE(before.has_value()) << before.error().message();
  ASSERT_EQ(kRows + 1, before->size());
  EXPECT_EQ(nullptr, before->at("missing"));
  for (size_t row = 0; row < kRows; ++row) {
    ASSERT_NE(nullptr, before->at(pk(row)));
    EXPECT_EQ(make_doc(row), *before->at(pk(row))) << row;
  }

  ASSERT_TRUE(collection_->flush().ok());
  ASSERT_NO_FATAL_FAILURE(reopen());
  ASSERT_NO_FATAL_FAILURE(compare_single_fetch(ids));
  ASSERT_NO_FATAL_FAILURE(compare_single_fetch(
      ids, std::vector<std::string>{"payload", "optional", "numbers"}, false));
  ASSERT_NO_FATAL_FAILURE(compare_single_fetch(
      {pk(3), pk(1002), pk(3), "missing"}, std::vector<std::string>{}, true));

  // Returned data owns its strings/lists/vectors even after later reads/close.
  collection_.reset();
  EXPECT_EQ(make_doc(7), *before->at(pk(7)));
  EXPECT_EQ(make_doc(1002), *before->at(pk(1002)));
}

TEST_P(CollectionBatchFetchTest, HandlesPartialUpdatesDeletesAndProjection) {
  ASSERT_NO_FATAL_FAILURE(create(1103));
  ASSERT_TRUE(collection_->flush().ok());
  ASSERT_NO_FATAL_FAILURE(reopen());
  auto retained = collection_->fetch({pk(1), pk(1001)});
  ASSERT_TRUE(retained.has_value());

  std::vector<Doc> updates(3);
  updates[0].set_pk(pk(1));
  updates[0].set("payload", std::string("updated payload"));
  updates[1].set_pk(pk(1001));
  updates[1].set("category", int64_t{-91});
  updates[2].set_pk(pk(1002));
  updates[2].set("emb", std::vector<float>{9.0F, 8.0F, 7.0F, 6.0F});
  ASSERT_NO_FATAL_FAILURE(check_write(collection_->update(updates), 3));
  ASSERT_NO_FATAL_FAILURE(
      check_write(collection_->delete_({pk(7), pk(1030)}), 2));

  std::vector<std::string> ids;
  for (size_t i = 0; i < 1103; ++i) {
    ids.push_back(pk(i));
  }
  ids.push_back("missing");
  ids.push_back(pk(1));
  ids.push_back(pk(7));
  ASSERT_NO_FATAL_FAILURE(compare_single_fetch(ids));
  ASSERT_TRUE(collection_->flush().ok());
  ASSERT_NO_FATAL_FAILURE(reopen());
  ASSERT_NO_FATAL_FAILURE(compare_single_fetch(ids));
  ASSERT_NO_FATAL_FAILURE(compare_single_fetch(
      ids, std::vector<std::string>{"category", "payload", "labels"}, false));

  auto projected =
      collection_->fetch({pk(1), pk(1001)}, std::vector<std::string>{}, false);
  ASSERT_TRUE(projected.has_value());
  for (const auto &[id, doc] : *projected) {
    ASSERT_NE(nullptr, doc);
    EXPECT_EQ(id, doc->pk());
    EXPECT_TRUE(doc->field_names().empty());
  }
  auto empty = collection_->fetch({});
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());

  collection_.reset();
  EXPECT_EQ(make_doc(1), *retained->at(pk(1)));
  EXPECT_EQ(make_doc(1001), *retained->at(pk(1001)));
}

INSTANTIATE_TEST_SUITE_P(StorageModes, CollectionBatchFetchTest,
                         ::testing::Bool());

class CollectionBatchFetchPressureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(128ULL * 1024 * 1024);
    ailego::FileHelper::RemoveDirectory(path_.c_str());
  }

  void TearDown() override {
    collection_.reset();
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    ailego::FileHelper::RemoveDirectory(path_.c_str());
  }

  std::string path_ = "collection_batch_fetch_pressure_test";
  Collection::Ptr collection_;
};

class ScopedPoolReservation {
 public:
  ~ScopedPoolReservation() {
    ailego::MemoryLimitPool::get_instance().release_external(bytes_);
  }

  bool reserve(size_t bytes) {
    if (!ailego::MemoryLimitPool::get_instance().try_charge_external(bytes)) {
      return false;
    }
    bytes_ = bytes;
    return true;
  }

 private:
  size_t bytes_{0};
};

TEST_F(CollectionBatchFetchPressureTest, BatchesDecodeOnceUnderCachePressure) {
  constexpr size_t kRows = 512;
  constexpr size_t kFetchRows = 32;
  static constexpr size_t kStringBytes = 2048;
  const std::vector<std::string> columns{"a", "b", "c", "d"};
  const auto text = [](const std::string &column, size_t row) {
    auto value = column + std::to_string(row);
    value.resize(kStringBytes, static_cast<char>('a' + row % 26));
    return value;
  };

  CollectionSchema schema("batch_fetch_pressure");
  for (const auto &column : columns) {
    ASSERT_TRUE(
        schema
            .add_field(std::make_shared<FieldSchema>(column, DataType::STRING))
            .ok());
  }
  auto created = Collection::CreateAndOpen(
      path_, schema,
      CollectionOptions{/*read_only=*/false, /*enable_mmap=*/false});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  collection_ = std::move(created.value());
  {
    std::vector<Doc> docs;
    for (size_t row = 0; row < kRows; ++row) {
      Doc doc;
      doc.set_pk(std::to_string(row));
      for (const auto &column : columns) {
        doc.set(column, text(column, row));
      }
      docs.push_back(std::move(doc));
    }
    auto inserted = collection_->insert(docs);
    ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
    for (const auto &status : *inserted) {
      ASSERT_TRUE(status.ok()) << status.message();
    }
  }
  ASSERT_TRUE(collection_->flush().ok());
  collection_.reset();
  auto reopened = Collection::Open(
      path_, CollectionOptions{/*read_only=*/true, /*enable_mmap=*/false});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  collection_ = std::move(reopened.value());

  auto &queue = ailego::BlockEvictionQueue::get_instance();
  auto &pool = ailego::MemoryLimitPool::get_instance();
  queue.batch_recycle(1024);
  // Each decoded string column is slightly over 1 MiB. One fits, but two
  // cannot coexist: per-document fetching necessarily reloads the columns.
  constexpr size_t kHeadroom = 1536 * 1024;
  ASSERT_GT(pool.available(), kHeadroom);
  ScopedPoolReservation reservation;
  ASSERT_TRUE(reservation.reserve(pool.available() - kHeadroom));

  std::vector<std::string> ids;
  for (size_t row = 0; row < kFetchRows; ++row) {
    ids.push_back(std::to_string(row));
  }
  auto allocator = detail::GetParquetMemoryPool();
  const int64_t before_batch = allocator->num_mapped_allocations();
  auto batched = collection_->fetch(ids, std::nullopt, false);
  const int64_t batch_allocations =
      allocator->num_mapped_allocations() - before_batch;
  ASSERT_TRUE(batched.has_value()) << batched.error().message();
  ASSERT_EQ(kFetchRows, batched->size());
  ASSERT_GT(batch_allocations, 0);
  for (size_t row = 0; row < kFetchRows; ++row) {
    const auto &doc = batched->at(ids[row]);
    ASSERT_NE(nullptr, doc);
    for (const auto &column : columns) {
      ASSERT_EQ(text(column, row), doc->get<std::string>(column));
    }
  }

  queue.batch_recycle(1024);
  const int64_t before_singles = allocator->num_mapped_allocations();
  for (const auto &id : ids) {
    auto single = collection_->fetch({id}, std::nullopt, false);
    ASSERT_TRUE(single.has_value()) << single.error().message();
    ASSERT_NE(nullptr, single->at(id));
    EXPECT_EQ(*batched->at(id), *single->at(id));
  }
  const int64_t single_allocations =
      allocator->num_mapped_allocations() - before_singles;
  // Use an allocation counter rather than a timing assertion. The old public
  // per-PK loop performs the same work for both calls and fails this check.
  EXPECT_GT(single_allocations, batch_allocations * 4);
}

TEST_F(CollectionBatchFetchPressureTest, FullPoolDoesNotReturnPartialDocs) {
  CollectionSchema schema("batch_fetch_failure");
  ASSERT_TRUE(
      schema.add_field(std::make_shared<FieldSchema>("value", DataType::INT64))
          .ok());
  auto created = Collection::CreateAndOpen(
      path_, schema,
      CollectionOptions{/*read_only=*/false, /*enable_mmap=*/false});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  collection_ = std::move(created.value());
  std::vector<Doc> docs(2);
  for (size_t row = 0; row < docs.size(); ++row) {
    docs[row].set_pk(std::to_string(row));
    docs[row].set("value", static_cast<int64_t>(row + 10));
  }
  auto inserted = collection_->insert(docs);
  ASSERT_TRUE(inserted.has_value());
  for (const auto &status : *inserted) ASSERT_TRUE(status.ok());
  ASSERT_TRUE(collection_->flush().ok());
  collection_.reset();
  auto opened = Collection::Open(
      path_, CollectionOptions{/*read_only=*/true, /*enable_mmap=*/false});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  collection_ = std::move(opened.value());
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
  {
    ScopedPoolReservation reservation;
    ASSERT_TRUE(reservation.reserve(pool.available()));
    auto single = collection_->fetch({"0"}, std::nullopt, false);
    ASSERT_TRUE(single.has_value());
    ASSERT_EQ(nullptr, single->at("0"));
    auto batch = collection_->fetch({"0", "1"}, std::nullopt, false);
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(2U, batch->size());
    EXPECT_EQ(nullptr, batch->at("0"));
    EXPECT_EQ(nullptr, batch->at("1"));
  }
  auto recovered = collection_->fetch({"0", "1"}, std::nullopt, false);
  ASSERT_TRUE(recovered.has_value());
  for (size_t row = 0; row < docs.size(); ++row) {
    const auto &doc = recovered->at(std::to_string(row));
    ASSERT_NE(nullptr, doc);
    EXPECT_EQ(docs[row], *doc);
  }
}

TEST_F(CollectionBatchFetchPressureTest,
       NestedCrossRowGroupFetchFallsBackSafely) {
  constexpr size_t kRows = 512;
  constexpr size_t kRowsPerGroup = 256;
  const auto values_for_row = [](size_t row) {
    return std::vector<std::string>{std::to_string(row) +
                                    std::string(8192, 'v')};
  };
  CollectionSchema schema("batch_fetch_nested");
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      "values", DataType::ARRAY_STRING))
                  .ok());
  auto created = Collection::CreateAndOpen(
      path_, schema,
      CollectionOptions{/*read_only=*/false, /*enable_mmap=*/false});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  collection_ = std::move(created.value());
  {
    std::vector<Doc> docs;
    for (size_t row = 0; row < kRows; ++row) {
      Doc doc;
      doc.set_pk(std::to_string(row));
      doc.set("values", values_for_row(row));
      docs.push_back(std::move(doc));
    }
    auto inserted = collection_->insert(docs);
    ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
    for (const auto &status : *inserted) ASSERT_TRUE(status.ok());
  }
  ASSERT_TRUE(collection_->flush().ok());
  collection_.reset();

  // Keep the collection metadata/rows unchanged, but exercise a valid older
  // file layout with multiple row groups in one persisted scalar block.
  std::vector<std::string> parquet_paths;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(path_)) {
    if (entry.path().extension() == ".parquet") {
      parquet_paths.push_back(entry.path().string());
    }
  }
  ASSERT_EQ(1U, parquet_paths.size());
  {
    auto input = arrow::io::ReadableFile::Open(parquet_paths.front());
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    auto reader =
        parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader)->ReadTable(&table).ok());
    ASSERT_EQ(kRows, static_cast<size_t>(table->num_rows()));
    reader.ValueOrDie().reset();
    ASSERT_TRUE((*input)->Close().ok());
    auto output = arrow::io::FileOutputStream::Open(parquet_paths.front());
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *output, kRowsPerGroup)
                    .ok());
    ASSERT_TRUE((*output)->Close().ok());
  }

  auto opened = Collection::Open(
      path_, CollectionOptions{/*read_only=*/true, /*enable_mmap=*/false});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  collection_ = std::move(opened.value());
  auto &queue = ailego::BlockEvictionQueue::get_instance();
  auto &pool = ailego::MemoryLimitPool::get_instance();
  queue.batch_recycle(1024);
  constexpr size_t kHeadroom = 3 * 1024 * 1024;
  ASSERT_GT(pool.available(), kHeadroom);
  ScopedPoolReservation reservation;
  ASSERT_TRUE(reservation.reserve(pool.available() - kHeadroom));
  // A LIST scalar retains its source row group. Each group fits alone, but
  // two 2-MiB groups cannot stay pinned together during the batched read.
  const std::vector<std::string> ids{"0", std::to_string(kRowsPerGroup)};
  for (const auto &id : ids) {
    auto single = collection_->fetch({id}, std::nullopt, false);
    ASSERT_TRUE(single.has_value());
    ASSERT_NE(nullptr, single->at(id));
  }
  queue.batch_recycle(1024);
  auto batch = collection_->fetch(ids, std::nullopt, false);
  ASSERT_TRUE(batch.has_value()) << batch.error().message();
  for (const auto &id : ids) {
    const auto &doc = batch->at(id);
    ASSERT_NE(nullptr, doc);
    EXPECT_EQ(id, doc->pk());
    EXPECT_EQ(values_for_row(std::stoul(id)),
              doc->get<std::vector<std::string>>("values"));
  }
}

}  // namespace
}  // namespace zvec
