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

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <tuple>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/db/collection.h>
#include "db/common/file_helper.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/common/manifest_codec.h"

using namespace zvec;

namespace {
using Values = std::optional<std::vector<float>>;
using Row = std::pair<Values, Values>;

class NullableVectorTest
    : public ::testing::TestWithParam<std::tuple<int, bool>> {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
    FileHelper::RemoveDirectory(path_);
  }
  void TearDown() override {
    collection_.reset();
    FileHelper::RemoveDirectory(path_);
  }
  IndexParams::Ptr params() {
    switch (std::get<0>(GetParam())) {
      case 1:
        return std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
      case 2:
        return std::make_shared<IVFIndexParams>(MetricType::L2, 2, 3);
      case 3:
        return std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100,
                                                 QuantizeType::FP16);
      case 4:
        return std::make_shared<VamanaIndexParams>(MetricType::L2, 16, 32);
#if RABITQ_SUPPORTED
      case 5:
        return std::make_shared<HnswRabitqIndexParams>(MetricType::L2, 4, 2, 16,
                                                       100);
      case 6:
        return std::make_shared<IvfRabitqIndexParams>(MetricType::L2, 2, 4);
#endif
#if DISKANN_SUPPORTED
      case 7:
        return std::make_shared<DiskAnnIndexParams>(MetricType::L2, 16, 32);
#endif
      default:
        return std::make_shared<FlatIndexParams>(MetricType::L2);
    }
  }
  void create() {
    CollectionSchema schema("nullable_vectors");
    schema.set_max_doc_count_per_segment(1000);
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "ordinal", DataType::INT32, false))
                    .ok());
    for (const auto *name : {"vec", "other"})
      ASSERT_TRUE(schema
                      .add_field(std::make_shared<FieldSchema>(
                          name, DataType::VECTOR_FP32, 64, true, params()))
                      .ok());
    options_.enable_mmap_ = std::get<1>(GetParam());
    auto result = Collection::CreateAndOpen(path_, schema, options_);
    ASSERT_TRUE(result) << result.error().message();
    collection_ = std::move(result.value());
  }
  bool reopen(bool read_only = false) {
    collection_.reset();
    auto options = options_;
    options.read_only_ = read_only;
    auto result = Collection::Open(path_, options);
    if (!result) {
      ADD_FAILURE() << result.error().message();
      return false;
    }
    collection_ = std::move(result.value());
    return true;
  }
  static std::vector<float> vector(int i) {
    std::vector<float> value(64, 0);
    value[0] = i;
    return value;
  }
  void write(int i, bool all_null = false) {
    Doc doc;
    const auto pk = "d" + std::to_string(i);
    doc.set_pk(pk);
    doc.set<int32_t>("ordinal", i);
    Row row;
    if (!all_null && i % 2 == 0) {
      row.first = vector(i);  // d0 is a genuine all-zero vector.
      doc.set("vec", *row.first);
    } else if (i % 4 == 3) {
      doc.set_null("vec");  // Both omission and explicit NULL are accepted.
    }
    if (!all_null && i % 3 == 0) {
      row.second = vector(100 + i);
      doc.set("other", *row.second);
    }
    std::vector<Doc> docs{doc};
    auto result = collection_->insert(docs);
    ASSERT_TRUE(result) << result.error().message();
    ASSERT_TRUE(result->at(0).ok()) << result->at(0).message();
    rows_[pk] = row;
  }
  void check_doc(const Doc &doc) {
    auto it = rows_.find(doc.pk());
    ASSERT_NE(it, rows_.end());
    for (const auto &entry : {std::make_pair("vec", it->second.first),
                              std::make_pair("other", it->second.second)}) {
      auto actual = doc.get<std::vector<float>>(entry.first);
      ASSERT_EQ(actual.has_value(), entry.second.has_value())
          << doc.pk() << ":" << entry.first;
      if (actual) {
        EXPECT_EQ(*actual, *entry.second) << doc.pk() << ":" << entry.first;
      }
    }
  }
  void check_hits(const std::set<std::string> &actual,
                  const std::set<std::string> &expected) {
    if (std::get<0>(GetParam()) == 7) {
      // DiskAnn can omit a valid neighbor even for this small collinear set.
      // Validate NULL exclusion and ID translation without requiring full ANN
      // recall; fetch/scan/iteration separately inspect every stored vector.
      for (const auto &pk : actual) EXPECT_NE(expected.count(pk), 0U);
      if (!expected.empty()) {
        EXPECT_FALSE(actual.empty());
      }
    } else {
      EXPECT_EQ(actual, expected) << "NULL vectors must not occupy top-k slots";
    }
  }
  void check() {
    ASSERT_NE(collection_, nullptr);
    std::vector<std::string> pks;
    for (const auto &row : rows_) pks.push_back(row.first);
    for (bool include : {false, true}) {
      auto fetched = collection_->fetch(pks, std::nullopt, include);
      ASSERT_TRUE(fetched) << fetched.error().message();
      ASSERT_EQ(fetched->size(), rows_.size());
      for (const auto &[pk, doc] : *fetched) {
        ASSERT_NE(doc, nullptr) << "lost document " << pk;
        if (include) check_doc(*doc);
      }
    }
    SearchQuery scan;
    scan.topk_ = 100;
    scan.filter_ = "ordinal >= 0";
    scan.include_vector_ = true;
    auto result = collection_->query(scan);
    ASSERT_TRUE(result) << result.error().message();
    ASSERT_EQ(result->size(), rows_.size());
    for (const auto &doc : *result) check_doc(*doc);

    SearchQuery knn;
    knn.target_.field_name_ = "vec";
    const auto zero = vector(std::get<0>(GetParam()) >= 5 ? 3 : 0);
    knn.target_.set_vector(
        std::string(reinterpret_cast<const char *>(zero.data()),
                    zero.size() * sizeof(float)));
    knn.topk_ = 100;
    knn.include_vector_ = true;
    if (std::get<0>(GetParam()) == 2)
      knn.target_.query_params_ = std::make_shared<IVFQueryParams>(2);
    if (std::get<0>(GetParam()) == 3)
      knn.target_.query_params_ =
          std::make_shared<HnswQueryParams>(100, 0, false, true);
    if (std::get<0>(GetParam()) == 6)
      knn.target_.query_params_ = std::make_shared<IvfRabitqQueryParams>(2);
    std::set<std::string> expected;
    for (const auto &[pk, row] : rows_)
      if (row.first) expected.insert(pk);
    for (const auto *filter : {"", "ordinal >= 8", "ordinal = 1"}) {
      knn.filter_ = filter;
      auto hits = collection_->query(knn);
      ASSERT_TRUE(hits) << hits.error().message();
      auto wanted = expected;
      if (*filter)
        for (auto it = wanted.begin(); it != wanted.end();) {
          const int ordinal = std::stoi(it->substr(1));
          if ((std::string(filter) == "ordinal = 1" && ordinal != 1) ||
              (std::string(filter) == "ordinal >= 8" && ordinal < 8))
            it = wanted.erase(it);
          else
            ++it;
        }
      std::set<std::string> actual;
      for (const auto &doc : *hits) {
        check_doc(*doc);
        actual.insert(doc->pk());
      }
      check_hits(actual, wanted);
    }
    if (collection_->options()->read_only_) {
      knn.filter_.clear();
      knn.include_vector_ = false;
      auto ids = collection_->query_internal_ids(knn, true);
      ASSERT_TRUE(ids) << ids.error().message();
      auto resolved = collection_->resolve_internal_ids(ids->ids);
      ASSERT_TRUE(resolved);
      std::set<std::string> actual;
      for (const auto &pk : *resolved)
        if (pk) actual.insert(*pk);
      check_hits(actual, expected);
    }
    auto iterator = collection_->create_iterator();
    ASSERT_TRUE(iterator) << iterator.error().message();
    std::set<std::string> seen;
    while (true) {
      auto next = (*iterator)->next();
      ASSERT_TRUE(next) << next.error().message();
      if (!*next) break;
      check_doc(**next);
      seen.insert((*next)->pk());
    }
    EXPECT_EQ(seen.size(), rows_.size());
    (*iterator)->close();
  }
  std::string path_ = "test_nullable_vectors";
  CollectionOptions options_;
  Collection::Ptr collection_;
  std::map<std::string, Row> rows_;
};

TEST_P(NullableVectorTest, MixedRowsSurviveFlushMergeReopenAndUpdates) {
  create();
  ASSERT_NE(collection_, nullptr);
  for (int i = 0; i < 24; ++i) {
    write(i);
    if (i % 4 == 3 && i != 23) {
      ASSERT_TRUE(collection_->flush().ok());
    }
  }
  check();
  ASSERT_TRUE(collection_->flush().ok());
  ASSERT_TRUE(reopen());
  check();
  ASSERT_TRUE(collection_->optimize().ok());
  check();
  ASSERT_TRUE(reopen(true));
  check();
  ASSERT_TRUE(reopen());

  Doc clear, fill;
  clear.set_pk("d0");
  clear.set_null("vec");
  fill.set_pk("d1");
  fill.set("vec", vector(77));
  std::vector<Doc> updates{clear, fill};
  auto result = collection_->update(updates);
  ASSERT_TRUE(result);
  for (const auto &status : *result)
    ASSERT_TRUE(status.ok()) << status.message();
  rows_["d0"].first.reset();
  rows_["d1"].first = vector(77);
  auto deleted = collection_->delete_({"d2", "d3"});
  ASSERT_TRUE(deleted);
  rows_.erase("d2");
  rows_.erase("d3");
  ASSERT_TRUE(collection_->optimize().ok());
  check();
  ASSERT_TRUE(reopen());
  check();
}

TEST_P(NullableVectorTest, AllNullColumnsKeepEveryDocument) {
  create();
  ASSERT_NE(collection_, nullptr);
  for (int i = 0; i < 16; ++i) write(i, true);
  check();
  ASSERT_TRUE(collection_->flush().ok());
  ASSERT_TRUE(reopen());
  check();
  ASSERT_TRUE(collection_->optimize().ok());
  ASSERT_TRUE(reopen());
  check();
}

class NullableVectorRolloverTest : public NullableVectorTest {};

TEST_P(NullableVectorRolloverTest, SegmentRolloverPreservesRowsAndVectors) {
  // One streaming and one builder-backed index exercise compaction across
  // the segment boundary as well as the minimum 1,000-row segment size.
  create();
  ASSERT_NE(collection_, nullptr);
  for (int i = 0; i < 1004; ++i) write(i);
  ASSERT_TRUE(collection_->flush().ok());
  ASSERT_TRUE(collection_->optimize().ok());
  ASSERT_TRUE(reopen());
  // check() caps scans at 100; use PK fetch to inspect every merged row.
  std::vector<std::string> pks;
  for (const auto &row : rows_) pks.push_back(row.first);
  auto fetched = collection_->fetch(pks, std::nullopt, true);
  ASSERT_TRUE(fetched);
  ASSERT_EQ(fetched->size(), rows_.size());
  for (const auto &[pk, doc] : *fetched) {
    ASSERT_NE(doc, nullptr) << pk;
    check_doc(*doc);
  }
}

std::vector<int> IndexTypes() {
  std::vector<int> types{0, 1, 2, 3, 4};
#if RABITQ_SUPPORTED
  types.insert(types.end(), {5, 6});
#endif
#if DISKANN_SUPPORTED
  types.push_back(7);
#endif
  return types;
}

INSTANTIATE_TEST_SUITE_P(IndexAndStorage, NullableVectorTest,
                         ::testing::Combine(::testing::ValuesIn(IndexTypes()),
                                            ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(IndexAndStorage, NullableVectorRolloverTest,
                         ::testing::Combine(::testing::Values(0, 1),
                                            ::testing::Bool()));

class NullableVectorStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  }
};

TEST_F(NullableVectorStorageTest, NonNullableRowCountUsesCoreDocumentRange) {
  const std::string path = "test_non_nullable_row_count";
  FileHelper::RemoveDirectory(path);
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  FieldSchema field("vec", DataType::VECTOR_FP32, 16, false,
                    std::make_shared<FlatIndexParams>(MetricType::L2));
  const auto file = path + "/index";
  VectorColumnIndexer writer(file, field);
  ASSERT_TRUE(writer.open({true, true}).ok());
  std::vector<float> values(16, 7);
  for (uint32_t id : {5U, 2U}) {
    ASSERT_TRUE(
        writer.insert({vector_column_params::DenseVector{values.data()}}, id)
            .ok());
    EXPECT_EQ(6U, writer.row_count());
  }
  ASSERT_TRUE(writer.close().ok());
  VectorColumnIndexer reader(file, field);
  ASSERT_TRUE(reader.open({true, false, true}).ok());
  EXPECT_EQ(6U, reader.row_count());
  EXPECT_EQ(2U, reader.debug_get_index()->get_vector_count());
  ASSERT_TRUE(reader.close().ok());
  FileHelper::RemoveDirectory(path);
}

TEST_F(NullableVectorStorageTest, NullableRowsPreserveIdsAcrossMerge) {
  const std::string path = "test_nullable_merge";
  FileHelper::RemoveDirectory(path);
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  FieldSchema nullable("vec", DataType::VECTOR_FP32, 16, true,
                       std::make_shared<FlatIndexParams>(MetricType::L2));
  std::vector<VectorColumnIndexer::Ptr> sources;
  for (uint32_t i = 0; i < 2; ++i) {
    const auto file = path + "/source" + std::to_string(i);
    VectorColumnIndexer writer(file, nullable);
    ASSERT_TRUE(writer.open({true, true}).ok());
    std::vector<float> data(16, 0);
    data[0] = i * 4;
    ASSERT_TRUE(
        writer.insert({vector_column_params::DenseVector{data.data()}}, 0)
            .ok());
    ASSERT_TRUE(writer.insert_null(1).ok());
    data[0] = i * 4 + 2;
    ASSERT_TRUE(
        writer.insert({vector_column_params::DenseVector{data.data()}}, 2)
            .ok());
    ASSERT_TRUE(writer.insert_null(3).ok());
    ASSERT_TRUE(writer.close().ok());
    auto source = std::make_shared<VectorColumnIndexer>(file, nullable);
    ASSERT_TRUE(source->open({true, false, true, 4, true}).ok());
    for (uint32_t missing : {1U, 3U}) {
      auto fetched = source->fetch(missing);
      ASSERT_FALSE(fetched);
      EXPECT_EQ(fetched.error().code(), StatusCode::NOT_FOUND);
    }
    sources.push_back(source);
  }
  nullable.set_index_params(std::make_shared<HnswIndexParams>(MetricType::L2));
  VectorColumnIndexer target(path + "/merged", nullable);
  ASSERT_TRUE(target.open({true, true}).ok());
  auto filter = EasyIndexFilter::Create([](uint64_t row) { return row == 2; });
  ASSERT_TRUE(target.merge(sources, filter).ok());
  EXPECT_EQ(target.row_count(), 7U);
  for (const auto &[row, value] :
       std::map<uint32_t, float>{{0, 0}, {3, 4}, {5, 6}}) {
    auto fetched = target.fetch(row);
    ASSERT_TRUE(fetched) << fetched.error().message();
    const auto &bytes = std::get<vector_column_params::DenseVectorBuffer>(
                            fetched->vector_buffer)
                            .data;
    float actual;
    std::memcpy(&actual, bytes.data(), sizeof(actual));
    EXPECT_EQ(actual, value) << row;
  }
  std::vector<float> query(16, 0);
  vector_column_params::QueryParams grouped;
  grouped.topk = 10;
  grouped.group_by = std::make_unique<vector_column_params::GroupByParams>(
      1, 3, [](uint64_t row) { return std::to_string(row); });
  auto result =
      target.search({vector_column_params::DenseVector{query.data()}}, grouped);
  ASSERT_TRUE(result) << result.error().message();
  auto iter = (*result)->create_iterator();
  std::set<uint32_t> grouped_rows;
  for (; iter->valid(); iter->next()) {
    EXPECT_EQ(iter->group_id(), std::to_string(iter->doc_id()));
    grouped_rows.insert(iter->doc_id());
  }
  EXPECT_EQ(grouped_rows, (std::set<uint32_t>{0, 3, 5}));
  auto outside = target.fetch(7);
  ASSERT_FALSE(outside);
  EXPECT_EQ(outside.error().code(), StatusCode::INTERNAL_ERROR);
  ASSERT_TRUE(target.close().ok());
  for (auto &source : sources) ASSERT_TRUE(source->close().ok());
  VectorColumnIndexer reopened(path + "/merged", nullable);
  ASSERT_TRUE(reopened.open({true, false, true, 7, true}).ok());
  EXPECT_FALSE(reopened.fetch(6));
  EXPECT_TRUE(reopened.fetch(0));  // The genuine zero vector remains present.
  ASSERT_TRUE(reopened.close().ok());
  FileHelper::RemoveDirectory(path);
}

TEST_F(NullableVectorStorageTest, CoreKeysStayEqualToDocumentIds) {
  const std::string path = "test_nullable_document_ids";
  for (bool graph : {false, true}) {
    FileHelper::RemoveDirectory(path);
    ASSERT_TRUE(FileHelper::CreateDirectory(path));
    FieldSchema field("vec", DataType::VECTOR_FP32, 16, true,
                      std::make_shared<FlatIndexParams>(MetricType::L2));
    if (graph)
      field.set_index_params(std::make_shared<HnswIndexParams>(MetricType::L2));
    VectorColumnIndexer indexer(path + "/index", field);
    ASSERT_TRUE(indexer.open({true, true}).ok());
    std::vector<float> data(16, 0);
    for (uint32_t row = 0; row < 6; ++row) {
      if (row == 1 || row == 3) {
        data[0] = row;
        ASSERT_TRUE(
            indexer
                .insert({vector_column_params::DenseVector{data.data()}}, row)
                .ok());
      } else {
        ASSERT_TRUE(indexer.insert_null(row).ok());
      }
    }
    auto check = [&] {
      // Bypass the column layer: core fetch must already accept document IDs.
      for (uint32_t row : {1U, 3U}) {
        core_interface::VectorDataBuffer buffer;
        ASSERT_EQ(0, indexer.debug_get_index()->fetch(row, &buffer));
        const auto &bytes =
            std::get<core_interface::DenseVectorBuffer>(buffer.vector_buffer)
                .data;
        float first;
        std::memcpy(&first, bytes.data(), sizeof(first));
        EXPECT_EQ(first, row);
      }
      auto provider = indexer.create_index_provider();
      ASSERT_NE(provider, nullptr);
      EXPECT_EQ(provider->count(), 2U);
      EXPECT_EQ(provider->slot_count(), 4U);
      auto iter = provider->create_iterator();
      ASSERT_NE(iter, nullptr);
      std::set<uint64_t> keys;
      for (; iter->is_valid(); iter->next()) keys.insert(iter->key());
      EXPECT_EQ(keys, (std::set<uint64_t>{1, 3}));
      EXPECT_EQ(iter->status(), 0);
    };
    check();
    ASSERT_TRUE(indexer.close().ok());
    ailego::File bitmap_file;
    ASSERT_TRUE(bitmap_file.open(path + "/index.nulls", true));
    std::string bytes(bitmap_file.size(), '\0');
    ASSERT_EQ(bitmap_file.read(0, bytes.data(), bytes.size()), bytes.size());
    auto nulls =
        roaring::Roaring::readSafe(bytes.data() + 16, bytes.size() - 20);
    EXPECT_EQ(nulls.cardinality(), 4U);
    for (uint32_t row = 0; row < 6; ++row)
      EXPECT_EQ(nulls.contains(row), row != 1 && row != 3);
    bitmap_file.close();
    ASSERT_TRUE(indexer.open({true, false, true, 6, true}).ok());
    check();
    ASSERT_TRUE(indexer.close().ok());
    FileHelper::RemoveDirectory(path);
  }
}

TEST_F(NullableVectorStorageTest, AllNullBackingAcceptsLaterVectors) {
  const std::string path = "test_nullable_empty_backing";
  for (bool ivf : {false, true}) {
    for (bool reopen : {false, true}) {
      SCOPED_TRACE(ivf);
      SCOPED_TRACE(reopen);
      FileHelper::RemoveDirectory(path);
      ASSERT_TRUE(FileHelper::CreateDirectory(path));
      FieldSchema field("vec", DataType::VECTOR_FP32, 16, true,
                        std::make_shared<FlatIndexParams>(MetricType::L2));
      auto source =
          std::make_shared<VectorColumnIndexer>(path + "/source", field);
      ASSERT_TRUE(source->open({true, true}).ok());
      ASSERT_TRUE(source->insert_null(0).ok());
      std::vector<float> data(16, 11);
      ASSERT_TRUE(
          source->insert({vector_column_params::DenseVector{data.data()}}, 1)
              .ok());
      if (ivf)
        field.set_index_params(
            std::make_shared<IVFIndexParams>(MetricType::L2, 1, 3));
      else
        field.set_index_params(
            std::make_shared<HnswIndexParams>(MetricType::L2));
      VectorColumnIndexer target(path + "/target", field);
      ASSERT_TRUE(target.open({true, true}).ok());
      ASSERT_TRUE(target.insert_null(0).ok());
      ASSERT_TRUE(target.insert_null(1).ok());
      ASSERT_TRUE(target.flush().ok());
      if (reopen) {
        ASSERT_TRUE(target.close().ok());
        ASSERT_TRUE(target.open({true, false, false, 2, true}).ok());
      }
      if (!ivf) {
        ASSERT_TRUE(
            target.insert({vector_column_params::DenseVector{data.data()}}, 2)
                .ok());
      }
      ASSERT_TRUE(target.merge({source}).ok());
      const uint32_t rows = ivf ? 4 : 5;
      EXPECT_EQ(target.row_count(), rows);
      ASSERT_TRUE(target.close().ok());
      auto opened = target.open({true, false, true, rows, true});
      ASSERT_TRUE(opened.ok()) << opened.message();
      for (uint32_t row = 0; row < rows; ++row) {
        const bool present = row == rows - 1 || (!ivf && row == 2);
        auto fetched = target.fetch(row);
        ASSERT_EQ(fetched.has_value(), present) << row;
        if (fetched) {
          const auto &bytes = std::get<vector_column_params::DenseVectorBuffer>(
                                  fetched->vector_buffer)
                                  .data;
          ASSERT_EQ(bytes.size(), data.size() * sizeof(float));
          EXPECT_EQ(std::memcmp(bytes.data(), data.data(), bytes.size()), 0);
        } else {
          EXPECT_EQ(fetched.error().code(), StatusCode::NOT_FOUND);
        }
      }
      ASSERT_TRUE(target.close().ok());
      ASSERT_TRUE(source->close().ok());
      FileHelper::RemoveDirectory(path);
    }
  }
}

TEST_F(NullableVectorStorageTest, LegacyNullableIndexesAreRejected) {
  const std::string path = "test_nullable_legacy";
  for (int kind : {0, 1, 2}) {
    for (bool mmap : {false, true}) {
      SCOPED_TRACE(kind);
      SCOPED_TRACE(mmap);
      FileHelper::RemoveDirectory(path);
      ASSERT_TRUE(FileHelper::CreateDirectory(path));
      const bool sparse = kind == 1;
      FieldSchema field(
          "vec", sparse ? DataType::SPARSE_VECTOR_FP32 : DataType::VECTOR_FP32,
          sparse ? 0 : 16, false,
          std::make_shared<FlatIndexParams>(MetricType::IP));
      const auto file = path + "/index";
      // A non-nullable writer produces the old core-only format, without
      // validity metadata. Include intermediate padding and trailing NULLs.
      VectorColumnIndexer writer(file, field);
      ASSERT_TRUE(writer.open({mmap, true}).ok());
      if (kind != 2) {
        std::vector<float> data(16, 0);
        const uint32_t index = 1;
        const float value = 1;
        for (uint32_t row : {0U, 2U}) {
          vector_column_params::VectorData vector;
          if (sparse)
            vector.vector =
                vector_column_params::SparseVector{1, &index, &value};
          else
            vector.vector = vector_column_params::DenseVector{data.data()};
          ASSERT_TRUE(writer.insert(vector, row).ok());
        }
      }
      ASSERT_TRUE(writer.close().ok());
      FieldSchema nullable(field);
      nullable.set_nullable(true);
      for (bool read_only : {false, true}) {
        VectorColumnIndexer rejected(file, nullable);
        auto status = rejected.open({mmap, false, read_only, 4, false});
        EXPECT_EQ(status.code(), StatusCode::NOT_SUPPORTED);
        EXPECT_EQ(rejected.debug_get_index(), nullptr);
        EXPECT_FALSE(FileHelper::FileExists(file + ".nulls"));
      }
      // Legacy non-nullable fields continue to use their existing read path.
      VectorColumnIndexer non_nullable(file, field);
      ASSERT_TRUE(non_nullable.open({mmap, false, true, 4, false}).ok());
      if (kind != 2) {
        EXPECT_TRUE(non_nullable.fetch(0));
        EXPECT_TRUE(non_nullable.fetch(2));
      }
      ASSERT_TRUE(non_nullable.close().ok());
      FileHelper::RemoveDirectory(path);
    }
  }
}

TEST_F(NullableVectorStorageTest, LegacyVectorLayoutsAreRejected) {
  const std::string path = "test_nullable_identity";
  FileHelper::RemoveDirectory(path);
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  FieldSchema field("vec", DataType::VECTOR_FP32, 16, true,
                    std::make_shared<FlatIndexParams>(MetricType::L2));
  const auto file = path + "/index";
  VectorColumnIndexer writer(file, field);
  ASSERT_TRUE(writer.open({true, true}).ok());
  ASSERT_TRUE(writer.insert_null(0).ok());
  ASSERT_TRUE(writer.close().ok());
  ailego::File validity;
  ASSERT_TRUE(validity.open(file + ".nulls", false));
  std::string bytes(validity.size(), '\0');
  ASSERT_EQ(validity.read(0, bytes.data(), bytes.size()), bytes.size());
  std::memset(bytes.data() + 12, 0, 4);  // Legacy identity-ID layout.
  const uint32_t crc = ailego::Crc32c::Hash(bytes.data(), bytes.size() - 4);
  for (unsigned i = 0; i < 4; ++i) bytes[bytes.size() - 4 + i] = crc >> (i * 8);
  ASSERT_EQ(validity.write(0, bytes.data(), bytes.size()), bytes.size());
  validity.close();
  for (bool read_only : {false, true}) {
    VectorColumnIndexer rejected(file, field);
    EXPECT_EQ(rejected.open({true, false, read_only, 1, true}).code(),
              StatusCode::NOT_SUPPORTED);
    EXPECT_EQ(rejected.debug_get_index(), nullptr);
  }
  // The previous compact-ID format used a .validity sidecar. It remains
  // unsupported even when the manifest requires nullable metadata.
  ASSERT_TRUE(FileHelper::MoveFile(file + ".nulls", file + ".validity"));
  VectorColumnIndexer compact(file, field);
  EXPECT_EQ(compact.open({true, false, true, 1, true}).code(),
            StatusCode::NOT_SUPPORTED);
  FileHelper::RemoveDirectory(path);
}

TEST_F(NullableVectorStorageTest, MissingOrCorruptNullBitmapIsAnError) {
  const std::string path = "test_nullable_corrupt";
  FileHelper::RemoveDirectory(path);
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  FieldSchema field("vec", DataType::VECTOR_FP32, 16, true,
                    std::make_shared<FlatIndexParams>(MetricType::L2));
  const auto file = path + "/index";
  VectorColumnIndexer indexer(file, field);
  ASSERT_TRUE(indexer.open({true, true}).ok());
  ASSERT_TRUE(indexer.insert_null(0).ok());
  ASSERT_TRUE(indexer.close().ok());
  ailego::File validity;
  ASSERT_TRUE(validity.open(file + ".nulls", false));
  const char bad = 0;
  ASSERT_EQ(validity.write(0, &bad, 1), 1U);
  validity.close();
  VectorColumnIndexer corrupt(file, field);
  EXPECT_EQ(corrupt.open({true, false, true, 1, true}).code(),
            StatusCode::INTERNAL_ERROR);
  ASSERT_TRUE(FileHelper::RemoveFile(file + ".nulls"));
  VectorColumnIndexer missing(file, field);
  EXPECT_EQ(missing.open({true, false, true, 1, true}).code(),
            StatusCode::INTERNAL_ERROR);
  FileHelper::RemoveDirectory(path);
}

TEST_F(NullableVectorStorageTest, NullBitmapRequirementRoundTripsInManifest) {
  BlockMeta meta(5, BlockType::VECTOR_INDEX, 0, 3, 4, {"vec"});
  meta.set_vector_validity(true);
  std::string encoded;
  ManifestCodec::EncodeBlockMeta(meta, &encoded);
  EXPECT_EQ(*ManifestCodec::DecodeBlockMeta(encoded), meta);
  EXPECT_TRUE(ManifestCodec::DecodeBlockMeta(encoded)->vector_validity());
}

TEST_F(NullableVectorStorageTest, SparseNullsSurviveReindexAndReopen) {
  using Sparse = std::pair<std::vector<uint32_t>, std::vector<float>>;
  const std::string path = "test_nullable_sparse";
  for (bool mmap : {false, true}) {
    FileHelper::RemoveDirectory(path);
    CollectionSchema schema("nullable_sparse");
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "ordinal", DataType::INT32, false))
                    .ok());
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "vec", DataType::SPARSE_VECTOR_FP32, 0, true,
                        std::make_shared<FlatIndexParams>(MetricType::IP)))
                    .ok());
    CollectionOptions options;
    options.enable_mmap_ = mmap;
    auto opened = Collection::CreateAndOpen(path, schema, options);
    ASSERT_TRUE(opened);
    auto collection = std::move(*opened);
    std::vector<std::string> pks;
    std::map<std::string, Sparse> vectors;
    for (int i = 0; i < 8; ++i) {
      Doc doc;
      doc.set_pk("d" + std::to_string(i));
      doc.set<int32_t>("ordinal", i);
      pks.push_back(doc.pk());
      if (i % 2 == 1 && i != 7) {
        Sparse value{{1, 3}, {float(i), float(i + 1)}};
        doc.set("vec", value);
        vectors[doc.pk()] = value;
      } else if (i == 7)
        doc.set_null("vec");
      std::vector<Doc> docs{doc};
      auto written = collection->insert(docs);
      ASSERT_TRUE(written);
      ASSERT_TRUE(written->at(0).ok());
      if (i == 3) {
        ASSERT_TRUE(collection->flush().ok());
      }
    }
    auto check = [&] {
      auto fetched = collection->fetch(pks, std::nullopt, true);
      ASSERT_TRUE(fetched);
      ASSERT_EQ(fetched->size(), 8U);
      for (const auto &[pk, doc] : *fetched) {
        ASSERT_NE(doc, nullptr) << pk;
        auto value = doc->get<Sparse>("vec");
        ASSERT_EQ(value.has_value(), vectors.count(pk) != 0) << pk;
        if (value) {
          EXPECT_EQ(*value, vectors.at(pk));
        }
      }
      SearchQuery scan;
      scan.filter_ = "ordinal >= 0";
      scan.topk_ = 10;
      scan.include_vector_ = true;
      auto rows = collection->query(scan);
      ASSERT_TRUE(rows) << rows.error().message();
      ASSERT_EQ(rows->size(), 8U);
      for (const auto &doc : *rows)
        EXPECT_EQ(doc->get<Sparse>("vec").has_value(),
                  vectors.count(doc->pk()) != 0);
      const uint32_t index = 1;
      const float one = 1;
      SearchQuery knn;
      knn.target_.field_name_ = "vec";
      knn.target_.set_sparse_vector(
          std::string(reinterpret_cast<const char *>(&index), sizeof(index)),
          std::string(reinterpret_cast<const char *>(&one), sizeof(one)));
      knn.topk_ = 10;
      knn.include_vector_ = true;
      auto hits = collection->query(knn);
      ASSERT_TRUE(hits) << hits.error().message();
      EXPECT_FALSE(hits->empty());
      for (const auto &doc : *hits) EXPECT_NE(vectors.count(doc->pk()), 0U);
      auto iter = collection->create_iterator();
      ASSERT_TRUE(iter);
      size_t count = 0;
      while (true) {
        auto next = (*iter)->next();
        ASSERT_TRUE(next) << next.error().message();
        if (!*next) break;
        EXPECT_EQ((*next)->get<Sparse>("vec").has_value(),
                  vectors.count((*next)->pk()) != 0);
        ++count;
      }
      EXPECT_EQ(count, 8U);
      (*iter)->close();
    };
    check();
    ASSERT_TRUE(collection->flush().ok());
    ASSERT_TRUE(collection
                    ->create_index("vec", std::make_shared<HnswIndexParams>(
                                              MetricType::IP))
                    .ok());
    check();
    ASSERT_TRUE(collection->optimize().ok());
    collection.reset();
    opened = Collection::Open(path, options);
    ASSERT_TRUE(opened) << opened.error().message();
    collection = std::move(*opened);
    check();
    collection.reset();
    FileHelper::RemoveDirectory(path);
  }
}

#if GTEST_HAS_DEATH_TEST
void WriteUnflushedNullableCollection() {
  const std::string path = "test_nullable_recovery";
  ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  CollectionSchema schema("nullable_recovery");
  schema.add_field(std::make_shared<FieldSchema>(
      "vec", DataType::VECTOR_FP32, 16, true,
      std::make_shared<HnswIndexParams>(MetricType::L2)));
  auto result = Collection::CreateAndOpen(path, schema, {});
  if (!result) std::_Exit(1);
  auto collection = std::move(*result);
  for (int i = 0; i < 8; ++i) {
    Doc doc;
    doc.set_pk("d" + std::to_string(i));
    if (i % 2 == 0)
      doc.set("vec", std::vector<float>(16, float(i)));
    else
      doc.set_null("vec");
    std::vector<Doc> docs{doc};
    auto written = collection->upsert(docs);
    if (!written || !written->at(0).ok()) std::_Exit(2);
  }
  std::_Exit(0);  // No destructor/flush: replay the committed WAL on reopen.
}

TEST(NullableVectorRecoveryTest, WalRebuildsNullBitmapIncludingTrailingNull) {
  const std::string path = "test_nullable_recovery";
  FileHelper::RemoveDirectory(path);
  const auto style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(WriteUnflushedNullableCollection(), ::testing::ExitedWithCode(0),
              "");
  ::testing::FLAGS_gtest_death_test_style = style;
  auto result = Collection::Open(path, {});
  ASSERT_TRUE(result) << result.error().message();
  auto collection = std::move(*result);
  for (int stage = 0; stage < 2; ++stage) {
    for (int i = 0; i < 8; ++i) {
      auto fetched =
          collection->fetch({"d" + std::to_string(i)}, std::nullopt, true);
      ASSERT_TRUE(fetched);
      auto doc = fetched->begin()->second;
      ASSERT_NE(doc, nullptr);
      auto value = doc->get<std::vector<float>>("vec");
      EXPECT_EQ(value.has_value(), i % 2 == 0);
      if (value) {
        EXPECT_EQ(*value, std::vector<float>(16, float(i)));
      }
    }
    ASSERT_TRUE(collection->flush().ok());
    collection.reset();
    result = Collection::Open(path, {});
    ASSERT_TRUE(result) << result.error().message();
    collection = std::move(*result);
  }
  collection.reset();
  FileHelper::RemoveDirectory(path);
}
#endif
}  // namespace
