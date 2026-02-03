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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include <zvec/ailego/logger/logger.h>
#include "db/index/column/common/index_results.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/segment/segment.h"
#include "index/column/inverted_column/inverted_column_indexer.h"
#include "index/column/vector_column/vector_column_params.h"
#include "index/common/index_filter.h"
namespace zvec {


class MockIndexResult : public InvertedSearchResult {
 public:
  MockIndexResult(const std::vector<idx_t> &doc_ids,
                  const std::vector<float> &scores)
      : doc_ids_(doc_ids), scores_(scores) {}

  MockIndexResult(const std::vector<idx_t> &doc_ids,
                  const std::vector<float> &scores,
                  const std::vector<std::string> &groups)
      : doc_ids_(doc_ids), scores_(scores), group_ids_(groups) {}

  size_t count() const override {
    return doc_ids_.size();
  }

  IteratorUPtr create_iterator() override {
    return std::make_unique<MockIterator>(*this);
  }

 private:
  struct MockIterator : public IndexResults::Iterator {
    MockIterator(MockIndexResult &parent) : parent_(parent) {}

    idx_t doc_id() const override {
      return parent_.doc_ids_[current_index_];
    }

    float score() const override {
      return parent_.scores_[current_index_];
    }

    void next() override {
      ++current_index_;
    }

    bool valid() const override {
      return current_index_ < parent_.count();
    }

    const std::string &group_id() const override {
      return parent_.group_ids_[current_index_];
    }

    MockIndexResult &parent_;
    size_t current_index_{0};
  };

  std::vector<idx_t> doc_ids_;
  std::vector<float> scores_;
  std::vector<std::string> group_ids_;
};

class MockVectorIndexer : public CombinedVectorColumnIndexer {
 public:
  //! Search results with query
  Result<IndexResults::Ptr> Search(
      const vector_column_params::VectorData &vector_data,
      const vector_column_params::QueryParams &query_params) override {
    // return tl::make_unexpected(Status::InternalError("err"));
    return std::make_shared<MockIndexResult>(
        std::vector<idx_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F, 0.9F,
                           1.0F},
        std::vector<std::string>{"group_0", "group_1", "group_2", "group_0",
                                 "group_1", "group_2", "group_0", "group_1",
                                 "group_2", "group_0"});
  }

  Result<vector_column_params::VectorDataBuffer> Fetch(
      uint32_t doc_id) const override {
    // float f = doc_id;
    // std::vector<float> v(4, f);
    // std::string v_str = std::string(reinterpret_cast<char *>(v.data()),
    //                                 v.size() * sizeof(float));
    // return vector_column_params::VectorDataBuffer{
    //     vector_column_params::DenseVectorBuffer{v_str}};

    // sparse
    uint32_t count = doc_id % 5;
    std::vector<uint32_t> indices(count);
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; i++) {
      indices[i] = i;
      values[i] = i / 100.0;
    }
    return vector_column_params::VectorDataBuffer{
        vector_column_params::SparseVectorBuffer{
            std::string(reinterpret_cast<char *>(indices.data()),
                        indices.size() * sizeof(uint32_t)),
            std::string(reinterpret_cast<char *>(values.data()),
                        values.size() * sizeof(float))}};
  }
};

class MockInvertIndexer : public InvertedColumnIndexer {
 public:
  MockInvertIndexer() : InvertedColumnIndexer(ctx) {}

  InvertedSearchResult::Ptr search(const std::string &value,
                                   CompareOp op) const override {
    return std::make_shared<MockIndexResult>(
        std::vector<idx_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F, 0.9F,
                           1.0F});
  }

  InvertedSearchResult::Ptr search_null() const override {
    return std::make_shared<MockIndexResult>(
        std::vector<idx_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F, 0.9F,
                           1.0F});
  }

  InvertedSearchResult::Ptr search_non_null() const override {
    return std::make_shared<MockIndexResult>(
        std::vector<idx_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F, 0.9F,
                           1.0F});
  }

 private:
  RocksdbContext ctx;
};

//   std::make_shared<FieldSchema>("id", DataType::INT32, false, 0, false,
//                                 nullptr),
//   std::make_shared<FieldSchema>("name", DataType::STRING, false, 0,
//                                 false, nullptr),
//   std::make_shared<FieldSchema>("age", DataType::INT64, false, 0,
//                                 false, nullptr),
//   std::make_shared<FieldSchema>("score", DataType::DOUBLE, false, 0,
//                                 false, nullptr),
inline arrow::Result<std::shared_ptr<arrow::Table>> CreateTable(
    int count = 10000000) {
  auto schema = arrow::schema({
      arrow::field("id", arrow::int32()),
      arrow::field("name", arrow::utf8()),
      arrow::field("age", arrow::int64()),
      arrow::field("score", arrow::float64()),
      arrow::field("_zvec_uid_", arrow::utf8()),
      arrow::field("_zvec_row_id_", arrow::uint64()),
      arrow::field("_zvec_g_doc_id_", arrow::uint64()),
      arrow::field("tag_list", arrow::list(arrow::int32())),
  });
  std::shared_ptr<arrow::Array> array_id;
  std::shared_ptr<arrow::Array> array_name;
  std::shared_ptr<arrow::Array> array_age;
  std::shared_ptr<arrow::Array> array_score;
  std::shared_ptr<arrow::Array> array_uid;
  arrow::NumericBuilder<arrow::Int64Type> builder;
  auto has_value = [](int i) { return i % 13 != 0; };
  ARROW_RETURN_NOT_OK(builder.Reserve(count));
  for (int i = 0; i < count; i++) {
    if (has_value(i)) {
      ARROW_RETURN_NOT_OK((builder.Append(i)));
    } else {
      ARROW_RETURN_NOT_OK((builder.AppendNull()));
    }
  }
  ARROW_RETURN_NOT_OK(builder.Finish(&array_age));
  builder.Reset();

  arrow::NumericBuilder<arrow::Int32Type> builder_id;
  ARROW_RETURN_NOT_OK(builder_id.Reserve(count));
  for (int i = 0; i < count; i++) {
    if (has_value(i)) {
      ARROW_RETURN_NOT_OK((builder_id.Append(i)));
    } else {
      ARROW_RETURN_NOT_OK((builder_id.AppendNull()));
    }
  }
  ARROW_RETURN_NOT_OK(builder_id.Finish(&array_id));

  arrow::NumericBuilder<arrow::DoubleType> builder_score;
  ARROW_RETURN_NOT_OK(builder_score.Reserve(count));
  for (int i = 0; i < count; i++) {
    if (has_value(i)) {
      ARROW_RETURN_NOT_OK((builder_score.Append(i / 100.0)));
    } else {
      ARROW_RETURN_NOT_OK((builder_score.AppendNull()));
    }
  }
  ARROW_RETURN_NOT_OK(builder_score.Finish(&array_score));


  arrow::StringBuilder builder_d;
  ARROW_RETURN_NOT_OK(builder_d.Reserve(count));
  for (int i = 0; i < count; i++) {
    if (has_value(i)) {
      ARROW_RETURN_NOT_OK((builder_d.Append("name_" + std::to_string(i))));
    } else {
      ARROW_RETURN_NOT_OK((builder_d.AppendNull()));
    }
  }
  ARROW_RETURN_NOT_OK(builder_d.Finish(&array_name));

  arrow::StringBuilder builder_uid;
  ARROW_RETURN_NOT_OK(builder_uid.Reserve(count));
  for (int i = 0; i < count; i++) {
    ARROW_RETURN_NOT_OK((builder_uid.Append("uid_" + std::to_string(i))));
  }
  ARROW_RETURN_NOT_OK(builder_uid.Finish(&array_uid));

  arrow::NumericBuilder<arrow::UInt64Type> builder_row_id;
  ARROW_RETURN_NOT_OK(builder_row_id.Reserve(count));
  for (int i = 0; i < count; i++) {
    ARROW_RETURN_NOT_OK((builder_row_id.Append(i)));
  }
  std::shared_ptr<arrow::Array> array_row_id;
  ARROW_RETURN_NOT_OK(builder_row_id.Finish(&array_row_id));

  arrow::NumericBuilder<arrow::UInt64Type> builder_doc_id;
  ARROW_RETURN_NOT_OK(builder_doc_id.Reserve(count));
  for (int i = 0; i < count; i++) {
    ARROW_RETURN_NOT_OK((builder_doc_id.Append(i)));
  }
  std::shared_ptr<arrow::Array> array_doc_id;
  ARROW_RETURN_NOT_OK(builder_doc_id.Finish(&array_doc_id));

  arrow::ListBuilder list_builder(arrow::default_memory_pool(),
                                  std::make_shared<arrow::Int32Builder>());
  auto *tag_value_builder =
      static_cast<arrow::Int32Builder *>(list_builder.value_builder());

  for (int i = 0; i < count; ++i) {
    // 开始一个新的 list
    ARROW_RETURN_NOT_OK(list_builder.Append());

    int idx = i % 5;  // 对应模式
    for (int j = 0; j < idx + 1; ++j) {
      ARROW_RETURN_NOT_OK(tag_value_builder->Append(j + 1));
    }
  }
  std::shared_ptr<arrow::Array> tag_list_array;
  auto status = list_builder.Finish(&tag_list_array);
  ;

  return arrow::Table::Make(
      schema, {array_id, array_name, array_age, array_score, array_uid,
               array_row_id, array_doc_id, tag_list_array});
}

class MockIndexFilter : public IndexFilter {
 public:
  bool is_filtered(uint64_t id) const override {
    return id % 2 == 1;
  }
};

inline arrow::Result<std::shared_ptr<Table>> TakeRowsByIndices(
    const std::shared_ptr<Table> &table, const std::vector<int> &row_indices) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  arrow::Int32Builder indices_builder(pool);
  ARROW_RETURN_NOT_OK(
      indices_builder.AppendValues(row_indices.data(), row_indices.size()));
  std::shared_ptr<arrow::Array> indices_array;
  ARROW_RETURN_NOT_OK(indices_builder.Finish(&indices_array));


  // 2. 对每一列执行 Take 操作
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
  for (const auto &column : table->columns()) {
    // 使用 Take 提取指定索引的元素
    ARROW_ASSIGN_OR_RAISE(auto taken_array, cp::Take(column, indices_array));
    new_columns.emplace_back(taken_array.chunked_array());
  }

  // 3. 构造新的 Table
  return arrow::Table::Make(table->schema(), new_columns, row_indices.size());
}


class MockSegment : public Segment {
 public:
  MockSegment() : Segment() {}

  virtual ~MockSegment() = default;

  SegmentID id() const override {
    return 0;
  }

  TablePtr fetch(const std::vector<std::string> &columns,
                 const std::vector<int> &indices) const override {
    std::string s = "";
    for (auto i : indices) {
      s += std::to_string(i);
      s += ",";
    }
    LOG_INFO("Fetch indices: %s %s", get_column_names(columns).c_str(),
             s.c_str());
    auto table = CreateTable(1000).MoveValueUnsafe();

    auto res = TakeRowsByIndices(table, indices);
    if (!res.ok()) {
      LOG_ERROR("Take error: %s", res.status().ToString().c_str());
      return nullptr;
    }
    LOG_INFO("Take: %s", res.ValueOrDie()->ToString().c_str());
    return res.MoveValueUnsafe();
  }

  ExecBatchPtr fetch(const std::vector<std::string> &columns,
                     int index) const override {
    LOG_ERROR("Not implemented");
    return nullptr;
  }

  static std::string get_column_names(const std::vector<std::string> &columns) {
    std::string s = "";
    for (auto i : columns) {
      s += i;
      s += ",";
    }
    return s;
  }

  RecordBatchReaderPtr scan(
      const std::vector<std::string> &columns) const override {
    auto table = CreateTable(10000);
    LOG_INFO("Scan return: %s %s", get_column_names(columns).c_str(),
             table.ValueOrDie()->ToString().c_str());
    return std::make_shared<arrow::TableBatchReader>(table.ValueOrDie());
  }

  const IndexFilter::Ptr get_filter() override {
    return std::make_shared<MockIndexFilter>();
  }

  CombinedVectorColumnIndexer::Ptr get_quant_combined_vector_indexer(
      const std::string &field_name) const override {
    return std::make_shared<MockVectorIndexer>();
  }

  CombinedVectorColumnIndexer::Ptr get_combined_vector_indexer(
      const std::string &field_name) const override {
    return std::make_shared<MockVectorIndexer>();
  }

  InvertedColumnIndexer::Ptr get_scalar_indexer(
      const std::string &field_name) const override {
    return std::make_shared<MockInvertIndexer>();
  }

  SegmentMeta::Ptr meta() const override {
    return nullptr;
  }

  uint64_t doc_count(const IndexFilter::Ptr filter = nullptr) override {
    return 0;
  }

  Status add_column(FieldSchema::Ptr column_schema,
                    const std::string &expression,
                    const AddColumnOptions &options) override {
    return Status::InternalError();
  }

  Status alter_column(const std::string &column_name,
                      const FieldSchema::Ptr &new_column_schema,
                      const AlterColumnOptions &options) override {
    return Status::InternalError();
  }


  Status drop_column(const std::string &column_name) override {
    return Status::OK();
  }

  Status create_all_vector_index(
      int concurrency, SegmentMeta::Ptr *new_segmnet_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) override {
    return Status::OK();
  }

  Status create_vector_index(
      const std::string &column, const IndexParams::Ptr &index_params,
      int concurrency, SegmentMeta::Ptr *new_segmnet_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) override {
    return Status::OK();
  }

  Status drop_vector_index(
      const std::string &column, SegmentMeta::Ptr *new_segmnet_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers) override {
    return Status::OK();
  }

  Status reload_vector_index(
      const CollectionSchema &schema, const SegmentMeta::Ptr &segment_meta,
      const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          &vector_indexers,
      const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          &quant_vector_indexers) override {
    return Status::OK();
  }

  bool vector_index_ready(const std::string &column,
                          const IndexParams::Ptr &index_params) const override {
    return true;
  }

  bool all_vector_index_ready() const override {
    return true;
  }

  Status create_scalar_index(
      const std::vector<std::string> &columns,
      const IndexParams::Ptr &index_params, SegmentMeta::Ptr *new_segment_meta,
      InvertedIndexer::Ptr *new_scalar_indexer) override {
    return Status::OK();
  }

  Status drop_scalar_index(const std::vector<std::string> &columns,
                           SegmentMeta::Ptr *new_segment_meta,
                           InvertedIndexer::Ptr *new_scalar_indexer) override {
    return Status::OK();
  }

  Status reload_scalar_index(
      const CollectionSchema &schema, const SegmentMeta::Ptr &segment_meta,
      const InvertedIndexer::Ptr &scalar_indexer) override {
    return Status::OK();
  }

  Status Insert(Doc &doc) override {
    return Status::OK();
  }

  Status Upsert(Doc &doc) override {
    return Status::OK();
  }

  Status Update(Doc &doc) override {
    return Status::OK();
  }

  Status Delete(const std::string &pk) override {
    return Status::OK();
  }

  Status Delete(uint64_t doc_id) override {
    return Status::OK();
  }

  Doc::Ptr Fetch(uint64_t doc_id) override {
    return nullptr;
  }

  std::vector<VectorColumnIndexer::Ptr> get_vector_indexer(
      const std::string &field_name) const override {
    return {};
  }

  std::vector<VectorColumnIndexer::Ptr> get_quant_vector_indexer(
      const std::string &field_name) const override {
    return {};
  }

  Status flush() override {
    return Status::OK();
  }

  Status dump() override {
    return Status::OK();
  }

  Status destroy() override {
    return Status::OK();
  }
};

}  // namespace zvec
