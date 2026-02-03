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
// limitations under the License

#include "db/sqlengine/sqlengine_impl.h"
#include <unordered_map>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/doc.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"
#include "db/sqlengine/analyzer/query_analyzer.h"
#include "db/sqlengine/parser/sql_info_helper.h"
#include "db/sqlengine/parser/zvec_parser.h"
#include "db/sqlengine/planner/op_register.h"
#include "db/sqlengine/planner/query_planner.h"

namespace zvec::sqlengine {

void global_init() {
  static std::once_flag once;
  // run once
  std::call_once(once, []() {
    auto status = arrow::compute::Initialize();
    if (!status.ok()) {
      LOG_ERROR("arrow compute init failed: [%s]", status.ToString().c_str());
      abort();
    }
    status = OpRegister::register_ops();
    if (!status.ok()) {
      LOG_ERROR("arrow compute register op failed: [%s]",
                status.ToString().c_str());
      abort();
    }
  });
}

SQLEngine::~SQLEngine() = default;

SQLEngineImpl::SQLEngineImpl(zvec::Profiler::Ptr profiler)
    : profiler_(std::move(profiler)) {}

Result<DocPtrList> SQLEngineImpl::execute(
    CollectionSchema::Ptr collection, const VectorQuery &query,
    const std::vector<Segment::Ptr> &segments) {
  if (segments.empty()) {
    return DocPtrList{};
  }

  auto query_info = parse_request(collection, query, nullptr);
  if (!query_info) {
    return tl::make_unexpected(query_info.error());
  }
  if (query_info.value()->is_filter_unsatisfiable()) {
    LOG_WARN("filter is unsatisfiable: %s",
             query_info.value()->to_string().c_str());
    return {};
  }
  const auto &select_item_meta_ptrs =
      query_info.value()->select_item_schema_ptrs();
  std::vector<QueryInfo::Ptr> query_infos(segments.size(), query_info.value());
  auto reader = search_by_query_info(collection, segments, &query_infos);
  if (!reader) {
    return tl::make_unexpected(
        Status::InternalError("Execute plan failed: ", reader.error().c_str()));
  }
  return fill_result(select_item_meta_ptrs, reader.value().get());
}

VectorQuery from_group_by(const GroupByVectorQuery &gq) {
  VectorQuery vq;
  vq.field_name_ = gq.field_name_;
  vq.query_vector_ = gq.query_vector_;
  vq.query_sparse_indices_ = gq.query_sparse_indices_;
  vq.query_sparse_values_ = gq.query_sparse_values_;
  vq.filter_ = gq.filter_;
  vq.include_vector_ = gq.include_vector_;
  vq.query_params_ = gq.query_params_;
  vq.output_fields_ = gq.output_fields_;
  vq.topk_ = 0;
  return vq;
}

Result<GroupResults> SQLEngineImpl::execute_group_by(
    CollectionSchema::Ptr collection, const GroupByVectorQuery &group_by_query,
    const std::vector<Segment::Ptr> &segments) {
  if (segments.empty()) {
    return GroupResults{};
  }

  VectorQuery query = from_group_by(group_by_query);
  auto query_info = parse_request(
      collection, query,
      std::make_shared<GroupBy>(group_by_query.group_by_field_name_,
                                group_by_query.group_topk_,
                                group_by_query.group_count_));
  if (!query_info) {
    return tl::make_unexpected(query_info.error());
  }
  if (query_info.value()->is_filter_unsatisfiable()) {
    LOG_WARN("filter is unsatisfiable: %s",
             query_info.value()->to_string().c_str());
    return {};
  }
  std::vector<QueryInfo::Ptr> query_infos(segments.size(), query_info.value());
  auto reader = search_by_query_info(collection, segments, &query_infos);
  if (!reader) {
    return tl::make_unexpected(
        Status::InternalError("Execute plan failed: ", reader.error().c_str()));
  }
  return fill_group_by_result(*query_info.value(), reader.value().get());
}

Result<QueryInfo::Ptr> SQLEngineImpl::parse_sql_info(
    const CollectionSchema &schema, const SQLInfo::Ptr &sql_info) {
  profiler_->open_stage("analyze stage");
  QueryAnalyzer analyzer;
  auto query_info = analyzer.analyze(schema, sql_info);
  if (!query_info) {
    return tl::make_unexpected(Status::InvalidArgument(
        "Analyze sql info failed:", query_info.error().c_str()));
  }
  profiler_->close_stage();
  LOG_DEBUG("query_info: [%s]", query_info.value()->to_string().c_str());
  return query_info.value();
}

Result<QueryInfo::Ptr> SQLEngineImpl::parse_request(
    CollectionSchema::Ptr collection, const VectorQuery &request,
    std::shared_ptr<GroupBy> group_by) {
  profiler_->open_stage("message_to_sqlinfo");
  sqlengine::SQLInfo::Ptr sql_info;
  std::string err_msg;
  Node::Ptr filter_node;
  if (!request.filter_.empty()) {
    ZVecParser::Ptr parser = ZVecParser::create();
    filter_node = parser->parse_filter(request.filter_);
    if (filter_node == nullptr) {
      LOG_ERROR("parse filter failed. reason:[%s] filter:[%s]",
                parser->err_msg().c_str(), request.filter_.c_str());
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid filter:", parser->err_msg()));
    }
  }
  if (group_by) {
    auto &group = *group_by;
    if (group.group_by_field.empty() || group.group_count == 0 ||
        group.group_topk == 0) {
      return tl::make_unexpected(Status::InvalidArgument(
          "Invalid group by request: group_by", group.group_by_field,
          " group_count: ", group.group_count,
          " group_topk: ", group.group_topk));
    }
  }

  sqlengine::SQLInfoHelper::MessageToSQLInfo(&request, std::move(filter_node),
                                             std::move(group_by), &sql_info,
                                             &err_msg);
  profiler_->close_stage();
  if (!err_msg.empty()) {
    LOG_ERROR("QueryAgent, message to sql info failed, err_msg: %s",
              err_msg.c_str());
    return tl::make_unexpected(
        Status::InvalidArgument("To sql info failed:", err_msg));
  }
  LOG_DEBUG("Sql info is %s", sql_info->to_string().c_str());
  return parse_sql_info(*collection, std::move(sql_info));
}

Result<std::unique_ptr<arrow::RecordBatchReader>>
SQLEngineImpl::search_by_query_info(
    CollectionSchema::Ptr collection, const std::vector<Segment::Ptr> &segments,
    std::vector<sqlengine::QueryInfo::Ptr> *query_infos) {
  global_init();

  profiler_->open_stage("plan stage");
  QueryPlanner planner(collection.get());
  auto plan_info =
      planner.make_plan(segments, profiler_->trace_id(), query_infos);
  if (!plan_info) {
    LOG_ERROR("plan query_info failed: [%s]", plan_info.error().c_str());
    return tl::make_unexpected(plan_info.error());
  }
  profiler_->close_stage();
  // LOG_DEBUG("plan_info: [%s]", plan_info->to_string().c_str());
  return plan_info.value()->execute_to_reader();
}

#define GET_FIELD_FROM_RECORD_BATCH(res, field_name)                         \
  auto res = record_batch.GetColumnByName(field_name);                       \
  if (!res) {                                                                \
    return Status::InternalError("Get column by name failed: ", field_name); \
  }

template <typename T>
std::vector<T> to_vector(const char *data, size_t size) {
  std::vector<T> vec(size);
  memcpy(vec.data(), data, size * sizeof(T));
  return vec;
}

template <typename VectorType>
Status fill_doc_sparse_vector(const arrow::StructArray *typed_arr,
                              const std::string &field_name,
                              DocPtrList::iterator doc_it) {
  auto *indices = (const arrow::BinaryArray *)typed_arr->field(0).get();
  auto *values = (const arrow::BinaryArray *)typed_arr->field(1).get();
  bool has_null = typed_arr->null_count() > 0;
  for (int64_t i = 0; i < typed_arr->length(); ++i, ++doc_it) {
    if (has_null && typed_arr->IsNull(i)) {
      continue;
    }
    auto indice_data = indices->GetView(i);
    auto value_data = values->GetView(i);
    uint32_t count = indice_data.size() / sizeof(uint32_t);
    if (count != value_data.size() / sizeof(VectorType)) {
      return Status::InvalidArgument("Dimension not match:", count, " vs ",
                                     value_data.size() / sizeof(VectorType));
    }
    (*doc_it)->set(
        field_name,
        std::make_pair(to_vector<uint32_t>(indice_data.data(), count),
                       to_vector<VectorType>(value_data.data(), count)));
  }
  return Status::OK();
}

template <typename VectorType>
Status fill_doc_vector(const arrow::BinaryArray *typed_arr,
                       const std::string &field_name, int dimension,
                       DocPtrList::iterator doc_it) {
  bool no_null = typed_arr->null_count() == 0;
  for (int64_t i = 0; i < typed_arr->length(); ++i, ++doc_it) {
    if (no_null || !typed_arr->IsNull(i)) {
      auto data = typed_arr->GetView(i);
      if ((size_t)dimension != data.size() / sizeof(VectorType)) {
        return Status::InvalidArgument("Dimension not match:", dimension,
                                       " vs ",
                                       data.size() / sizeof(VectorType));
      }
      (*doc_it)->set(field_name, std::vector<VectorType>(
                                     (const VectorType *)&data[0],
                                     (const VectorType *)&data[0] + dimension));
    }
  }
  return Status::OK();
}

template <typename ArrowArrayType>
Status fill_doc_field(const arrow::Array *arr, const std::string &field_name,
                      DocPtrList::iterator doc_it) {
  auto *typed_arr = static_cast<const ArrowArrayType *>(arr);
  bool no_null = typed_arr->null_count() == 0;
  for (int64_t i = 0; i < typed_arr->length(); ++i, ++doc_it) {
    if (no_null || !typed_arr->IsNull(i)) {
      if constexpr (std::is_same_v<ArrowArrayType, arrow::StringArray> ||
                    std::is_same_v<ArrowArrayType, arrow::LargeStringArray> ||
                    std::is_same_v<ArrowArrayType, arrow::BinaryArray> ||
                    std::is_same_v<ArrowArrayType, arrow::LargeBinaryArray>) {
        (*doc_it)->set(field_name, typed_arr->GetString(i));
      } else {
        (*doc_it)->set(field_name, typed_arr->Value(i));
      }
    }
  }
  return Status::OK();
}

template <typename ArrowArrayType, typename ElementType>
Status fill_doc_array_field(const arrow::Array *arr,
                            const std::string &field_name,
                            DocPtrList::iterator doc_it) {
  const auto *list_arr = static_cast<const arrow::ListArray *>(arr);
  auto *typed_arr =
      dynamic_cast<const ArrowArrayType *>(list_arr->values().get());
  bool has_null = list_arr->null_count() > 0;
  for (int64_t i = 0; i < list_arr->length(); ++i, ++doc_it) {
    if (has_null && list_arr->IsNull(i)) {
      continue;
    }
    int64_t offset = list_arr->value_offset(i);
    int64_t length = list_arr->value_length(i);
    std::vector<ElementType> vec(length);
    for (int64_t j = 0; j < length; ++j) {
      vec[j] = typed_arr->Value(offset + j);
    }
    (*doc_it)->set(field_name, std::move(vec));
  }
  return Status::OK();
}

Status fill_doc_field(const std::shared_ptr<arrow::Array> &chunk,
                      const FieldSchema &field_schema,
                      DocPtrList::iterator doc_it) {
  switch (field_schema.data_type()) {
    case DataType::INT32:
      return fill_doc_field<arrow::Int32Array>(chunk.get(), field_schema.name(),
                                               doc_it);
    case DataType::UINT32:
      return fill_doc_field<arrow::UInt32Array>(chunk.get(),
                                                field_schema.name(), doc_it);
    case DataType::INT64:
      return fill_doc_field<arrow::Int64Array>(chunk.get(), field_schema.name(),
                                               doc_it);
    case DataType::UINT64:
      return fill_doc_field<arrow::UInt64Array>(chunk.get(),
                                                field_schema.name(), doc_it);
    case DataType::FLOAT:
      return fill_doc_field<arrow::FloatArray>(chunk.get(), field_schema.name(),
                                               doc_it);
    case DataType::DOUBLE:
      return fill_doc_field<arrow::DoubleArray>(chunk.get(),
                                                field_schema.name(), doc_it);
    case DataType::BOOL:
      return fill_doc_field<arrow::BooleanArray>(chunk.get(),
                                                 field_schema.name(), doc_it);
    case DataType::BINARY:
      return fill_doc_field<arrow::BinaryArray>(chunk.get(),
                                                field_schema.name(), doc_it);

    case DataType::STRING:
      return fill_doc_field<arrow::StringArray>(chunk.get(),
                                                field_schema.name(), doc_it);

    case DataType::ARRAY_INT32:
      return fill_doc_array_field<arrow::Int32Array, int32_t>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_INT64:
      return fill_doc_array_field<arrow::Int64Array, int64_t>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_UINT32:
      return fill_doc_array_field<arrow::UInt32Array, uint32_t>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_UINT64:
      return fill_doc_array_field<arrow::UInt64Array, uint64_t>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_FLOAT:
      return fill_doc_array_field<arrow::FloatArray, float>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_DOUBLE:
      return fill_doc_array_field<arrow::DoubleArray, double>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_STRING:
      return fill_doc_array_field<arrow::StringArray, std::string>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_BINARY:
      return fill_doc_array_field<arrow::BinaryArray, std::string>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::ARRAY_BOOL:
      return fill_doc_array_field<arrow::BooleanArray, bool>(
          chunk.get(), field_schema.name(), doc_it);

    case DataType::VECTOR_FP32:
      return fill_doc_vector<float>((arrow::BinaryArray *)chunk.get(),
                                    field_schema.name(),
                                    field_schema.dimension(), doc_it);

    case DataType::VECTOR_FP64:
      return fill_doc_vector<double>((arrow::BinaryArray *)chunk.get(),
                                     field_schema.name(),
                                     field_schema.dimension(), doc_it);
    case DataType::VECTOR_FP16:
      return fill_doc_vector<float16_t>((arrow::BinaryArray *)chunk.get(),
                                        field_schema.name(),
                                        field_schema.dimension(), doc_it);

    case DataType::VECTOR_INT16:
      return fill_doc_vector<int16_t>((arrow::BinaryArray *)chunk.get(),
                                      field_schema.name(),
                                      field_schema.dimension(), doc_it);

    case DataType::VECTOR_INT8:
      return fill_doc_vector<int8_t>((arrow::BinaryArray *)chunk.get(),
                                     field_schema.name(),
                                     field_schema.dimension(), doc_it);

    case DataType::VECTOR_BINARY32:
      return fill_doc_vector<uint32_t>(
          (arrow::BinaryArray *)chunk.get(), field_schema.name(),
          field_schema.dimension() / sizeof(uint32_t), doc_it);

    case DataType::VECTOR_BINARY64:
      return fill_doc_vector<uint64_t>(
          (arrow::BinaryArray *)chunk.get(), field_schema.name(),
          field_schema.dimension() / sizeof(uint64_t), doc_it);

    case DataType::SPARSE_VECTOR_FP32:
      return fill_doc_sparse_vector<float>((arrow::StructArray *)chunk.get(),
                                           field_schema.name(), doc_it);

    case DataType::SPARSE_VECTOR_FP16:
      return fill_doc_sparse_vector<float16_t>(
          (arrow::StructArray *)chunk.get(), field_schema.name(), doc_it);

    default:
      return Status::InvalidArgument("Datatype not supported:",
                                     field_schema.data_type());
  }
  return Status::OK();
}

void fill_doc_id(const std::shared_ptr<arrow::Array> &doc_id_array,
                 DocPtrList::iterator doc_it) {
  arrow::UInt64Array *typed_arr =
      static_cast<arrow::UInt64Array *>(doc_id_array.get());
  for (int64_t i = 0; i < typed_arr->length(); ++i, ++doc_it) {
    // doc_id is non-null
    (*doc_it)->set_doc_id(typed_arr->Value(i));
  }
}

void fill_doc_score(const std::shared_ptr<arrow::Array> &doc_id_array,
                    DocPtrList::iterator doc_it) {
  arrow::FloatArray *typed_arr =
      static_cast<arrow::FloatArray *>(doc_id_array.get());
  for (int64_t i = 0; i < typed_arr->length(); ++i, ++doc_it) {
    // doc_score is non-null
    (*doc_it)->set_score(typed_arr->Value(i));
  }
}

void fill_user_id(const std::shared_ptr<arrow::Array> &user_id_array,
                  DocPtrList::iterator doc_it) {
  arrow::StringArray *typed_arr =
      static_cast<arrow::StringArray *>(user_id_array.get());
  for (int64_t i = 0; i < typed_arr->length(); ++i, ++doc_it) {
    // user_id is non-null
    (*doc_it)->set_pk(typed_arr->GetString(i));
  }
}

Status record_batch_to_doc_list(
    const std::vector<FieldAndSchema> &output_fields,
    const arrow::RecordBatch &record_batch, DocPtrList::iterator doc_it) {
  GET_FIELD_FROM_RECORD_BATCH(user_id_array, USER_ID);
  fill_user_id(user_id_array, doc_it);
  if (auto doc_id_array = record_batch.GetColumnByName(GLOBAL_DOC_ID);
      doc_id_array != nullptr) {
    fill_doc_id(doc_id_array, doc_it);
  }
  if (auto score_array = record_batch.GetColumnByName(kFieldScore);
      score_array != nullptr) {
    fill_doc_score(score_array, doc_it);
  }

  for (auto &[field_name, field_schema] : output_fields) {
    GET_FIELD_FROM_RECORD_BATCH(field_array, field_name);
    if (auto status = fill_doc_field(field_array, *field_schema, doc_it);
        !status.ok()) {
      return status;
    }
  }
  if (ailego::LoggerBroker::IsLevelEnabled(ailego::Logger::LEVEL_DEBUG)) {
    for (int i = 0; i < record_batch.num_rows(); i++) {
      LOG_DEBUG("Doc: %s", (*(doc_it + i))->to_detail_string().c_str());
    }
  }
  return Status::OK();
}

Result<DocPtrList> SQLEngineImpl::fill_result(
    const std::vector<FieldAndSchema> &output_fields,
    arrow::RecordBatchReader *reader) {
  DocPtrList docs;
  std::shared_ptr<RecordBatch> record_batch;
  while (true) {
    auto read_res = reader->ReadNext(&record_batch);
    if (!read_res.ok()) {
      return tl::make_unexpected(Status::InternalError(
          "Read record batch failed: ", read_res.ToString()));
    }
    if (record_batch == nullptr) {
      break;
    }
    size_t cur_size = docs.size();
    docs.resize(docs.size() + record_batch->num_rows());
    for (int i = 0; i < record_batch->num_rows(); i++) {
      docs[cur_size + i] = std::make_shared<Doc>();
    }
    auto status = record_batch_to_doc_list(output_fields, *record_batch,
                                           docs.begin() + cur_size);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  return docs;
}


Result<GroupResults> SQLEngineImpl::fill_group_by_result(
    const QueryInfo &query_info, arrow::RecordBatchReader *reader) {
  const std::vector<FieldAndSchema> &output_fields =
      query_info.select_item_schema_ptrs();
  uint32_t group_count = query_info.group_by()->group_count;
  uint32_t group_topk = query_info.group_by()->group_topk;
  std::shared_ptr<RecordBatch> record_batch;
  std::unordered_map<std::string, std::vector<Doc>> group_to_docs;
  while (true) {
    auto read_res = reader->ReadNext(&record_batch);
    if (!read_res.ok()) {
      return tl::make_unexpected(Status::InternalError(
          "Read record batch failed: ", read_res.ToString()));
    }
    if (record_batch == nullptr) {
      break;
    }
    DocPtrList docs(record_batch->num_rows());
    for (int i = 0; i < record_batch->num_rows(); i++) {
      docs[i] = std::make_shared<Doc>();
    }
    auto status =
        record_batch_to_doc_list(output_fields, *record_batch, docs.begin());
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
    auto group_id_array = record_batch->GetColumnByName(kFieldGroupId);
    if (!group_id_array) {
      return tl::make_unexpected(
          Status::InternalError("Get group_id_array failed"));
    }
    arrow::StringArray *typed_arr =
        static_cast<arrow::StringArray *>(group_id_array.get());
    for (int i = 0; i < record_batch->num_rows(); i++) {
      if (!typed_arr->IsNull(i)) {
        // docs already order by score
        auto &group_docs = group_to_docs[typed_arr->GetString(i)];
        if (group_docs.size() < group_count) {
          group_docs.push_back(std::move(*docs[i]));
        }
      }
    }
  }
  GroupResults group_results;
  for (auto &kv : group_to_docs) {
    group_results.emplace_back(
        GroupResult{std::move(kv.first), std::move(kv.second)});
  }
  std::sort(group_results.begin(), group_results.end(),
            [&query_info](GroupResult &a, GroupResult &b) {
              if (query_info.vector_cond_info()->is_reverse_sort()) {
                return a.docs_[0].score() > b.docs_[0].score();
              }
              return a.docs_[0].score() < b.docs_[0].score();
            });
  if (group_results.size() > group_topk) {
    group_results.resize(group_topk);
  }
  for (auto &group_result : group_results) {
    LOG_DEBUG("Group: %s", group_result.group_by_value_.c_str());
    for (auto &doc : group_result.docs_) {
      LOG_DEBUG("\tDoc: %s", doc.to_detail_string().c_str());
    }
  }
  return group_results;
}

}  // namespace zvec::sqlengine