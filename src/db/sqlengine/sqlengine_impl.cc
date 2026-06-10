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
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"
#include "db/index/column/fts_column/fts_ast_rewriter.h"
#include "db/index/column/fts_column/fts_pipeline.h"
#include "db/index/column/fts_column/fts_query_ast.h"
#include "db/sqlengine/analyzer/query_analyzer.h"
#include "db/sqlengine/parser/select_info.h"
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
    CollectionSchema::Ptr collection, SearchQuery query,
    const std::vector<Segment::Ptr> &segments) {
  if (segments.empty()) {
    return DocPtrList{};
  }

  // Check filter satisfiability once before the loop (result depends only on
  // the query, not on segment data).
  auto first_query_info = build_query_info(collection, query, nullptr);
  if (!first_query_info) {
    return tl::make_unexpected(first_query_info.error());
  }
  if (first_query_info.value()->is_filter_unsatisfiable()) {
    LOG_WARN("filter is unsatisfiable: %s",
             first_query_info.value()->to_string().c_str());
    return {};
  }
  // Capture output field schema before query_infos are moved into the planner.
  auto select_item_meta_ptrs =
      first_query_info.value()->select_item_schema_ptrs();

  // Build a separate QueryInfo per segment so the optimizer can mutate each
  // independently.  Reuse first_query_info for segment 0.
  std::vector<QueryInfo::Ptr> query_infos;
  query_infos.reserve(segments.size());
  query_infos.emplace_back(std::move(first_query_info.value()));
  for (size_t i = 1; i < segments.size(); ++i) {
    auto query_info = build_query_info(collection, query, nullptr);
    if (!query_info) {
      return tl::make_unexpected(query_info.error());
    }
    query_infos.emplace_back(std::move(query_info.value()));
  }
  auto reader = search_by_query_info(collection, segments, &query_infos);
  if (!reader) {
    return tl::make_unexpected(Status::InternalError(
        "Execute plan failed (query): ", reader.error().c_str()));
  }
  return fill_result(select_item_meta_ptrs, reader.value().get());
}

SearchQuery from_group_by(const GroupByVectorQuery &gq) {
  SearchQuery sq;
  sq.target_ = gq.target_;
  sq.filter_ = gq.filter_;
  sq.include_vector_ = gq.include_vector_;
  sq.output_fields_ = gq.output_fields_;
  sq.topk_ = 0;
  return sq;
}

Result<GroupResults> SQLEngineImpl::execute_group_by(
    CollectionSchema::Ptr collection, const GroupByVectorQuery &group_by_query,
    const std::vector<Segment::Ptr> &segments) {
  if (segments.empty()) {
    return GroupResults{};
  }

  SearchQuery query = from_group_by(group_by_query);

  // Check filter satisfiability once before the loop (result depends only on
  // the query, not on segment data).
  auto first_query_info = build_query_info(
      collection, query,
      std::make_shared<GroupBy>(group_by_query.group_by_field_name_,
                                group_by_query.group_topk_,
                                group_by_query.group_count_));
  if (!first_query_info) {
    return tl::make_unexpected(first_query_info.error());
  }
  if (first_query_info.value()->is_filter_unsatisfiable()) {
    LOG_WARN("filter is unsatisfiable: %s",
             first_query_info.value()->to_string().c_str());
    return {};
  }

  // Keep a copy, the planner will move elements out of query_infos.
  QueryInfo::Ptr group_by_query_info = first_query_info.value();

  // Build a separate QueryInfo per segment so the optimizer can mutate each
  // independently.  Reuse first_query_info for segment 0.
  std::vector<QueryInfo::Ptr> query_infos;
  query_infos.reserve(segments.size());
  query_infos.emplace_back(std::move(first_query_info.value()));
  for (size_t i = 1; i < segments.size(); ++i) {
    auto query_info = build_query_info(
        collection, query,
        std::make_shared<GroupBy>(group_by_query.group_by_field_name_,
                                  group_by_query.group_topk_,
                                  group_by_query.group_count_));
    if (!query_info) {
      return tl::make_unexpected(query_info.error());
    }
    query_infos.emplace_back(std::move(query_info.value()));
  }
  auto reader = search_by_query_info(collection, segments, &query_infos);
  if (!reader) {
    return tl::make_unexpected(Status::InternalError(
        "Execute plan failed (group_by): ", reader.error().c_str()));
  }
  return fill_group_by_result(*group_by_query_info, reader.value().get());
}

Result<FtsCondInfo::Ptr> SQLEngineImpl::parse_fts_query(
    CollectionSchema::Ptr collection, const std::string &field_name,
    const FtsClause &fts, const QueryParams::Ptr &query_params) {
  // Exactly one of query_string_ or match_string_ must be provided.
  bool has_query = !fts.query_string_.empty();
  bool has_match_string = !fts.match_string_.empty();
  if (has_query == has_match_string) {
    return tl::make_unexpected(Status::InvalidArgument(
        "Exactly one of query_string or match_string must be provided"));
  }

  auto *fts_query_param = dynamic_cast<FtsQueryParams *>(query_params.get());

  // Determine default operator once, shared by both query_string and
  // match_string paths. Accept "and"/"or" case-insensitively, empty means OR;
  // any other value is a user error and must be reported, not silently
  // downgraded to OR. strcasecmp is mapped to _stricmp on MSVC by platform.h.
  fts::FtsDefaultOperator default_op = fts::FtsDefaultOperator::OR;
  if (fts_query_param) {
    const auto &op_str = fts_query_param->default_operator();
    if (op_str.empty() || strcasecmp(op_str.c_str(), "or") == 0) {
      default_op = fts::FtsDefaultOperator::OR;
    } else if (strcasecmp(op_str.c_str(), "and") == 0) {
      default_op = fts::FtsDefaultOperator::AND;
    } else {
      return tl::make_unexpected(Status::InvalidArgument(
          "FTS default_operator must be empty, 'and' or 'or' (case-insensitive)"
          ", got: ",
          op_str));
    }
  }

  // Tokenizer pipeline is required by both branches: query_string needs it to
  // tokenize phrase contents and bare terms, match_string needs it to split
  // the natural-language input. Resolve once and share.
  auto *field_schema = collection->get_field(field_name);
  if (!field_schema) {
    return tl::make_unexpected(
        Status::InvalidArgument("FTS field not found: ", field_name));
  }
  auto fts_idx_param =
      std::dynamic_pointer_cast<FtsIndexParams>(field_schema->index_params());
  if (!fts_idx_param) {
    return tl::make_unexpected(Status::InvalidArgument(
        "FTS field has no FtsIndexParams: ", field_name));
  }
  auto pipeline_result = detail::AcquireFtsPipeline(*fts_idx_param);
  if (!pipeline_result.has_value()) {
    return tl::make_unexpected(Status::InternalError(
        "Failed to create tokenizer pipeline for field: ", field_name, " ",
        pipeline_result.error().message()));
  }
  auto &pipeline = pipeline_result.value();

  fts::FtsAstNodePtr ast;
  if (has_query) {
    // Structured query expression: parse via ANTLR grammar; phrase/term
    // bodies are tokenized through the same pipeline used at index time.
    fts::FtsQueryParser fts_parser;
    ast = fts_parser.parse(fts.query_string_, pipeline, default_op);
    if (!ast) {
      LOG_ERROR("FTS query parse failed: %s", fts_parser.err_msg().c_str());
      return tl::make_unexpected(Status::InvalidArgument(
          "FTS query parse failed: ", fts_parser.err_msg()));
    }
  } else {
    // Natural language match_string: tokenize and combine with default_op.
    auto tokens = pipeline->process(fts.match_string_);
    if (tokens.empty()) {
      // Analyzer dropped everything → zero-doc query, not an error.
      return std::make_shared<FtsCondInfo>(field_name,
                                           std::make_unique<fts::EmptyNode>());
    }
    if (tokens.size() == 1) {
      ast = std::make_unique<fts::TermNode>(std::move(tokens[0].text));
    } else {
      if (default_op == fts::FtsDefaultOperator::AND) {
        auto and_node = std::make_unique<fts::AndNode>();
        for (auto &token : tokens) {
          and_node->children.push_back(
              std::make_unique<fts::TermNode>(std::move(token.text)));
        }
        ast = std::move(and_node);
      } else {
        auto or_node = std::make_unique<fts::OrNode>();
        for (auto &token : tokens) {
          or_node->children.push_back(
              std::make_unique<fts::TermNode>(std::move(token.text)));
        }
        ast = std::move(or_node);
      }
    }
  }

  // Structural rewrite: dedup repeated terms (collapsed into a single node
  // with summed boost), flatten same-type composites for better WAND pruning,
  // propagate EmptyNode, and detect must/must_not contradictions. The pre-
  // rewrite AST is logged at DEBUG so the transform is auditable. LOG_DEBUG
  // is gated by the configured log level, so ast->text() is only built when
  // debug logging is enabled.
  LOG_DEBUG("FTS AST before rewrite: %s", ast ? ast->text().c_str() : "<null>");
  fts::simplify(ast);
  LOG_DEBUG("FTS AST after rewrite : %s", ast ? ast->text().c_str() : "<null>");

  return std::make_shared<FtsCondInfo>(field_name, std::move(ast));
}

Result<QueryInfo::Ptr> SQLEngineImpl::parse_sql_info(
    const CollectionSchema &schema, const SQLInfo::Ptr &sql_info) {
  ScopedProfilerStage stage_guard(profiler_, "analyze_sql_info");
  QueryAnalyzer analyzer;
  auto query_info = analyzer.analyze(schema, sql_info);
  if (!query_info) {
    return tl::make_unexpected(Status::InvalidArgument(
        "Analyze SQL info failed: ", query_info.error().c_str()));
  }
  LOG_DEBUG("query_info: [%s]", query_info.value()->to_string().c_str());
  return query_info.value();
}

Result<QueryInfo::Ptr> SQLEngineImpl::build_query_info(
    CollectionSchema::Ptr collection, SearchQuery request,
    std::shared_ptr<GroupBy> group_by) {
  ScopedProfilerStage stage_guard(profiler_, "build_sql_info");
  Node::Ptr filter_node;
  if (!request.filter_.empty()) {
    ZVecParser::Ptr parser = ZVecParser::create();
    filter_node = parser->parse_filter(request.filter_);
    if (filter_node == nullptr) {
      LOG_ERROR("parse filter failed. reason:[%s] filter:[%s]",
                parser->err_msg().c_str(), request.filter_.c_str());
      return tl::make_unexpected(Status::InvalidArgument(
          "Invalid filter [", request.filter_, "]: ", parser->err_msg()));
    }
  }
  if (group_by) {
    auto &group = *group_by;
    if (group.group_by_field.empty() || group.group_count == 0 ||
        group.group_topk == 0) {
      return tl::make_unexpected(Status::InvalidArgument(
          "Invalid group_by request: group_by_field='", group.group_by_field,
          "', group_count=", group.group_count,
          ", group_topk=", group.group_topk));
    }
  }

  FtsCondInfo::Ptr fts_cond_info;
  if (const auto *fts_clause = request.target_.get_fts_clause()) {
    auto fts_result =
        parse_fts_query(collection, request.target_.field_name_, *fts_clause,
                        request.target_.query_params_);
    if (!fts_result) {
      return tl::make_unexpected(fts_result.error());
    }
    fts_cond_info = std::move(fts_result.value());
  }

  auto sql_info = sqlengine::SQLInfoHelper::BuildSQLInfoFromSearchQuery(
      std::move(request), std::move(filter_node), std::move(group_by));
  if (!sql_info) {
    return tl::make_unexpected(sql_info.error());
  }

  // Attach FTS info to SelectInfo so query_analyzer can propagate it to
  // QueryInfo.
  if (fts_cond_info) {
    auto select_info =
        std::dynamic_pointer_cast<SelectInfo>(sql_info.value()->base_info());
    if (!select_info) {
      return tl::make_unexpected(Status::InternalError(
          "BuildSQLInfoFromSearchQuery did not produce SelectInfo"));
    }
    select_info->set_fts_cond_info(std::move(fts_cond_info));
  }

  LOG_DEBUG("Sql info is %s", sql_info.value()->to_string().c_str());
  stage_guard.close();
  return parse_sql_info(*collection, std::move(sql_info.value()));
}

Result<std::unique_ptr<arrow::RecordBatchReader>>
SQLEngineImpl::search_by_query_info(
    CollectionSchema::Ptr collection, const std::vector<Segment::Ptr> &segments,
    std::vector<sqlengine::QueryInfo::Ptr> *query_infos) {
  global_init();

  ScopedProfilerStage stage_guard(profiler_, "build_query_plan");
  QueryPlanner planner(collection.get());
  auto plan_info =
      planner.make_plan(segments, profiler_->trace_id(), query_infos);
  if (!plan_info) {
    LOG_ERROR("plan query_info failed: [%s]", plan_info.error().c_str());
    return tl::make_unexpected(plan_info.error());
  }
  // LOG_DEBUG("plan_info: [%s]", plan_info->to_string().c_str());
  return plan_info.value()->execute_to_reader();
}

#define GET_FIELD_FROM_RECORD_BATCH(res, field_name)                    \
  auto res = record_batch.GetColumnByName(field_name);                  \
  if (!res) {                                                           \
    return Status::InternalError("Column not found in record batch: [", \
                                 field_name, "]");                      \
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
      return Status::InvalidArgument(
          "Sparse vector indices and values size mismatch [", field_name,
          "]: indices count=", count,
          " vs values count=", value_data.size() / sizeof(VectorType));
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
        return Status::InvalidArgument(
            "Vector dimension not match [", field_name,
            "]: expected dimension=", dimension,
            " vs actual dimension=", data.size() / sizeof(VectorType));
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
      return Status::InvalidArgument("Unsupported data type for field [",
                                     field_schema.name(),
                                     "]: data_type=", field_schema.data_type());
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
      return tl::make_unexpected(
          Status::InternalError("Read next record batch failed (fill_result): ",
                                read_res.ToString()));
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
          "Read next record batch failed (group_by): ", read_res.ToString()));
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
      return tl::make_unexpected(Status::InternalError(
          "Column not found in record batch: [", kFieldGroupId, "]"));
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
