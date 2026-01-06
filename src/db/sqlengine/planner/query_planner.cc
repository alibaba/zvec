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

#include "query_planner.h"
#include <memory>
#include <utility>
#include <vector>
#include <ailego/logger/logger.h>
#include <ailego/parallel/thread_pool.h>
#include <arrow/acero/api.h>
#include <arrow/compute/api.h>
#include "analyzer/query_info.h"
#include "analyzer/query_node.h"
#include "db/common/constants.h"
#include "db/common/global_resource.h"
#include "db/common/status.h"
#include "db/index/common/schema.h"
#include "db/index/common/type.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/invert_recall_node.h"
#include "db/sqlengine/planner/ops/check_not_filtered_op.h"
#include "db/sqlengine/planner/ops/contain_op.h"
#include "db/sqlengine/planner/ops/fetch_vector_op.h"
#include "db/sqlengine/planner/segment_node.h"
#include "db/sqlengine/planner/vector_recall_node.h"
#include "planner/plan_info.h"
#include "optimizer.h"

namespace zvec::sqlengine {

namespace cp = ::arrow::compute;
namespace ac = ::arrow::acero;

QueryPlanner::QueryPlanner(CollectionSchema *schema) : schema_(schema) {}

template <typename T>
auto convert_node_to_value(const QueryNode::Ptr &node) {
  const std::string &str = node->text();
  T value;
  if constexpr (std::is_same_v<T, int32_t>) {
    ailego::StringHelper::ToInt32(str, &value);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    ailego::StringHelper::ToInt64(str, &value);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    ailego::StringHelper::ToUint32(str, &value);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    ailego::StringHelper::ToUint64(str, &value);
  } else if constexpr (std::is_same_v<T, float>) {
    ailego::StringHelper::ToFloat(str, &value);
  } else if constexpr (std::is_same_v<T, double>) {
    ailego::StringHelper::ToDouble(str, &value);
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type for conversion");
  }
  return value;
}


template <typename ArrowType>
arrow::Result<std::shared_ptr<arrow::Array>> to_arrow_array(
    const std::vector<QueryNode::Ptr> &input) {
  using CType = typename ArrowType::c_type;
  typename arrow::TypeTraits<ArrowType>::BuilderType builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(input.size()));

  for (auto &s : input) {
    ARROW_RETURN_NOT_OK(builder.Append(convert_node_to_value<CType>(s)));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> to_arrow_string_array(
    const std::vector<QueryNode::Ptr> &input) {
  arrow::StringBuilder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(input.size()));

  for (auto &s : input) {
    ARROW_RETURN_NOT_OK(builder.Append(s->text()));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> to_arrow_bool_array(
    const std::vector<QueryNode::Ptr> &input) {
  arrow::BooleanBuilder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(input.size()));

  for (auto &s : input) {
    // input is normalized to "true" or "false"
    ARROW_RETURN_NOT_OK(builder.Append(s->text() == "true"));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> create_array_from_list_node(
    DataType data_type, const QueryListNode *list_node) {
  auto const &value_expr_list = list_node->value_expr_list();
  switch (data_type) {
    case DataType::INT32:
      return to_arrow_array<arrow::Int32Type>(value_expr_list);
    case DataType::UINT32:
      return to_arrow_array<arrow::UInt32Type>(value_expr_list);
    case DataType::INT64:
      return to_arrow_array<arrow::Int64Type>(value_expr_list);
    case DataType::UINT64:
      return to_arrow_array<arrow::UInt64Type>(value_expr_list);
    case DataType::FLOAT:
      return to_arrow_array<arrow::FloatType>(value_expr_list);
    case DataType::DOUBLE:
      return to_arrow_array<arrow::DoubleType>(value_expr_list);
    case DataType::STRING:
      return to_arrow_string_array(value_expr_list);
    case DataType::BOOL:
      return to_arrow_bool_array(value_expr_list);
    default:
      LOG_ERROR("Unsupported data type for list node. %d", (int)data_type);
      return arrow::Status::Invalid("Unsupported data type for list node.");
  }
}

Result<cp::Expression> QueryPlanner::create_filter_node(
    const QueryNode *query_node) {
  const QueryNode *left = query_node->left_node();
  const QueryNode *right = query_node->right_node();

  arrow::Expression left_exp;
  DataType data_type;
  if (left->op() == QueryNodeOp::Q_ID) {
    left_exp = cp::field_ref(left->text());
    auto field_schema = schema_->get_forward_field(left->text());
    data_type = field_schema->data_type();
  } else if (left->op() == QueryNodeOp::Q_FUNCTION_CALL) {
    const QueryFuncNode *func_node = dynamic_cast<const QueryFuncNode *>(left);
    const auto &func_name = func_node->get_func_name_node()->text();
    const auto &arguments = func_node->arguments();
    if (func_name == kFuncArrayLength) {
      left_exp =
          cp::call("list_value_length", {cp::field_ref(arguments[0]->text())});
      // assume array_length argument is uint32
      data_type = DataType::UINT32;
    } else {
      return tl::make_unexpected(
          Status::InvalidArgument("unexpected function call", func_name));
    }
  } else {
    LOG_ERROR("Unexpected left op. expr[%s]", query_node->text().c_str());
    return tl::make_unexpected(
        Status::InvalidArgument("unexpected left op", left->text()));
  }

  cp::Expression right_exp;
  const std::string &filter_value = right->text();
  auto op = query_node->op();
  if (op == QueryNodeOp::Q_IS_NULL) {
    return cp::is_null(std::move(left_exp));
  } else if (op == QueryNodeOp::Q_IS_NOT_NULL) {
    return cp::is_valid(std::move(left_exp));
  }

  // TODO: check invalid filter
  if (op == QueryNodeOp::Q_IN || op == QueryNodeOp::Q_CONTAIN_ALL ||
      op == QueryNodeOp::Q_CONTAIN_ANY) {
    const QueryListNode *list_node = dynamic_cast<const QueryListNode *>(right);
    auto array_res = create_array_from_list_node(
        FieldSchema::get_element_data_type(data_type), list_node);
    if (!array_res.ok()) {
      return tl::make_unexpected(Status::InvalidArgument(
          "create array failed", array_res.status().ToString()));
    }
    if (op == QueryNodeOp::Q_IN) {
      auto in_filter = cp::call(
          "is_in", {std::move(left_exp)},
          std::make_shared<cp::SetLookupOptions>(array_res.MoveValueUnsafe()));
      if (list_node->exclude()) {
        return cp::not_(std::move(in_filter));
      }
      return in_filter;
    }
    auto contain_filter =
        cp::call(op == QueryNodeOp::Q_CONTAIN_ALL ? kContainAll : kContainAny,
                 {std::move(left_exp)},
                 std::make_shared<ContainOp::Options>(
                     ContainOp::Args{array_res.MoveValueUnsafe(), data_type}));
    if (list_node->exclude()) {
      return cp::not_(std::move(contain_filter));
    }
    return contain_filter;
  }

  switch (data_type) {
    case DataType::STRING: {
      if (op == sqlengine::QueryNodeOp::Q_LIKE) {
        return cp::call("match_like", {std::move(left_exp)},
                        cp::MatchSubstringOptions(filter_value));
      } else {
        right_exp = cp::literal(filter_value);
      }
      break;
    }
    case DataType::INT32: {
      int32_t int32_value;
      ailego::StringHelper::ToInt32(filter_value, &int32_value);
      right_exp = cp::literal(int32_value);
      break;
    }
    case DataType::UINT32: {
      uint32_t uint32_value;
      ailego::StringHelper::ToUint32(filter_value, &uint32_value);
      right_exp = cp::literal(uint32_value);
      break;
    }
    case DataType::INT64: {
      int64_t int64_value;
      ailego::StringHelper::ToInt64(filter_value, &int64_value);
      right_exp = cp::literal(int64_value);
      break;
    }
    case DataType::UINT64: {
      uint64_t uint64_value;
      ailego::StringHelper::ToUint64(filter_value, &uint64_value);
      right_exp = cp::literal(uint64_value);
      break;
    }
    case DataType::FLOAT: {
      float float_value;
      ailego::StringHelper::ToFloat(filter_value, &float_value);
      right_exp = cp::literal(float_value);
      break;
    }
    case DataType::DOUBLE: {
      double double_value;
      ailego::StringHelper::ToDouble(filter_value, &double_value);
      right_exp = cp::literal(double_value);
      break;
    }
    case DataType::BOOL: {
      std::string lower_filter_value;
      lower_filter_value.resize(filter_value.size());
      bool bool_value;
      std::transform(filter_value.begin(), filter_value.end(),
                     lower_filter_value.begin(), ::tolower);
      if (lower_filter_value == "true") {
        bool_value = true;
      } else if (lower_filter_value == "false") {
        bool_value = false;
      } else {
        LOG_ERROR("Unrecognized bool value: %s", filter_value.c_str());
        return tl::make_unexpected(
            Status::InvalidArgument("unexpected bool value", filter_value));
      }
      right_exp = cp::literal(bool_value);
      break;
    }
    default: {
      LOG_ERROR("filter to data type is not supported.");
      return tl::make_unexpected(Status::InvalidArgument(
          "filter to data type is not supported", data_type));
      break;
    }
  }

  switch (op) {
    case sqlengine::QueryNodeOp::Q_EQ:
      return cp::equal(std::move(left_exp), std::move(right_exp));
    case sqlengine::QueryNodeOp::Q_NE:
      return cp::not_equal(std::move(left_exp), std::move(right_exp));
    case sqlengine::QueryNodeOp::Q_GT:
      return cp::greater(std::move(left_exp), std::move(right_exp));
    case sqlengine::QueryNodeOp::Q_LT:
      return cp::less(std::move(left_exp), std::move(right_exp));
    case sqlengine::QueryNodeOp::Q_GE:
      return cp::greater_equal(std::move(left_exp), std::move(right_exp));
    case sqlengine::QueryNodeOp::Q_LE:
      return cp::less_equal(std::move(left_exp), std::move(right_exp));
      // NOTE: Q_LIKE already handled above

    default:
      return tl::make_unexpected(Status::InvalidArgument("unexpected op", op));
      break;
  }
  return tl::make_unexpected(Status::InvalidArgument("unexpected op", op));
}

Result<cp::Expression> QueryPlanner::parse_filter(const QueryNode *query_node) {
  if (!query_node) {
    return cp::literal(true);
  }
  if (query_node->type() == QueryNode::QueryNodeType::REL_EXPR) {
    return create_filter_node(query_node);
  }
  if (query_node->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
    auto left = parse_filter(query_node->left_node());
    if (!left) {
      return left;
    }
    auto right = parse_filter(query_node->right_node());
    if (!right) {
      return right;
    }
    if (query_node->op() == QueryNodeOp::Q_AND) {
      return cp::and_(std::move(left.value()), std::move(right.value()));
    } else if (query_node->op() == QueryNodeOp::Q_OR) {
      return cp::or_(std::move(left.value()), std::move(right.value()));
    }
  }
  return tl::make_unexpected(
      Status::InvalidArgument("unexpected ", query_node->text()));
}


Result<PlanInfo::Ptr> QueryPlanner::make_plan(
    const std::vector<Segment::Ptr> &segments, const std::string &trace_id,
    std::vector<sqlengine::QueryInfo::Ptr> *query_infos) {
  // make logic plan from query_info
  // PlanInfo::Ptr logical_plan = make_logical_plan(query_info);

  // do logic optimization here

  // as we don't have logic optimization in a period of time,
  // simply make physical plan directly from query info
  return make_physical_plan(segments, trace_id, query_infos);
}

Result<PlanInfo::Ptr> QueryPlanner::make_physical_plan(
    const std::vector<Segment::Ptr> &segments, const std::string & /*trace_id*/,
    std::vector<sqlengine::QueryInfo::Ptr> *query_infos) {
  const std::string &table_name = schema_->name();
  if (segments.empty()) {
    LOG_ERROR("Segment not found [%s]", table_name.c_str());
    return tl::make_unexpected(
        Status::InvalidArgument("segment not found:", table_name));
  }

  QueryInfo *query_info = (*query_infos)[0].get();
  LOG_DEBUG("Making plan for collection[%s] query_info[%s]", table_name.c_str(),
            query_info->to_string().c_str());
  int topn = query_info->query_topn();
  auto vector_cond = query_info->vector_cond_info();
  bool has_group_by = query_info->group_by() != nullptr;

  // optimize plan by instrument query info condition, eg adjust invert cond
  Optimizer::Ptr optimizer =
      InvertCondOptimizer::CreateInvertCondOptimizer(schema_);
  int num_segments = segments.size();
  std::vector<PlanInfo::Ptr> segment_plans(segments.size());
  for (int idx = 0; idx < num_segments; ++idx) {
    auto &segment = segments[idx];
    auto &segment_query_info = (*query_infos)[idx];
    bool only_invert_before_opt =
        segment_query_info->invert_cond() != nullptr &&
        segment_query_info->filter_cond() == nullptr;
    if (optimizer) {
      // Optimize by change query info if needed.
      if (!optimizer->optimize(segment.get(), segment_query_info.get())) {
        LOG_DEBUG(
            "Not optimized. collection[%s] segment[%zu] "
            "segment_query_info[%s]",
            table_name.c_str(), (size_t)segment->id(),
            segment_query_info->to_string().c_str());
      } else {
        LOG_DEBUG(
            "Optimized. collection[%s] segment[%zu] segment_query_info[%s]",
            table_name.c_str(), (size_t)segment->id(),
            segment_query_info->to_string().c_str());
      }
    }
    bool only_forward_after_opt =
        segment_query_info->invert_cond() == nullptr &&
        segment_query_info->filter_cond() != nullptr;
    // if only invert cond before opt and only forward cond after opt,
    // single stage search should be performed as large ratio of docs match
    // with filter
    bool single_stage_search = only_invert_before_opt && only_forward_after_opt;
    std::unique_ptr<arrow::compute::Expression> forward_filter;
    if (query_info->filter_cond()) {
      auto filter = parse_filter(query_info->filter_cond().get());
      if (!filter) {
        LOG_ERROR("Parse filter failed: %s", filter.error().c_str());
        return tl::make_unexpected(filter.error());
      }
      forward_filter =
          std::make_unique<cp::Expression>(std::move(filter.value()));
    }

    Result<PlanInfo::Ptr> seg_plan;
    if (query_info->vector_cond_info()) {
      seg_plan = vector_scan(segment, std::move(segment_query_info),
                             std::move(forward_filter), single_stage_search);
    } else if (query_info->invert_cond()) {
      seg_plan = invert_scan(segment, std::move(segment_query_info),
                             std::move(forward_filter));
    } else {
      seg_plan = forward_scan(segment, std::move(segment_query_info),
                              std::move(forward_filter));
    }
    if (!seg_plan) {
      LOG_ERROR("Make plan failed: %s", seg_plan.error().c_str());
      return seg_plan;
    }
    if (segments.size() == 1) {
      return seg_plan;
    }
    segment_plans[idx] = std::move(seg_plan.value());
  }

  // multi segment logic
  ailego::ThreadPool *pool = GlobalResource::Instance().query_thread_pool();
  auto recall_node =
      std::make_shared<SegmentNode>(std::move(segment_plans), pool);
  auto source_node_options =
      arrow::acero::SourceNodeOptions{recall_node->schema(), recall_node->gen(),
                                      arrow::compute::Ordering::Implicit()};
  ac::Declaration node{"source", source_node_options};

  if (vector_cond) {
    node = ac::Declaration{"order_by",
                           {std::move(node)},
                           ac::OrderByNodeOptions{cp::Ordering{{cp::SortKey{
                               kFieldScore, vector_cond->is_reverse_sort()
                                                ? cp::SortOrder::Descending
                                                : cp::SortOrder::Ascending}}}}};
  }

  // group by need to collect all docs
  if (!has_group_by) {
    node = ac::Declaration{
        "fetch", {std::move(node)}, ac::FetchNodeOptions{0, topn}};
  }
  return std::make_shared<PlanInfo>(std::move(node), recall_node->schema());
}

Result<PlanInfo::Ptr> QueryPlanner::forward_scan(
    Segment::Ptr seg, QueryInfo::Ptr query_info,
    std::unique_ptr<arrow::compute::Expression> forward_filter) {
  auto reader = seg->scan(query_info->get_all_fetched_scalar_field_names());
  auto schema = reader->schema();
  ac::Declaration node{
      "record_batch_reader_source",
      ac::RecordBatchReaderSourceNodeOptions{std::move(reader)}};

  auto seg_filter = seg->get_filter();
  if (seg_filter) {
    cp::Expression check_not_filtered =
        cp::call(kCheckNotFiltered, {cp::field_ref(LOCAL_ROW_ID)},
                 std::make_shared<CheckNotFilteredOp::Options>(seg_filter));
    node =
        ac::Declaration{"filter",
                        {std::move(node)},
                        ac::FilterNodeOptions(std::move(check_not_filtered))};
  }

  if (forward_filter) {
    node = ac::Declaration{"filter",
                           {std::move(node)},
                           ac::FilterNodeOptions(std::move(*forward_filter))};
  }

  if (query_info->is_include_vector()) {
    std::vector<cp::Expression> expressions;
    std::vector<std::string> names =
        query_info->get_all_fetched_scalar_field_names();
    for (const auto &field_name : names) {
      expressions.emplace_back(cp::field_ref(field_name));
    }
    for (const auto &vector_field : query_info->selected_vector_fields()) {
      auto indexer = seg->get_combined_vector_indexer(vector_field.field_name);
      if (!indexer) {
        return tl::make_unexpected(Status::InvalidArgument(
            "vector indexer not found:", vector_field.field_name));
      }
      if (vector_field.field_schema_ptr->is_dense_vector()) {
        expressions.emplace_back(
            cp::call("fetch_vector", {cp::field_ref(LOCAL_ROW_ID)},
                     std::make_shared<FetchVectorOp::Options>(indexer, true)));
        schema = Util::append_field(*schema, vector_field.field_name,
                                    arrow::binary());
      } else {
        expressions.emplace_back(
            cp::call("fetch_sparse_vector", {cp::field_ref(LOCAL_ROW_ID)},
                     std::make_shared<FetchVectorOp::Options>(indexer, false)));
        schema = Util::append_field(*schema, vector_field.field_name,
                                    Util::sparse_type());
      }
      names.emplace_back(vector_field.field_name);
    }
    node = ac::Declaration{
        "project",
        {std::move(node)},
        ac::ProjectNodeOptions{std::move(expressions), std::move(names)}};
  }

  node = ac::Declaration{"fetch",
                         {std::move(node)},
                         ac::FetchNodeOptions{0, query_info->query_topn()}};
  return std::make_shared<PlanInfo>(std::move(node), std::move(schema));
}

Result<PlanInfo::Ptr> QueryPlanner::vector_scan(
    Segment::Ptr seg, QueryInfo::Ptr query_info,
    std::unique_ptr<arrow::compute::Expression> forward_filter,
    bool single_stage_search) {
  std::unique_ptr<ac::Declaration> forward_filter_plan;
  // if single stage search is not enabled, first run acero plan to get
  // forward bitmap, then filter during vector search. otherwise, filter
  // forward during forward search.
  if (forward_filter && !single_stage_search) {
    ac::RecordBatchReaderSourceNodeOptions source_options{
        seg->scan(query_info->get_forward_filter_field_names())};
    forward_filter_plan.reset(new ac::Declaration{ac::Declaration::Sequence({
        {"record_batch_reader_source", std::move(source_options)},
        {
            "project",
            ac::ProjectNodeOptions{{std::move(*forward_filter)},
                                   {kFieldIsValid}},
        },
    })});
    forward_filter.reset();
  }
  auto doc_filter = std::make_shared<DocFilter>(seg, query_info,
                                                std::move(forward_filter_plan),
                                                std::move(forward_filter));

  int topn = query_info->query_topn();
  int batch_size = get_batch_size(*query_info, false);
  auto recall_node = std::make_shared<VectorRecallNode>(
      std::move(seg), std::move(query_info), std::move(doc_filter), batch_size,
      single_stage_search);

  auto source_node_options =
      arrow::acero::SourceNodeOptions{recall_node->schema(), recall_node->gen(),
                                      arrow::compute::Ordering::Implicit()};
  ac::Declaration node{"source", source_node_options};
  // group by need to collect all docs
  if (!recall_node->query_info()->group_by()) {
    node = ac::Declaration{
        "fetch", {std::move(node)}, ac::FetchNodeOptions{0, topn}};
  }
  return std::make_shared<PlanInfo>(std::move(node), recall_node->schema());
}

Result<PlanInfo::Ptr> QueryPlanner::invert_scan(
    Segment::Ptr seg, QueryInfo::Ptr query_info,
    std::unique_ptr<arrow::compute::Expression> forward_filter) {
  auto topn = query_info->query_topn();
  int batch_size = get_batch_size(*query_info, forward_filter != nullptr);
  auto recall_node =
      std::make_shared<InvertRecallNode>(seg, query_info, batch_size);

  auto source_node_options =
      arrow::acero::SourceNodeOptions{recall_node->schema(), recall_node->gen(),
                                      arrow::compute::Ordering::Implicit()};
  ac::Declaration node{"source", source_node_options};
  if (forward_filter) {
    node = ac::Declaration{"filter",
                           {std::move(node)},
                           ac::FilterNodeOptions(std::move(*forward_filter))};
  }

  auto schema = recall_node->schema();
  if (query_info->is_include_vector()) {
    std::vector<cp::Expression> expressions;
    std::vector<std::string> names =
        query_info->get_all_fetched_scalar_field_names();
    for (const auto &field_name : names) {
      expressions.emplace_back(cp::field_ref(field_name));
    }
    for (const auto &vector_field : query_info->selected_vector_fields()) {
      auto indexer = seg->get_combined_vector_indexer(vector_field.field_name);
      if (!indexer) {
        return tl::make_unexpected(Status::InvalidArgument(
            "vector indexer not found:", vector_field.field_name));
      }
      if (vector_field.field_schema_ptr->is_dense_vector()) {
        expressions.emplace_back(
            cp::call("fetch_vector", {cp::field_ref(LOCAL_ROW_ID)},
                     std::make_shared<FetchVectorOp::Options>(indexer, true)));
        schema = Util::append_field(*schema, vector_field.field_name,
                                    arrow::binary());
      } else {
        expressions.emplace_back(
            cp::call("fetch_sparse_vector", {cp::field_ref(LOCAL_ROW_ID)},
                     std::make_shared<FetchVectorOp::Options>(indexer, false)));
        schema = Util::append_field(*schema, vector_field.field_name,
                                    Util::sparse_type());
      }
      names.emplace_back(vector_field.field_name);
    }
    node = ac::Declaration{
        "project",
        {std::move(node)},
        ac::ProjectNodeOptions{std::move(expressions), std::move(names)}};
  }

  node = ac::Declaration{
      "fetch", {std::move(node)}, ac::FetchNodeOptions{0, topn}};
  return std::make_shared<PlanInfo>(std::move(node), std::move(schema));
}

int QueryPlanner::get_batch_size(const QueryInfo &info, bool has_later_filter) {
  // ref https://arrow.apache.org/docs/developers/cpp/acero.html#batch-size
  if (!info.query_orderbys().empty() || has_later_filter) {
    return 32 * 1024;
  }
  return std::min(info.query_topn(), 32U * 1024);
}

}  // namespace zvec::sqlengine
