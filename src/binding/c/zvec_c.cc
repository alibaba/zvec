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

#include "zvec_c.h"
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/query_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>

using json = nlohmann::json;
using namespace zvec;

// ============================================================================
// Internal helpers: status
// ============================================================================

static zvec_status_t make_ok() {
  return {0, nullptr};
}

static zvec_status_t make_error(const Status &s) {
  if (s.ok()) return make_ok();
  const std::string &msg = s.message();
  char *buf = new char[msg.size() + 1];
  std::memcpy(buf, msg.c_str(), msg.size() + 1);
  return {static_cast<int>(s.code()), buf};
}

static zvec_status_t make_exception_error(const std::exception &e) {
  std::string msg = e.what();
  char *buf = new char[msg.size() + 1];
  std::memcpy(buf, msg.c_str(), msg.size() + 1);
  return {ZVEC_UNKNOWN, buf};
}

static char *strdup_new(const std::string &s) {
  char *buf = new char[s.size() + 1];
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return buf;
}

// ============================================================================
// Internal helpers: type parsing
// ============================================================================

static DataType parse_data_type(const std::string &s) {
  if (s == "BINARY") return DataType::BINARY;
  if (s == "STRING") return DataType::STRING;
  if (s == "BOOL") return DataType::BOOL;
  if (s == "INT32") return DataType::INT32;
  if (s == "INT64") return DataType::INT64;
  if (s == "UINT32") return DataType::UINT32;
  if (s == "UINT64") return DataType::UINT64;
  if (s == "FLOAT") return DataType::FLOAT;
  if (s == "DOUBLE") return DataType::DOUBLE;
  if (s == "VECTOR_BINARY32") return DataType::VECTOR_BINARY32;
  if (s == "VECTOR_BINARY64") return DataType::VECTOR_BINARY64;
  if (s == "VECTOR_FP16") return DataType::VECTOR_FP16;
  if (s == "VECTOR_FP32") return DataType::VECTOR_FP32;
  if (s == "VECTOR_FP64") return DataType::VECTOR_FP64;
  if (s == "VECTOR_INT4") return DataType::VECTOR_INT4;
  if (s == "VECTOR_INT8") return DataType::VECTOR_INT8;
  if (s == "VECTOR_INT16") return DataType::VECTOR_INT16;
  if (s == "SPARSE_VECTOR_FP16") return DataType::SPARSE_VECTOR_FP16;
  if (s == "SPARSE_VECTOR_FP32") return DataType::SPARSE_VECTOR_FP32;
  if (s == "ARRAY_BINARY") return DataType::ARRAY_BINARY;
  if (s == "ARRAY_STRING") return DataType::ARRAY_STRING;
  if (s == "ARRAY_BOOL") return DataType::ARRAY_BOOL;
  if (s == "ARRAY_INT32") return DataType::ARRAY_INT32;
  if (s == "ARRAY_INT64") return DataType::ARRAY_INT64;
  if (s == "ARRAY_UINT32") return DataType::ARRAY_UINT32;
  if (s == "ARRAY_UINT64") return DataType::ARRAY_UINT64;
  if (s == "ARRAY_FLOAT") return DataType::ARRAY_FLOAT;
  if (s == "ARRAY_DOUBLE") return DataType::ARRAY_DOUBLE;
  return DataType::UNDEFINED;
}

static MetricType parse_metric_type(const std::string &s) {
  if (s == "L2") return MetricType::L2;
  if (s == "IP") return MetricType::IP;
  if (s == "COSINE") return MetricType::COSINE;
  if (s == "MIPSL2") return MetricType::MIPSL2;
  return MetricType::UNDEFINED;
}

static QuantizeType parse_quantize_type(const json &j) {
  if (!j.contains("quantize")) return QuantizeType::UNDEFINED;
  const std::string s = j["quantize"].get<std::string>();
  if (s == "FP16") return QuantizeType::FP16;
  if (s == "INT8") return QuantizeType::INT8;
  if (s == "INT4") return QuantizeType::INT4;
  return QuantizeType::UNDEFINED;
}

static IndexParams::Ptr parse_index_params(const json &j) {
  std::string type = j["type"].get<std::string>();
  if (type == "HNSW") {
    auto metric = parse_metric_type(j.value("metric", std::string("L2")));
    int m = j.value("m", 16);
    int ef = j.value("ef_construction", 200);
    auto quant = parse_quantize_type(j);
    return std::make_shared<HnswIndexParams>(metric, m, ef, quant);
  }
  if (type == "FLAT") {
    auto metric = parse_metric_type(j.value("metric", std::string("L2")));
    auto quant = parse_quantize_type(j);
    return std::make_shared<FlatIndexParams>(metric, quant);
  }
  if (type == "IVF") {
    auto metric = parse_metric_type(j.value("metric", std::string("L2")));
    int n_list = j.value("n_list", 1024);
    int n_iters = j.value("n_iters", 10);
    bool use_soar = j.value("use_soar", false);
    auto quant = parse_quantize_type(j);
    return std::make_shared<IVFIndexParams>(metric, n_list, n_iters, use_soar,
                                            quant);
  }
  if (type == "INVERT") {
    bool range_opt = j.value("enable_range_optimization", true);
    bool ext_wildcard = j.value("enable_extended_wildcard", false);
    return std::make_shared<InvertIndexParams>(range_opt, ext_wildcard);
  }
  return nullptr;
}

// ============================================================================
// Internal helpers: schema parsing
// ============================================================================

static FieldSchema::Ptr parse_field_schema(const json &j) {
  std::string name = j["name"].get<std::string>();
  DataType data_type = parse_data_type(j["data_type"].get<std::string>());
  bool nullable = j.value("nullable", false);
  uint32_t dimension = j.value("dimension", 0);
  IndexParams::Ptr index_params = nullptr;
  if (j.contains("index")) {
    index_params = parse_index_params(j["index"]);
  }
  return std::make_shared<FieldSchema>(name, data_type, dimension, nullable,
                                       index_params);
}

static CollectionSchema parse_schema(const json &j) {
  CollectionSchema schema(j["name"].get<std::string>());
  for (const auto &field_j : j["fields"]) {
    schema.add_field(parse_field_schema(field_j));
  }
  return schema;
}

// ============================================================================
// Internal helpers: Doc serialization/deserialization
// ============================================================================

static void set_doc_field(Doc &doc, const std::string &name, const json &val,
                          DataType dt) {
  if (val.is_null()) {
    doc.set_null(name);
    return;
  }
  switch (dt) {
    case DataType::BOOL:
      doc.set(name, val.get<bool>());
      break;
    case DataType::INT32:
      doc.set(name, val.get<int32_t>());
      break;
    case DataType::INT64:
      doc.set(name, val.get<int64_t>());
      break;
    case DataType::UINT32:
      doc.set(name, val.get<uint32_t>());
      break;
    case DataType::UINT64:
      doc.set(name, val.get<uint64_t>());
      break;
    case DataType::FLOAT:
      doc.set(name, val.get<float>());
      break;
    case DataType::DOUBLE:
      doc.set(name, val.get<double>());
      break;
    case DataType::STRING:
    case DataType::BINARY:
      doc.set(name, val.get<std::string>());
      break;
    case DataType::VECTOR_FP32:
      doc.set(name, val.get<std::vector<float>>());
      break;
    case DataType::VECTOR_FP16: {
      auto floats = val.get<std::vector<float>>();
      std::vector<float16_t> fp16(floats.size());
      for (size_t i = 0; i < floats.size(); ++i) fp16[i] = float16_t(floats[i]);
      doc.set(name, fp16);
      break;
    }
    case DataType::VECTOR_INT8: {
      doc.set(name, val.get<std::vector<int8_t>>());
      break;
    }
    case DataType::VECTOR_INT16: {
      doc.set(name, val.get<std::vector<int16_t>>());
      break;
    }
    case DataType::SPARSE_VECTOR_FP32: {
      auto indices = val["indices"].get<std::vector<uint32_t>>();
      auto values = val["values"].get<std::vector<float>>();
      doc.set(name, std::make_pair(indices, values));
      break;
    }
    case DataType::SPARSE_VECTOR_FP16: {
      auto indices = val["indices"].get<std::vector<uint32_t>>();
      auto fvals = val["values"].get<std::vector<float>>();
      std::vector<float16_t> fp16v(fvals.size());
      for (size_t i = 0; i < fvals.size(); ++i) fp16v[i] = float16_t(fvals[i]);
      doc.set(name, std::make_pair(indices, fp16v));
      break;
    }
    case DataType::ARRAY_BOOL:
      doc.set(name, val.get<std::vector<bool>>());
      break;
    case DataType::ARRAY_INT32:
      doc.set(name, val.get<std::vector<int32_t>>());
      break;
    case DataType::ARRAY_INT64:
      doc.set(name, val.get<std::vector<int64_t>>());
      break;
    case DataType::ARRAY_UINT32:
      doc.set(name, val.get<std::vector<uint32_t>>());
      break;
    case DataType::ARRAY_UINT64:
      doc.set(name, val.get<std::vector<uint64_t>>());
      break;
    case DataType::ARRAY_FLOAT:
      doc.set(name, val.get<std::vector<float>>());
      break;
    case DataType::ARRAY_DOUBLE:
      doc.set(name, val.get<std::vector<double>>());
      break;
    case DataType::ARRAY_STRING:
      doc.set(name, val.get<std::vector<std::string>>());
      break;
    default:
      break;
  }
}

static std::vector<Doc> parse_docs(const json &docs_j,
                                   const CollectionSchema &schema) {
  std::vector<Doc> docs;
  docs.reserve(docs_j.size());
  for (const auto &doc_j : docs_j) {
    Doc doc;
    if (doc_j.contains("pk")) {
      doc.set_pk(doc_j["pk"].get<std::string>());
    }
    if (doc_j.contains("fields")) {
      for (const auto &[field_name, value_j] : doc_j["fields"].items()) {
        const FieldSchema *field = schema.get_field(field_name);
        if (!field) continue;
        set_doc_field(doc, field_name, value_j, field->data_type());
      }
    }
    docs.push_back(std::move(doc));
  }
  return docs;
}

static json field_to_json(const Doc &doc, const std::string &name,
                          DataType dt) {
  switch (dt) {
    case DataType::BOOL: {
      auto v = doc.get<bool>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::INT32: {
      auto v = doc.get<int32_t>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::INT64: {
      auto v = doc.get<int64_t>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::UINT32: {
      auto v = doc.get<uint32_t>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::UINT64: {
      auto v = doc.get<uint64_t>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::FLOAT: {
      auto v = doc.get<float>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::DOUBLE: {
      auto v = doc.get<double>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::STRING:
    case DataType::BINARY: {
      auto v = doc.get<std::string>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::VECTOR_FP32: {
      auto v = doc.get<std::vector<float>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::VECTOR_FP16: {
      auto v = doc.get<std::vector<float16_t>>(name);
      if (!v) return json(nullptr);
      std::vector<float> floats(v->size());
      for (size_t i = 0; i < v->size(); ++i)
        floats[i] = static_cast<float>((*v)[i]);
      return json(floats);
    }
    case DataType::VECTOR_INT8: {
      auto v = doc.get<std::vector<int8_t>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::VECTOR_INT16: {
      auto v = doc.get<std::vector<int16_t>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::SPARSE_VECTOR_FP32: {
      auto v =
          doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(name);
      if (!v) return json(nullptr);
      return json{{"indices", v->first}, {"values", v->second}};
    }
    case DataType::SPARSE_VECTOR_FP16: {
      auto v =
          doc.get<std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
              name);
      if (!v) return json(nullptr);
      std::vector<float> fvals(v->second.size());
      for (size_t i = 0; i < v->second.size(); ++i)
        fvals[i] = static_cast<float>(v->second[i]);
      return json{{"indices", v->first}, {"values", fvals}};
    }
    case DataType::ARRAY_BOOL: {
      auto v = doc.get<std::vector<bool>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::ARRAY_INT32: {
      auto v = doc.get<std::vector<int32_t>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::ARRAY_INT64: {
      auto v = doc.get<std::vector<int64_t>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::ARRAY_FLOAT: {
      auto v = doc.get<std::vector<float>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::ARRAY_DOUBLE: {
      auto v = doc.get<std::vector<double>>(name);
      return v ? json(*v) : json(nullptr);
    }
    case DataType::ARRAY_STRING: {
      auto v = doc.get<std::vector<std::string>>(name);
      return v ? json(*v) : json(nullptr);
    }
    default:
      return json(nullptr);
  }
}

static json doc_to_json(const Doc &doc, const CollectionSchema &schema) {
  json j;
  j["pk"] = doc.pk();
  j["score"] = doc.score();
  json fields = json::object();
  for (const auto &name : doc.field_names()) {
    const FieldSchema *field = schema.get_field(name);
    if (field) {
      fields[name] = field_to_json(doc, name, field->data_type());
    }
  }
  j["fields"] = fields;
  return j;
}

static json write_results_to_json(const WriteResults &results,
                                  const std::vector<Doc> &docs) {
  json arr = json::array();
  for (size_t i = 0; i < results.size(); ++i) {
    json r;
    r["pk"] = (i < docs.size()) ? docs[i].pk() : "";
    r["code"] = static_cast<int>(results[i].code());
    r["message"] = results[i].message();
    arr.push_back(r);
  }
  return arr;
}

// ============================================================================
// C API implementation
// ============================================================================

extern "C" {

zvec_status_t zvec_create_and_open(const char *path, const char *schema_json,
                                   zvec_collection_t *out) {
  if (!path || !schema_json || !out)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    CollectionSchema schema = parse_schema(json::parse(schema_json));
    auto result = Collection::CreateAndOpen(path, schema, CollectionOptions{});
    if (!result) return make_error(result.error());
    *out =
        static_cast<zvec_collection_t>(new Collection::Ptr(std::move(*result)));
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_open(const char *path, zvec_collection_t *out) {
  if (!path || !out)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto result = Collection::Open(path, CollectionOptions{});
    if (!result) return make_error(result.error());
    *out =
        static_cast<zvec_collection_t>(new Collection::Ptr(std::move(*result)));
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_open_read_only(const char *path, zvec_collection_t *out) {
  if (!path || !out)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    CollectionOptions options{};
    options.read_only_ = true;
    auto result = Collection::Open(path, options);
    if (!result) return make_error(result.error());
    *out =
        static_cast<zvec_collection_t>(new Collection::Ptr(std::move(*result)));
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

void zvec_collection_free(zvec_collection_t col) {
  delete static_cast<Collection::Ptr *>(col);
}

zvec_status_t zvec_collection_destroy(zvec_collection_t col) {
  if (!col) return make_error(Status::InvalidArgument("null collection"));
  try {
    return make_error((*static_cast<Collection::Ptr *>(col))->Destroy());
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_collection_flush(zvec_collection_t col) {
  if (!col) return make_error(Status::InvalidArgument("null collection"));
  try {
    return make_error((*static_cast<Collection::Ptr *>(col))->Flush());
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

// ── DML ──────────────────────────────────────────────────────────────────────

enum class WriteOp { INSERT, UPSERT, UPDATE };

static zvec_status_t do_write(zvec_collection_t col, const char *docs_json,
                              char **results_json, WriteOp op) {
  if (!col || !docs_json)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    auto schema_result = collection->Schema();
    if (!schema_result) return make_error(schema_result.error());
    const CollectionSchema &schema = *schema_result;

    auto docs = parse_docs(json::parse(docs_json), schema);

    Result<WriteResults> result{tl::unexpect,
                                Status::InternalError("uninitialized")};
    switch (op) {
      case WriteOp::INSERT:
        result = collection->Insert(docs);
        break;
      case WriteOp::UPSERT:
        result = collection->Upsert(docs);
        break;
      case WriteOp::UPDATE:
        result = collection->Update(docs);
        break;
    }
    if (!result) return make_error(result.error());

    if (results_json) {
      *results_json = strdup_new(write_results_to_json(*result, docs).dump());
    }
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_insert(zvec_collection_t col, const char *docs_json,
                          char **results_json) {
  return do_write(col, docs_json, results_json, WriteOp::INSERT);
}

zvec_status_t zvec_upsert(zvec_collection_t col, const char *docs_json,
                          char **results_json) {
  return do_write(col, docs_json, results_json, WriteOp::UPSERT);
}

zvec_status_t zvec_update(zvec_collection_t col, const char *docs_json,
                          char **results_json) {
  return do_write(col, docs_json, results_json, WriteOp::UPDATE);
}

zvec_status_t zvec_delete_by_pks(zvec_collection_t col, const char *pks_json,
                                 char **results_json) {
  if (!col || !pks_json)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    auto pks = json::parse(pks_json).get<std::vector<std::string>>();

    auto result = collection->Delete(pks);
    if (!result) return make_error(result.error());

    if (results_json) {
      json arr = json::array();
      for (size_t i = 0; i < result->size(); ++i) {
        arr.push_back({{"pk", i < pks.size() ? pks[i] : ""},
                       {"code", static_cast<int>((*result)[i].code())},
                       {"message", (*result)[i].message()}});
      }
      *results_json = strdup_new(arr.dump());
    }
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_delete_by_filter(zvec_collection_t col, const char *filter) {
  if (!col || !filter)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    return make_error(
        (*static_cast<Collection::Ptr *>(col))->DeleteByFilter(filter));
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

// ── DQL ──────────────────────────────────────────────────────────────────────

zvec_status_t zvec_query(zvec_collection_t col, const char *query_json,
                         char **results_json) {
  if (!col || !query_json)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    auto schema_result = collection->Schema();
    if (!schema_result) return make_error(schema_result.error());
    const CollectionSchema &schema = *schema_result;

    const auto q = json::parse(query_json);

    VectorQuery query;
    query.field_name_ = q["field_name"].get<std::string>();
    query.topk_ = q["topk"].get<int>();

    auto floats = q["vector"].get<std::vector<float>>();
    const FieldSchema *vfield = schema.get_vector_field(query.field_name_);
    if (vfield && vfield->data_type() == DataType::VECTOR_FP16) {
      std::vector<float16_t> fp16(floats.size());
      for (size_t i = 0; i < floats.size(); ++i) fp16[i] = float16_t(floats[i]);
      query.query_vector_ =
          std::string(reinterpret_cast<const char *>(fp16.data()),
                      fp16.size() * sizeof(float16_t));
    } else {
      query.query_vector_ =
          std::string(reinterpret_cast<const char *>(floats.data()),
                      floats.size() * sizeof(float));
    }

    if (q.contains("filter") && q["filter"].is_string()) {
      query.filter_ = q["filter"].get<std::string>();
    }
    if (q.contains("include_vector")) {
      query.include_vector_ = q["include_vector"].get<bool>();
    }
    if (q.contains("output_fields") && q["output_fields"].is_array()) {
      query.output_fields_ = q["output_fields"].get<std::vector<std::string>>();
    }

    auto result = collection->Query(query);
    if (!result) return make_error(result.error());

    json arr = json::array();
    for (const auto &doc_ptr : *result) {
      if (doc_ptr) arr.push_back(doc_to_json(*doc_ptr, schema));
    }
    if (results_json) *results_json = strdup_new(arr.dump());
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_fetch(zvec_collection_t col, const char *pks_json,
                         char **results_json) {
  if (!col || !pks_json)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    auto schema_result = collection->Schema();
    if (!schema_result) return make_error(schema_result.error());
    const CollectionSchema &schema = *schema_result;

    auto pks = json::parse(pks_json).get<std::vector<std::string>>();
    auto result = collection->Fetch(pks);
    if (!result) return make_error(result.error());

    json arr = json::array();
    for (const auto &[pk, doc_ptr] : *result) {
      if (doc_ptr) arr.push_back(doc_to_json(*doc_ptr, schema));
    }
    if (results_json) *results_json = strdup_new(arr.dump());
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_sparse_query(zvec_collection_t col, const char *query_json,
                                char **results_json) {
  if (!col || !query_json)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    auto schema_result = collection->Schema();
    if (!schema_result) return make_error(schema_result.error());
    const CollectionSchema &schema = *schema_result;

    const auto q = json::parse(query_json);

    VectorQuery query;
    query.field_name_ = q["field_name"].get<std::string>();
    query.topk_ = q["topk"].get<int>();

    // Pack sparse indices as raw uint32_t bytes.
    auto indices = q["indices"].get<std::vector<uint32_t>>();
    query.query_sparse_indices_ =
        std::string(reinterpret_cast<const char *>(indices.data()),
                    indices.size() * sizeof(uint32_t));

    // Pack sparse values as raw float bytes, converting to fp16 if needed.
    const FieldSchema *vfield = schema.get_vector_field(query.field_name_);
    auto fvals = q["values"].get<std::vector<float>>();
    if (vfield && vfield->data_type() == DataType::SPARSE_VECTOR_FP16) {
      std::vector<float16_t> fp16(fvals.size());
      for (size_t i = 0; i < fvals.size(); ++i) fp16[i] = float16_t(fvals[i]);
      query.query_sparse_values_ =
          std::string(reinterpret_cast<const char *>(fp16.data()),
                      fp16.size() * sizeof(float16_t));
    } else {
      query.query_sparse_values_ =
          std::string(reinterpret_cast<const char *>(fvals.data()),
                      fvals.size() * sizeof(float));
    }

    if (q.contains("filter") && q["filter"].is_string()) {
      query.filter_ = q["filter"].get<std::string>();
    }
    if (q.contains("include_vector")) {
      query.include_vector_ = q["include_vector"].get<bool>();
    }
    if (q.contains("output_fields") && q["output_fields"].is_array()) {
      query.output_fields_ = q["output_fields"].get<std::vector<std::string>>();
    }

    auto result = collection->Query(query);
    if (!result) return make_error(result.error());

    json arr = json::array();
    for (const auto &doc_ptr : *result) {
      if (doc_ptr) arr.push_back(doc_to_json(*doc_ptr, schema));
    }
    if (results_json) *results_json = strdup_new(arr.dump());
    return make_ok();
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

// ── DDL ──────────────────────────────────────────────────────────────────────

zvec_status_t zvec_create_index(zvec_collection_t col, const char *column_name,
                                const char *index_params_json) {
  if (!col || !column_name || !index_params_json)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    auto params = parse_index_params(json::parse(index_params_json));
    if (!params)
      return make_error(Status::InvalidArgument("unknown index type"));
    return make_error(collection->CreateIndex(column_name, params));
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_drop_index(zvec_collection_t col, const char *column_name) {
  if (!col || !column_name)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    return make_error(
        (*static_cast<Collection::Ptr *>(col))->DropIndex(column_name));
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_add_column(zvec_collection_t col,
                              const char *field_schema_json,
                              const char *expression) {
  if (!col || !field_schema_json)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    auto field = parse_field_schema(json::parse(field_schema_json));
    std::string expr = expression ? expression : "";
    return make_error(collection->AddColumn(field, expr));
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_drop_column(zvec_collection_t col, const char *column_name) {
  if (!col || !column_name)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    return make_error(
        (*static_cast<Collection::Ptr *>(col))->DropColumn(column_name));
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

zvec_status_t zvec_alter_column(zvec_collection_t col, const char *column_name,
                                const char *new_name,
                                const char *field_schema_json) {
  if (!col || !column_name)
    return make_error(Status::InvalidArgument("null argument"));
  try {
    auto &collection = *static_cast<Collection::Ptr *>(col);
    std::string rename = (new_name && *new_name) ? new_name : "";
    FieldSchema::Ptr new_schema;
    if (field_schema_json && *field_schema_json) {
      new_schema = parse_field_schema(json::parse(field_schema_json));
    }
    return make_error(collection->AlterColumn(column_name, rename, new_schema));
  } catch (const std::exception &e) {
    return make_exception_error(e);
  }
}

// ── Memory management ────────────────────────────────────────────────────────

void zvec_free_string(char *s) {
  delete[] s;
}

void zvec_status_free(zvec_status_t *status) {
  if (status && status->message) {
    delete[] status->message;
    status->message = nullptr;
  }
}

}  // extern "C"
