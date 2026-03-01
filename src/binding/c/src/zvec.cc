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

#include "zvec/zvec.h"
#include <memory>
#include <string>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>

using namespace zvec;

// --- Internal Wrappers ---

struct zvec_status {
  Status status;
};

struct zvec_collection {
  Collection::Ptr collection;
};

struct zvec_schema {
  CollectionSchema::Ptr schema;
};

struct zvec_doc {
  std::shared_ptr<Doc> doc;
};

struct zvec_doc_list {
  std::vector<std::shared_ptr<Doc>> docs;
};

struct zvec_stats {
  CollectionStats stats;
};

// --- Helpers ---

static zvec_status_code_t map_status_code(StatusCode code) {
  switch (code) {
    case StatusCode::OK:
      return ZVEC_OK;
    case StatusCode::NOT_FOUND:
      return ZVEC_NOT_FOUND;
    case StatusCode::ALREADY_EXISTS:
      return ZVEC_ALREADY_EXISTS;
    case StatusCode::INVALID_ARGUMENT:
      return ZVEC_INVALID_ARGUMENT;
    case StatusCode::PERMISSION_DENIED:
      return ZVEC_PERMISSION_DENIED;
    case StatusCode::FAILED_PRECONDITION:
      return ZVEC_FAILED_PRECONDITION;
    case StatusCode::RESOURCE_EXHAUSTED:
      return ZVEC_RESOURCE_EXHAUSTED;
    case StatusCode::UNAVAILABLE:
      return ZVEC_UNAVAILABLE;
    case StatusCode::INTERNAL_ERROR:
      return ZVEC_INTERNAL_ERROR;
    case StatusCode::NOT_SUPPORTED:
      return ZVEC_NOT_SUPPORTED;
    case StatusCode::UNKNOWN:
      return ZVEC_UNKNOWN;
    default:
      return ZVEC_UNKNOWN;
  }
}

static DataType map_data_type(zvec_data_type_t type) {
  switch (type) {
    case ZVEC_TYPE_BINARY:
      return DataType::BINARY;
    case ZVEC_TYPE_STRING:
      return DataType::STRING;
    case ZVEC_TYPE_BOOL:
      return DataType::BOOL;
    case ZVEC_TYPE_INT32:
      return DataType::INT32;
    case ZVEC_TYPE_INT64:
      return DataType::INT64;
    case ZVEC_TYPE_UINT32:
      return DataType::UINT32;
    case ZVEC_TYPE_UINT64:
      return DataType::UINT64;
    case ZVEC_TYPE_FLOAT:
      return DataType::FLOAT;
    case ZVEC_TYPE_DOUBLE:
      return DataType::DOUBLE;
    case ZVEC_TYPE_VECTOR_BINARY32:
      return DataType::VECTOR_BINARY32;
    case ZVEC_TYPE_VECTOR_FP16:
      return DataType::VECTOR_FP16;
    case ZVEC_TYPE_VECTOR_FP32:
      return DataType::VECTOR_FP32;
    case ZVEC_TYPE_VECTOR_FP64:
      return DataType::VECTOR_FP64;
    case ZVEC_TYPE_VECTOR_INT8:
      return DataType::VECTOR_INT8;
    case ZVEC_TYPE_SPARSE_VECTOR_FP16:
      return DataType::SPARSE_VECTOR_FP16;
    case ZVEC_TYPE_SPARSE_VECTOR_FP32:
      return DataType::SPARSE_VECTOR_FP32;
    case ZVEC_TYPE_ARRAY_STRING:
      return DataType::ARRAY_STRING;
    case ZVEC_TYPE_ARRAY_INT32:
      return DataType::ARRAY_INT32;
    case ZVEC_TYPE_ARRAY_FLOAT:
      return DataType::ARRAY_FLOAT;
    default:
      return DataType::UNDEFINED;
  }
}

// --- Status API Implementation ---

zvec_status_code_t zvec_status_code(zvec_status_t *status) {
  return status ? map_status_code(status->status.code()) : ZVEC_OK;
}

const char *zvec_status_message(zvec_status_t *status) {
  return status ? status->status.message().c_str() : "";
}

void zvec_status_destroy(zvec_status_t *status) {
  delete status;
}

// --- Schema API Implementation ---

zvec_schema_t *zvec_schema_create(const char *name) {
  auto schema = new zvec_schema();
  schema->schema = std::make_shared<CollectionSchema>(name ? name : "");
  return schema;
}

void zvec_schema_destroy(zvec_schema_t *schema) {
  delete schema;
}

zvec_status_t *zvec_schema_add_field(zvec_schema_t *schema, const char *name,
                                     zvec_data_type_t type,
                                     uint32_t dimension) {
  auto field = std::make_shared<FieldSchema>(
      name ? name : "", map_data_type(type), dimension, false);
  Status s = schema->schema->add_field(field);
  if (s.ok()) return nullptr;
  return new zvec_status{s};
}

// --- Doc API Implementation ---

zvec_doc_t *zvec_doc_create() {
  auto doc = new zvec_doc();
  doc->doc = std::make_shared<Doc>();
  return doc;
}

void zvec_doc_destroy(zvec_doc_t *doc) {
  delete doc;
}

void zvec_doc_set_pk(zvec_doc_t *doc, const char *pk) {
  if (doc && pk) doc->doc->set_pk(pk);
}

const char *zvec_doc_pk(zvec_doc_t *doc) {
  return doc ? doc->doc->pk().c_str() : "";
}

void zvec_doc_set_score(zvec_doc_t *doc, float score) {
  if (doc) doc->doc->set_score(score);
}

float zvec_doc_score(zvec_doc_t *doc) {
  return doc ? doc->doc->score() : 0.0f;
}

zvec_status_t *zvec_doc_set_string(zvec_doc_t *doc, const char *field,
                                   const char *value) {
  if (doc && field && value) doc->doc->set(field, std::string(value));
  return nullptr;
}

zvec_status_t *zvec_doc_set_int32(zvec_doc_t *doc, const char *field,
                                  int32_t value) {
  if (doc && field) doc->doc->set(field, value);
  return nullptr;
}

zvec_status_t *zvec_doc_set_float(zvec_doc_t *doc, const char *field,
                                  float value) {
  if (doc && field) doc->doc->set(field, value);
  return nullptr;
}

zvec_status_t *zvec_doc_set_float_vector(zvec_doc_t *doc,
                                         const char *field_name,
                                         const float *data, uint32_t count) {
  std::vector<float> vec(data, data + count);
  doc->doc->set(field_name, std::move(vec));
  return nullptr;
}

// --- Collection API Implementation ---

zvec_status_t *zvec_collection_create_and_open(
    const char *path, zvec_schema_t *schema,
    zvec_collection_t **out_collection) {
  CollectionOptions options;
  auto result =
      Collection::CreateAndOpen(path ? path : "", *schema->schema, options);
  if (!result) {
    return new zvec_status{result.error()};
  }
  *out_collection = new zvec_collection{result.value()};
  return nullptr;
}

zvec_status_t *zvec_collection_open(const char *path,
                                    zvec_collection_t **out_collection) {
  CollectionOptions options;
  auto result = Collection::Open(path ? path : "", options);
  if (!result) {
    return new zvec_status{result.error()};
  }
  *out_collection = new zvec_collection{result.value()};
  return nullptr;
}

void zvec_collection_destroy(zvec_collection_t *collection) {
  delete collection;
}

zvec_status_t *zvec_collection_flush(zvec_collection_t *collection) {
  Status s = collection->collection->Flush();
  if (s.ok()) return nullptr;
  return new zvec_status{s};
}

zvec_status_t *zvec_collection_destroy_physical(zvec_collection_t *collection) {
  Status s = collection->collection->Destroy();
  if (s.ok()) return nullptr;
  return new zvec_status{s};
}

zvec_status_t *zvec_collection_get_stats(zvec_collection_t *collection,
                                         zvec_stats_t **out_stats) {
  auto stats = collection->collection->Stats();
  if (!stats) return new zvec_status{stats.error()};
  *out_stats = new zvec_stats{stats.value()};
  return nullptr;
}

static zvec_status_t *handle_write_results(
    const tl::expected<WriteResults, Status> &result) {
  if (!result) return new zvec_status{result.error()};
  for (const auto &s : result.value()) {
    if (!s.ok()) return new zvec_status{s};
  }
  return nullptr;
}

zvec_status_t *zvec_collection_insert(zvec_collection_t *collection,
                                      zvec_doc_t **docs, size_t count) {
  std::vector<Doc> cpp_docs;
  cpp_docs.reserve(count);
  for (size_t i = 0; i < count; ++i) cpp_docs.push_back(*(docs[i]->doc));
  return handle_write_results(collection->collection->Insert(cpp_docs));
}

zvec_status_t *zvec_collection_upsert(zvec_collection_t *collection,
                                      zvec_doc_t **docs, size_t count) {
  std::vector<Doc> cpp_docs;
  cpp_docs.reserve(count);
  for (size_t i = 0; i < count; ++i) cpp_docs.push_back(*(docs[i]->doc));
  return handle_write_results(collection->collection->Upsert(cpp_docs));
}

zvec_status_t *zvec_collection_update(zvec_collection_t *collection,
                                      zvec_doc_t **docs, size_t count) {
  std::vector<Doc> cpp_docs;
  cpp_docs.reserve(count);
  for (size_t i = 0; i < count; ++i) cpp_docs.push_back(*(docs[i]->doc));
  return handle_write_results(collection->collection->Update(cpp_docs));
}

zvec_status_t *zvec_collection_delete(zvec_collection_t *collection,
                                      const char **pks, size_t count) {
  std::vector<std::string> cpp_pks;
  cpp_pks.reserve(count);
  for (size_t i = 0; i < count; ++i) cpp_pks.push_back(pks[i]);
  return handle_write_results(collection->collection->Delete(cpp_pks));
}

zvec_status_t *zvec_collection_query(zvec_collection_t *collection,
                                     const char *field_name,
                                     const float *vector, uint32_t count,
                                     int topk, zvec_doc_list_t **out_results) {
  VectorQuery vq;
  vq.field_name_ = field_name ? field_name : "";
  vq.topk_ = topk;
  vq.query_vector_.assign((const char *)vector, count * sizeof(float));
  auto result = collection->collection->Query(vq);
  if (!result) return new zvec_status{result.error()};
  *out_results = new zvec_doc_list{result.value()};
  return nullptr;
}

zvec_status_t *zvec_collection_fetch(zvec_collection_t *collection,
                                     const char **pks, size_t count,
                                     zvec_doc_list_t **out_results) {
  std::vector<std::string> cpp_pks;
  cpp_pks.reserve(count);
  for (size_t i = 0; i < count; ++i) cpp_pks.push_back(pks[i]);
  auto result = collection->collection->Fetch(cpp_pks);
  if (!result) return new zvec_status{result.error()};

  zvec_doc_list *list = new zvec_doc_list();
  for (auto &pair : result.value()) {
    if (pair.second) list->docs.push_back(pair.second);
  }
  *out_results = list;
  return nullptr;
}

// --- Doc List API Implementation ---

size_t zvec_doc_list_size(zvec_doc_list_t *list) {
  return list ? list->docs.size() : 0;
}

zvec_doc_t *zvec_doc_list_get(zvec_doc_list_t *list, size_t index) {
  if (!list || index >= list->docs.size()) return nullptr;
  auto doc = new zvec_doc();
  doc->doc = list->docs[index];
  return doc;
}

void zvec_doc_list_destroy(zvec_doc_list_t *list) {
  delete list;
}

// --- Stats API Implementation ---

uint64_t zvec_stats_total_docs(zvec_stats_t *stats) {
  return stats ? stats->stats.doc_count : 0;
}

void zvec_stats_destroy(zvec_stats_t *stats) {
  delete stats;
}
