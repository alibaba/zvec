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

#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// Internal Helper Functions
// =============================================================================

static char *strdup_safe(const char *str) {
  if (!str) return NULL;
  size_t len = strlen(str) + 1;
  char *copy = (char *)malloc(len);
  if (copy) {
    memcpy(copy, str, len);
  }
  return copy;
}

// =============================================================================
// Schema Creation Helper Functions Implementation
// =============================================================================

ZVecCollectionSchema *zvec_test_create_temp_schema(void) {
  // Create collection schema using C API
  ZVecCollectionSchema *schema = zvec_collection_schema_create("demo");
  schema->max_doc_count_per_segment = 1000;

  // Create index parameters using C API
  ZVecInvertIndexParams *invert_params =
      zvec_index_params_invert_create(true, true);
  ZVecHnswIndexParams *dense_hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_L2, 16, 100, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  ZVecHnswIndexParams *sparse_hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_IP, 16, 100, 50, ZVEC_QUANTIZE_TYPE_UNDEFINED);


  // Create and add fields
  ZVecFieldSchema *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
  zvec_field_schema_set_invert_index(id_field, invert_params);
  zvec_collection_schema_add_field(schema, id_field);

  // Create name field (inverted index without optimization)
  ZVecInvertIndexParams *name_invert_params =
      zvec_index_params_invert_create(false, false);
  ZVecFieldSchema *name_field =
      zvec_field_schema_create("name", ZVEC_DATA_TYPE_STRING, false, 0);
  zvec_field_schema_set_invert_index(name_field, name_invert_params);
  zvec_collection_schema_add_field(schema, name_field);

  // Create weight field (no index)
  ZVecFieldSchema *weight_field =
      zvec_field_schema_create("weight", ZVEC_DATA_TYPE_FLOAT, true, 0);
  zvec_collection_schema_add_field(schema, weight_field);

  // Create dense field (HNSW index)
  ZVecFieldSchema *dense_field =
      zvec_field_schema_create("dense", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  zvec_field_schema_set_hnsw_index(dense_field, dense_hnsw_params);
  zvec_collection_schema_add_field(schema, dense_field);

  // Create sparse field (HNSW index)
  ZVecFieldSchema *sparse_field = zvec_field_schema_create(
      "sparse", ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32, false, 0);
  zvec_field_schema_set_hnsw_index(sparse_field, sparse_hnsw_params);
  zvec_collection_schema_add_field(schema, sparse_field);

  return schema;
}

ZVecCollectionSchema *zvec_test_create_scalar_schema(void) {
  // Create collection schema using C API
  ZVecCollectionSchema *schema = zvec_collection_schema_create("demo");

  // Create fields
  ZVecFieldSchema *int32_field =
      zvec_field_schema_create("int32", ZVEC_DATA_TYPE_INT32, false, 0);
  zvec_collection_schema_add_field(schema, int32_field);

  ZVecFieldSchema *string_field =
      zvec_field_schema_create("string", ZVEC_DATA_TYPE_STRING, false, 0);
  zvec_collection_schema_add_field(schema, string_field);

  return schema;
}

ZVecCollectionSchema *zvec_test_create_normal_schema(
    bool nullable, const char *name,
    const ZVecInvertIndexParams *scalar_index_params,
    const ZVecHnswIndexParams *vector_index_params, uint64_t max_doc_count) {
  // Create collection schema using C API
  ZVecCollectionSchema *schema =
      zvec_collection_schema_create(name ? name : "demo");
  schema->max_doc_count_per_segment = max_doc_count;

  // Create scalar fields (8)
  const char *scalar_names[] = {"int32", "string", "uint32", "bool",
                                "float", "double", "int64",  "uint64"};
  ZVecDataType scalar_types[] = {ZVEC_DATA_TYPE_INT32,  ZVEC_DATA_TYPE_STRING,
                                 ZVEC_DATA_TYPE_UINT32, ZVEC_DATA_TYPE_BOOL,
                                 ZVEC_DATA_TYPE_FLOAT,  ZVEC_DATA_TYPE_DOUBLE,
                                 ZVEC_DATA_TYPE_INT64,  ZVEC_DATA_TYPE_UINT64};

  for (int i = 0; i < 8; i++) {
    ZVecFieldSchema *field =
        zvec_field_schema_create(scalar_names[i], scalar_types[i], nullable, 0);
    if (scalar_index_params) {
      zvec_field_schema_set_invert_index(
          field, (ZVecInvertIndexParams *)scalar_index_params);
    }
    zvec_collection_schema_add_field(schema, field);
  }

  // Create array fields (8)
  const char *array_names[] = {"array_int32", "array_string", "array_uint32",
                               "array_bool",  "array_float",  "array_double",
                               "array_int64", "array_uint64"};
  ZVecDataType array_types[] = {
      ZVEC_DATA_TYPE_ARRAY_INT32,  ZVEC_DATA_TYPE_ARRAY_STRING,
      ZVEC_DATA_TYPE_ARRAY_UINT32, ZVEC_DATA_TYPE_ARRAY_BOOL,
      ZVEC_DATA_TYPE_ARRAY_FLOAT,  ZVEC_DATA_TYPE_ARRAY_DOUBLE,
      ZVEC_DATA_TYPE_ARRAY_INT64,  ZVEC_DATA_TYPE_ARRAY_UINT64};

  for (int i = 0; i < 8; i++) {
    ZVecFieldSchema *field =
        zvec_field_schema_create(array_names[i], array_types[i], nullable, 0);
    if (scalar_index_params) {
      zvec_field_schema_set_invert_index(
          field, (ZVecInvertIndexParams *)scalar_index_params);
    }
    zvec_collection_schema_add_field(schema, field);
  }

  // Create vector fields (5)
  // dense vectors
  ZVecFieldSchema *dense_fp32 = zvec_field_schema_create(
      "dense_fp32", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
  if (vector_index_params) {
    zvec_field_schema_set_hnsw_index(
        dense_fp32, (ZVecHnswIndexParams *)vector_index_params);
  }
  zvec_collection_schema_add_field(schema, dense_fp32);

  ZVecFieldSchema *dense_fp16 = zvec_field_schema_create(
      "dense_fp16", ZVEC_DATA_TYPE_VECTOR_FP16, false, 128);
  ZVecFlatIndexParams *flat_params1 = zvec_index_params_flat_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  zvec_field_schema_set_flat_index(dense_fp16, flat_params1);
  zvec_collection_schema_add_field(schema, dense_fp16);

  ZVecFieldSchema *dense_int8 = zvec_field_schema_create(
      "dense_int8", ZVEC_DATA_TYPE_VECTOR_INT8, false, 128);
  ZVecFlatIndexParams *flat_params2 = zvec_index_params_flat_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  zvec_field_schema_set_flat_index(dense_int8, flat_params2);
  zvec_collection_schema_add_field(schema, dense_int8);

  // sparse vectors
  ZVecFieldSchema *sparse_fp32 = zvec_field_schema_create(
      "sparse_fp32", ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32, false, 0);
  if (vector_index_params) {
    zvec_field_schema_set_hnsw_index(
        sparse_fp32, (ZVecHnswIndexParams *)vector_index_params);
  }
  zvec_collection_schema_add_field(schema, sparse_fp32);

  ZVecFieldSchema *sparse_fp16 = zvec_field_schema_create(
      "sparse_fp16", ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16, false, 0);
  ZVecFlatIndexParams *flat_params3 = zvec_index_params_flat_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  zvec_field_schema_set_flat_index(sparse_fp16, flat_params3);
  zvec_collection_schema_add_field(schema, sparse_fp16);

  return schema;
}

ZVecCollectionSchema *zvec_test_create_schema_with_scalar_index(
    bool nullable, bool enable_optimize, const char *name) {
  ZVecInvertIndexParams *invert_params =
      zvec_test_create_default_invert_params(enable_optimize);
  ZVecCollectionSchema *schema =
      zvec_test_create_normal_schema(nullable, name, invert_params, NULL, 1000);
  free(invert_params);
  return schema;
}

ZVecCollectionSchema *zvec_test_create_schema_with_vector_index(
    bool nullable, const char *name,
    const ZVecHnswIndexParams *vector_index_params) {
  ZVecHnswIndexParams *default_params = NULL;
  if (!vector_index_params) {
    default_params = zvec_test_create_default_hnsw_params();
  }

  ZVecCollectionSchema *schema = zvec_test_create_normal_schema(
      nullable, name, NULL,
      vector_index_params ? vector_index_params : default_params, 1000);

  if (default_params) {
    free(default_params);
  }

  return schema;
}

ZVecCollectionSchema *zvec_test_create_schema_with_max_doc_count(
    uint64_t doc_count) {
  return zvec_test_create_normal_schema(false, "demo", NULL, NULL, doc_count);
}

// =============================================================================
// Document Creation Helper Functions Implementation
// =============================================================================

char *zvec_test_make_pk(uint64_t doc_id) {
  char *pk = (char *)malloc(32);  // Sufficiently large buffer
  if (pk) {
    snprintf(pk, 32, "pk_%llu", (unsigned long long)doc_id);
  }
  return pk;
}

uint64_t zvec_test_extract_doc_id(const char *pk) {
  if (!pk || strlen(pk) < 4) return 0;
  return strtoull(pk + 3, NULL, 10);
}

ZVecDoc *zvec_test_create_doc(uint64_t doc_id,
                              const ZVecCollectionSchema *schema,
                              const char *pk) {
  if (!schema) return NULL;
  ZVecDoc *doc = zvec_doc_create();
  if (!doc) return NULL;

  // Set primary key
  char *primary_key = pk ? strdup_safe(pk) : zvec_test_make_pk(doc_id);
  if (primary_key) {
    zvec_doc_set_pk(doc, primary_key);
    free(primary_key);
  }

  // Create test data for each field
  for (size_t i = 0; i < schema->field_count; i++) {
    // Fix type mismatch issue - remove address operator
    const ZVecFieldSchema *field = schema->fields[i];
    // Remove unused variable
    // ZVecErrorCode err = ZVEC_OK;

    switch (field->data_type) {
      case ZVEC_DATA_TYPE_BINARY: {
        char binary_str[32];
        snprintf(binary_str, sizeof(binary_str), "binary_%llu",
                 (unsigned long long)doc_id);
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    binary_str, strlen(binary_str));
        break;
      }
      case ZVEC_DATA_TYPE_BOOL: {
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    &(bool){doc_id % 10 == 0}, sizeof(bool));
        break;
      }
      case ZVEC_DATA_TYPE_INT32: {
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    &(int32_t){(int32_t)doc_id},
                                    sizeof(int32_t));
        break;
      }
      case ZVEC_DATA_TYPE_INT64: {
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    &(int64_t){(int64_t)doc_id},
                                    sizeof(int64_t));
        break;
      }
      case ZVEC_DATA_TYPE_UINT32: {
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    &(uint32_t){(uint32_t)doc_id},
                                    sizeof(uint32_t));
        break;
      }
      case ZVEC_DATA_TYPE_UINT64: {
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    &(uint64_t){(uint64_t)doc_id},
                                    sizeof(uint64_t));
        break;
      }
      case ZVEC_DATA_TYPE_FLOAT: {
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    &(float){(float)doc_id}, sizeof(float));
        break;
      }
      case ZVEC_DATA_TYPE_DOUBLE: {
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    &(double){(double)doc_id}, sizeof(double));
        break;
      }
      case ZVEC_DATA_TYPE_STRING: {
        char string_val[64];
        snprintf(string_val, sizeof(string_val), "value_%llu",
                 (unsigned long long)doc_id);
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    string_val, strlen(string_val));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_BOOL: {
        bool bool_array[10];
        for (int j = 0; j < 10; j++) {
          bool_array[j] = (doc_id + j) % 2 == 0;
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    bool_array, sizeof(bool_array));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_INT32: {
        int32_t int32_array[10];
        for (int j = 0; j < 10; j++) {
          int32_array[j] = (int32_t)doc_id;
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    int32_array, sizeof(int32_array));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_INT64: {
        int64_t int64_array[10];
        for (int j = 0; j < 10; j++) {
          int64_array[j] = (int64_t)doc_id;
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    int64_array, sizeof(int64_array));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_UINT32: {
        uint32_t uint32_array[10];
        for (int j = 0; j < 10; j++) {
          uint32_array[j] = (uint32_t)doc_id;
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    uint32_array, sizeof(uint32_array));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_UINT64: {
        uint64_t uint64_array[10];
        for (int j = 0; j < 10; j++) {
          uint64_array[j] = (uint64_t)doc_id;
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    uint64_array, sizeof(uint64_array));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_FLOAT: {
        float float_array[10];
        for (int j = 0; j < 10; j++) {
          float_array[j] = (float)doc_id;
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    float_array, sizeof(float_array));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_DOUBLE: {
        double double_array[10];
        for (int j = 0; j < 10; j++) {
          double_array[j] = (double)doc_id;
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    double_array, sizeof(double_array));
        break;
      }
      case ZVEC_DATA_TYPE_ARRAY_STRING: {
        // String arrays need special handling
        char string_data[256];
        size_t offset = 0;
        for (int j = 0; j < 10; j++) {
          char temp_str[32];
          snprintf(temp_str, sizeof(temp_str), "value_%llu_%d",
                   (unsigned long long)doc_id, j);
          size_t len = strlen(temp_str);
          if (offset + len + 1 < sizeof(string_data)) {
            strcpy(string_data + offset, temp_str);
            offset += len + 1;
          }
        }
        zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                    string_data, offset);
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_BINARY32: {
        uint32_t *vector_data =
            (uint32_t *)malloc(field->dimension * sizeof(uint32_t));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (uint32_t)(doc_id + j);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      vector_data,
                                      field->dimension * sizeof(uint32_t));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_BINARY64: {
        uint64_t *vector_data =
            (uint64_t *)malloc(field->dimension * sizeof(uint64_t));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (uint64_t)(doc_id + j);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      vector_data,
                                      field->dimension * sizeof(uint64_t));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_FP32: {
        float *vector_data = (float *)malloc(field->dimension * sizeof(float));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (float)(doc_id + j * 0.1);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      vector_data,
                                      field->dimension * sizeof(float));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_FP64: {
        double *vector_data =
            (double *)malloc(field->dimension * sizeof(double));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (double)(doc_id + j * 0.1);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      vector_data,
                                      field->dimension * sizeof(double));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_FP16: {
        // FP16 needs special handling, simplified to FP32 here
        float *vector_data = (float *)malloc(field->dimension * sizeof(float));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (float)(doc_id + j * 0.1);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      vector_data,
                                      field->dimension * sizeof(float));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_INT8: {
        int8_t *vector_data =
            (int8_t *)malloc(field->dimension * sizeof(int8_t));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (int8_t)((doc_id + j) % 256);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      vector_data,
                                      field->dimension * sizeof(int8_t));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_INT16: {
        int16_t *vector_data =
            (int16_t *)malloc(field->dimension * sizeof(int16_t));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (int16_t)((doc_id + j) % 65536);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      vector_data,
                                      field->dimension * sizeof(int16_t));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
        // Sparse vectors need special handling
        uint32_t nnz = field->dimension > 0
                           ? field->dimension / 10
                           : 10;  // Number of non-zero elements
        size_t sparse_size =
            sizeof(uint32_t) + nnz * (sizeof(uint32_t) + sizeof(float));
        void *sparse_data = malloc(sparse_size);
        if (sparse_data) {
          uint32_t *data_ptr = (uint32_t *)sparse_data;
          *data_ptr = nnz;  // Set number of non-zero elements
          uint32_t *indices = data_ptr + 1;
          float *values = (float *)(indices + nnz);
          for (uint32_t j = 0; j < nnz; j++) {
            indices[j] = j * 10;                    // Index
            values[j] = (float)(doc_id + j * 0.1);  // Value
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      sparse_data, sparse_size);
          free(sparse_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16: {
        // Sparse FP16 vectors, simplified handling
        uint32_t nnz = field->dimension > 0 ? field->dimension / 10 : 10;
        size_t sparse_size =
            sizeof(uint32_t) +
            nnz * (sizeof(uint32_t) +
                   sizeof(float));  // Still use float for storage
        void *sparse_data = malloc(sparse_size);
        if (sparse_data) {
          uint32_t *data_ptr = (uint32_t *)sparse_data;
          *data_ptr = nnz;
          uint32_t *indices = data_ptr + 1;
          float *values = (float *)(indices + nnz);
          for (uint32_t j = 0; j < nnz; j++) {
            indices[j] = j * 10;
            values[j] = (float)(doc_id + j * 0.1);
          }
          zvec_doc_add_field_by_value(doc, field->name->data, field->data_type,
                                      sparse_data, sparse_size);
          free(sparse_data);
        }
        break;
      }

      default:
        // Unsupported data type
        break;
    }

    // Remove reference to removed variable err
    /*
    if (err != ZVEC_OK) {
        // Error handling: continue processing other fields
    }
    */
  }

  return doc;
}

ZVecDoc *zvec_test_create_doc_null(uint64_t doc_id,
                                   const ZVecCollectionSchema *schema,
                                   const char *pk) {
  // Reuse create_doc function, but only process vector fields
  ZVecDoc *doc = zvec_doc_create();
  if (!doc) return NULL;

  // Set primary key
  char *primary_key = pk ? strdup_safe(pk) : zvec_test_make_pk(doc_id);
  if (primary_key) {
    zvec_doc_set_pk(doc, primary_key);
    free(primary_key);
  }

  // Only create data for vector fields
  for (size_t i = 0; i < schema->field_count; i++) {
    const ZVecFieldSchema *field = schema->fields[i];

    // Only process specific vector type fields
    if (field->data_type != ZVEC_DATA_TYPE_VECTOR_FP32 &&
        field->data_type != ZVEC_DATA_TYPE_VECTOR_FP16 &&
        field->data_type != ZVEC_DATA_TYPE_VECTOR_INT8 &&
        field->data_type != ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 &&
        field->data_type != ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16) {
      continue;
    }

    ZVecErrorCode err = ZVEC_OK;

    switch (field->data_type) {
      case ZVEC_DATA_TYPE_VECTOR_FP32: {
        float *vector_data = (float *)malloc(field->dimension * sizeof(float));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (float)(doc_id + j * 0.1);
          }
          err = zvec_doc_add_field_by_value(doc, field->name->data,
                                            field->data_type, vector_data,
                                            field->dimension * sizeof(float));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_FP64: {
        double *vector_data =
            (double *)malloc(field->dimension * sizeof(double));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (double)(doc_id + j * 0.1);
          }
          err = zvec_doc_add_field_by_value(doc, field->name->data,
                                            field->data_type, vector_data,
                                            field->dimension * sizeof(double));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_FP16: {
        float *vector_data = (float *)malloc(field->dimension * sizeof(float));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (float)(doc_id + j * 0.1);
          }
          err = zvec_doc_add_field_by_value(doc, field->name->data,
                                            field->data_type, vector_data,
                                            field->dimension * sizeof(float));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_INT8: {
        int8_t *vector_data =
            (int8_t *)malloc(field->dimension * sizeof(int8_t));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (int8_t)(doc_id % 128);
          }
          err = zvec_doc_add_field_by_value(doc, field->name->data,
                                            field->data_type, vector_data,
                                            field->dimension * sizeof(int8_t));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_VECTOR_INT16: {
        int16_t *vector_data =
            (int16_t *)malloc(field->dimension * sizeof(int16_t));
        if (vector_data) {
          for (uint32_t j = 0; j < field->dimension; j++) {
            vector_data[j] = (int16_t)(doc_id % 32768);
          }
          err = zvec_doc_add_field_by_value(doc, field->name->data,
                                            field->data_type, vector_data,
                                            field->dimension * sizeof(int16_t));
          free(vector_data);
        }
        break;
      }
      case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16:
      case ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32: {
        const size_t nnz = 100;
        size_t sparse_size =
            sizeof(size_t) + nnz * (sizeof(uint32_t) + sizeof(float));
        char *sparse_data = (char *)malloc(sparse_size);
        if (sparse_data) {
          char *ptr = sparse_data;
          *((size_t *)ptr) = nnz;
          ptr += sizeof(size_t);

          for (size_t j = 0; j < nnz; j++) {
            *((uint32_t *)ptr) = (uint32_t)j;
            ptr += sizeof(uint32_t);
            *((float *)ptr) = (float)(doc_id + j * 0.1);
            ptr += sizeof(float);
          }
          err = zvec_doc_add_field_by_value(doc, field->name->data,
                                            field->data_type, sparse_data,
                                            sparse_size);
          free(sparse_data);
        }
        break;
      }
      default:
        break;
    }


    if (err != ZVEC_OK) {
      zvec_doc_destroy(doc);
      return NULL;
    }
  }

  return doc;
}

ZVecDoc *zvec_test_create_doc_with_fields(uint64_t doc_id,
                                          const char **field_names,
                                          const ZVecDataType *field_types,
                                          size_t field_count, const char *pk) {
  ZVecDoc *doc = zvec_doc_create();
  if (!doc) return NULL;

  // Set primary key
  char *primary_key = pk ? strdup_safe(pk) : zvec_test_make_pk(doc_id);
  if (primary_key) {
    zvec_doc_set_pk(doc, primary_key);
    free(primary_key);
  }

  // Create data for specified fields
  for (size_t i = 0; i < field_count; i++) {
    ZVecErrorCode err = ZVEC_OK;

    switch (field_types[i]) {
      case ZVEC_DATA_TYPE_INT32:
        err = zvec_doc_add_field_by_value(doc, field_names[i], field_types[i],
                                          &(int32_t){(int32_t)doc_id},
                                          sizeof(int32_t));
        break;
      case ZVEC_DATA_TYPE_STRING: {
        char string_val[64];
        snprintf(string_val, sizeof(string_val), "value_%llu",
                 (unsigned long long)doc_id);
        err = zvec_doc_add_field_by_value(doc, field_names[i], field_types[i],
                                          string_val, strlen(string_val));
        break;
      }
      case ZVEC_DATA_TYPE_FLOAT:
        err =
            zvec_doc_add_field_by_value(doc, field_names[i], field_types[i],
                                        &(float){(float)doc_id}, sizeof(float));
        break;
      case ZVEC_DATA_TYPE_VECTOR_FP32: {
        float vector_data[128];
        for (int j = 0; j < 128; j++) {
          vector_data[j] = (float)(doc_id + j * 0.1);
        }
        err = zvec_doc_add_field_by_value(doc, field_names[i], field_types[i],
                                          vector_data, sizeof(vector_data));
        break;
      }
      default:
        // Other types can be added here
        break;
    }

    if (err != ZVEC_OK) {
      zvec_doc_destroy(doc);
      return NULL;
    }
  }

  return doc;
}

// =============================================================================
// Index Parameter Creation Helper Functions Implementation
// =============================================================================

ZVecHnswIndexParams *zvec_test_create_default_hnsw_params(void) {
  ZVecHnswIndexParams *params =
      (ZVecHnswIndexParams *)malloc(sizeof(ZVecHnswIndexParams));
  if (!params) return NULL;

  params->base.base.index_type = ZVEC_INDEX_TYPE_HNSW;
  params->base.metric_type = ZVEC_METRIC_TYPE_IP;
  params->base.quantize_type = ZVEC_QUANTIZE_TYPE_UNDEFINED;
  params->m = 16;
  params->ef_construction = 100;

  return params;
}

ZVecFlatIndexParams *zvec_test_create_default_flat_params(void) {
  ZVecFlatIndexParams *params =
      (ZVecFlatIndexParams *)malloc(sizeof(ZVecFlatIndexParams));
  if (!params) return NULL;

  params->base.base.index_type = ZVEC_INDEX_TYPE_FLAT;
  params->base.metric_type = ZVEC_METRIC_TYPE_IP;
  params->base.quantize_type = ZVEC_QUANTIZE_TYPE_UNDEFINED;

  return params;
}

ZVecInvertIndexParams *zvec_test_create_default_invert_params(
    bool enable_optimize) {
  ZVecInvertIndexParams *params =
      (ZVecInvertIndexParams *)malloc(sizeof(ZVecInvertIndexParams));
  if (!params) return NULL;

  params->base.index_type = ZVEC_INDEX_TYPE_INVERT;
  params->enable_range_optimization = enable_optimize;
  params->enable_extended_wildcard = enable_optimize;

  return params;
}

// =============================================================================
// Field Schema Creation Helper Functions Implementation
// =============================================================================

ZVecFieldSchema *zvec_test_create_scalar_field(
    const char *name, ZVecDataType data_type, bool nullable,
    const ZVecInvertIndexParams *invert_params) {
  ZVecFieldSchema *field = (ZVecFieldSchema *)malloc(sizeof(ZVecFieldSchema));
  if (!field) return NULL;

  field->name = (ZVecString *)malloc(sizeof(ZVecString));
  if (!field->name) {
    free(field);
    return NULL;
  }
  // Fix const qualifier issue - create string copy
  field->name->data = name ? strdup(name) : NULL;
  field->name->length = name ? strlen(name) : 0;
  field->name->capacity = name ? strlen(name) + 1 : 0;
  field->data_type = data_type;
  field->nullable = nullable;
  field->dimension = 0;
  field->index_params = invert_params ? (ZVecIndexParams *)invert_params : NULL;

  return field;
}

ZVecFieldSchema *zvec_test_create_vector_field(
    const char *name, ZVecDataType data_type, uint32_t dimension, bool nullable,
    const ZVecHnswIndexParams *vector_index_params) {
  ZVecFieldSchema *field = (ZVecFieldSchema *)malloc(sizeof(ZVecFieldSchema));
  if (!field) return NULL;

  field->name = (ZVecString *)malloc(sizeof(ZVecString));
  if (!field->name) {
    free(field);
    return NULL;
  }
  // Fix const qualifier issue - create string copy
  field->name->data = name ? strdup(name) : NULL;
  field->name->length = name ? strlen(name) : 0;
  field->name->capacity = name ? strlen(name) + 1 : 0;
  field->data_type = data_type;
  field->nullable = nullable;
  field->dimension = dimension;
  field->index_params =
      vector_index_params ? (ZVecIndexParams *)vector_index_params : NULL;

  return field;
}

ZVecFieldSchema *zvec_test_create_sparse_vector_field(
    const char *name, ZVecDataType data_type, bool nullable,
    const ZVecHnswIndexParams *vector_index_params) {
  ZVecFieldSchema *field = (ZVecFieldSchema *)malloc(sizeof(ZVecFieldSchema));
  if (!field) return NULL;

  field->name = (ZVecString *)malloc(sizeof(ZVecString));
  if (!field->name) {
    free(field);
    return NULL;
  }
  // Fix const qualifier issue - create string copy
  field->name->data = name ? strdup(name) : NULL;
  field->name->length = name ? strlen(name) : 0;
  field->name->capacity = name ? strlen(name) + 1 : 0;
  field->data_type = data_type;
  field->nullable = nullable;
  field->dimension = 0;  // Sparse vectors don't need fixed dimension
  field->index_params =
      vector_index_params ? (ZVecIndexParams *)vector_index_params : NULL;

  return field;
}

// =============================================================================
// Memory Management Helper Functions Implementation
// =============================================================================

void zvec_test_free_field_schemas(ZVecFieldSchema *fields, size_t count) {
  if (!fields) return;

  for (size_t i = 0; i < count; i++) {
    if (fields[i].name) {
      // Free string memory allocated by strdup
      if (fields[i].name->data) {
        free(fields[i].name->data);
      }
      free(fields[i].name);
    }
    // Free index parameter memory
    if (fields[i].index_params) {
      zvec_index_params_destroy(fields[i].index_params);
      free(fields[i].index_params);
    }
  }
  free(fields);
}

void zvec_test_free_strings(char **strings, size_t count) {
  if (!strings) return;

  for (size_t i = 0; i < count; i++) {
    if (strings[i]) {
      free(strings[i]);
    }
  }

  free(strings);
}

// =============================================================================
// File System Helper Functions Implementation
// =============================================================================

/**
 * @brief Delete directory and all its contents (wrapper function)
 *
 * @param dir_path Directory path
 * @return int 0 for success, -1 for failure
 */
int zvec_test_delete_dir(const char *dir_path) {
  if (!dir_path) {
    return -1;
  }

#ifdef _WIN32
  // Windows platform implementation
  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "rd /s /q \"%s\" >nul 2>&1", dir_path);
  int result = system(cmd);
  return (result == 0) ? 0 : -1;
#else
  // Unix/Linux/macOS platform implementation
  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "rm -rf \"%s\" 2>/dev/null", dir_path);
  int result = system(cmd);
  return (result == 0) ? 0 : -1;
#endif
}
