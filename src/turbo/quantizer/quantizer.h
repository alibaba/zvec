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

#include <cstdint>
#include <memory>
#include <string>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/turbo/turbo.h>
#include "distance.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

//! Self-describing, fixed-size header that prefixes every serialized quantizer.
//! The type-specific payload (scalar params, codebook, rotation matrix, ...)
//! follows immediately after this header.
struct QuantizerSerHeader {
  uint32_t magic;         // kQuantizerMagic
  uint16_t version;       // kQuantizerSerVersion
  uint16_t quant_type;    // QuantizeType
  uint32_t dim;           // original dim (sanity check)
  uint32_t metric;        // MetricType  (sanity check)
  uint32_t payload_size;  // bytes following the header
  uint32_t reserved;      // 0, for future use / alignment
};
static_assert(sizeof(QuantizerSerHeader) == 24,
              "QuantizerSerHeader must be 24 bytes");

class Quantizer {
 public:
  typedef std::shared_ptr<Quantizer> Pointer;

  Quantizer() {}
  virtual ~Quantizer() {}

  //! Initialize quantizer with index metadata and parameters
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Get the output metadata after initialization
  virtual const IndexMeta &meta() const = 0;

  //! Input data type accepted by the quantizer
  virtual DataType input_data_type() const = 0;

  //! Data type
  virtual QuantizeType type() const {
    return type_;
  }

  //! Dimensionality of the input vectors
  virtual int dim() const = 0;

  //! Train the quantizer with a contiguous batch of data
  virtual int train(const void * /*data*/, size_t /*num*/, size_t /*stride*/) {
    return 0;
  }

  //! Whether the quantizer requires training before use
  virtual bool require_train() const = 0;

  //! Train the quantizer with data from an IndexHolder
  virtual int train(IndexHolder::Pointer /*holder*/) {
    return 0;
  }

  //! Byte length of a quantized datapoint vector
  virtual size_t quantized_datapoint_vector_length() const = 0;

  //! Byte length of a quantized query vector
  virtual size_t quantized_query_vector_length() const = 0;

  //! Quantize a datapoint vector
  virtual void quantize_data(const void *input, void *output) const = 0;

  //! Quantize a query vector
  virtual void quantize_query(const void *input, void *output) const = 0;

  //! Distance between a quantized datapoint and a quantized query
  virtual float calc_distance_dp_query(const void *dp,
                                       const void *query) const = 0;

  //! Batched distance between quantized datapoints and a quantized query
  virtual void calc_distance_dp_query_batch(const void *const *dp_list,
                                            int dp_num, const void *query,
                                            float *dist_list) const = 0;

  //! Distance between a quantized datapoint and an unquantized query
  virtual float calc_distance_dp_query_unquantized(const void *dp,
                                                   const void *query) const = 0;

  //! Batched distance between quantized datapoints and an unquantized query
  virtual void calc_distance_dp_query_batch_unquantized(
      const void *const *dp_list, int dp_num, const void *query,
      float *dist_list) const = 0;

  //! Distance between two quantized datapoints
  virtual float calc_distance_dp_dp(const void *dp1, const void *dp2) const = 0;

  //! Quantize a query vector for search
  virtual int quantize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                       std::string * /*out*/,
                       IndexQueryMeta * /*ometa*/) const {
    return 0;
  }

  //! Dequantize a result vector back to original format
  virtual int dequantize(const void * /*in*/, const IndexQueryMeta & /*qmeta*/,
                         std::string * /*out*/) const {
    return 0;
  }

  virtual DistanceImpl distance(const void * /*query*/,
                                const IndexQueryMeta & /*qmeta*/) const {
    return DistanceImpl{};
  }

  //! Serialize quantizer parameters
  virtual int serialize(std::string * /*out*/) const {
    return 0;
  }

  //! Deserialize quantizer parameters
  virtual int deserialize(std::string & /*in*/) {
    return 0;
  }

  //! Deserialize quantizer parameters from a raw, possibly mmap-backed buffer
  //! (zero-copy entry point for large payloads such as codebooks/matrices).
  virtual int deserialize(const void * /*data*/, size_t /*len*/) {
    return 0;
  }

 protected:
  //! Map a metric name (e.g. "SquaredEuclidean", "Cosine",
  //! "InnerProduct", "MipsSquaredEuclidean") to its MetricType.
  static MetricType metric_from_name(const std::string &name) {
    if (name == "SquaredEuclidean") {
      return MetricType::kSquaredEuclidean;
    }
    if (name == "Cosine") {
      return MetricType::kCosine;
    }
    if (name == "InnerProduct") {
      return MetricType::kInnerProduct;
    }
    if (name == "MipsSquaredEuclidean") {
      return MetricType::kMipsSquaredEuclidean;
    }
    return MetricType::kUnknown;
  }

  QuantizeType type_{QuantizeType::kDefault};
  uint32_t extra_meta_size_{0};
};

}  // namespace turbo
}  // namespace zvec
