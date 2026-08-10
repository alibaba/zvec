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

#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include "quantizer/quantizer.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

class Fp32Quantizer : public Quantizer {
 public:
  Fp32Quantizer() {
    type_ = QuantizeType::kFp32;
  }

  virtual ~Fp32Quantizer() {}

 public:
  int init(const core::IndexMeta &meta, const ailego::Params &params) override;

  const core::IndexMeta &meta(void) const override {
    return meta_;
  }

  DataType input_data_type() const override {
    return DataType::kFp32;
  }

  QuantizeType type() const override {
    return type_;
  }

  int dim() const override {
    return static_cast<int>(original_dim_);
  }

  bool require_train() const override {
    return false;
  }

  int train(core::IndexHolder::Pointer /*holder*/) override {
    return 0;
  }

  size_t quantized_datapoint_vector_length() const override {
    return quantized_length();
  }

  size_t quantized_query_vector_length() const override {
    return quantized_length();
  }

  void quantize_data(const void *input, void *output) const override {
    quantize_one(input, output);
  }

  void quantize_query(const void *input, void *output) const override {
    quantize_one(input, output);
  }

  float calc_distance_dp_query(const void *dp,
                               const void *query) const override;

  void calc_distance_dp_query_batch(const void *const *dp_list, int dp_num,
                                    const void *query,
                                    float *dist_list) const override;

  float calc_distance_dp_query_unquantized(const void *dp,
                                           const void *query) const override;

  void calc_distance_dp_query_batch_unquantized(
      const void *const *dp_list, int dp_num, const void *query,
      float *dist_list) const override;

  float calc_distance_dp_dp(const void *dp1, const void *dp2) const override;

  int quantize(const void *query, const core::IndexQueryMeta &qmeta,
               std::string *out, core::IndexQueryMeta *ometa) const override;

  int dequantize(const void *in, const core::IndexQueryMeta &qmeta,
                 std::string *out) const override;

  DistanceImpl distance(const void *query,
                        const core::IndexQueryMeta &qmeta) const override;

 private:
  //! Byte length of a quantized vector (raw fp32 data + extra meta).
  size_t quantized_length() const {
    return static_cast<size_t>(original_dim_) * sizeof(float) +
           extra_meta_size_;
  }

  //! Quantize a single fp32 vector into a caller-provided buffer of
  //! quantized_length() bytes.
  void quantize_one(const void *input, void *output) const;

  static constexpr uint32_t EXTRA_META_SIZE_COSINE = 4;

  IndexMeta meta_{};
  uint32_t original_dim_{0};

  //! Cached distance dispatch (bound in init()).
  DistanceFunc dp_query_func_{};
  BatchDistanceFunc dp_query_batch_func_{};
};


}  // namespace turbo
}  // namespace zvec
