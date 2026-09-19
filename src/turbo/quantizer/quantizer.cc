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

#include "quantizer/quantizer.h"
#include <string>
#include <vector>

namespace zvec::turbo {

namespace {

struct InputDistanceScratch {
  std::string datapoint;
  std::string query;
  std::string datapoints;
  std::vector<const void *> datapoint_ptrs;
};

InputDistanceScratch &GetInputDistanceScratch() {
  thread_local InputDistanceScratch scratch;
  return scratch;
}

void QuantizeDatapoints(const Quantizer &quantizer, const void *const *dp_list,
                        int dp_num, InputDistanceScratch *scratch) {
  const size_t length = quantizer.quantized_datapoint_vector_length();
  scratch->datapoints.resize(length * static_cast<size_t>(dp_num));
  scratch->datapoint_ptrs.resize(static_cast<size_t>(dp_num));
  for (int i = 0; i < dp_num; ++i) {
    void *output = scratch->datapoints.data() + length * i;
    quantizer.quantize_data(dp_list[i], output);
    scratch->datapoint_ptrs[i] = output;
  }
}

}  // namespace

float Quantizer::calc_distance_input_query(const void *dp,
                                           const void *query) const {
  auto &scratch = GetInputDistanceScratch();
  scratch.datapoint.resize(quantized_datapoint_vector_length());
  quantize_data(dp, scratch.datapoint.data());
  return calc_distance_dp_query(scratch.datapoint.data(), query);
}

void Quantizer::calc_distance_input_query_batch(const void *const *dp_list,
                                                int dp_num, const void *query,
                                                float *dist_list) const {
  if (dp_num <= 0) {
    return;
  }
  auto &scratch = GetInputDistanceScratch();
  QuantizeDatapoints(*this, dp_list, dp_num, &scratch);
  calc_distance_dp_query_batch(scratch.datapoint_ptrs.data(), dp_num, query,
                               dist_list);
}

float Quantizer::calc_distance_input_input(const void *dp1,
                                           const void *dp2) const {
  auto &scratch = GetInputDistanceScratch();
  scratch.datapoint.resize(quantized_datapoint_vector_length());
  scratch.query.resize(quantized_query_vector_length());
  quantize_data(dp1, scratch.datapoint.data());
  quantize_query(dp2, scratch.query.data());
  return calc_distance_dp_query(scratch.datapoint.data(), scratch.query.data());
}

void Quantizer::calc_distance_input_input_batch(const void *const *dp_list,
                                                int dp_num, const void *query,
                                                float *dist_list) const {
  if (dp_num <= 0) {
    return;
  }
  auto &scratch = GetInputDistanceScratch();
  QuantizeDatapoints(*this, dp_list, dp_num, &scratch);
  scratch.query.resize(quantized_query_vector_length());
  quantize_query(query, scratch.query.data());
  calc_distance_dp_query_batch(scratch.datapoint_ptrs.data(), dp_num,
                               scratch.query.data(), dist_list);
}

}  // namespace zvec::turbo
