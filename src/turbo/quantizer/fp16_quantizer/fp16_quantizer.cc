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

#include "quantizer/fp16_quantizer/fp16_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <ailego/math/normalizer.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace turbo {

int Fp16Quantizer::init(const IndexMeta &meta,
                        const ailego::Params & /*params*/) {
  input_data_type_ = (meta.data_type() == IndexMeta::DataType::DT_FP32)
                         ? DataType::kFp32
                         : DataType::kFp16;

  meta_ = meta;

  meta_.set_meta(IndexMeta::DataType::DT_FP16, meta.dimension());

  original_dim_ = meta.dimension();
  auto metric_name = meta.metric_name();
  if (metric_name == "Cosine") {
    extra_meta_size_ = EXTRA_META_SIZE_COSINE;
    meta_.set_extra_meta_size(extra_meta_size_);
    if (input_data_type_ == DataType::kFp32) {
      meta_.set_reformer("Fp16Quantizer", 0, ailego::Params());
    }
  }

  // Cache the distance dispatch for the new Quantizer interface.
  dp_query_func_ =
      get_distance_func(metric_from_name(metric_name), DataType::kFp16,
                        QuantizeType::kDefault, CpuArchType::kAuto);
  dp_query_batch_func_ =
      get_batch_distance_func(metric_from_name(metric_name), DataType::kFp16,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  return 0;
}

int Fp16Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  bool is_fp32 = (qmeta.unit_size() == sizeof(float));
  if (!is_fp32 && qmeta.unit_size() != sizeof(uint16_t)) {
    return kErrUnsupported;
  }

  size_t raw_dim = (original_dim_ != 0 && qmeta.dimension() >= original_dim_)
                       ? original_dim_
                       : qmeta.dimension();
  size_t byte_size = raw_dim * sizeof(uint16_t) + extra_meta_size_;
  out->resize(byte_size);

  if (meta_.metric_name() == "Cosine") {
    std::vector<float> fp32_buf(raw_dim);
    if (is_fp32) {
      std::memcpy(fp32_buf.data(), query, raw_dim * sizeof(float));
    } else {
      ailego::FloatHelper::ToFP32(reinterpret_cast<const uint16_t *>(query),
                                  raw_dim, fp32_buf.data());
    }
    float norm = 0.0f;
    ailego::Normalizer<float>::L2(fp32_buf.data(), raw_dim, &norm);
    ailego::FloatHelper::ToFP16(fp32_buf.data(), raw_dim,
                                reinterpret_cast<uint16_t *>(&(*out)[0]));
    std::memcpy(
        reinterpret_cast<uint8_t *>(&(*out)[0]) + raw_dim * sizeof(uint16_t),
        &norm, extra_meta_size_);
  } else {
    if (is_fp32) {
      ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(query),
                                  raw_dim,
                                  reinterpret_cast<uint16_t *>(&(*out)[0]));
    } else {
      std::memcpy(&(*out)[0], query, byte_size);
    }
  }

  *ometa = qmeta;
  uint32_t extra_dim = extra_meta_size_ / sizeof(uint16_t);
  ometa->set_meta(IndexMeta::DataType::DT_FP16, raw_dim + extra_dim);

  return 0;
}

int Fp16Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  size_t raw_dim = (original_dim_ != 0 && qmeta.dimension() >= original_dim_)
                       ? original_dim_
                       : qmeta.dimension();
  size_t byte_size = raw_dim * sizeof(uint16_t);

  if (meta_.metric_name() == "Cosine") {
    // Denormalize the vector using the stored norm (in FP32 space).
    out->resize(byte_size);
    const uint16_t *in_buf = reinterpret_cast<const uint16_t *>(in);
    uint16_t *out_buf = reinterpret_cast<uint16_t *>(&(*out)[0]);
    float norm = 0.0f;
    std::memcpy(&norm, reinterpret_cast<const uint8_t *>(in) + byte_size,
                extra_meta_size_);
    for (size_t i = 0; i < raw_dim; ++i) {
      out_buf[i] = ailego::FloatHelper::ToFP16(
          ailego::FloatHelper::ToFP32(in_buf[i]) * norm);
    }
  } else {
    out->resize(byte_size);
    std::memcpy(out->data(), in, byte_size);
  }
  return 0;
}

DistanceImpl Fp16Quantizer::distance(const void *query,
                                     const IndexQueryMeta &qmeta) const {
  auto metric = metric_from_name(meta_.metric_name());
  auto func = get_distance_func(metric, DataType::kFp16, QuantizeType::kDefault,
                                CpuArchType::kAuto);
  if (!func) {
    return DistanceImpl{};
  }
  auto batch_func = get_batch_distance_func(
      metric, DataType::kFp16, QuantizeType::kDefault, CpuArchType::kAuto);

  // The query is assumed to be already quantized — copy it directly.
  std::string quantized_query(static_cast<const char *>(query),
                              qmeta.element_size());
  return DistanceImpl(std::move(func), std::move(batch_func),
                      std::move(quantized_query), original_dim_);
}

void Fp16Quantizer::quantize_one(const void *input, void *output) const {
  size_t byte_size = static_cast<size_t>(original_dim_) * sizeof(uint16_t);

  if (meta_.metric_name() == "Cosine") {
    // L2-normalize in FP32 space and store the norm at the end.
    std::vector<float> fp32_buf(original_dim_);
    if (input_data_type_ == DataType::kFp32) {
      std::memcpy(fp32_buf.data(), input, original_dim_ * sizeof(float));
    } else {
      ailego::FloatHelper::ToFP32(reinterpret_cast<const uint16_t *>(input),
                                  original_dim_, fp32_buf.data());
    }
    float norm = 0.0f;
    ailego::Normalizer<float>::L2(fp32_buf.data(), original_dim_, &norm);
    ailego::FloatHelper::ToFP16(fp32_buf.data(), original_dim_,
                                reinterpret_cast<uint16_t *>(output));
    std::memcpy(reinterpret_cast<uint8_t *>(output) + byte_size, &norm,
                extra_meta_size_);
  } else {
    if (input_data_type_ == DataType::kFp32) {
      ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(input),
                                  original_dim_,
                                  reinterpret_cast<uint16_t *>(output));
    } else {
      std::memcpy(output, input, byte_size);
    }
  }
}

float Fp16Quantizer::calc_distance_dp_query(const void *dp,
                                            const void *query) const {
  float d = 0.0f;
  if (dp_query_func_) {
    dp_query_func_(dp, query, original_dim_, &d);
  }
  return d;
}

void Fp16Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                 int dp_num, const void *query,
                                                 float *dist_list) const {
  if (dp_query_batch_func_) {
    dp_query_batch_func_(const_cast<const void **>(dp_list), query,
                         static_cast<size_t>(dp_num), original_dim_, dist_list);
    return;
  }
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = calc_distance_dp_query(dp_list[i], query);
  }
}

float Fp16Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void Fp16Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float Fp16Quantizer::calc_distance_dp_dp(const void *dp1,
                                         const void *dp2) const {
  return calc_distance_dp_query(dp1, dp2);
}

INDEX_FACTORY_REGISTER_QUANTIZER(Fp16Quantizer);

}  // namespace turbo
}  // namespace zvec
