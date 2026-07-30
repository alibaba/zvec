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
  meta_ = meta;

  meta_.set_meta(IndexMeta::DataType::DT_FP16, meta.dimension());

  original_dim_ = meta.dimension();
  auto metric_name = meta.metric_name();
  if (metric_name == "Cosine") {
    extra_meta_size_ = EXTRA_META_SIZE_COSINE;
    meta_.set_extra_meta_size(extra_meta_size_);
  }

  // Cache the distance dispatch for the new Quantizer interface.
  auto kernels =
      get_distance_kernels(metric_from_name(metric_name), DataType::kFp16,
                           QuantizeType::kFp16, CpuArchType::kAuto);
  dp_query_func_ = std::move(kernels.dist);
  dp_query_batch_func_ = std::move(kernels.batch);

  return 0;
}

int Fp16Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  if (qmeta.unit_size() != sizeof(float)) {
    return kErrUnsupported;
  }

  // qmeta.dimension() may be the inflated (data + extras) dimension when the
  // caller uses meta_.dimension() directly. Use the raw original dim we
  // recorded at init() to avoid over-reading the query.
  size_t raw_dim = (original_dim_ != 0 && qmeta.dimension() >= original_dim_)
                       ? original_dim_
                       : qmeta.dimension();
  size_t byte_size = raw_dim * sizeof(uint16_t) + extra_meta_size_;
  out->resize(byte_size);
  uint16_t *out_buf = reinterpret_cast<uint16_t *>(&(*out)[0]);

  if (meta_.metric_name() == "Cosine") {
    // L2-normalize the vector before converting to fp16 and store the norm
    // at the end so the original vector can be reconstructed during
    // dequantize.
    std::vector<float> buf(raw_dim);
    std::memcpy(buf.data(), query, raw_dim * sizeof(float));
    float norm = 0.0f;
    ailego::Normalizer<float>::L2(buf.data(), raw_dim, &norm);
    ailego::FloatHelper::ToFP16(buf.data(), raw_dim, out_buf);
    std::memcpy(
        reinterpret_cast<uint8_t *>(&(*out)[0]) + raw_dim * sizeof(uint16_t),
        &norm, extra_meta_size_);
  } else {
    ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(query), raw_dim,
                                out_buf);
  }

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_FP16, raw_dim,
                  static_cast<uint32_t>(type_), extra_meta_size_);

  return 0;
}

int Fp16Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  size_t raw_dim = (original_dim_ != 0 && qmeta.dimension() >= original_dim_)
                       ? original_dim_
                       : qmeta.dimension();
  size_t byte_size = raw_dim * sizeof(float);

  out->resize(byte_size);
  const uint16_t *in_buf = reinterpret_cast<const uint16_t *>(in);
  float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
  ailego::FloatHelper::ToFP32(in_buf, raw_dim, out_buf);

  if (meta_.metric_name() == "Cosine") {
    // Denormalize the vector using the stored norm.
    float norm = 0.0f;
    std::memcpy(
        &norm,
        reinterpret_cast<const uint8_t *>(in) + raw_dim * sizeof(uint16_t),
        extra_meta_size_);
    for (size_t i = 0; i < raw_dim; ++i) {
      out_buf[i] *= norm;
    }
  }
  return 0;
}

DistanceImpl Fp16Quantizer::distance(const void *query,
                                     const IndexQueryMeta &qmeta) const {
  // Reuse the dispatch cached at init().
  if (!dp_query_func_) {
    return DistanceImpl{};
  }

  // The query is assumed to be already quantized — copy it directly.
  std::string quantized_query(static_cast<const char *>(query),
                              qmeta.element_size());
  return DistanceImpl(dp_query_func_, dp_query_batch_func_,
                      std::move(quantized_query), original_dim_);
}

void Fp16Quantizer::quantize_one(const void *input, void *output) const {
  uint16_t *out_buf = reinterpret_cast<uint16_t *>(output);

  if (meta_.metric_name() == "Cosine") {
    // L2-normalize before converting and store the norm at the end.
    std::vector<float> buf(original_dim_);
    std::memcpy(buf.data(), input, original_dim_ * sizeof(float));
    float norm = 0.0f;
    ailego::Normalizer<float>::L2(buf.data(), original_dim_, &norm);
    ailego::FloatHelper::ToFP16(buf.data(), original_dim_, out_buf);
    std::memcpy(reinterpret_cast<uint8_t *>(output) +
                    static_cast<size_t>(original_dim_) * sizeof(uint16_t),
                &norm, extra_meta_size_);
  } else {
    ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(input),
                                original_dim_, out_buf);
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
