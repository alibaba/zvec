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

#include "quantizer/int8_quantizer/int8_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <ailego/math/normalizer.h>
#include <zvec/core/framework/index_factory.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int Int8Quantizer::init(const IndexMeta &meta,
                        const ailego::Params & /*params*/) {
  meta_ = meta;

  meta_.set_meta(IndexMeta::DataType::DT_INT8, meta.dimension());

  original_dim_ = meta.dimension();
  auto metric_name = meta.metric_name();
  is_cosine_ = (metric_name == "Cosine");
  is_euclidean_ =
      (metric_name == "SquaredEuclidean" || metric_name == "Euclidean" ||
       metric_name == "MipsSquaredEuclidean");

  extra_meta_size_ = RECORD_TAIL_SIZE;
  if (is_cosine_) {
    // The raw kernels return -cos; offset to 1 - cos for a non-negative,
    // monotonically equivalent distance (same convention as Fp32Quantizer).
    distance_offset_ = 1.0f;
    extra_meta_size_ += EXTRA_META_SIZE_COSINE;
  }
  meta_.set_extra_meta_size(extra_meta_size_);

  // Distance kernels take the full encoded size in bytes.
  dist_dim_ = static_cast<size_t>(original_dim_) + extra_meta_size_;

  // Cache the distance dispatch for the new Quantizer interface.
  auto kernels =
      get_distance_kernels(metric_from_name(metric_name), DataType::kInt8,
                           QuantizeType::kRecord, CpuArchType::kAuto);
  if (!kernels.dist || !kernels.batch) {
    LOG_ERROR("Unsupported metric %s for INT8 quantizer", metric_name.c_str());
    return kErrUnsupported;
  }
  dp_query_func_ = std::move(kernels.dist);
  dp_query_batch_func_ = std::move(kernels.batch);
  dp_query_preprocess_func_ = kernels.preprocess;

  return 0;
}

void Int8Quantizer::quantize_one(const void *input, void *output) const {
  const float *vec = reinterpret_cast<const float *>(input);

  std::vector<float> normalized;
  float norm = 0.0f;
  if (is_cosine_) {
    // L2-normalize before quantization; the norm is stored after the record
    // tail so the original vector can be reconstructed during dequantize.
    normalized.assign(vec, vec + original_dim_);
    ailego::Normalizer<float>::L2(normalized.data(), original_dim_, &norm);
    vec = normalized.data();
  }

  RecordQuantizer::quantize_record(
      vec, original_dim_, IndexMeta::DataType::DT_INT8, is_euclidean_, output);

  if (is_cosine_) {
    std::memcpy(
        reinterpret_cast<uint8_t *>(output) + original_dim_ + RECORD_TAIL_SIZE,
        &norm, EXTRA_META_SIZE_COSINE);
  }
}

int Int8Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  if (qmeta.unit_size() != sizeof(float)) {
    return kErrUnsupported;
  }

  size_t raw_dim = (original_dim_ != 0 && qmeta.dimension() >= original_dim_)
                       ? original_dim_
                       : qmeta.dimension();
  if (raw_dim != original_dim_) {
    return kErrUnsupported;
  }

  out->resize(quantized_length());
  quantize_one(query, &(*out)[0]);

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_INT8, raw_dim,
                  static_cast<uint32_t>(type_), extra_meta_size_);

  return 0;
}

int Int8Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  size_t raw_dim = (original_dim_ != 0 && qmeta.dimension() >= original_dim_)
                       ? original_dim_
                       : qmeta.dimension();

  out->resize(raw_dim * sizeof(float));
  float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
  RecordQuantizer::unquantize_record(in, raw_dim, IndexMeta::DataType::DT_INT8,
                                     out_buf);

  if (is_cosine_) {
    // Denormalize the vector using the stored norm.
    float norm = 0.0f;
    std::memcpy(
        &norm,
        reinterpret_cast<const uint8_t *>(in) + raw_dim + RECORD_TAIL_SIZE,
        EXTRA_META_SIZE_COSINE);
    for (size_t i = 0; i < raw_dim; ++i) {
      out_buf[i] *= norm;
    }
  }
  return 0;
}

DistanceImpl Int8Quantizer::distance(const void *query,
                                     const IndexQueryMeta &qmeta) const {
  DistanceFunc func = dp_query_func_;
  if (!func) {
    return DistanceImpl{};
  }

  // Batch kernels that require a preprocessed (uint8-shifted) query cannot
  // share the stored query with the single-pair kernel; fall back to the
  // scalar path inside DistanceImpl in that case.
  BatchDistanceFunc batch_func;
  if (!dp_query_preprocess_func_) {
    batch_func = dp_query_batch_func_;
  }

  if (distance_offset_ != 0.0f) {
    float offset = distance_offset_;
    DistanceFunc base = std::move(func);
    func = [base, offset](const void *a, const void *b, size_t dim,
                          float *out) {
      base(a, b, dim, out);
      *out += offset;
    };
    if (batch_func) {
      BatchDistanceFunc batch_base = std::move(batch_func);
      batch_func = [batch_base, offset](const void **m, const void *q,
                                        size_t num, size_t dim, float *out) {
        batch_base(m, q, num, dim, out);
        for (size_t i = 0; i < num; ++i) {
          out[i] += offset;
        }
      };
    }
  }

  // The query is assumed to be already quantized — copy it directly.
  std::string quantized_query(static_cast<const char *>(query),
                              qmeta.element_size());
  return DistanceImpl(std::move(func), std::move(batch_func),
                      std::move(quantized_query), dist_dim_);
}

float Int8Quantizer::calc_distance_dp_query(const void *dp,
                                            const void *query) const {
  float d = 0.0f;
  if (dp_query_func_) {
    dp_query_func_(dp, query, dist_dim_, &d);
    d += distance_offset_;
  }
  return d;
}

void Int8Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                 int dp_num, const void *query,
                                                 float *dist_list) const {
  if (dp_query_batch_func_) {
    if (dp_query_preprocess_func_) {
      // The batch kernel expects the query shifted to uint8; preprocess a
      // private copy to keep the caller's buffer intact.
      std::string buf(static_cast<const char *>(query), quantized_length());
      dp_query_preprocess_func_(&buf[0], dist_dim_);
      dp_query_batch_func_(const_cast<const void **>(dp_list), buf.data(),
                           static_cast<size_t>(dp_num), dist_dim_, dist_list);
    } else {
      dp_query_batch_func_(const_cast<const void **>(dp_list), query,
                           static_cast<size_t>(dp_num), dist_dim_, dist_list);
    }
    if (distance_offset_ != 0.0f) {
      for (int i = 0; i < dp_num; ++i) {
        dist_list[i] += distance_offset_;
      }
    }
    return;
  }
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = calc_distance_dp_query(dp_list[i], query);
  }
}

float Int8Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void Int8Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float Int8Quantizer::calc_distance_dp_dp(const void *dp1,
                                         const void *dp2) const {
  return calc_distance_dp_query(dp1, dp2);
}

INDEX_FACTORY_REGISTER_QUANTIZER(Int8Quantizer);

}  // namespace turbo
}  // namespace zvec
