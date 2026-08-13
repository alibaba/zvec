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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace core {

/*! Reformer for Uniform Uint7 Quantization (Global Scale)
 *
 * Uses a global scale/bias (computed by UniformUint7Converter) to
 * quantize query vectors and build-time record vectors to int8.
 * No per-vector extras are appended — the output is pure int8.
 */
class UniformUint7Reformer : public IndexReformer {
 public:
  //! Constructor.
  //! `dst_type` is required by the INDEX_FACTORY_REGISTER_REFORMER_ALIAS
  //! macro signature but is unused here: the quantization output is
  //! always int8, governed by the (scale, bias) pair received in init().
  UniformUint7Reformer(IndexMeta::DataType /*dst_type*/) {}

  //! Initialize Reformer
  //!
  //! Scale/bias are supplied through reformer params after converter training.
  int init(const ailego::Params &params) override {
    Reset();

    float scale = 0.0f;
    float bias = 0.0f;
    const bool has_scale = params.get(UNIFORM_UINT7_REFORMER_SCALE, &scale);
    const bool has_bias = params.get(UNIFORM_UINT7_REFORMER_BIAS, &bias);
    if (!has_scale || !has_bias) {
      LOG_ERROR(
          "UniformUint7Reformer: missing scale/bias params "
          "(scale_present=%d, bias_present=%d)",
          static_cast<int>(has_scale), static_cast<int>(has_bias));
      return IndexError_InvalidArgument;
    }
    if (!std::isfinite(scale) || scale <= 0.0f || !std::isfinite(bias)) {
      LOG_ERROR("UniformUint7Reformer: invalid params scale=%f, bias=%f", scale,
                bias);
      return IndexError_InvalidArgument;
    }

    scale_ = scale;
    bias_ = bias;
    scale_reciprocal_sq_ = 1.0f / (scale_ * scale_);
    initialized_ = true;
    quantize_func_ = turbo::get_uniform_quantize_func(turbo::DataType::kInt8);

    LOG_INFO("UniformUint7Reformer init: scale=%f, bias=%f, simd=%s", scale_,
             bias_, quantize_func_ != nullptr ? "avx512" : "scalar");
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    Reset();
    return 0;
  }

  //! Load index from container
  int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  int unload(void) override {
    return 0;
  }

  //! Transform a single query: float → int8
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(query, qmeta, 1, out, ometa);
  }

  //! Transform batch queries: float → int8
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(query, qmeta, count, out, ometa);
  }

  //! Convert a single record: float → int8 (used during build)
  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    return do_quantize(record, rmeta, 1, out, ometa);
  }

  //! Convert batch records: float → int8
  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(records, rmeta, count, out, ometa);
  }

  //! Normalize results: convert int8 L2 distances back to float L2 distances
  int normalize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                IndexDocumentList &result) const override {
    if (!initialized_) {
      LOG_ERROR(
          "UniformUint7Reformer::normalize called before init "
          "with valid params");
      return IndexError_Runtime;
    }
    for (auto &it : result) {
      *it.mutable_score() *= scale_reciprocal_sq_;
    }
    return 0;
  }

  //! Support revert (int8 → float)
  bool need_revert() const override {
    return true;
  }

  //! Revert: convert int8 vector back to float
  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    if (!initialized_) {
      LOG_ERROR(
          "UniformUint7Reformer::revert called before init "
          "with valid params");
      return IndexError_Runtime;
    }
    if (!in || !out || qmeta.data_type() != IndexMeta::DataType::DT_INT8 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_INT8)) {
      return IndexError_InvalidArgument;
    }
    const size_t dim = qmeta.dimension();
    if (dim == 0 || dim > MAX_DIMENSION) {
      LOG_ERROR("UniformUint7Reformer: dimension=%zu must be in [1, %d]", dim,
                MAX_DIMENSION);
      return IndexError_InvalidArgument;
    }
    out->resize(dim * sizeof(float));
    char *out_buf = out->data();
    const auto *buf = static_cast<const int8_t *>(in);

    // Approximate dequantization (lossy):
    //   forward:
    //     int8 = clip(round-to-nearest-even(float * scale + bias), 0, 127)
    //   inverse:  float ≈ (int8 - bias) / scale
    // initialized_ guarantees scale_ != 0 and finite.
    float inv_scale = 1.0f / scale_;
    for (size_t i = 0; i < dim; ++i) {
      const float value = (static_cast<float>(buf[i]) - bias_) * inv_scale;
      std::memcpy(out_buf + i * sizeof(value), &value, sizeof(value));
    }

    return 0;
  }

 private:
  void Reset() {
    scale_ = 0.0f;
    bias_ = 0.0f;
    scale_reciprocal_sq_ = 1.0f;
    initialized_ = false;
    quantize_func_ = nullptr;
  }

  //! Common quantization path shared by transform()/convert() (single & batch)
  int do_quantize(const void *src, const IndexQueryMeta &smeta, uint32_t count,
                  std::string *out, IndexQueryMeta *ometa) const {
    if (!initialized_) {
      LOG_ERROR(
          "UniformUint7Reformer: quantize called before init "
          "with valid params");
      return IndexError_Runtime;
    }
    if (!src || !out || !ometa || count == 0) {
      return IndexError_InvalidArgument;
    }
    if (smeta.data_type() != IndexMeta::DataType::DT_FP32 ||
        smeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    const size_t dim = smeta.dimension();
    if (dim == 0 || dim > MAX_DIMENSION) {
      LOG_ERROR("UniformUint7Reformer: dimension=%zu must be in [1, %d]", dim,
                MAX_DIMENSION);
      return IndexError_InvalidArgument;
    }
    *ometa = smeta;
    ometa->set_meta(IndexMeta::DataType::DT_INT8, dim);
    const size_t out_stride = ometa->element_size();
    out->resize(static_cast<size_t>(count) * out_stride);

    const auto *vec = static_cast<const float *>(src);
    auto *ovec = reinterpret_cast<int8_t *>(out->data());
    for (uint32_t i = 0; i < count; ++i) {
      quantize(vec + i * dim, dim, ovec + i * out_stride);
    }
    return 0;
  }

  //! Quantize float vector to int8 using global scale/bias.
  //! Output values are in [0, 127] to enable the VNNI abs trick.
  //! Uses the SIMD kernel resolved in init() when available, otherwise
  //! falls back to the scalar reference implementation.
  inline void quantize(const float *in, size_t dim, int8_t *out) const {
    if (quantize_func_ != nullptr) {
      quantize_func_(in, dim, scale_, bias_, out);
      return;
    }
    for (size_t i = 0; i < dim; ++i) {
      // Match _mm512_cvtps_epi32 used by the SIMD kernel under the default
      // floating-point rounding mode.
      float v = std::nearbyint(in[i] * scale_ + bias_);
      v = std::max(0.0f, std::min(127.0f, v));
      out[i] = static_cast<int8_t>(v);
    }
  }

  //! Members
  float scale_{0.0f};
  float bias_{0.0f};
  float scale_reciprocal_sq_{1.0f};
  bool initialized_{false};
  turbo::UniformQuantizeFunc quantize_func_{nullptr};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(UniformUint7Reformer,
                                      UniformUint7Reformer,
                                      IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
