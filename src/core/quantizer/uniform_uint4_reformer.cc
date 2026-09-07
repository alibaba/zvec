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
#include <cstdint>
#include <cstring>
#include <vector>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace core {
namespace {

constexpr float kAlmostHalf = 0.4999999701976776123046875f;

size_t EncodedDimension(size_t dimension) {
  return ((dimension + 127U) / 128U * 128U) / 2U;
}

void QuantizeScalar(const float *input, size_t dimension, float minimum,
                    float range, uint8_t *output, size_t encoded_dimension) {
  std::memset(output, 0, encoded_dimension);
  for (size_t d = 0; d < dimension; ++d) {
    float normalized = (input[d] - minimum) / range;
    normalized = std::min(1.0f, std::max(0.0f, normalized));
    const auto code = static_cast<uint8_t>(
        static_cast<int>(normalized * 15.0f + kAlmostHalf));
    if ((d & 1U) == 0) {
      output[d >> 1U] = code;
    } else {
      output[d >> 1U] |= static_cast<uint8_t>(code << 4U);
    }
  }
}

}  // namespace

class UniformUint4Reformer : public IndexReformer {
 public:
  UniformUint4Reformer(IndexMeta::DataType /*dst_type*/) {}
  ~UniformUint4Reformer() override = default;

  int init(const ailego::Params &params) override {
    uint32_t original_dimension = 0;
    const bool has_minimum =
        params.get(UNIFORM_UINT4_REFORMER_MINIMUM, &minimum_);
    const bool has_range = params.get(UNIFORM_UINT4_REFORMER_RANGE, &range_);
    const bool has_dimension = params.get(
        UNIFORM_UINT4_REFORMER_ORIGINAL_DIMENSION, &original_dimension);
    if (!has_minimum || !has_range || !has_dimension ||
        !std::isfinite(minimum_) || !std::isfinite(range_) ||
        !(range_ > 0.0f) || original_dimension == 0 ||
        original_dimension > MAX_DIMENSION) {
      LOG_ERROR("UniformUint4Reformer: invalid or missing params");
      return IndexError_InvalidArgument;
    }
    original_dimension_ = original_dimension;
    encoded_dimension_ = EncodedDimension(original_dimension_);
    const float step = range_ / 15.0f;
    distance_scale_ = step * step;
    quantize_func_ =
        turbo::get_uniform_uint4_quantize_func(turbo::DataType::kUint4);
    initialized_ = true;
    return 0;
  }

  int cleanup(void) override {
    initialized_ = false;
    return 0;
  }
  int load(IndexStorage::Pointer) override {
    return 0;
  }
  int unload(void) override {
    return 0;
  }

  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    return Quantize(query, qmeta, 1, out, ometa);
  }
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    return Quantize(query, qmeta, count, out, ometa);
  }
  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    return Quantize(record, rmeta, 1, out, ometa);
  }
  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    return Quantize(records, rmeta, count, out, ometa);
  }

  int normalize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                IndexDocumentList &result) const override {
    if (!initialized_) return IndexError_Runtime;
    for (auto &item : result) {
      *item.mutable_score() *= distance_scale_;
    }
    return 0;
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const void *input, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    if (!initialized_) return IndexError_Runtime;
    if (qmeta.data_type() != IndexMeta::DataType::DT_INT8 ||
        qmeta.dimension() != encoded_dimension_) {
      return IndexError_Mismatch;
    }
    out->resize(original_dimension_ * sizeof(float));
    auto *decoded = reinterpret_cast<float *>(out->data());
    const auto *packed = static_cast<const uint8_t *>(input);
    const float step = range_ / 15.0f;
    for (size_t d = 0; d < original_dimension_; ++d) {
      const uint8_t byte = packed[d >> 1U];
      const uint8_t code = (d & 1U) == 0 ? byte & 0x0fU : (byte >> 4U) & 0x0fU;
      decoded[d] = minimum_ + static_cast<float>(code) * step;
    }
    return 0;
  }

 private:
  int Quantize(const void *source, const IndexQueryMeta &source_meta,
               uint32_t count, std::string *out,
               IndexQueryMeta *output_meta) const {
    if (!initialized_) return IndexError_Runtime;
    const auto source_type = source_meta.data_type();
    const bool is_fp32 = source_type == IndexMeta::DataType::DT_FP32;
    const bool is_fp16 = source_type == IndexMeta::DataType::DT_FP16;
    if (!source || !out || !output_meta || (!is_fp32 && !is_fp16) ||
        source_meta.dimension() != original_dimension_) {
      return IndexError_Mismatch;
    }

    *output_meta = source_meta;
    output_meta->set_meta(IndexMeta::DataType::DT_INT8, encoded_dimension_);
    const size_t output_stride = output_meta->element_size();
    out->resize(static_cast<size_t>(count) * output_stride);
    auto *output = reinterpret_cast<uint8_t *>(out->data());
    const auto *source_bytes = static_cast<const uint8_t *>(source);
    static thread_local std::vector<float> decoded;
    if (!is_fp32) decoded.resize(original_dimension_);
    for (uint32_t i = 0; i < count; ++i) {
      const void *source_row =
          source_bytes + static_cast<size_t>(i) * source_meta.element_size();
      const float *row = nullptr;
      if (is_fp32) {
        row = static_cast<const float *>(source_row);
      } else {
        // fp16
        const auto *input = static_cast<const ailego::Float16 *>(source_row);
        for (size_t d = 0; d < original_dimension_; ++d) {
          decoded[d] = static_cast<float>(input[d]);
        }
        row = decoded.data();
      }
      for (size_t d = 0; d < original_dimension_; ++d) {
        if (!std::isfinite(row[d])) {
          LOG_ERROR("UniformUint4Reformer: non-finite input value");
          return IndexError_InvalidArgument;
        }
      }
      uint8_t *encoded = output + static_cast<size_t>(i) * output_stride;
      if (quantize_func_) {
        quantize_func_(row, original_dimension_, minimum_, range_, encoded);
      } else {
        QuantizeScalar(row, original_dimension_, minimum_, range_, encoded,
                       encoded_dimension_);
      }
    }
    return 0;
  }

  float minimum_{0.0f};
  float range_{0.0f};
  float distance_scale_{1.0f};
  size_t original_dimension_{0};
  size_t encoded_dimension_{0};
  bool initialized_{false};
  turbo::UniformUint4QuantizeFunc quantize_func_{nullptr};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(UniformUint4Reformer,
                                      UniformUint4Reformer,
                                      IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
