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
#include <limits>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>

namespace zvec {
namespace core {

class UniformUint8Reformer : public IndexReformer {
 public:
  explicit UniformUint8Reformer(IndexMeta::DataType /*destination_type*/) {}

  int init(const ailego::Params &params) override {
    Reset();

    float scale = 0.0f;
    float bias = 0.0f;
    const bool has_scale = params.get(UNIFORM_UINT8_REFORMER_SCALE, &scale);
    const bool has_bias = params.get(UNIFORM_UINT8_REFORMER_BIAS, &bias);
    if (!has_scale || !has_bias) {
      LOG_ERROR(
          "UniformUint8Reformer: missing scale/bias params "
          "(scale_present=%d, bias_present=%d)",
          static_cast<int>(has_scale), static_cast<int>(has_bias));
      return IndexError_InvalidArgument;
    }
    return SetParams(scale, bias);
  }

  int cleanup(void) override {
    Reset();
    return 0;
  }

  int load(IndexStorage::Pointer) override {
    return 0;
  }

  int unload(void) override {
    return 0;
  }

  int transform(const void *query, const IndexQueryMeta &query_meta,
                std::string *output,
                IndexQueryMeta *output_meta) const override {
    return Encode(query, query_meta, 1, output, output_meta);
  }

  int transform(const void *queries, const IndexQueryMeta &query_meta,
                uint32_t count, std::string *output,
                IndexQueryMeta *output_meta) const override {
    return Encode(queries, query_meta, count, output, output_meta);
  }

  int convert(const void *record, const IndexQueryMeta &record_meta,
              std::string *output, IndexQueryMeta *output_meta) const override {
    return Encode(record, record_meta, 1, output, output_meta);
  }

  int convert(const void *records, const IndexQueryMeta &record_meta,
              uint32_t count, std::string *output,
              IndexQueryMeta *output_meta) const override {
    return Encode(records, record_meta, count, output, output_meta);
  }

  int normalize(const void * /*query*/, const IndexQueryMeta & /*query_meta*/,
                IndexDocumentList &result) const override {
    if (!initialized_) {
      return IndexError_Runtime;
    }
    for (auto &document : result) {
      *document.mutable_score() *= scale_reciprocal_squared_;
    }
    return 0;
  }

  bool need_revert(void) const override {
    return true;
  }

  // Both transform() and convert() emit the canonical shifted layout.
  int revert(const void *input, const IndexQueryMeta &record_meta,
             std::string *output) const override {
    if (!initialized_ || !input || !output ||
        record_meta.data_type() != IndexMeta::DataType::DT_INT8 ||
        record_meta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_INT8)) {
      return IndexError_InvalidArgument;
    }

    if (record_meta.dimension() <= kTailBytes) {
      return IndexError_InvalidArgument;
    }
    const size_t dimension = record_meta.dimension() - kTailBytes;
    if (dimension > MAX_DIMENSION) {
      LOG_ERROR("UniformUint8Reformer: dimension=%zu must be in [1, %d]",
                dimension, MAX_DIMENSION);
      return IndexError_InvalidArgument;
    }
    output->resize(dimension * sizeof(float));
    auto *destination = output->data();
    const auto *source = static_cast<const int8_t *>(input);
    const float inverse_scale = 1.0f / scale_;
    for (size_t i = 0; i < dimension; ++i) {
      const float raw_code = static_cast<float>(source[i]) + 128.0f;
      const float value = (raw_code - bias_) * inverse_scale;
      std::memcpy(destination + i * sizeof(value), &value, sizeof(value));
    }
    return 0;
  }

 private:
  void Reset() {
    initialized_ = false;
    scale_ = 0.0f;
    bias_ = 0.0f;
    scale_reciprocal_squared_ = 1.0f;
  }

  int SetParams(float scale, float bias) {
    if (!std::isfinite(scale) || scale <= 0.0f || !std::isfinite(bias)) {
      LOG_ERROR("UniformUint8Reformer: invalid params scale=%f bias=%f", scale,
                bias);
      initialized_ = false;
      return IndexError_InvalidArgument;
    }
    scale_ = scale;
    bias_ = bias;
    scale_reciprocal_squared_ = 1.0f / (scale_ * scale_);
    initialized_ = true;
    return 0;
  }

  int Encode(const void *input, const IndexQueryMeta &input_meta,
             uint32_t count, std::string *output,
             IndexQueryMeta *output_meta) const {
    if (!initialized_ || !input || !output || !output_meta || count == 0) {
      return IndexError_InvalidArgument;
    }
    if (input_meta.data_type() != IndexMeta::DataType::DT_FP32 ||
        input_meta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    const size_t dimension = input_meta.dimension();
    if (dimension == 0 || dimension > MAX_DIMENSION) {
      LOG_ERROR("UniformUint8Reformer: dimension=%zu must be in [1, %d]",
                dimension, MAX_DIMENSION);
      return IndexError_InvalidArgument;
    }
    const size_t encoded_dimension = dimension + kTailBytes;
    *output_meta = input_meta;
    output_meta->set_meta(IndexMeta::DataType::DT_INT8, encoded_dimension);
    const size_t output_stride = output_meta->element_size();
    output->resize(static_cast<size_t>(count) * output_stride);

    const auto *source = static_cast<const float *>(input);
    auto *destination = reinterpret_cast<int8_t *>(output->data());
    for (uint32_t i = 0; i < count; ++i) {
      EncodeOne(source + static_cast<size_t>(i) * dimension, dimension,
                destination + static_cast<size_t>(i) * output_stride);
    }
    return 0;
  }

  void EncodeOne(const float *input, size_t dimension, int8_t *output) const {
    int64_t sum_squared = 0;
    for (size_t i = 0; i < dimension; ++i) {
      float value = std::round(input[i] * scale_ + bias_);
      value = std::max(0.0f, std::min(255.0f, value));
      const int code = static_cast<int>(value);
      output[i] = static_cast<int8_t>(code - 128);
      sum_squared += code * code;
    }
    const uint32_t tail = static_cast<uint32_t>(sum_squared);
    std::memcpy(output + dimension, &tail, sizeof(tail));
  }

  // Keep the encoded-tail and exact-query assumptions local to this
  // independent encoding entry point.
  static_assert(uint64_t{MAX_DIMENSION} * 255 * 255 <=
                    (std::numeric_limits<uint32_t>::max)(),
                "Uniform UINT8 norm and exact L2 must fit uint32_t");
  static_assert(uint64_t{MAX_DIMENSION} * 255 * 128 <=
                    (std::numeric_limits<int32_t>::max)(),
                "Uniform UINT8 VNNI dot product must fit int32_t");
  static_assert(uint64_t{MAX_DIMENSION} * 128 * 128 <=
                    (std::numeric_limits<int32_t>::max)(),
                "Uniform UINT8 query correction must fit int32_t");
  static constexpr size_t kTailBytes = sizeof(uint32_t);
  float scale_{0.0f};
  float bias_{0.0f};
  float scale_reciprocal_squared_{1.0f};
  bool initialized_{false};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(UniformUint8Reformer,
                                      UniformUint8Reformer,
                                      IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
