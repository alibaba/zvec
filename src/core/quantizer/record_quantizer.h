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
#include <cstring>
#include <zvec/core/framework/index_meta.h>

#pragma once

namespace zvec {
namespace core {

class RecordQuantizer {
 public:
  //! Convert the float feature to int8 or int4 feature
  static inline void quantize_record(const float *vec, size_t dim,
                                     IndexMeta::DataType type,
                                     bool is_euclidean, void *out) {
    if (type == IndexMeta::DataType::DT_FP16) {
      ailego::FloatHelper::ToFP16(vec, dim, reinterpret_cast<uint16_t *>(out));
    } else if (type == IndexMeta::DataType::DT_INT4 ||
               type == IndexMeta::DataType::DT_INT8) {
      float min = std::numeric_limits<float>::max();
      float max = std::numeric_limits<float>::lowest();
      constexpr float epsilon = std::numeric_limits<float>::epsilon();
      for (size_t i = 0; i < dim; ++i) {
        min = std::min(min, vec[i]);
        max = std::max(max, vec[i]);
      }

      float sum = 0.0f;
      float squared_sum = 0.0f;
      int int8_sum = 0;
      uint8_t *extras;
      float scale, bias;
      if (type == IndexMeta::DataType::DT_INT8) {
        scale = 254 / std::max(max - min, epsilon);
        bias = -min * scale - 127;
        for (size_t i = 0; i < dim; ++i) {
          float v = std::round(vec[i] * scale + bias);
          squared_sum += v * v;
          sum += v;
          (reinterpret_cast<int8_t *>(out))[i] = static_cast<int8_t>(v);
          int8_sum += (reinterpret_cast<int8_t *>(out))[i];
        }
        extras = static_cast<uint8_t *>(out) + dim;
      } else {
        scale = 15 / std::max(max - min, epsilon);
        bias = -min * scale - 8;
        // Accumulate the rounded codes: the stored sum must match the packed
        // nibbles, otherwise QuantizedInteger scoring (which reconstructs
        // scores from sum) ranks with a per-record error.
        for (size_t i = 0; i < dim; i += 2) {
          float lo = std::round(vec[i] * scale + bias);
          float hi = std::round(vec[i + 1] * scale + bias);
          squared_sum += lo * lo;
          sum += lo;
          squared_sum += hi * hi;
          sum += hi;
          (reinterpret_cast<uint8_t *>(out))[i / 2] =
              (static_cast_from_float_to_uint8(hi) << 4) |
              (static_cast_from_float_to_uint8(lo) & 0xF);
        }
        extras = static_cast<uint8_t *>(out) + dim / 2;
      }

      // Save the feature quantization params for IndexMeasure
      // Metadata follows packed codes and need not be float-aligned.
      const float params[] = {1.0f / scale, -bias / scale, sum};
      std::memcpy(extras, params, sizeof(params));
      const int32_t code_sum = int8_sum;

      if (type == IndexMeta::DataType::DT_INT8) {
        std::memcpy(extras + 3 * sizeof(float), &squared_sum,
                    sizeof(squared_sum));
        std::memcpy(extras + 4 * sizeof(float), &code_sum, sizeof(code_sum));
      } else {
        if (is_euclidean) {
          std::memcpy(extras + 3 * sizeof(float), &squared_sum,
                      sizeof(squared_sum));
        } else {
          std::memcpy(extras + 3 * sizeof(float), &code_sum, sizeof(code_sum));
        }
      }
    }
  }

  static inline void unquantize_record(const void *vec, size_t origin_dim,
                                       IndexMeta::DataType type, float *out) {
    if (type == IndexMeta::DataType::DT_INT8) {
      float extras[2];
      std::memcpy(extras, static_cast<const uint8_t *>(vec) + origin_dim,
                  sizeof(extras));

      const int8_t *buf = reinterpret_cast<const int8_t *>(vec);
      for (size_t i = 0; i < origin_dim; ++i) {
        out[i] = buf[i] * extras[0] + extras[1];
      }

    } else if (type == IndexMeta::DataType::DT_INT4) {
      float extras[2];
      std::memcpy(extras, static_cast<const uint8_t *>(vec) + origin_dim / 2,
                  sizeof(extras));

      const uint8_t *buf = reinterpret_cast<const uint8_t *>(vec);

      for (size_t i = 0; i < origin_dim / 2; ++i) {
        int8_t lo = (static_cast<int8_t>(buf[i] << 4) >> 4);
        int8_t hi = (static_cast<int8_t>(buf[i] & 0xf0) >> 4);

        out[2 * i] = lo * extras[0] + extras[1];
        out[2 * i + 1] = hi * extras[0] + extras[1];
      }
    } else if (type == IndexMeta::DataType::DT_FP16) {
      const uint16_t *in_buf = reinterpret_cast<const uint16_t *>(vec);
      for (size_t i = 0; i < origin_dim; ++i) {
        out[i] = ailego::FloatHelper::ToFP32(in_buf[i]);
      }
    }
  }

  static inline void unquantize_sparse_record(const void *sparse_value,
                                              size_t sparse_count,
                                              IndexMeta::DataType type,
                                              float *sparse_value_out) {
    if (type == IndexMeta::DataType::DT_FP16) {
      const uint16_t *in_buf = reinterpret_cast<const uint16_t *>(sparse_value);
      for (size_t i = 0; i < sparse_count; ++i) {
        sparse_value_out[i] = ailego::FloatHelper::ToFP32(in_buf[i]);
      }
    }
  }
};

}  // namespace core
}  // namespace zvec
