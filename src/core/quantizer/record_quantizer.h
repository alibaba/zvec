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
      float *extras, scale, bias;
      if (type == IndexMeta::DataType::DT_INT8) {
        scale = 254 / std::max(max - min, epsilon);
        bias = -min * scale - 127;
        for (size_t i = 0; i < dim; ++i) {
          float v = vec[i] * scale + bias;
          squared_sum += v * v;
          sum += v;
          (reinterpret_cast<int8_t *>(out))[i] =
              static_cast<int8_t>(std::round(v));
          int8_sum += (reinterpret_cast<int8_t *>(out))[i];
        }
        extras = reinterpret_cast<float *>(static_cast<int8_t *>(out) + dim);
      } else {
        scale = 15 / std::max(max - min, epsilon);
        bias = -min * scale - 8;
        for (size_t i = 0; i < dim; i += 2) {
          float lo = vec[i] * scale + bias;
          float hi = vec[i + 1] * scale + bias;
          squared_sum += lo * lo;
          sum += lo;
          squared_sum += hi * hi;
          sum += hi;
          (reinterpret_cast<uint8_t *>(out))[i / 2] =
              (static_cast_from_float_to_uint8(std::round(hi)) << 4) |
              (static_cast_from_float_to_uint8(std::round(lo)) & 0xF);
        }
        extras =
            reinterpret_cast<float *>(static_cast<uint8_t *>(out) + dim / 2);
      }

      // Save the feature quantization params for IndexMeasure
      extras[0] = 1.0f / scale;
      extras[1] = -bias / scale;
      extras[2] = sum;
      if (is_euclidean) {
        extras[3] = squared_sum;
      } else {
        reinterpret_cast<int *>(extras)[3] = int8_sum;
      }
    }
  }

  static inline void unquantize_record(const void *vec, size_t origin_dim,
                                       IndexMeta::DataType type, float *out) {
    if (type == IndexMeta::DataType::DT_INT8) {
      const float *extras = reinterpret_cast<const float *>(
          static_cast<const int8_t *>(vec) + origin_dim);

      const int8_t *buf = reinterpret_cast<const int8_t *>(vec);
      for (size_t i = 0; i < origin_dim; ++i) {
        out[i] = buf[i] * extras[0] + extras[1];
      }

    } else if (type == IndexMeta::DataType::DT_INT4) {
      const float *extras = reinterpret_cast<const float *>(
          static_cast<const uint8_t *>(vec) + origin_dim / 2);

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