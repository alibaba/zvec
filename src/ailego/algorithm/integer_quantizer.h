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

#include <limits>
#include <vector>
#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

/*! Entropy-based Integer Quantization Algorithm
 */
template <typename T, int RANGE_MIN, int RANGE_MAX>
class EntropyIntegerQuantizer {
 public:
  //! Primitive Built-in Types to store the quantized data
  using ValueType = typename std::remove_cv<T>::type;

  //! Constants
  constexpr static int MIN_VALUE = RANGE_MIN;
  constexpr static int MAX_VALUE = RANGE_MAX;

  // Check supporting type
  static_assert(std::is_integral<T>::value, "ValueType must be integral");

  // Check template values
  static_assert(RANGE_MIN < RANGE_MAX, "Invalid value range");

  //! Constructor
  EntropyIntegerQuantizer(void) {}

  //! Set histogram bins in train
  void set_histogram_bins(size_t bins) {
    if (bins > (RANGE_MAX - RANGE_MIN)) {
      histogram_bins_ = bins;
    }
  }

  //! Set quantization params scale
  void set_scale(float val) {
    if (val > 0.0f) {
      scale_ = val;
      scale_reciprocal_ = 1 / scale_;
    }
  }

  //! Set quantization params bias
  void set_bias(float val) {
    bias_ = val;
  }

  //! Set quantization params max
  void set_max(float val) {
    max_ = val;
  }

  //! Set quantization params min
  void set_min(float val) {
    min_ = val;
  }

  //! Set quantization params non bias
  void set_non_bias(bool val) {
    non_bias_ = val;
  }

  //! Get histogram bins in train
  size_t histogram_bins(void) const {
    return histogram_bins_;
  }

  //! Get quantization params scale
  float scale(void) const {
    return scale_;
  }

  //! Get quantization params bias
  float bias(void) const {
    return bias_;
  }

  //! Get quantization params max
  float max(void) const {
    return max_;
  }

  //! Get quantization params min
  float min(void) const {
    return min_;
  }

  //! Get quantization params non bias
  bool non_bias(void) const {
    return non_bias_;
  }

  //! Retrieve the scale reciprocal for decoding
  float scale_reciprocal(void) const {
    return scale_reciprocal_;
  }

 protected:
  //! Disable them
  EntropyIntegerQuantizer(const EntropyIntegerQuantizer &) = delete;
  EntropyIntegerQuantizer &operator=(const EntropyIntegerQuantizer &) = delete;

  //! Members
  size_t histogram_bins_{0};
  float hist_interval_{1.0f};
  float max_{std::numeric_limits<float>::min()};
  float min_{std::numeric_limits<float>::max()};
  float bias_{0.0f};
  float scale_{0.0f};
  float scale_reciprocal_{0.0f};
  float left_boundary_{0.0f};
  bool non_bias_{false};
  std::vector<uint32_t> histogram_{};
};

/*! INT16 Quantizer
 */
class EntropyInt16Quantizer
    : public EntropyIntegerQuantizer<int16_t, -32767, 32767> {
 public:
  //! Feed the training data
  bool feed(const float *vec, size_t dim);

  //! Train the quantizer
  bool train(void);

  //! Encode float vector to int16
  void encode(const float *in, size_t dim, ValueType *out) const;

  //! Decode to float vector from int16
  void decode(const ValueType *in, size_t dim, float *out) const;
};

/*! UINT16 Quantizer
 */
class EntropyUInt16Quantizer
    : public EntropyIntegerQuantizer<uint16_t, 0, 65535> {
 public:
  //! Feed the training data
  bool feed(const float *vec, size_t dim);

  //! Train the quantizer
  bool train(void);

  //! Encode float vector to uint16
  void encode(const float *in, size_t dim, ValueType *out) const;

  //! Decode to float vector from uint16
  void decode(const ValueType *in, size_t dim, float *out) const;
};

/*! INT8 Quantizer
 */
class EntropyInt8Quantizer : public EntropyIntegerQuantizer<int8_t, -127, 127> {
 public:
  //! Feed the training data
  bool feed(const float *vec, size_t dim);

  //! Train the quantizer
  bool train(void);

  //! Encode float vector to int8
  void encode(const float *in, size_t dim, ValueType *out) const;

  //! Decode to float vector from int8
  void decode(const ValueType *in, size_t dim, float *out) const;
};

/*! UINT8 Quantizer
 */
class EntropyUInt8Quantizer : public EntropyIntegerQuantizer<uint8_t, 0, 255> {
 public:
  //! Feed the training data
  bool feed(const float *vec, size_t dim);

  //! Train the quantizer
  bool train(void);

  //! Encode float vector to uint8
  void encode(const float *in, size_t dim, ValueType *out) const;

  //! Decode to float vector from uint8
  void decode(const ValueType *in, size_t dim, float *out) const;
};

/*! INT4 Quantizer
 */
class EntropyInt4Quantizer : public EntropyIntegerQuantizer<uint8_t, -8, 7> {
 public:
  //! Feed the training data
  bool feed(const float *vec, size_t dim);

  //! Train the quantizer
  bool train(void);

  //! Encode float vector to int4
  void encode(const float *in, size_t dim, ValueType *out) const;

  //! Decode to float vector from int4
  void decode(const ValueType *in, size_t dim, float *out) const;
};

/*! UINT4 Quantizer
 */
class EntropyUInt4Quantizer : public EntropyIntegerQuantizer<uint8_t, 0, 15> {
 public:
  //! Feed the training data
  bool feed(const float *vec, size_t dim);

  //! Train the quantizer
  bool train(void);

  //! Encode float vector to uint4
  void encode(const float *in, size_t dim, ValueType *out) const;

  //! Decode to float vector from uint4
  void decode(const ValueType *in, size_t dim, float *out) const;
};

}  // namespace ailego
}  // namespace zvec