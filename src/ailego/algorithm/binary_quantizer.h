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

#include <vector>
#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

/*! Binary Quantization Algorithm
 */
class BinaryQuantizer {
 public:
  //! Constructor
  BinaryQuantizer(void) {}

  //! Feed the training data
  bool feed(const float *vec, size_t dim);

  //! Train the quantizer
  bool train(void);

  //! Quantize data: encode the float input to uint32_t output
  void encode(const float *in, size_t dim, uint32_t *out) const;

  //! De-quantize data: decode the input uint32_t to float output
  void decode(const uint32_t *in, size_t dim, float *out) const;

  //! Get encoded elements in type of uint32_t
  static size_t EncodedSizeInBinary32(size_t dim) {
    return (dim + 31) / 32;
  }

  //! Set quantization threshold
  void set_threshold(float threshold) {
    threshold_ = threshold;
  }

  //! Get quantization threshold
  float threshold(void) const {
    return threshold_;
  }

 private:
  //! Disable them
  BinaryQuantizer(const BinaryQuantizer &) = delete;
  BinaryQuantizer &operator=(const BinaryQuantizer &) = delete;

 private:
  //! Members
  std::vector<float> data_{};
  float threshold_{0.0f};
};

}  // namespace ailego
}  // namespace zvec