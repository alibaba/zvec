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

#include "binary_quantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <ailego/math/normalizer.h>

namespace zvec {
namespace ailego {

//! Feed the training data
bool BinaryQuantizer::feed(const float *vec, size_t dim) {
  for (size_t i = 0; i < dim; ++i) {
    data_.emplace_back(vec[i]);
  }
  return true;
}

//! Train the quantizer
bool BinaryQuantizer::train() {
  return true;
}

//! Quantize data: encode the float input to uint32_t output
void BinaryQuantizer::encode(const float *in, size_t dim, uint32_t *out) const {
  for (size_t i = 0; i < dim; i += 32) {
    size_t remain = i + 32 <= dim ? 32 : dim - i;
    uint32_t data = 0;
    uint32_t mask = 1;

    for (size_t j = 0; j < remain; j++) {
      if (in[i + j] >= threshold_) {
        data |= mask;
      }

      mask <<= 1;
    }

    *out = data;
    out++;
  }
}

//! De-quantize data: decode the input uint32_t to float output
//!   bit value 1 will be mapped to 1.0
//!   bit value 0 will be mapped to -1.0
void BinaryQuantizer::decode(const uint32_t *in, size_t dim, float *out) const {
  for (size_t i = 0; i < dim; ++i) {
    uint8_t bit = (in[i >> 5] >> (i & 31)) & 0x01;

    if (bit == 1) {
      out[i] = 1.0f;
    } else {
      out[i] = -1.0f;
    }

    // std::cout << "dim: " << i << ", value: " << (size_t)bit << std::endl;
  }
}

}  // namespace ailego
}  // namespace zvec