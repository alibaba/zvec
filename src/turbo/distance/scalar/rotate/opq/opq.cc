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

#include "opq.h"

namespace zvec::turbo::scalar {

void opq_rotate(const float *in, float *out, size_t in_dim, size_t /*out_dim*/,
                void *ctx) {
  // ctx is the dim x dim row-major rotation matrix; out = R * in.
  const float *matrix = reinterpret_cast<const float *>(ctx);
  for (size_t r = 0; r < in_dim; ++r) {
    const float *row = matrix + r * in_dim;
    float acc = 0.0f;
    for (size_t c = 0; c < in_dim; ++c) {
      acc += row[c] * in[c];
    }
    out[r] = acc;
  }
}

void opq_unrotate(const float *in, float *out, size_t in_dim,
                  size_t /*out_dim*/, void *ctx) {
  // out = R^T * in via outer-product accumulation, keeping all accesses
  // sequential.
  const float *matrix = reinterpret_cast<const float *>(ctx);
  for (size_t r = 0; r < in_dim; ++r) {
    out[r] = 0.0f;
  }
  for (size_t c = 0; c < in_dim; ++c) {
    const float xc = in[c];
    const float *row = matrix + c * in_dim;
    for (size_t r = 0; r < in_dim; ++r) {
      out[r] += row[r] * xc;
    }
  }
}

}  // namespace zvec::turbo::scalar
