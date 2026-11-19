// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>

namespace zvec::turbo {

// x86 converts interleaved FP16 matrices in tiles. Both single and joint
// inner products must add the same partial sums at the same boundaries.
inline constexpr size_t kFp16MatrixTile = 64;

template <size_t Rows, bool Pair, typename T, typename Kernel>
inline void fp16_ip_tiles(Kernel kernel, const T *matrix, const T *q0,
                          const T *q1, size_t dim, float *out0, float *out1) {
  kernel(matrix, q0, q1, std::min(dim, kFp16MatrixTile), out0, out1);
  for (size_t d = kFp16MatrixTile; d < dim; d += kFp16MatrixTile) {
    float a[Rows], b[Rows];
    kernel(matrix + d * Rows, q0 + d, Pair ? q1 + d : nullptr,
           std::min(dim - d, kFp16MatrixTile), a, b);
    for (size_t i = 0; i < Rows; ++i) {
      out0[i] += a[i];
      if constexpr (Pair) out1[i] += b[i];
    }
  }
}

}  // namespace zvec::turbo
