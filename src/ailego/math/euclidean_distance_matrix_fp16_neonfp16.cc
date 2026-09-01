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

// Native ARMv8.2 FP16 (FEAT_FP16) variant of the FP16 squared-euclidean
// kernel. This TU is compiled with an FP16-capable -march (e.g.
// -march=armv8.2-a+fp16, see src/ailego/CMakeLists.txt), which defines
// __ARM_FEATURE_FP16_VECTOR_ARITHMETIC so the ACCUM_FP16 macros expand to
// the half-precision arithmetic path. Callers must runtime-gate invocation
// on CpuFeatures FP16.

#include "distance_matrix_accum_fp16.i"
#include "distance_matrix_euclidean_utility.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
float SquaredEuclideanDistanceFp16NEONFP16(const Float16 *lhs,
                                           const Float16 *rhs, size_t size) {
  float score{0.0f};

  ACCUM_FP16_1X1_NEON(lhs, rhs, size, &score, 0ull, )

  return score;
}
#endif

}  // namespace ailego
}  // namespace zvec
