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

#include "bitpacked_simd_dispatch.h"
#include <ailego/internal/cpu_features.h>
#include "bitpacked_simd_scalar.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#include "bitpacked_simd_avx2.h"
#include "bitpacked_simd_sse41.h"
#endif

namespace zvec::fts::simd {

static DispatchTable init_dispatch() {
  DispatchTable t{};
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    t.max_128 = avx2_max_128;
    t.pack_uint32_128 = avx2_pack_uint32_128;
    t.unpack_uint32_128 = avx2_unpack_uint32_128;
    t.prefix_sum_128 = avx2_prefix_sum_128;
    t.find_first_ge = avx2_find_first_ge;
    return t;
  }
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    t.max_128 = sse41_max_128;
    t.pack_uint32_128 = sse41_pack_uint32_128;
    t.unpack_uint32_128 = sse41_unpack_uint32_128;
    t.prefix_sum_128 = sse41_prefix_sum_128;
    t.find_first_ge = sse41_find_first_ge;
    return t;
  }
#endif
  t.max_128 = scalar_max_128;
  t.pack_uint32_128 = scalar_pack_uint32_128;
  t.unpack_uint32_128 = scalar_unpack_uint32_128;
  t.prefix_sum_128 = scalar_prefix_sum_128;
  t.find_first_ge = scalar_find_first_ge;
  return t;
}

const DispatchTable &get_dispatch() {
  static const DispatchTable table = init_dispatch();
  return table;
}

}  // namespace zvec::fts::simd
