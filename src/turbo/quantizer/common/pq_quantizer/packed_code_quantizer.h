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

#include <cstddef>

namespace zvec {
namespace turbo {

//! Optional capability: a quantizer whose codes must be stored in packed
//! 32-vector blocks (e.g. FastScan, where one SIMD byte shuffle looks up
//! 32 codes per sub-space).  Orthogonal to the Quantizer base contract,
//! so storage layers discover it via dynamic_cast: a successful cast both
//! requires packing and provides the packer.  Storage layers that cannot
//! honor packing must not use such a quantizer.
//!
//! The write side (pack_codes) and the read side (calc_distance_packed_block)
//! share the same block layout contract; generic gather-style quantizers
//! keep using Quantizer::calc_distance_dp_query_batch.
class PackedCodeQuantizer {
 public:
  virtual ~PackedCodeQuantizer() = default;

  //! Pack up to 32 plain codes (laid out `stride` bytes apart) into one
  //! packed block consumable by calc_distance_packed_block.  Slots beyond
  //! `num` are zero-filled; `out` must hold one full packed block.
  virtual int pack_codes(const void *codes, size_t num, size_t stride,
                         void *out) const = 0;

  //! Scan one or several back-to-back packed blocks (produced by
  //! pack_codes) against a quantized query, writing `num` distances.
  virtual void calc_distance_packed_block(const void *block, size_t num,
                                          const void *query,
                                          float *dist_list) const = 0;
};

}  // namespace turbo
}  // namespace zvec
