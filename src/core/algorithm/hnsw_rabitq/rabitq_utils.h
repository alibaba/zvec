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

#include <memory>
#include <vector>
#include <rabitqlib/utils/rotator.hpp>
#include "zvec/core/framework/index_dumper.h"

namespace zvec {
namespace core {

inline const std::string RABITQ_CONVERTER_SEG_ID{"rabitq.converter"};

struct RabitqConverterHeader {
  uint32_t num_clusters;
  uint32_t dim;
  uint32_t padded_dim;
  uint32_t rotator_size;
  uint8_t ex_bits;
  uint8_t rotator_type;
  uint8_t padding[2];
  uint32_t reserve[3];

  RabitqConverterHeader() {
    memset(this, 0, sizeof(RabitqConverterHeader));
  }
};
static_assert(sizeof(RabitqConverterHeader) % 32 == 0,
              "RabitqConverterHeader must be aligned with 32 bytes");

// Common dump implementation for RabitqConverter and RabitqReformer
int dump_rabitq_centroids(
    const IndexDumper::Pointer &dumper, size_t dimension, size_t padded_dim,
    size_t ex_bits, size_t num_clusters, rabitqlib::RotatorType rotator_type,
    const std::vector<float> &rotated_centroids,
    const std::vector<float> &centroids,
    const std::unique_ptr<rabitqlib::Rotator<float>> &rotator,
    size_t *out_dumped_size = nullptr);

}  // namespace core
}  // namespace zvec
