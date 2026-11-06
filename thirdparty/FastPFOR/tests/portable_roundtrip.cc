// Copyright 2026 Alibaba Group Holding Limited. All Rights Reserved.
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

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>
#include "compositecodec.h"
#include "simdfastpfor.h"
#include "variablebyte.h"

namespace {

std::vector<uint32_t> make_input(size_t length, size_t block_size,
                                 bool full_width) {
  std::vector<uint32_t> input(length);
  uint32_t state = 0x9e3779b9U;
  for (size_t i = 0; i < length; ++i) {
    // Explicit integer arithmetic keeps the data identical across platforms.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    input[i] = full_width ? state : state & 0x3ffU;
  }

  // Small posting-list-like integers with sparse wide exceptions exercise the
  // SIMD block codec; values after the block boundary use the VariableByte
  // tail.
  const uint32_t boundaries[] = {0U,
                                 1U,
                                 127U,
                                 128U,
                                 255U,
                                 256U,
                                 65535U,
                                 65536U,
                                 0x7fffffffU,
                                 0x80000000U,
                                 std::numeric_limits<uint32_t>::max()};
  for (size_t i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); ++i) {
    input[i] = boundaries[i];
  }
  input[block_size - 2] = std::numeric_limits<uint32_t>::max();
  if (length >= block_size) {
    input[block_size - 1] = 0x80000000U;
  }
  if (length > block_size) {
    input[block_size] = std::numeric_limits<uint32_t>::max();
  }
  input.back() = std::numeric_limits<uint32_t>::max();
  return input;
}

template <uint32_t BlockUnits>
bool check_roundtrip(size_t length, bool full_width) {
  using BlockCodec = FastPForLib::SIMDFastPFor<BlockUnits>;
  FastPForLib::CompositeCodec<BlockCodec, FastPForLib::VariableByte> codec;
  const auto input = make_input(length, BlockCodec::BlockSize, full_width);
  std::vector<uint32_t> encoded(input.size() * 2 + 1024);
  size_t encoded_size = encoded.size();
  codec.encodeArray(input.data(), input.size(), encoded.data(), encoded_size);
  if (encoded_size == 0 || encoded_size > encoded.size()) {
    std::cerr << "Invalid encoded size for " << codec.name() << '\n';
    return false;
  }

  std::vector<uint32_t> decoded(input.size(), 0xdeadbeefU);
  size_t decoded_size = decoded.size();
  const uint32_t *end = codec.decodeArray(encoded.data(), encoded_size,
                                          decoded.data(), decoded_size);
  if (decoded_size != input.size() || end != encoded.data() + encoded_size) {
    std::cerr << "Incomplete decode for " << codec.name()
              << " length=" << length << " full_width=" << full_width << '\n';
    return false;
  }
  for (size_t i = 0; i < input.size(); ++i) {
    if (decoded[i] != input[i]) {
      std::cerr << "Round-trip mismatch for " << codec.name()
                << " length=" << length << " full_width=" << full_width
                << " index=" << i << ": expected " << input[i] << ", got "
                << decoded[i] << '\n';
      return false;
    }
  }
  return true;
}

template <uint32_t BlockUnits>
bool check_boundaries() {
  constexpr size_t block_size =
      FastPForLib::SIMDFastPFor<BlockUnits>::BlockSize;
  const size_t lengths[] = {block_size - 1, block_size, block_size + 1,
                            2 * block_size + 17};
  for (size_t length : lengths) {
    for (bool full_width : {false, true}) {
      if (!check_roundtrip<BlockUnits>(length, full_width)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  try {
    return check_boundaries<4>() && check_boundaries<8>() ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FastPFOR round-trip failed: " << error.what() << '\n';
    return 1;
  }
}
