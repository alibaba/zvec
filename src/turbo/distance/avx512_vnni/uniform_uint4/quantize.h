// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx512_vnni {

void uniform_uint4_quantize(const float *input, std::size_t dimension,
                            float minimum, float range, std::uint8_t *output);

}  // namespace zvec::turbo::avx512_vnni
