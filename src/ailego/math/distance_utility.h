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

#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

/*! Four-bits Squared Difference Table
 */
static const AILEGO_ALIGNED(64) uint8_t Int4SquaredDiffTable[256] = {
    0,  1,  4,   9,   16,  25,  36,  49,  64,  49,  36,  25,  16,  9,   4,  1,
    1,  0,  1,   4,   9,   16,  25,  36,  81,  64,  49,  36,  25,  16,  9,  4,
    4,  1,  0,   1,   4,   9,   16,  25,  100, 81,  64,  49,  36,  25,  16, 9,
    9,  4,  1,   0,   1,   4,   9,   16,  121, 100, 81,  64,  49,  36,  25, 16,
    16, 9,  4,   1,   0,   1,   4,   9,   144, 121, 100, 81,  64,  49,  36, 25,
    25, 16, 9,   4,   1,   0,   1,   4,   169, 144, 121, 100, 81,  64,  49, 36,
    36, 25, 16,  9,   4,   1,   0,   1,   196, 169, 144, 121, 100, 81,  64, 49,
    49, 36, 25,  16,  9,   4,   1,   0,   225, 196, 169, 144, 121, 100, 81, 64,
    64, 81, 100, 121, 144, 169, 196, 225, 0,   1,   4,   9,   16,  25,  36, 49,
    49, 64, 81,  100, 121, 144, 169, 196, 1,   0,   1,   4,   9,   16,  25, 36,
    36, 49, 64,  81,  100, 121, 144, 169, 4,   1,   0,   1,   4,   9,   16, 25,
    25, 36, 49,  64,  81,  100, 121, 144, 9,   4,   1,   0,   1,   4,   9,  16,
    16, 25, 36,  49,  64,  81,  100, 121, 16,  9,   4,   1,   0,   1,   4,  9,
    9,  16, 25,  36,  49,  64,  81,  100, 25,  16,  9,   4,   1,   0,   1,  4,
    4,  9,  16,  25,  36,  49,  64,  81,  36,  25,  16,  9,   4,   1,   0,  1,
    1,  4,  9,   16,  25,  36,  49,  64,  49,  36,  25,  16,  9,   4,   1,  0,
};

/*! Four-bits Integer Multiplication Table
 */
static const AILEGO_ALIGNED(64) int8_t Int4MulTable[256] = {
    0, 0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0, 1,  2,   3,   4,   5,   6,   7,   -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,
    0, 2,  4,   6,   8,   10,  12,  14,  -16, -14, -12, -10, -8,  -6,  -4,  -2,
    0, 3,  6,   9,   12,  15,  18,  21,  -24, -21, -18, -15, -12, -9,  -6,  -3,
    0, 4,  8,   12,  16,  20,  24,  28,  -32, -28, -24, -20, -16, -12, -8,  -4,
    0, 5,  10,  15,  20,  25,  30,  35,  -40, -35, -30, -25, -20, -15, -10, -5,
    0, 6,  12,  18,  24,  30,  36,  42,  -48, -42, -36, -30, -24, -18, -12, -6,
    0, 7,  14,  21,  28,  35,  42,  49,  -56, -49, -42, -35, -28, -21, -14, -7,
    0, -8, -16, -24, -32, -40, -48, -56, 64,  56,  48,  40,  32,  24,  16,  8,
    0, -7, -14, -21, -28, -35, -42, -49, 56,  49,  42,  35,  28,  21,  14,  7,
    0, -6, -12, -18, -24, -30, -36, -42, 48,  42,  36,  30,  24,  18,  12,  6,
    0, -5, -10, -15, -20, -25, -30, -35, 40,  35,  30,  25,  20,  15,  10,  5,
    0, -4, -8,  -12, -16, -20, -24, -28, 32,  28,  24,  20,  16,  12,  8,   4,
    0, -3, -6,  -9,  -12, -15, -18, -21, 24,  21,  18,  15,  12,  9,   6,   3,
    0, -2, -4,  -6,  -8,  -10, -12, -14, 16,  14,  12,  10,  8,   6,   4,   2,
    0, -1, -2,  -3,  -4,  -5,  -6,  -7,  8,   7,   6,   5,   4,   3,   2,   1,
};

}  // namespace ailego
}  // namespace zvec
