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

#include "quantizer/rotator/fht_rotator.h"
#include <vector>
#include <gtest/gtest.h>

namespace zvec::core {
namespace {

class LegacyFhtFixture : public FhtRotator {
 public:
  void restore(size_t dim, const std::vector<char> &payload) {
    dimension_ = dim;
    load_blob(payload.data());
  }

  std::vector<char> payload() const {
    std::vector<char> result(blob_bytes());
    save_blob(result.data());
    return result;
  }
};

TEST(CoreFhtRotator, LegacyPayloadCompatibility) {
  // Outputs recorded from the Ailego implementation before its removal.
  // Cover both power-of-two transforms and odd dimensions with SIMD tails.
  const std::vector<std::vector<float>> expected = {
      {0.56249994f, -1.1875f, -0.6875f, 0.0624999627f, 1.68749988f, -1.0625f,
       2.93749976f, -0.0625000447f},
      {-0.0999501944f, 0.987252235f, -2.61998558f, -0.199672937f, -0.886320353f,
       -0.338618219f, 1.23634613f, 2.68937016f, 0.297273308f, 0.476163983f,
       -1.64819348f, -1.04912591f, -1.95380306f},
      {-0.0624998994f, 0.85937506f, -1.4375f,      0.640624762f, 1.50000024f,
       0.29687503f,    0.37499997f, -1.546875f,    1.60937488f,  0.18749994f,
       -0.42187494f,   1.93749964f, -0.171875104f, -1.40624976f, -2.70312476f,
       -0.28124997f,   -1.125f,     1.04687476f,   -3.40625024f, -2.32812476f,
       -0.0937500149f, 0.515625f,   -0.125000045f, 0.640625f,    -0.578125119f,
       -2.125f,        -1.703125f,  0.0312501192f, -1.76562488f, -1.0625f,
       -1.26562488f,   1.84374988f},
      {-0.758491993f, 1.10270405f,    0.154842347f,  0.832951486f,
       -1.91311431f,  0.380737245f,   -2.97885323f,  -0.848319292f,
       -1.81309533f,  -1.73712897f,   -2.73360062f,  -1.63505292f,
       -1.30537999f,  1.47828913f,    0.16802673f,   1.05678391f,
       -0.20405671f,  2.2601285f,     -2.32434988f,  -1.27150369f,
       -1.46672845f,  -1.74106979f,   -0.821945071f, -0.653419316f,
       -0.531985402f, 0.687504292f,   2.05014062f,   1.85613477f,
       3.90745926f,   -1.8369503f,    0.0596558452f, 0.511027277f,
       0.519282222f,  -0.0314112306f, 0.136300206f,  0.129508972f,
       -1.3379606f,   -1.32524908f,   1.81530142f,   -2.52661633f,
       0.347391844f,  -0.851854861f,  0.706693232f,  0.0705927908f,
       0.882308483f,  0.472256392f,   2.38945007f,   2.62175918f,
       1.23784006f,   0.431680202f,   -1.54591417f,  1.11937404f,
       -1.39181232f,  2.18033004f,    0.356999338f,  0.731980681f,
       -0.200637221f, 0.934967816f,   1.25248671f,   -0.409617543f,
       2.17438245f,   1.28635013f,    -1.78452396f,  0.139531478f,
       -0.23540172f,  0.704256535f,   0.688521504f,  1.01937151f,
       2.96606135f,   -0.295190156f,  0.254893899f,  -1.56239092f,
       -0.230148152f, -0.349212646f,  0.489522815f,  -0.157410324f,
       -1.47589147f,  1.9084698f,     0.0614075661f, 0.11402005f,
       0.0248839259f, -2.92599916f,   1.14671135f,   0.571595967f,
       -1.58671141f,  0.290792584f,   -0.409218669f, 1.66512156f,
       -0.381273985f, -0.988983393f,  1.2441709f,    -1.62465358f,
       -1.04002166f,  -1.3895812f,    0.24763605f,   0.65419054f,
       1.60267234f},
  };
  for (const auto &golden : expected) {
    const size_t dim = golden.size();
    SCOPED_TRACE(dim);
    std::vector<char> flip(4 * ((dim + 7) / 8));
    for (size_t i = 0; i < flip.size(); ++i) {
      flip[i] = static_cast<char>((i * 37 + 11) & 255);
    }
    LegacyFhtFixture rotator;
    rotator.restore(dim, flip);
    EXPECT_EQ(flip, rotator.payload());
    EXPECT_EQ(24 + flip.size(), rotator.dump_bytes());

    std::vector<float> input(dim), output(dim), recovered(dim);
    for (size_t i = 0; i < dim; ++i) {
      input[i] = float(int(i * 7 % 19) - 9) * 0.25f;
    }
    rotator.rotate(input.data(), output.data());
    // Decode vectors produced before migration, not just this implementation's
    // own output: a changed forward/inverse pair could still round-trip.
    rotator.unrotate(golden.data(), recovered.data());
    for (size_t i = 0; i < dim; ++i) {
      EXPECT_NEAR(golden[i], output[i], 2e-6f);
      EXPECT_NEAR(input[i], recovered[i], 2e-6f);
    }

    auto inplace = input;
    rotator.rotate(inplace.data(), inplace.data());
    for (size_t i = 0; i < dim; ++i) {
      EXPECT_NEAR(golden[i], inplace[i], 2e-6f);
    }
    rotator.unrotate(inplace.data(), inplace.data());
    for (size_t i = 0; i < dim; ++i) {
      EXPECT_NEAR(input[i], inplace[i], 2e-6f);
    }
    EXPECT_EQ(flip, rotator.payload());
  }
}

}  // namespace
}  // namespace zvec::core
