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

#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "quantizer/preprocessor/fht_rotator.h"

using namespace zvec::turbo;

namespace {

// Helper: fill a vector with random floats.
void fill_random(float *data, size_t dim, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < dim; ++i) data[i] = dist(gen);
}

// Helper: check round-trip (apply_inverse(apply(x)) == x) within tolerance.
void check_round_trip(const FhtRotator &rot, const std::vector<float> &input,
                      float tol = 1e-3f) {
  const int dim = rot.in_dim();
  std::vector<float> rotated(dim);
  std::vector<float> recovered(dim);

  rot.apply(input.data(), rotated.data());
  rot.apply_inverse(rotated.data(), recovered.data());

  for (int i = 0; i < dim; ++i) {
    EXPECT_NEAR(input[i], recovered[i], tol)
        << "mismatch at i=" << i << " dim=" << dim;
  }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Power-of-2 dimensions
// ---------------------------------------------------------------------------

TEST(FhtRotator, PowerOf2RoundTrip) {
  std::mt19937 gen(42);
  for (int dim : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot) << "create failed for dim=" << dim;
    rot->train(nullptr, 0, 0);

    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);
    check_round_trip(*rot, input);
  }
}

// ---------------------------------------------------------------------------
// Non-power-of-2 dimensions
// ---------------------------------------------------------------------------

TEST(FhtRotator, NonPowerOf2RoundTrip) {
  std::mt19937 gen(123);
  for (int dim : {3, 5, 7, 10, 13, 31, 50, 97, 100, 127, 192, 320}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot) << "create failed for dim=" << dim;
    rot->train(nullptr, 0, 0);

    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);
    check_round_trip(*rot, input);
  }
}

// ---------------------------------------------------------------------------
// Serialize / Deserialize round-trip
// ---------------------------------------------------------------------------

TEST(FhtRotator, SerializeDeserialize) {
  std::mt19937 gen(999);
  for (int dim : {32, 97, 128}) {
    // Build and train original rotator.
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);
    rot->train(nullptr, 0, 0);

    // Serialize.
    std::string blob;
    ASSERT_EQ(0, rot->serialize(&blob));
    ASSERT_GT(blob.size(), sizeof(RotatorSerHeader));

    // Restore from blob.
    auto rot2 = FhtRotator::from_blob(blob.data(), blob.size());
    ASSERT_TRUE(rot2) << "from_blob failed for dim=" << dim;

    // Dimensions must match.
    EXPECT_EQ(rot2->in_dim(), dim);
    EXPECT_EQ(rot2->out_dim(), dim);

    // Round-trip via the restored rotator must produce the same result
    // as the original (same flip signs).
    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);

    std::vector<float> r1(dim), r2(dim);
    rot->apply(input.data(), r1.data());
    rot2->apply(input.data(), r2.data());
    for (int i = 0; i < dim; ++i) {
      EXPECT_FLOAT_EQ(r1[i], r2[i]) << "apply mismatch at i=" << i;
    }

    // Inverse via restored rotator must recover the input.
    check_round_trip(*rot2, input);
  }
}

// ---------------------------------------------------------------------------
// Dimension preserved
// ---------------------------------------------------------------------------

TEST(FhtRotator, DimensionPreserved) {
  for (int dim : {1, 7, 64, 97, 128}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);
    EXPECT_EQ(rot->in_dim(), dim);
    EXPECT_EQ(rot->out_dim(), dim);
  }
}

// ---------------------------------------------------------------------------
// Train generates non-zero flip signs
// ---------------------------------------------------------------------------

TEST(FhtRotator, TrainGeneratesFlip) {
  for (int dim : {8, 64, 97}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

    // Before train, apply should not crash but flip is empty — we skip
    // calling apply before train.  After train, serialize must succeed
    // (which requires flip to be populated).
    rot->train(nullptr, 0, 0);

    std::string blob;
    EXPECT_EQ(0, rot->serialize(&blob))
        << "serialize failed after train for dim=" << dim;

    // Verify the payload is not all zeros (extremely unlikely for random bits).
    const auto *hdr = reinterpret_cast<const RotatorSerHeader *>(blob.data());
    EXPECT_EQ(hdr->magic, kRotatorMagic);
    EXPECT_EQ(hdr->version, kRotatorSerVersion);
    EXPECT_EQ(static_cast<RotatorType>(hdr->rotator_type), RotatorType::kFht);
    EXPECT_EQ(static_cast<int>(hdr->in_dim), dim);
    EXPECT_EQ(static_cast<int>(hdr->out_dim), dim);
    EXPECT_GT(hdr->payload_size, 0u);

    // Check that the flip payload is not all-zero.
    const uint8_t *payload =
        reinterpret_cast<const uint8_t *>(blob.data()) + sizeof(RotatorSerHeader);
    bool any_nonzero = false;
    for (uint32_t i = 0; i < hdr->payload_size; ++i) {
      if (payload[i] != 0) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero) << "flip payload is all-zero for dim=" << dim;
  }
}

// ---------------------------------------------------------------------------
// Create with invalid dimension returns nullptr
// ---------------------------------------------------------------------------

TEST(FhtRotator, InvalidDimension) {
  EXPECT_EQ(FhtRotator::create(0), nullptr);
  EXPECT_EQ(FhtRotator::create(-1), nullptr);
}

// ---------------------------------------------------------------------------
// from_blob with malformed input returns nullptr
// ---------------------------------------------------------------------------

TEST(FhtRotator, FromBlobMalformed) {
  EXPECT_EQ(FhtRotator::from_blob(nullptr, 0), nullptr);

  // Too short.
  char buf[4] = {};
  EXPECT_EQ(FhtRotator::from_blob(buf, sizeof(buf)), nullptr);

  // Wrong magic.
  RotatorSerHeader hdr{};
  hdr.magic = 0xDEADBEEF;
  hdr.version = kRotatorSerVersion;
  hdr.rotator_type = static_cast<uint16_t>(RotatorType::kFht);
  hdr.payload_size = 0;
  EXPECT_EQ(FhtRotator::from_blob(&hdr, sizeof(hdr)), nullptr);
}
