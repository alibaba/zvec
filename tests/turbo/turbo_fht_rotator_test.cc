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
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "preprocessor/fht_rotator/fht_rotator.h"

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
    // Build original rotator.
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

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

TEST(FhtRotator, CreateGeneratesFlip) {
  for (int dim : {8, 64, 97}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

    // After create, flip is already populated, so serialize must succeed.

    std::string blob;
    EXPECT_EQ(0, rot->serialize(&blob))
        << "serialize failed after create for dim=" << dim;

    // Verify the serialized structure deterministically. Copy the header into
    // an aligned local first: blob.data() may be unaligned and dereferencing a
    // reinterpret_cast<const RotatorSerHeader *> would be undefined behavior on
    // architectures that require alignment.
    RotatorSerHeader hdr;
    std::memcpy(&hdr, blob.data(), sizeof(RotatorSerHeader));
    EXPECT_EQ(hdr.magic, kRotatorMagic);
    EXPECT_EQ(hdr.version, kRotatorSerVersion);
    EXPECT_EQ(static_cast<RotateType>(hdr.rotator_type), RotateType::kFht);
    EXPECT_EQ(static_cast<int>(hdr.in_dim), dim);
    EXPECT_EQ(static_cast<int>(hdr.out_dim), dim);

    // Payload holds 4 rounds of ceil(dim/8) flip bytes. Checking the exact size
    // (and total blob length) keeps the test deterministic: we avoid asserting
    // on random bit values, which could theoretically be all-zero on a platform
    // where std::random_device has low entropy.
    const uint32_t expected_flip_size =
        4u * ((static_cast<uint32_t>(dim) + 7u) / 8u);
    EXPECT_EQ(hdr.payload_size, expected_flip_size);
    EXPECT_EQ(blob.size(), sizeof(RotatorSerHeader) + expected_flip_size);
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
  hdr.rotator_type = static_cast<uint16_t>(RotateType::kFht);
  hdr.payload_size = 0;
  EXPECT_EQ(FhtRotator::from_blob(&hdr, sizeof(hdr)), nullptr);
}

// ---------------------------------------------------------------------------
// L2 distance preserved (orthogonal transform)
// ---------------------------------------------------------------------------

TEST(FhtRotator, L2DistancePreserved) {
  std::mt19937 gen(2024);

  for (int dim : {32, 64, 97, 128}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

    const int N = 50;
    std::vector<std::vector<float>> raw(N, std::vector<float>(dim));
    std::vector<std::vector<float>> rotated(N, std::vector<float>(dim));
    for (int i = 0; i < N; ++i) {
      fill_random(raw[i].data(), dim, gen);
      rot->apply(raw[i].data(), rotated[i].data());
    }

    // Check that ||rotated[i] - rotated[j]|| ≈ ||raw[i] - raw[j]||.
    for (int i = 1; i < N; ++i) {
      float d_raw = 0.0f, d_rot = 0.0f;
      for (int j = 0; j < dim; ++j) {
        float dr = raw[i][j] - raw[0][j];
        float dt = rotated[i][j] - rotated[0][j];
        d_raw += dr * dr;
        d_rot += dt * dt;
      }
      EXPECT_NEAR(d_raw, d_rot, 1e-2f)
          << "L2 mismatch for dim=" << dim << " i=" << i;
    }
  }
}

// ---------------------------------------------------------------------------
// Cosine distance preserved (orthogonal transform)
// ---------------------------------------------------------------------------

TEST(FhtRotator, CosineDistancePreserved) {
  std::mt19937 gen(777);

  auto cosine_dist = [](const float *a, const float *b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; ++i) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return (denom < 1e-12f) ? 1.0f : 1.0f - dot / denom;
  };

  for (int dim : {32, 97, 128}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

    const int N = 50;
    std::vector<std::vector<float>> raw(N, std::vector<float>(dim));
    std::vector<std::vector<float>> rotated(N, std::vector<float>(dim));
    for (int i = 0; i < N; ++i) {
      fill_random(raw[i].data(), dim, gen);
      rot->apply(raw[i].data(), rotated[i].data());
    }

    for (int i = 1; i < N; ++i) {
      float d_raw = cosine_dist(raw[i].data(), raw[0].data(), dim);
      float d_rot = cosine_dist(rotated[i].data(), rotated[0].data(), dim);
      EXPECT_NEAR(d_raw, d_rot, 1e-3f)
          << "Cosine mismatch for dim=" << dim << " i=" << i;
    }
  }
}

// ---------------------------------------------------------------------------
// Apply is non-trivial (not identity)
// ---------------------------------------------------------------------------

TEST(FhtRotator, ApplyIsNonTrivial) {
  std::mt19937 gen(42);
  for (int dim : {32, 97, 128}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);

    std::vector<float> output(dim);
    rot->apply(input.data(), output.data());

    // At least some elements should differ from the input.
    bool any_diff = false;
    for (int i = 0; i < dim; ++i) {
      if (std::abs(input[i] - output[i]) > 1e-6f) {
        any_diff = true;
        break;
      }
    }
    EXPECT_TRUE(any_diff) << "apply is identity for dim=" << dim;
  }
}

// ---------------------------------------------------------------------------
// Apply is deterministic (same input → same output)
// ---------------------------------------------------------------------------

TEST(FhtRotator, ApplyDeterministic) {
  std::mt19937 gen(55);
  for (int dim : {32, 97, 128}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);

    std::vector<float> r1(dim), r2(dim);
    rot->apply(input.data(), r1.data());
    rot->apply(input.data(), r2.data());

    for (int i = 0; i < dim; ++i) {
      EXPECT_FLOAT_EQ(r1[i], r2[i])
          << "non-deterministic apply at i=" << i << " dim=" << dim;
    }
  }
}

// ---------------------------------------------------------------------------
// Deserialize on existing object (init → serialize → deserialize on new object)
// ---------------------------------------------------------------------------

TEST(FhtRotator, DeserializeOnExistingObject) {
  std::mt19937 gen(314);
  for (int dim : {32, 97, 128}) {
    // Build original.
    auto rot1 = FhtRotator::create(dim);
    ASSERT_TRUE(rot1);

    std::string blob;
    ASSERT_EQ(0, rot1->serialize(&blob));

    // Create a fresh rotator, then call deserialize() on it.
    auto rot2 = FhtRotator::create(dim);
    ASSERT_TRUE(rot2);
    ASSERT_EQ(0, rot2->deserialize(blob.data(), blob.size()));

    EXPECT_EQ(rot2->in_dim(), dim);
    EXPECT_EQ(rot2->out_dim(), dim);

    // Both rotators should produce identical results.
    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);

    std::vector<float> r1(dim), r2(dim);
    rot1->apply(input.data(), r1.data());
    rot2->apply(input.data(), r2.data());
    for (int i = 0; i < dim; ++i) {
      EXPECT_FLOAT_EQ(r1[i], r2[i])
          << "apply mismatch at i=" << i << " dim=" << dim;
    }

    // Inverse via rot2 should recover input.
    check_round_trip(*rot2, input);
  }
}

// ---------------------------------------------------------------------------
// Deserialize with truncated payload fails
// ---------------------------------------------------------------------------

TEST(FhtRotator, DeserializeTruncatedPayload) {
  std::mt19937 gen(42);
  auto rot = FhtRotator::create(64);
  ASSERT_TRUE(rot);

  std::string blob;
  ASSERT_EQ(0, rot->serialize(&blob));

  // Truncate the blob: keep header but cut half the payload.
  // Copy the header into an aligned local before reading fields (blob.data()
  // may be unaligned; a direct reinterpret_cast dereference is UB on
  // alignment-sensitive architectures).
  RotatorSerHeader hdr;
  std::memcpy(&hdr, blob.data(), sizeof(RotatorSerHeader));
  size_t truncated_len = sizeof(RotatorSerHeader) + hdr.payload_size / 2;

  auto rot2 = FhtRotator::create(64);
  ASSERT_TRUE(rot2);
  EXPECT_NE(0, rot2->deserialize(blob.data(), truncated_len));

  // Also test from_blob with truncated data.
  EXPECT_EQ(FhtRotator::from_blob(blob.data(), truncated_len), nullptr);
}

// ---------------------------------------------------------------------------
// Large dimension stress test
// ---------------------------------------------------------------------------

TEST(FhtRotator, LargeDimension) {
  std::mt19937 gen(2025);
  for (int dim : {1024, 2048, 4096}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot) << "create failed for dim=" << dim;

    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);
    check_round_trip(*rot, input, 1e-2f);

    // Verify serialize/deserialize round-trip.
    std::string blob;
    ASSERT_EQ(0, rot->serialize(&blob));
    auto rot2 = FhtRotator::from_blob(blob.data(), blob.size());
    ASSERT_TRUE(rot2);

    std::vector<float> r1(dim), r2(dim);
    rot->apply(input.data(), r1.data());
    rot2->apply(input.data(), r2.data());
    for (int i = 0; i < dim; ++i) {
      EXPECT_FLOAT_EQ(r1[i], r2[i]) << "mismatch at i=" << i << " dim=" << dim;
    }
  }
}

// ---------------------------------------------------------------------------
// Norm preserved (orthogonal transform preserves vector norm)
// ---------------------------------------------------------------------------

TEST(FhtRotator, NormPreserved) {
  std::mt19937 gen(99);
  for (int dim : {32, 97, 128}) {
    auto rot = FhtRotator::create(dim);
    ASSERT_TRUE(rot);

    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);

    float norm_in = 0.0f;
    for (int i = 0; i < dim; ++i) norm_in += input[i] * input[i];

    std::vector<float> output(dim);
    rot->apply(input.data(), output.data());

    float norm_out = 0.0f;
    for (int i = 0; i < dim; ++i) norm_out += output[i] * output[i];

    EXPECT_NEAR(norm_in, norm_out, 1e-2f)
        << "norm not preserved for dim=" << dim;
  }
}
