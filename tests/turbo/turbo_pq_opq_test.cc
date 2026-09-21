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

#include <algorithm>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "preprocessor/opq_rotator/opq_rotator.h"
#include "quantizer/common/pq_quantizer/pq_opq.h"

namespace zvec {
namespace turbo {
namespace {

ailego::Params opq_params() {
  ailego::Params params;
  params.set("rotate_type", std::string("opq"));
  return params;
}

template <size_t NumCentroids>
void check_reconstruction(const std::vector<size_t> &widths) {
  size_t dim = 0;
  std::vector<std::vector<float>> codebook(widths.size());
  std::vector<std::vector<const void *>> pointers(widths.size());
  std::vector<float> expected;
  for (size_t m = 0; m < widths.size(); ++m) {
    const size_t width = widths[m];
    auto &block = codebook[m];
    block.assign(NumCentroids * width, 100.0f);
    for (size_t d = 0; d < width; ++d) {
      block[d] = std::numeric_limits<float>::quiet_NaN();
      block[width + d] = static_cast<float>(dim + d + 1);
      block[2 * width + d] = -block[width + d];
      expected.push_back(block[width + d]);
    }
    for (size_t j = 0; j < NumCentroids; ++j) {
      pointers[m].push_back(block.data() + j * width);
    }
    dim += width;
  }
  std::vector<float> input(dim * 2);
  for (size_t d = 0; d < dim; ++d) {
    input[d] = expected[d] + 0.5f;
    input[dim + d] = -input[d];
  }
  std::vector<float> output(input.size());
  const BatchDistanceFunc l2 = [](const void **centroids, const void *query,
                                  size_t count, size_t width, float *out,
                                  const void **) {
    const auto *q = static_cast<const float *>(query);
    for (size_t j = 0; j < count; ++j) {
      const auto *c = static_cast<const float *>(centroids[j]);
      out[j] = 0.0f;
      for (size_t d = 0; d < width; ++d) {
        float diff = q[d] - c[d];
        out[j] += diff * diff;
      }
    }
  };
  float mse = pq_reconstruct_batch<NumCentroids>(
      input.data(), 2, dim, pointers, l2, [&](size_t m) { return widths[m]; },
      output.data());
  EXPECT_FLOAT_EQ(static_cast<float>(dim) * 0.25f, mse);
  for (size_t d = 0; d < dim; ++d) {
    EXPECT_FLOAT_EQ(expected[d], output[d]);
    EXPECT_FLOAT_EQ(-expected[d], output[dim + d]);
  }
}

TEST(PqOpq, ReconstructionSkipsEmptyCentroidsForBothCodebookSizes) {
  check_reconstruction<16>({2, 2});
  check_reconstruction<256>({2, 2});
}

TEST(PqOpq, ReconstructionSupportsUnequalChunks) {
  check_reconstruction<256>({3, 2, 2});
}

TEST(PqOpq, ReinitializationClearsRotation) {
  PqOpq opq;
  for (bool explicit_none : {false, true}) {
    ASSERT_EQ(0, opq.init(2, DataType::kFp32, opq_params()));
    ASSERT_TRUE(opq.enabled());
    ailego::Params disabled;
    if (explicit_none) disabled.set("rotate_type", std::string("none"));
    ASSERT_EQ(0, opq.init(3, DataType::kFp16, disabled));
    EXPECT_FALSE(opq.enabled());
    EXPECT_EQ(0, opq.rotate_type());
    float input[] = {1.0f, 2.0f, 3.0f};
    std::vector<float> buffer;
    EXPECT_EQ(static_cast<const void *>(input), opq.apply(input, &buffer));
    EXPECT_TRUE(buffer.empty());
    std::string blob("old contents");
    ASSERT_EQ(0, opq.serialize(&blob));
    EXPECT_TRUE(blob.empty());
  }
}

TEST(PqOpq, InvalidConfiguration) {
  PqOpq opq;
  auto params = opq_params();
  EXPECT_EQ(kErrUnsupported, opq.init(2, DataType::kFp16, params));
  for (const char *key : {"opq_iter", "opq_pq_iter"}) {
    params = opq_params();
    params.set(key, uint32_t{0});
    EXPECT_EQ(kErrInvalidArgument, opq.init(2, DataType::kFp32, params));
  }
  params.set("rotate_type", std::string("unknown"));
  EXPECT_EQ(kErrUnsupported, opq.init(2, DataType::kFp32, params));
}

TEST(PqOpq, DisabledAndEmptyTrainingSkipCallbacks) {
  PqOpq opq;
  auto train = [](uint32_t) { ADD_FAILURE() << "unexpected training"; };
  auto reconstruct = [](const float *, size_t, float *) {
    ADD_FAILURE() << "unexpected reconstruction";
    return 0.0f;
  };
  float data[] = {1.0f, 2.0f};
  opq.train(data, 1, "test", train, reconstruct);
  EXPECT_FLOAT_EQ(1.0f, data[0]);
  EXPECT_FLOAT_EQ(2.0f, data[1]);
  ASSERT_EQ(0, opq.init(2, DataType::kFp32, opq_params()));
  opq.train(nullptr, 0, "test", train, reconstruct);
}

TEST(PqOpq, TrainingAppliesFinalMatrixAndStopsOnPlateau) {
  for (uint32_t rounds : {1u, 4u}) {
    PqOpq opq;
    auto params = opq_params();
    params.set("opq_iter", rounds);
    params.set("opq_pq_iter", uint32_t{2});
    ASSERT_EQ(0, opq.init(2, DataType::kFp32, params));
    const std::vector<float> original{1, 0, 0, 1, 1, 1};
    const std::vector<float> target{0, 1, -1, 0, -1, 1};
    std::vector<float> data = original;
    size_t trained = 0;
    size_t reconstructed = 0;
    opq.train(
        data.data(), 3, "test",
        [&](uint32_t max_iters) {
          EXPECT_EQ(2u, max_iters);
          EXPECT_EQ(reconstructed, trained);
          ++trained;
        },
        [&](const float *rotated, size_t num, float *x_hat) {
          EXPECT_EQ(trained, reconstructed + 1);
          EXPECT_EQ(3u, num);
          EXPECT_EQ(data.data(), rotated);
          std::copy(target.begin(), target.end(), x_hat);
          ++reconstructed;
          return 1.0f;  // Constant error triggers the plateau condition.
        });
    EXPECT_EQ(rounds == 1 ? 1u : 2u, trained);
    EXPECT_EQ(trained, reconstructed);
    for (size_t i = 0; i < data.size(); ++i) {
      EXPECT_NEAR(target[i], data[i], 1e-5f);
    }
    float restored[2];
    opq.rotate_inverse(data.data(), restored);
    EXPECT_NEAR(original[0], restored[0], 1e-5f);
    EXPECT_NEAR(original[1], restored[1], 1e-5f);
  }
}

TEST(PqOpq, PreservesRotatorBlobAndRejectsInvalidDimension) {
  auto rotator = OpqRotator::create(3);
  ASSERT_TRUE(rotator);
  std::string blob;
  ASSERT_EQ(0, rotator->serialize(&blob));
  PqOpq opq;
  const auto type = static_cast<uint8_t>(RotateType::kOpq);
  ASSERT_EQ(0, opq.deserialize(type, 3, blob.data(), blob.size()));
  std::string saved;
  ASSERT_EQ(0, opq.serialize(&saved));
  EXPECT_EQ(blob, saved);
  float input[] = {1, 2, 3};
  float expected[3];
  rotator->apply(input, expected);
  std::vector<float> actual;
  opq.apply(input, &actual);
  ASSERT_EQ(3u, actual.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i], actual[i]);
  }
  EXPECT_EQ(kErrInvalidArgument,
            opq.deserialize(type, 2, blob.data(), blob.size()));
  EXPECT_FALSE(opq.enabled());
  EXPECT_EQ(kErrInvalidArgument,
            opq.deserialize(type, 3, blob.data(), blob.size() - 1));
  EXPECT_EQ(0, opq.deserialize(0, 3, nullptr, 0));
  EXPECT_FALSE(opq.enabled());
}

}  // namespace
}  // namespace turbo
}  // namespace zvec
