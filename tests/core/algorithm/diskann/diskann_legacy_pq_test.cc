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

#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/turbo/turbo.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

namespace {

constexpr size_t kClusterNum = 256;

//! Lay out a codebook the way DiskAnn dumped it before the codebook moved into
//! the quantizer: 256 pivot rows of `dim` components, a per-dimension mean,
//! then chunk_num + 1 chunk offsets.
std::string make_legacy_payload(const std::vector<float> &pivots,
                                const std::vector<float> &mean,
                                const std::vector<uint32_t> &offsets) {
  std::string payload;
  payload.append(reinterpret_cast<const char *>(pivots.data()),
                 pivots.size() * sizeof(float));
  payload.append(reinterpret_cast<const char *>(mean.data()),
                 mean.size() * sizeof(float));
  payload.append(reinterpret_cast<const char *>(offsets.data()),
                 offsets.size() * sizeof(uint32_t));
  return payload;
}

std::vector<float> make_random_pivots(size_t dim, uint32_t seed = 7) {
  std::vector<float> pivots(kClusterNum * dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &value : pivots) {
    value = dist(gen);
  }
  return pivots;
}

std::vector<uint32_t> make_uniform_offsets(size_t dim, uint32_t chunk_num) {
  std::vector<uint32_t> offsets(chunk_num + 1);
  for (size_t m = 0; m <= chunk_num; ++m) {
    offsets[m] = static_cast<uint32_t>(m * (dim / chunk_num));
  }
  return offsets;
}

IndexMeta make_meta(size_t dim) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(dim));
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  return meta;
}

}  // namespace

//! The repacked codebook has to be the one the index was built with, otherwise
//! the stored codes decode to different vectors.  Encoding through the
//! quantizer is compared against a brute force search over the on-disk pivots.
TEST(DiskAnnLegacyPq, EncodeMatchesLegacyPivots) {
  const size_t DIM = 16;
  const uint32_t CHUNK_NUM = 4;
  const size_t SUB_DIM = DIM / CHUNK_NUM;

  auto pivots = make_random_pivots(DIM);
  std::vector<float> mean(DIM, 0.0f);
  auto payload =
      make_legacy_payload(pivots, mean, make_uniform_offsets(DIM, CHUNK_NUM));

  IndexMeta meta = make_meta(DIM);
  auto quantizer =
      DiskAnnUtil::create_quantizer_from_meta_buffer(payload, meta, CHUNK_NUM);
  ASSERT_TRUE(quantizer);

  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> vec(DIM);
  for (auto &value : vec) {
    value = dist(gen);
  }

  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(vec.data(), code.data());
  ASSERT_EQ(code.size(), CHUNK_NUM);

  for (size_t m = 0; m < CHUNK_NUM; ++m) {
    size_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t c = 0; c < kClusterNum; ++c) {
      float sum = 0.0f;
      for (size_t j = 0; j < SUB_DIM; ++j) {
        float diff = vec[m * SUB_DIM + j] - pivots[c * DIM + m * SUB_DIM + j];
        sum += diff * diff;
      }
      if (sum < best_dist) {
        best_dist = sum;
        best = c;
      }
    }
    EXPECT_EQ(code[m], best) << "m=" << m;
  }

  // Decoding must return the selected pivots bit for bit: only true if the
  // repacking picked the right components for every chunk.
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
                       static_cast<uint32_t>(DIM));
  std::string decoded;
  ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &decoded));
  ASSERT_EQ(decoded.size(), DIM * sizeof(float));
  const float *recon = reinterpret_cast<const float *>(decoded.data());
  for (size_t m = 0; m < CHUNK_NUM; ++m) {
    for (size_t j = 0; j < SUB_DIM; ++j) {
      EXPECT_FLOAT_EQ(recon[m * SUB_DIM + j],
                      pivots[code[m] * DIM + m * SUB_DIM + j])
          << "m=" << m << " j=" << j;
    }
  }
}

//! The lookup table the search path uses must reproduce the squared euclidean
//! distance to the decoded vector, which is what the legacy search computed.
TEST(DiskAnnLegacyPq, AdcDistanceMatchesReference) {
  const size_t DIM = 32;
  const uint32_t CHUNK_NUM = 8;

  auto pivots = make_random_pivots(DIM, 11);
  std::vector<float> mean(DIM, 0.0f);
  auto payload =
      make_legacy_payload(pivots, mean, make_uniform_offsets(DIM, CHUNK_NUM));

  IndexMeta meta = make_meta(DIM);
  auto quantizer =
      DiskAnnUtil::create_quantizer_from_meta_buffer(payload, meta, CHUNK_NUM);
  ASSERT_TRUE(quantizer);

  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> vec(DIM);
  std::vector<float> query(DIM);
  for (size_t i = 0; i < DIM; ++i) {
    vec[i] = dist(gen);
    query[i] = dist(gen);
  }

  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(vec.data(), code.data());

  std::vector<float> lut(quantizer->quantized_query_vector_length() /
                         sizeof(float));
  quantizer->quantize_query(query.data(), lut.data());
  float adc = quantizer->calc_distance_dp_query(code.data(), lut.data());

  float expected = 0.0f;
  for (size_t i = 0; i < DIM; ++i) {
    const size_t m = i / (DIM / CHUNK_NUM);
    float diff = query[i] - pivots[code[m] * DIM + i];
    expected += diff * diff;
  }

  EXPECT_NEAR(adc, expected, 1e-4f);
}

//! A codebook trained on residuals cannot be loaded: the quantizer runs without
//! a mean and would decode against the wrong origin.
TEST(DiskAnnLegacyPq, RejectsNonZeroMean) {
  const size_t DIM = 16;
  const uint32_t CHUNK_NUM = 4;

  auto pivots = make_random_pivots(DIM);
  std::vector<float> mean(DIM, 0.5f);
  auto payload =
      make_legacy_payload(pivots, mean, make_uniform_offsets(DIM, CHUNK_NUM));

  IndexMeta meta = make_meta(DIM);
  EXPECT_FALSE(
      DiskAnnUtil::create_quantizer_from_meta_buffer(payload, meta, CHUNK_NUM));
}

//! Chunks of unequal width have no counterpart in the quantizer, so the index
//! has to be rebuilt instead of being silently misread.
TEST(DiskAnnLegacyPq, RejectsNonUniformChunks) {
  const size_t DIM = 16;
  const uint32_t CHUNK_NUM = 4;

  auto pivots = make_random_pivots(DIM);
  std::vector<float> mean(DIM, 0.0f);
  auto offsets = make_uniform_offsets(DIM, CHUNK_NUM);
  offsets[1] += 1;
  auto payload = make_legacy_payload(pivots, mean, offsets);

  IndexMeta meta = make_meta(DIM);
  EXPECT_FALSE(
      DiskAnnUtil::create_quantizer_from_meta_buffer(payload, meta, CHUNK_NUM));
}

TEST(DiskAnnLegacyPq, RejectsTruncatedCodebook) {
  const size_t DIM = 16;
  const uint32_t CHUNK_NUM = 4;

  auto pivots = make_random_pivots(DIM);
  std::vector<float> mean(DIM, 0.0f);
  auto payload =
      make_legacy_payload(pivots, mean, make_uniform_offsets(DIM, CHUNK_NUM));
  payload.resize(payload.size() - sizeof(uint32_t));

  IndexMeta meta = make_meta(DIM);
  EXPECT_FALSE(
      DiskAnnUtil::create_quantizer_from_meta_buffer(payload, meta, CHUNK_NUM));
}

//! Only a headerless buffer is treated as legacy; a buffer written by the
//! quantizer keeps going through deserialize().
TEST(DiskAnnLegacyPq, DetectsQuantizerHeader) {
  std::string blob(sizeof(turbo::QuantizerSerHeader), '\0');
  uint32_t magic = turbo::kQuantizerMagic;
  std::memcpy(&blob[0], &magic, sizeof(magic));
  EXPECT_FALSE(DiskAnnUtil::is_legacy_pq_meta_buffer(blob));

  std::string legacy(sizeof(turbo::QuantizerSerHeader), '\0');
  EXPECT_TRUE(DiskAnnUtil::is_legacy_pq_meta_buffer(legacy));
}

}  // namespace core
}  // namespace zvec
