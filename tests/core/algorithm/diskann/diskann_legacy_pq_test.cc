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
#include "diskann_quantizer.h"
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

// The adapter must preserve PQ's defaults, budget clamping, metric conversion,
// serialized model, encoded bytes and distance values.
TEST(DiskAnnLegacyPq, BuildAdapterPreservesPqBehavior) {
  for (const char *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    SCOPED_TRACE(metric);
    IndexMeta meta = make_meta(17);
    meta.set_metric(metric, 0, ailego::Params{});
    if (std::strcmp(metric, "Cosine") == 0)
      meta.set_extra_meta_size(sizeof(float));
    DiskAnnQuantizer adapter;
    ASSERT_EQ(0, adapter.init(&meta, {}));
    ASSERT_TRUE(meta.quantizer_name().empty());
    ASSERT_EQ(0, adapter.prepare(100, false, 0, 1));
    EXPECT_EQ(8u, adapter.code_size());
    ASSERT_EQ(0, adapter.prepare(100, true, 400, 1));
    EXPECT_EQ(4u, adapter.code_size());
    EXPECT_NE(0, adapter.prepare(100, true, 99, 1));
    EXPECT_NE(0, adapter.prepare(0, false, 0, 1));

    IndexMeta pq_meta;
    ASSERT_EQ(0, DiskAnnUtil::quantizer_init_meta(meta, &pq_meta));
    ailego::Params settings;
    settings.set("num_chunk", 4U);
    settings.set("thread_count", 1U);
    settings.set("use_zero_mean", false);
    auto reference = IndexFactory::CreateQuantizer("PqInt8Quantizer");
    ASSERT_TRUE(reference);
    ASSERT_EQ(0, reference->init(pq_meta, settings));
    auto pivots = make_random_pivots(17);
    const size_t bytes = pivots.size() * sizeof(float);
    ASSERT_EQ(0, reference->import_codebook(pivots.data(), bytes));
    ASSERT_EQ(0, adapter.get()->import_codebook(pivots.data(), bytes));
    std::string expected, actual;
    ASSERT_EQ(0, reference->serialize(&expected));
    ASSERT_EQ(0, adapter.get()->serialize(&actual));
    EXPECT_EQ(expected, actual);
    auto restored = DiskAnnQuantizer::create_quantizer_from_meta_buffer(
        actual, meta, adapter.code_size(), false);
    ASSERT_TRUE(restored);
    std::vector<uint8_t> expected_code(4), actual_code(4);
    std::vector<uint8_t> expected_query(
        reference->quantized_query_vector_length());
    std::vector<uint8_t> actual_query(
        restored->quantized_query_vector_length());
    reference->quantize_query(pivots.data(), expected_query.data());
    restored->quantize_query(pivots.data(), actual_query.data());
    EXPECT_EQ(expected_query, actual_query);
    for (size_t row = 0; row < 32; ++row) {
      const float *vector = pivots.data() + row * 17;
      reference->quantize_data(vector, expected_code.data());
      restored->quantize_data(vector, actual_code.data());
      EXPECT_EQ(expected_code, actual_code);
      EXPECT_FLOAT_EQ(reference->calc_distance_dp_query(expected_code.data(),
                                                        expected_query.data()),
                      restored->calc_distance_dp_query(actual_code.data(),
                                                       actual_query.data()));
    }
  }
}

TEST(DiskAnnLegacyPq, BuildAdapterValidatesSelectionAndChunks) {
  IndexMeta meta = make_meta(17);
  DiskAnnQuantizer adapter;
  ailego::Params params;
  params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 18U);
  EXPECT_NE(0, adapter.init(&meta, params));
  params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 0U);
  ASSERT_EQ(0, adapter.init(&meta, params));
  ASSERT_EQ(0, adapter.prepare(100, false, 0, 1));
  EXPECT_EQ(8u, adapter.code_size());
  params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 6U);
  params.set(PARAM_DISKANN_BUILDER_QUANTIZER_CLASS, "PqInt8Quantizer");
  meta.set_quantizer("RabitqQuantizer", 0, ailego::Params{});
  ASSERT_EQ(0, adapter.init(&meta, params));
  EXPECT_TRUE(meta.quantizer_name().empty());
  ASSERT_EQ(0, adapter.prepare(100, false, 0, 1));
  EXPECT_EQ(6u, adapter.code_size());
  params.set(PARAM_DISKANN_BUILDER_QUANTIZER_CLASS, "UnknownQuantizer");
  EXPECT_EQ(IndexError_Unsupported, adapter.init(&meta, params));
}

//! The repacked codebook has to be the one the index was built with, otherwise
//! the stored codes decode to different vectors.  Encoding through the
//! quantizer is compared against a brute force search over the on-disk pivots.
TEST(DiskAnnLegacyPq, EncodeMatchesLegacyPivots) {
  const size_t DIM = 17;
  const uint32_t CHUNK_NUM = 4;
  // Legacy chunking distributes the remainder to the leading chunks.
  const std::vector<uint32_t> offsets{0, 5, 9, 13, 17};

  auto pivots = make_random_pivots(DIM);
  std::vector<float> mean(DIM, 0.0f);
  auto payload = make_legacy_payload(pivots, mean, offsets);

  IndexMeta meta = make_meta(DIM);
  auto quantizer = DiskAnnQuantizer::create_quantizer_from_meta_buffer(
      payload, meta, CHUNK_NUM, true);
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
      for (size_t j = offsets[m]; j < offsets[m + 1]; ++j) {
        float diff = vec[j] - pivots[c * DIM + j];
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
    for (size_t j = offsets[m]; j < offsets[m + 1]; ++j) {
      EXPECT_FLOAT_EQ(recon[j], pivots[code[m] * DIM + j])
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
  auto quantizer = DiskAnnQuantizer::create_quantizer_from_meta_buffer(
      payload, meta, CHUNK_NUM, true);
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
  EXPECT_FALSE(DiskAnnQuantizer::create_quantizer_from_meta_buffer(
      payload, meta, CHUNK_NUM, true));
}

//! Only the legacy balanced geometry is supported; arbitrary boundaries would
//! silently change the meaning of persisted codes.
TEST(DiskAnnLegacyPq, RejectsInvalidChunkOffsets) {
  const size_t DIM = 16;
  const uint32_t CHUNK_NUM = 4;

  auto pivots = make_random_pivots(DIM);
  std::vector<float> mean(DIM, 0.0f);
  auto offsets = make_uniform_offsets(DIM, CHUNK_NUM);
  offsets[1] += 1;
  auto payload = make_legacy_payload(pivots, mean, offsets);

  IndexMeta meta = make_meta(DIM);
  EXPECT_FALSE(DiskAnnQuantizer::create_quantizer_from_meta_buffer(
      payload, meta, CHUNK_NUM, true));
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
  EXPECT_FALSE(DiskAnnQuantizer::create_quantizer_from_meta_buffer(
      payload, meta, CHUNK_NUM, true));
}

//! Legacy centroid values can match the quantizer magic. The enclosing header
//! must select import versus deserialize independently of these values.
TEST(DiskAnnLegacyPq, DetectsQuantizerHeader) {
  constexpr uint32_t DIM = 5, CHUNKS = 2;
  IndexMeta meta = make_meta(DIM);
  meta.set_meta(IndexMeta::DataType::DT_FP16, DIM);
  std::vector<ailego::Float16> pivots(kClusterNum * DIM, 0.0f);
  pivots[0] = 202.125f;
  pivots[1] = 50.625f;
  std::string payload(reinterpret_cast<const char *>(pivots.data()),
                      pivots.size() * sizeof(ailego::Float16));
  uint32_t magic;
  std::memcpy(&magic, payload.data(), sizeof(magic));
  ASSERT_EQ(magic, turbo::kQuantizerMagic);
  payload.append(DIM * sizeof(ailego::Float16), '\0');
  const uint32_t offsets[] = {0, 3, DIM};
  payload.append(reinterpret_cast<const char *>(offsets), sizeof(offsets));

  DiskAnnLegacyPqMeta old_header;
  old_header.chunk_num = CHUNKS;
  old_header.centroid_data_size = DIM * sizeof(ailego::Float16);
  old_header.full_pivot_data_size = kClusterNum * old_header.centroid_data_size;
  DiskAnnPqMeta header;
  std::memcpy(static_cast<void *>(&header), &old_header, sizeof(header));
  bool legacy = false;
  ASSERT_EQ(0, DiskAnnQuantizer::normalize_pq_meta(meta, &header, &legacy));
  ASSERT_TRUE(legacy);
  ASSERT_EQ(payload.size(), header.quantizer_meta_buffer_size);
  auto quantizer = DiskAnnQuantizer::create_quantizer_from_meta_buffer(
      payload, meta, static_cast<uint32_t>(header.chunk_num), legacy);
  ASSERT_TRUE(quantizer);

  ASSERT_EQ(0, quantizer->serialize(&payload));
  header.clear();
  header.chunk_num = CHUNKS;
  header.quantizer_meta_buffer_size = payload.size();
  ASSERT_EQ(0, DiskAnnQuantizer::normalize_pq_meta(meta, &header, &legacy));
  ASSERT_FALSE(legacy);
  EXPECT_TRUE(DiskAnnQuantizer::create_quantizer_from_meta_buffer(
      payload, meta, CHUNKS, legacy));

  ++old_header.full_pivot_data_size;
  std::memcpy(static_cast<void *>(&header), &old_header, sizeof(header));
  EXPECT_NE(0, DiskAnnQuantizer::normalize_pq_meta(meta, &header, &legacy));
}

}  // namespace core
}  // namespace zvec
