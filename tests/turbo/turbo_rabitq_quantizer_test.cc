// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <gtest/gtest.h>
#include <turbo/quantizer/rabitq_quantizer/rabitq_quantizer.h>
#include <zvec/core/framework/index_factory.h>
#if RABITQ_SUPPORTED
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#endif

namespace zvec::turbo {
namespace {

TEST(RabitqQuantizer, BitsMetricsAsymmetricEncodingAndPersistence) {
  constexpr int dim =
      73;  // non-power-of-two rotation and non-byte-aligned codes
  std::mt19937 rng(123);
  std::normal_distribution<float> random;
  std::vector<float> x(dim), q(dim), zero(dim, 0);
  for (int i = 0; i < dim; ++i) {
    x[i] = random(rng);
    q[i] = random(rng);
  }
  for (const char *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    for (int bits = 1; bits <= 9; ++bits) {
      SCOPED_TRACE(metric);
      SCOPED_TRACE(bits);
      IndexMeta meta(IndexMeta::DT_FP32, dim);
      meta.set_metric(metric, 0, ailego::Params{});
      ailego::Params params;
      params.set(RABITQ_TOTAL_BITS, bits);
      auto quantizer = IndexFactory::CreateQuantizer("RabitqQuantizer");
      ASSERT_NE(nullptr, quantizer);
      ASSERT_EQ(0, quantizer->init(meta, params));
      // Use the same reproducible rotation across bit counts and metrics.
      // Production initialization intentionally draws fresh random signs.
      std::string initial_state;
      ASSERT_EQ(0, quantizer->serialize(&initial_state));
      std::mt19937 signs(42);
      const size_t signs_offset = sizeof(QuantizerSerHeader) +
                                  5 * sizeof(uint32_t) +
                                  sizeof(RotatorSerHeader);
      for (size_t i = signs_offset; i < initial_state.size(); ++i)
        initial_state[i] = static_cast<char>(signs() & 0xff);
      ASSERT_EQ(0, quantizer->deserialize(initial_state));
      ASSERT_TRUE(quantizer->requires_original_vectors());
      ASSERT_TRUE(quantizer->require_train());
      auto training =
          std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
      ailego::NumericalVector<float> training_zero(dim);
      std::fill(training_zero.begin(), training_zero.end(), 0);
      ASSERT_TRUE(training->emplace(0, training_zero));
      ASSERT_EQ(0, quantizer->train(training));
      ASSERT_FALSE(quantizer->require_train());
      IndexQueryMeta input(IndexMeta::DT_FP32, dim), dp_meta, query_meta;
      std::string dp, query, self, z;
      ASSERT_EQ(0,
                quantizer->quantize_datapoint(x.data(), input, &dp, &dp_meta));
      ASSERT_EQ(36u + ((dim + 63) / 64 * 64) * bits / 8, dp.size());
      ASSERT_EQ(dp.size(), quantizer->meta().element_size());
      ASSERT_EQ(dp.size(), dp_meta.element_size());
      ASSERT_EQ(0, quantizer->quantize(q.data(), input, &query, &query_meta));
      ASSERT_EQ(quantizer->quantized_query_vector_length(), query.size());
      ASSERT_EQ(query.size(), query_meta.element_size());
      ASSERT_EQ(0, quantizer->quantize(x.data(), input, &self, &query_meta));
      ASSERT_EQ(
          0, quantizer->quantize_datapoint(zero.data(), input, &z, &dp_meta));
      const float actual =
          quantizer->calc_distance_dp_query(dp.data(), query.data());
      EXPECT_TRUE(std::isfinite(actual));
      EXPECT_FLOAT_EQ(actual, quantizer->calc_distance_dp_query_unquantized(
                                  dp.data(), q.data()));
      const void *points[] = {dp.data(), z.data()};
      float batch[2], raw_batch[2];
      quantizer->calc_distance_dp_query_batch(points, 2, query.data(), batch);
      quantizer->calc_distance_dp_query_batch_unquantized(points, 2, q.data(),
                                                          raw_batch);
      auto bound = quantizer->distance(query.data(), query_meta);
      ASSERT_TRUE(bound.valid());
      ASSERT_TRUE(bound.batch_valid());
      float bound_batch[2];
      bound.batch(points, 2, bound_batch);
      EXPECT_FLOAT_EQ(batch[0], bound_batch[0]);
      EXPECT_FLOAT_EQ(batch[1], bound_batch[1]);
      EXPECT_FLOAT_EQ(actual, batch[0]);
      EXPECT_FLOAT_EQ(batch[0], raw_batch[0]);
      EXPECT_FLOAT_EQ(batch[1], raw_batch[1]);
      float norm2 = 0, query_norm2 = 0;
      for (int i = 0; i < dim; ++i) {
        norm2 += x[i] * x[i];
        query_norm2 += q[i] * q[i];
      }
      const bool l2 = std::strcmp(metric, "SquaredEuclidean") == 0;
      const bool cosine = std::strcmp(metric, "Cosine") == 0;
      if (bits > 1) {
        EXPECT_NEAR(l2 || cosine ? 0 : -norm2,
                    quantizer->calc_distance_dp_query(dp.data(), self.data()),
                    1e-3f);
      }
      EXPECT_NEAR(l2 ? query_norm2 : (cosine ? 1 : 0), batch[1], 1e-3f);
      std::string reconstructed;
      ASSERT_EQ(0, quantizer->dequantize(dp.data(), dp_meta, &reconstructed));
      double error2 = 0;
      for (int i = 0; i < dim; ++i) {
        float v;
        std::memcpy(&v, reconstructed.data() + i * sizeof(float), sizeof(v));
        EXPECT_TRUE(std::isfinite(v));
        error2 += (v - x[i]) * (v - x[i]);
      }
      if (bits >= 7) {
        EXPECT_LT(error2 / norm2, 0.01);
      }
      ASSERT_EQ(0, quantizer->dequantize(z.data(), dp_meta, &reconstructed));
      for (int i = 0; i < dim; ++i) {
        float value;
        std::memcpy(&value, reconstructed.data() + i * sizeof(float),
                    sizeof(value));
        EXPECT_FLOAT_EQ(0.0f, value);  // Rotation may preserve a negative zero.
      }
      std::string saved;
      ASSERT_EQ(0, quantizer->serialize(&saved));
      RabitqQuantizer restored;
      ASSERT_EQ(0, restored.init(meta, params));
      // Both the raw-buffer and string restoration entrypoints are supported.
      ASSERT_EQ(0, restored.deserialize(saved));
      std::string encoded, restored_query;
      ASSERT_EQ(
          0, restored.quantize_datapoint(x.data(), input, &encoded, &dp_meta));
      ASSERT_EQ(
          0, restored.quantize(q.data(), input, &restored_query, &query_meta));
      EXPECT_EQ(dp, encoded);
      EXPECT_EQ(query, restored_query);
      EXPECT_FLOAT_EQ(actual, restored.calc_distance_dp_query(
                                  dp.data(), restored_query.data()));
      EXPECT_NE(0, restored.deserialize(saved.data(), saved.size() - 1));
      saved[0] ^= 1;
      EXPECT_NE(0, restored.deserialize(saved));
      EXPECT_FLOAT_EQ(actual,
                      restored.calc_distance_dp_query(dp.data(), query.data()));
    }
  }
}

TEST(RabitqQuantizer, RejectsInvalidConfigurationAndMismatchedState) {
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  ailego::Params params;
  RabitqQuantizer q;
  for (int bits : {-1, 0, 10}) {
    params.set(RABITQ_TOTAL_BITS, bits);
    EXPECT_NE(0, q.init(meta, params));
  }
  params.set(RABITQ_TOTAL_BITS, 7);
  for (int niters : {-1, 0}) {
    params.set(RABITQ_NITERS, niters);
    EXPECT_NE(0, q.init(meta, params));
  }
  params.set(RABITQ_NITERS, 3);
  ASSERT_EQ(0, q.init(meta, params));
  std::string state;
  ASSERT_EQ(0, q.serialize(&state));
  params.set(RABITQ_TOTAL_BITS, 8);
  ASSERT_EQ(0, q.init(meta, params));
  EXPECT_NE(0, q.deserialize(state));
  meta.set_dimension(0);
  EXPECT_NE(0, q.init(meta, params));
  meta.set_dimension(4096);
  EXPECT_NE(0, q.init(meta, params));
  meta.set_dimension(64);
  meta.set_metric("MipsSquaredEuclidean", 0, ailego::Params{});
  EXPECT_NE(0, q.init(meta, params));
}


TEST(RabitqQuantizer, TrainedCentroidZeroResidualAndStateValidation) {
  constexpr int dim = 64;
  IndexMeta meta(IndexMeta::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  RabitqQuantizer q;
  ailego::Params params;
  params.set(RABITQ_NUM_CLUSTERS, 2);
  params.set(RABITQ_SAMPLE_COUNT, 1);
  ASSERT_EQ(0, q.init(meta, params));
  IndexQueryMeta input(IndexMeta::DT_FP32, dim), encoded_meta, query_meta;
  std::vector<float> raw(dim, 3.0f);
  std::string encoded, query;
  EXPECT_NE(0,
            q.quantize_datapoint(raw.data(), input, &encoded, &encoded_meta));
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  ailego::NumericalVector<float> vector(dim);
  std::fill(vector.begin(), vector.end(), 3.0f);
  ASSERT_TRUE(holder->emplace(0, vector));
  ASSERT_TRUE(holder->emplace(1, vector));
  ASSERT_EQ(0, q.train(holder));
  EXPECT_NE(0, q.train(holder));  // The model is frozen once trained.
  ASSERT_EQ(0,
            q.quantize_datapoint(raw.data(), input, &encoded, &encoded_meta));
  ASSERT_EQ(0, q.quantize(raw.data(), input, &query, &query_meta));
  auto estimate = q.estimate_distance_dp_query(encoded.data(), query.data());
  EXPECT_FLOAT_EQ(0, estimate.distance);
  EXPECT_FLOAT_EQ(0, estimate.lower_bound);
  EXPECT_FLOAT_EQ(0, q.calc_distance_dp_query(encoded.data(), query.data()));
  std::string restored;
  ASSERT_EQ(0, q.dequantize(encoded.data(), encoded_meta, &restored));
  for (int i = 0; i < dim; ++i) {
    float value;
    std::memcpy(&value, restored.data() + i * sizeof(float), sizeof(value));
    EXPECT_NEAR(3.0f, value, 1e-5);
  }
  std::string state;
  ASSERT_EQ(0, q.serialize(&state));
  std::string invalid = state;
  // An old single-stage layout must never be interpreted as split codes.
  const uint32_t old_format = 7;
  std::memcpy(invalid.data() + sizeof(QuantizerSerHeader), &old_format, 4);
  EXPECT_NE(0, q.deserialize(invalid));
  invalid = state;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(invalid.data() + invalid.size() - sizeof(float), &nan,
              sizeof(nan));
  EXPECT_NE(0, q.deserialize(invalid));
  std::string after_failure;
  ASSERT_EQ(0, q.serialize(&after_failure));
  EXPECT_EQ(state, after_failure);
  params.set(RABITQ_NUM_CLUSTERS, 3);
  RabitqQuantizer different;
  ASSERT_EQ(0, different.init(meta, params));
  EXPECT_NE(0, different.deserialize(state));
}

#if RABITQ_SUPPORTED
// Compare the portable split representation with the actual estimator used by
// HnswRabitqQueryAlgorithm, including its 4-bit query warmup and IP offset.
TEST(RabitqQuantizer, MatchesDedicatedRabitqEstimator) {
  constexpr int dim = 128;
  std::mt19937 rng(72);
  std::normal_distribution<float> random;
  std::vector<float> x(dim), query(dim), zero(dim, 0);
  for (int i = 0; i < dim; ++i) {
    x[i] = random(rng);
    query[i] = random(rng);
  }
  for (const char *name : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    for (int bits : {1, 2, 7, 9}) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(bits);
      IndexMeta meta(IndexMeta::DT_FP32, dim);
      meta.set_metric(name, 0, ailego::Params{});
      ailego::Params params;
      params.set(RABITQ_TOTAL_BITS, bits);
      RabitqQuantizer quantizer;
      ASSERT_EQ(0, quantizer.init(meta, params));
      auto holder =
          std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
      ailego::NumericalVector<float> center(dim);
      std::fill(center.begin(), center.end(), 0);
      ASSERT_TRUE(holder->emplace(0, center));
      ASSERT_EQ(0, quantizer.train(holder));
      std::string state;
      ASSERT_EQ(0, quantizer.serialize(&state));
      uint32_t rotation_size;
      std::memcpy(&rotation_size,
                  state.data() + sizeof(QuantizerSerHeader) + 16, 4);
      auto rotation = FhtRotator::from_blob(
          state.data() + sizeof(QuantizerSerHeader) + 20, rotation_size);
      ASSERT_NE(nullptr, rotation);
      std::vector<float> raw = x, rotated(dim);
      if (std::strcmp(name, "Cosine") == 0) {
        double norm2 = 0;
        for (float v : raw) norm2 += static_cast<double>(v) * v;
        const float norm = std::sqrt(norm2);
        for (float &v : raw) v /= norm;
      }
      rotation->apply(raw.data(), rotated.data());
      IndexQueryMeta input(IndexMeta::DT_FP32, dim), dp_meta, q_meta;
      std::string dp, q;
      ASSERT_EQ(0,
                quantizer.quantize_datapoint(x.data(), input, &dp, &dp_meta));
      ASSERT_EQ(0, quantizer.quantize(query.data(), input, &q, &q_meta));
      std::vector<float> rotated_query(dim);
      std::memcpy(rotated_query.data(), q.data(), dim * sizeof(float));
      const auto metric = std::strcmp(name, "SquaredEuclidean") == 0
                              ? rabitqlib::METRIC_L2
                              : rabitqlib::METRIC_IP;
      const size_t extra_bits = bits - 1;
      std::vector<char> bin(rabitqlib::BinDataMap<float>::data_bytes(dim));
      std::vector<char> extra(
          rabitqlib::ExDataMap<float>::data_bytes(dim, extra_bits));
      rabitqlib::quant::quantize_split_single(
          rotated.data(), zero.data(), dim, extra_bits, bin.data(),
          extra.data(), metric, rabitqlib::quant::faster_config(dim, bits));
      rabitqlib::SplitSingleQuery<float> wrapper(
          rotated_query.data(), dim, extra_bits,
          rabitqlib::quant::faster_config(dim, 4), metric);
      float norm2 = 0;
      for (float v : rotated_query) norm2 += v * v;
      const float g_add = metric == rabitqlib::METRIC_L2 ? norm2 : 0;
      float ip, expected, lower;
      rabitqlib::split_single_estdist(bin.data(), wrapper, dim, ip, expected,
                                      lower, g_add, std::sqrt(norm2));
      const float offset = std::strcmp(name, "InnerProduct") == 0 ? 1 : 0;
      auto coarse = quantizer.estimate_distance_dp_query(dp.data(), q.data());
      EXPECT_NEAR(expected - offset, coarse.distance, 1e-3);
      EXPECT_NEAR((bits == 1 ? expected : lower) - offset, coarse.lower_bound,
                  1e-3);
      if (bits > 1) {
        rabitqlib::split_single_fulldist(
            bin.data(), extra.data(),
            rabitqlib::select_excode_ipfunc(extra_bits), wrapper, dim,
            extra_bits, expected, lower, ip, g_add, std::sqrt(norm2));
      }
      EXPECT_NEAR(expected - offset,
                  quantizer.calc_distance_dp_query(dp.data(), q.data()), 1e-3);
    }
  }
}

// Verify uint64 single-code -> uint8 FastScan packing against the independent
// batch encoder, including partial lanes and the >1024-dimension accumulator.
TEST(RabitqQuantizer, PackedScanMatchesDedicatedBatchEstimator) {
  std::mt19937 rng(91);
  std::normal_distribution<float> random;
  for (int dim : {128, 1536}) {
    for (const char *metric_name :
         {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
      for (int bits : {1, 7, 9}) {
        SCOPED_TRACE(testing::Message()
                     << dim << ':' << metric_name << ':' << bits);
        IndexMeta meta(IndexMeta::DT_FP32, dim);
        meta.set_metric(metric_name, 0, ailego::Params{});
        ailego::Params params;
        params.set(RABITQ_TOTAL_BITS, bits);
        params.set(RABITQ_NUM_CLUSTERS, 1);
        RabitqQuantizer quantizer;
        ASSERT_EQ(0, quantizer.init(meta, params));
        auto holder =
            std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
        ailego::NumericalVector<float> zero(dim);
        std::fill(zero.begin(), zero.end(), 0);
        ASSERT_TRUE(holder->emplace(0, zero));
        ASSERT_EQ(0, quantizer.train(holder));
        std::string state;
        ASSERT_EQ(0, quantizer.serialize(&state));
        uint32_t rotation_size;
        std::memcpy(&rotation_size,
                    state.data() + sizeof(QuantizerSerHeader) + 16, 4);
        auto rotation = FhtRotator::from_blob(
            state.data() + sizeof(QuantizerSerHeader) + 20, rotation_size);
        ASSERT_NE(nullptr, rotation);
        std::vector<float> raw(dim), rotated(32 * dim), query(dim);
        for (float &v : query) v = random(rng);
        const size_t stride = quantizer.quantized_datapoint_vector_length();
        std::string rows(32 * stride, '\0');
        for (size_t i = 0; i < 32; ++i) {
          double norm2 = 0;
          for (float &v : raw) {
            v = random(rng);
            norm2 += double(v) * v;
          }
          quantizer.quantize_data(raw.data(), rows.data() + i * stride);
          if (std::strcmp(metric_name, "Cosine") == 0) {
            float norm = std::sqrt(norm2);
            for (float &v : raw) v /= norm;
          }
          rotation->apply(raw.data(), rotated.data() + i * dim);
        }
        IndexQueryMeta input(IndexMeta::DT_FP32, dim), encoded_meta;
        std::string encoded;
        ASSERT_EQ(0, quantizer.quantize(query.data(), input, &encoded,
                                        &encoded_meta));
        auto distance = quantizer.distance(encoded.data(), encoded_meta);
        auto scanner = quantizer.block_scanner(distance);
        ASSERT_NE(nullptr, scanner);
        const bool ip = std::strcmp(metric_name, "InnerProduct") == 0;
        const auto metric = std::strcmp(metric_name, "SquaredEuclidean") == 0
                                ? rabitqlib::METRIC_L2
                                : rabitqlib::METRIC_IP;
        rabitqlib::SplitBatchQuery<float> reference_query(
            reinterpret_cast<const float *>(encoded.data()), dim, bits - 1,
            metric, true);
        const float *scalars = reinterpret_cast<const float *>(
            encoded.data() + dim * sizeof(float) + dim / 2);
        reference_query.set_g_add(
            scalars[5]);  // zero centroid => IP correction is zero
        for (size_t count : {size_t(1), size_t(17), size_t(32)}) {
          SCOPED_TRACE(count);
          std::string packed(quantizer.scan_block_size(), '\0');
          ASSERT_EQ(0, quantizer.pack_scan_block(rows.data(), count, stride,
                                                 packed.data()));
          std::string reference(rabitqlib::BatchDataMap<float>::data_bytes(dim),
                                '\0');
          rabitqlib::quant::quantize_one_batch(rotated.data(), zero.data(),
                                               count, dim, reference.data(),
                                               metric);
          EXPECT_EQ(0, std::memcmp(packed.data() + 32 * sizeof(uint32_t),
                                   reference.data(), reference.size()));
          float expected[32], lower[32], dot[32];
          rabitqlib::split_batch_estdist(reference.data(), reference_query, dim,
                                         expected, lower, dot, true);
          turbo::DistanceEstimate estimates[32];
          ASSERT_EQ(0, scanner->estimate(packed.data(), count, estimates));
          for (size_t i = 0; i < count; ++i) {
            const float tolerance =
                1e-5f * std::max(1.0f, std::abs(expected[i]));
            EXPECT_NEAR(expected[i] - (ip ? 1 : 0), estimates[i].distance,
                        tolerance);
            EXPECT_NEAR((bits == 1 ? expected[i] : lower[i]) - (ip ? 1 : 0),
                        estimates[i].lower_bound, tolerance);
            EXPECT_FLOAT_EQ(
                bits == 1 ? estimates[i].distance
                          : distance(rows.data() + i * stride),
                scanner->refine(rows.data() + i * stride, estimates[i]));
            ASSERT_TRUE(scanner->set_reuse_coarse_dot(true));
            const float boosted =
                bits == 1 ? expected[i]
                          : rabitqlib::split_distance_boosting(
                                rows.data() + i * stride + 8 + dim / 8 + 12,
                                rabitqlib::select_excode_ipfunc(bits - 1),
                                reference_query, dim, bits - 1, dot[i]);
            EXPECT_NEAR(boosted - (ip ? 1 : 0),
                        scanner->refine(rows.data() + i * stride, estimates[i]),
                        1e-5f * std::max(1.0f, std::abs(boosted)));
            ASSERT_TRUE(scanner->set_reuse_coarse_dot(false));
          }
          const uint32_t bad_cluster = 1;
          std::memcpy(packed.data(), &bad_cluster, sizeof(bad_cluster));
          EXPECT_NE(0, scanner->estimate(packed.data(), count, estimates));
        }
        // Empty-query LUTs must fall back without dividing by a zero range.
        ASSERT_EQ(
            0, quantizer.quantize(zero.data(), input, &encoded, &encoded_meta));
        EXPECT_EQ(nullptr, quantizer.block_scanner(quantizer.distance(
                               encoded.data(), encoded_meta)));
      }
    }
  }
}

TEST(RabitqQuantizer, PackedScanUsesEachRowsModelCentroid) {
  constexpr int dim = 73, padded = 128, clusters = 3;
  std::mt19937 rng(1234);
  std::normal_distribution<float> random;
  for (const char *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    IndexMeta meta(IndexMeta::DT_FP32, dim);
    meta.set_metric(metric, 0, ailego::Params{});
    ailego::Params params;
    params.set(RABITQ_TOTAL_BITS, 7);
    params.set(RABITQ_NUM_CLUSTERS, clusters);
    RabitqQuantizer quantizer;
    ASSERT_EQ(0, quantizer.init(meta, params));
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
    std::vector<std::vector<float>> centers;
    for (int c = 0; c < clusters; ++c) {
      ailego::NumericalVector<float> center(dim);
      for (float &v : center) v = random(rng);
      centers.emplace_back(center.begin(), center.end());
      ASSERT_TRUE(holder->emplace(c, center));
    }
    ASSERT_EQ(0, quantizer.train(holder));
    std::vector<float> query(dim);
    for (float &v : query) v = random(rng);
    IndexQueryMeta input(IndexMeta::DT_FP32, dim), qmeta;
    std::string encoded;
    ASSERT_EQ(0, quantizer.quantize(query.data(), input, &encoded, &qmeta));
    auto distance = quantizer.distance(encoded.data(), qmeta);
    auto scanner = quantizer.block_scanner(distance);
    ASSERT_NE(nullptr, scanner);
    distance =
        {};  // scanner owns its query, including the LUT's source pointer
    const size_t stride = quantizer.quantized_datapoint_vector_length();
    std::string rows(17 * stride, '\0'),
        block(quantizer.scan_block_size(), '\0');
    for (size_t i = 0; i < 17; ++i) {
      auto row = centers[i % clusters];
      if (i >= clusters)
        for (float &v : row) v += random(rng) * .1f;
      quantizer.quantize_data(row.data(), rows.data() + i * stride);
    }
    ASSERT_EQ(0,
              quantizer.pack_scan_block(rows.data(), 17, stride, block.data()));
    DistanceEstimate estimates[32];
    ASSERT_EQ(0, scanner->estimate(block.data(), 17, estimates));
    const auto *scalars = reinterpret_cast<const float *>(
        encoded.data() + padded * sizeof(float) + padded / 2);
    const auto library_metric = std::strcmp(metric, "SquaredEuclidean") == 0
                                    ? rabitqlib::METRIC_L2
                                    : rabitqlib::METRIC_IP;
    rabitqlib::SplitBatchQuery<float> reference_query(
        reinterpret_cast<const float *>(encoded.data()), padded, 6,
        library_metric, true);
    // The reference uses the same 16-bit LUT estimator, rather than comparing
    // it with an exact floating-point dot product at an arbitrary tolerance.
    // Evaluate each model centroid independently, then select its own lanes.
    for (uint32_t c = 0; c < clusters; ++c) {
      reference_query.set_g_add(scalars[4 + clusters + c], -scalars[4 + c]);
      float expected[32], lower[32], dot[32];
      rabitqlib::split_batch_estdist(block.data() + 32 * sizeof(uint32_t),
                                     reference_query, padded, expected, lower,
                                     dot, true);
      for (size_t i = 0; i < 17; ++i) {
        const char *row = rows.data() + i * stride;
        uint32_t cluster;
        std::memcpy(&cluster, row, sizeof(cluster));
        ASSERT_LT(cluster, clusters);
        if (i < clusters) EXPECT_EQ(i, cluster);
        if (cluster != c) continue;
        const float offset = std::strcmp(metric, "InnerProduct") == 0 ? 1 : 0;
        const float tolerance = 1e-5f * std::max(1.0f, std::abs(expected[i]));
        EXPECT_NEAR(expected[i] - offset, estimates[i].distance, tolerance);
        EXPECT_NEAR(lower[i] - offset, estimates[i].lower_bound, tolerance);
        EXPECT_FLOAT_EQ(quantizer.calc_distance_dp_query(row, encoded.data()),
                        scanner->refine(row, estimates[i]));
        ASSERT_TRUE(scanner->set_reuse_coarse_dot(true));
        const float boosted = rabitqlib::split_distance_boosting(
            row + 8 + padded / 8 + 12, rabitqlib::select_excode_ipfunc(6),
            reference_query, padded, 6, dot[i]);
        EXPECT_NEAR(boosted - offset, scanner->refine(row, estimates[i]),
                    1e-5f * std::max(1.0f, std::abs(boosted)));
        ASSERT_TRUE(scanner->set_reuse_coarse_dot(false));
      }
    }
  }
}
#endif

}  // namespace
}  // namespace zvec::turbo
