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

// Micro-benchmark for the RaBitQ batch-estimate hot path on synthetic data.
// All loops scan identical packed 32-row blocks and differ only in scope:
//   (a) turbo RabitqQuantizer::block_scanner()->estimate()  [integrated]
//   (b) rabitqlib::split_batch_estdist()                    [dedicated]
//   (c) rabitqlib::fastscan::accumulate_hacc() alone        [SIMD kernel]
// Also confirms which SIMD implementation the runtime dispatch selected and
// cross-checks turbo's estimator against split_batch_estdist on blocks whose
// 32 rows share one model cluster.
//
// Usage:
//   ivf_rabitq_kernel_bench [--dim=768] [--bits=7] [--clusters=16]
//       [--metric=Cosine] [--rows=65536] [--reps=300] [--seed=123]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <turbo/quantizer/rabitq_quantizer/rabitq_quantizer.h>
#include <zvec/core/framework/index_factory.h>
#if RABITQ_SUPPORTED
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include <rabitqlib/utils/cpu_features.hpp>
#endif

DEFINE_int32(dim, 768, "raw vector dimension");
DEFINE_int32(bits, 7, "RaBitQ total bits");
DEFINE_int32(clusters, 16, "model cluster count");
DEFINE_string(metric, "Cosine", "SquaredEuclidean|InnerProduct|Cosine");
DEFINE_int32(rows, 65536, "datapoint count (rounded up to a multiple of 32)");
DEFINE_int32(reps, 300, "timed repetitions per loop");
DEFINE_int32(warmup, 5, "untimed repetitions per loop");
DEFINE_int32(seed, 123, "random seed");

#if RABITQ_SUPPORTED
namespace zvec::turbo {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedNs(const Clock::time_point &a, const Clock::time_point &b) {
  return std::chrono::duration<double, std::nano>(b - a).count();
}

void Report(const char *name, double pass_ns, size_t blocks, double checksum) {
  const double rows_per_pass = static_cast<double>(blocks * 32);
  std::printf(
      "%-26s %10.3f ms/pass %8.2f ns/block(32) %7.3f ns/row %8.2f Mrow/s "
      "checksum=%.2f\n",
      name, pass_ns / 1e6, pass_ns / blocks, pass_ns / (blocks * 32),
      rows_per_pass / (pass_ns / 1000.0), checksum);
}

}  // namespace

int RunKernelBench() {
  const int dim = FLAGS_dim;
  const size_t padded = static_cast<size_t>((dim + 63) / 64) * 64;
  const size_t rows = static_cast<size_t>((FLAGS_rows + 31) / 32) * 32;
  const size_t blocks = rows / 32;
  const size_t clusters = static_cast<size_t>(FLAGS_clusters);
  const bool ip_metric = FLAGS_metric == "InnerProduct";
  const rabitqlib::MetricType mtype =
      FLAGS_metric == "SquaredEuclidean" ? rabitqlib::METRIC_L2
                                         : rabitqlib::METRIC_IP;

  const rabitqlib::cpu::Features &feat = rabitqlib::cpu::features();
  std::printf(
      "config: dim=%d padded=%zu bits=%d clusters=%d metric=%s rows=%zu "
      "blocks=%zu reps=%d warmup=%d seed=%d\n",
      dim, padded, FLAGS_bits, FLAGS_clusters, FLAGS_metric.c_str(), rows,
      blocks, FLAGS_reps, FLAGS_warmup, FLAGS_seed);
  std::printf(
      "cpu: avx2=%d fma=%d avx512f=%d avx512bw=%d avx512dq=%d "
      "avx512vpopcntdq=%d has_avx512_core=%d has_avx512_popcnt=%d\n",
      feat.avx2, feat.fma, feat.avx512f, feat.avx512bw, feat.avx512dq,
      feat.avx512vpopcntdq, rabitqlib::cpu::has_avx512_core(),
      rabitqlib::cpu::has_avx512_popcnt());
  // kAccumulateHaccFn has internal linkage in rabitqlib's dispatch.cpp
  // (namespace-scope const without extern), so the selected branch
  // cannot be read here. The printed CPU inputs drive the same
  // static-init decision; the pointer value itself is verified
  // out-of-band (gdb) in the v9 record.
  const char *dispatch =
      rabitqlib::cpu::has_avx512_core()
          ? "AVX512 (has_avx512_core)"
          : rabitqlib::cpu::has_avx2() ? "AVX2 (has_avx2)" : "FALLBACK";
  std::printf("dispatch: accumulate_hacc -> %s\n", dispatch);
  IndexMeta meta(IndexMeta::DT_FP32, dim);
  meta.set_metric(FLAGS_metric.c_str(), 0, ailego::Params{});
  ailego::Params params;
  params.set(RABITQ_TOTAL_BITS, FLAGS_bits);
  params.set(RABITQ_NUM_CLUSTERS, FLAGS_clusters);
  auto quantizer = IndexFactory::CreateQuantizer("RabitqQuantizer");
  if (!quantizer || quantizer->init(meta, params) != 0) {
    std::fprintf(stderr, "quantizer init failed\n");
    return 1;
  }

  std::mt19937 rng(FLAGS_seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> centers(clusters * dim);
  for (float &v : centers) v = gauss(rng) * 2.0f;
  const auto sample = [&](size_t c, float scale, float *out) {
    for (int i = 0; i < dim; ++i)
      out[i] = centers[c * dim + i] + gauss(rng) * scale;
  };

  // Cluster-structured training data keeps the model centroids apart.
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  {
    std::vector<float> buf(dim);
    ailego::NumericalVector<float> row(dim);
    uint64_t id = 0;
    for (size_t c = 0; c < clusters; ++c) {
      for (int k = 0; k < 256; ++k) {
        sample(c, 0.3f, buf.data());
        std::copy(buf.begin(), buf.end(), row.begin());
        holder->emplace(id++, row);
      }
    }
    if (quantizer->train(holder) != 0) {
      std::fprintf(stderr, "quantizer train failed\n");
      return 1;
    }
  }

  // Encode datapoints; each 32-row block is sampled around one cluster so
  // some blocks share a single model centroid (like a dedicated IVF list).
  const size_t dp_len = quantizer->quantized_datapoint_vector_length();
  std::vector<char> encoded(rows * dp_len);
  {
    std::vector<float> buf(dim);
    for (size_t r = 0; r < rows; ++r) {
      sample((r / 32) % clusters, 0.3f, buf.data());
      quantizer->quantize_data(buf.data(), encoded.data() + r * dp_len);
    }
  }
  const size_t block_len = quantizer->scan_block_size();
  std::vector<char> block_data(blocks * block_len);
  for (size_t b = 0; b < blocks; ++b) {
    if (quantizer->pack_scan_block(encoded.data() + b * 32 * dp_len, 32,
                                   dp_len,
                                   block_data.data() + b * block_len) != 0) {
      std::fprintf(stderr, "pack_scan_block failed at block %zu\n", b);
      return 1;
    }
  }
  std::printf("layout: dp_len=%zu block_len=%zu query_len=%zu\n", dp_len,
              block_len, quantizer->quantized_query_vector_length());

  std::vector<float> qbuf(dim);
  sample(0, 0.3f, qbuf.data());
  IndexQueryMeta input(IndexMeta::DT_FP32, dim), qmeta;
  std::string query;
  if (quantizer->quantize(qbuf.data(), input, &query, &qmeta) != 0 ||
      query.size() != quantizer->quantized_query_vector_length()) {
    std::fprintf(stderr, "quantize query failed\n");
    return 1;
  }
  auto distance = quantizer->distance(query.data(), qmeta);
  auto scanner = quantizer->block_scanner(distance);
  if (!scanner) {
    std::fprintf(stderr, "block_scanner unavailable\n");
    return 1;
  }

  // Dedicated-side query state over the same rotated-query bytes.
  rabitqlib::SplitBatchQuery<float> bq(
      reinterpret_cast<const float *>(query.data()), padded,
      static_cast<size_t>(FLAGS_bits - 1), mtype, true);
  // Turbo appends per-cluster g_add/g_error after the transposed query codes.
  const float *scalars = reinterpret_cast<const float *>(
      query.data() + padded * sizeof(float) + padded / 2);
  const float *g_add = scalars + 4;
  const float *g_error = g_add + clusters;

  // Cross-check the two estimators on uniform-cluster blocks only, where
  // the dedicated single-g formulation is directly comparable.
  DistanceEstimate turbo_out[32];
  float est[32], low[32], dot[32];
  size_t checked = 0;
  double max_diff = 0.0;
  for (size_t b = 0; b < blocks && checked < 64; ++b) {
    const char *blk = block_data.data() + b * block_len;
    uint32_t c0;
    std::memcpy(&c0, blk, sizeof(c0));
    if (c0 >= clusters) continue;
    bool uniform = true;
    for (int i = 1; i < 32; ++i) {
      uint32_t ci;
      std::memcpy(&ci, blk + i * sizeof(ci), sizeof(ci));
      uniform = uniform && ci == c0;
    }
    if (!uniform) continue;
    if (mtype == rabitqlib::METRIC_L2) {
      bq.set_g_add(g_error[c0]);
    } else {
      bq.set_g_add(g_error[c0], -g_add[c0]);
    }
    if (scanner->estimate(blk, 32, turbo_out) != 0) continue;
    rabitqlib::split_batch_estdist(blk + 32 * sizeof(uint32_t), bq, padded,
                                  est, low, dot, true);
    ++checked;
    for (int i = 0; i < 32; ++i) {
      // Turbo's InnerProduct estimator subtracts 1; the library does not.
      const double expected =
          static_cast<double>(turbo_out[i].distance) + (ip_metric ? 1.0 : 0.0);
      max_diff = std::max(max_diff, std::fabs(expected - est[i]));
    }
  }
  std::printf("crosscheck: uniform blocks compared=%zu max|turbo-dedicated|=%.3g\n",
              checked, max_diff);

  // Fix the batch query's g values outside the timed loops (per-list work in
  // the dedicated index; values do not affect kernel cost).
  if (mtype == rabitqlib::METRIC_L2) {
    bq.set_g_add(g_error[0]);
  } else {
    bq.set_g_add(g_error[0], -g_add[0]);
  }

  double turbo_ns = 0, ded_ns = 0, kern_ns = 0;

  // (a) turbo whole-scanner estimate.
  {
    double checksum = 0;
    for (int w = 0; w < FLAGS_warmup; ++w)
      for (size_t b = 0; b < blocks; ++b)
        scanner->estimate(block_data.data() + b * block_len, 32, turbo_out);
    const auto t0 = Clock::now();
    for (int r = 0; r < FLAGS_reps; ++r) {
      for (size_t b = 0; b < blocks; ++b) {
        scanner->estimate(block_data.data() + b * block_len, 32, turbo_out);
        checksum += turbo_out[0].distance + turbo_out[31].distance;
      }
    }
    turbo_ns = ElapsedNs(t0, Clock::now()) / FLAGS_reps;
    Report("turbo scanner.estimate", turbo_ns, blocks, checksum);
  }

  // (b) dedicated whole split_batch_estdist.
  {
    double checksum = 0;
    for (int w = 0; w < FLAGS_warmup; ++w)
      for (size_t b = 0; b < blocks; ++b)
        rabitqlib::split_batch_estdist(
            block_data.data() + b * block_len + 32 * sizeof(uint32_t), bq,
            padded, est, low, dot, true);
    const auto t0 = Clock::now();
    for (int r = 0; r < FLAGS_reps; ++r) {
      for (size_t b = 0; b < blocks; ++b) {
        rabitqlib::split_batch_estdist(
            block_data.data() + b * block_len + 32 * sizeof(uint32_t), bq,
            padded, est, low, dot, true);
        checksum += est[0] + est[31];
      }
    }
    ded_ns = ElapsedNs(t0, Clock::now()) / FLAGS_reps;
    Report("ded split_batch_estdist", ded_ns, blocks, checksum);
  }

  // (c) dispatched accumulate_hacc kernel alone.
  {
    std::array<int32_t, 32> sums{};
    double checksum = 0;
    for (int w = 0; w < FLAGS_warmup; ++w)
      for (size_t b = 0; b < blocks; ++b)
        rabitqlib::fastscan::accumulate_hacc(
            rabitqlib::ConstBatchDataMap<float>(
                block_data.data() + b * block_len + 32 * sizeof(uint32_t),
                padded)
                .bin_code(),
            bq.lut(), sums.data(), padded);
    const auto t0 = Clock::now();
    for (int r = 0; r < FLAGS_reps; ++r) {
      for (size_t b = 0; b < blocks; ++b) {
        rabitqlib::fastscan::accumulate_hacc(
            rabitqlib::ConstBatchDataMap<float>(
                block_data.data() + b * block_len + 32 * sizeof(uint32_t),
                padded)
                .bin_code(),
            bq.lut(), sums.data(), padded);
        checksum += sums[0] + sums[31];
      }
    }
    kern_ns = ElapsedNs(t0, Clock::now()) / FLAGS_reps;
    Report("kernel accumulate_hacc", kern_ns, blocks, checksum);
  }

  std::printf("kernel share of whole estimate: turbo %.1f%% dedicated %.1f%%\n",
              100.0 * kern_ns / turbo_ns, 100.0 * kern_ns / ded_ns);
  return 0;
}

}  // namespace zvec::turbo
#endif

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
#if RABITQ_SUPPORTED
  return zvec::turbo::RunKernelBench();
#else
  std::fprintf(stderr, "RABITQ_SUPPORTED is off; rebuild with RaBitQ enabled\n");
  return 1;
#endif
}
