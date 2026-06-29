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
//
// Opt-in micro-benchmark for the core FP32 SquaredEuclidean distance kernel.
//
// Models a brute-force scan: for each query, the squared-Euclidean distance to
// every database vector is computed via
// zvec::ailego::Distance::SquaredEuclidean (which dispatches to AVX / AVX-512 /
// NEON). We report its throughput against a plain scalar reference across a
// range of dims, and self-validate that the SIMD result agrees with the scalar
// one before trusting any timing.
//
// Standalone executable (built under BUILD_TOOLS); intentionally NOT registered
// as a ctest -- micro-benchmark numbers are machine dependent. It exits
// non-zero only if the SIMD kernel disagrees with the scalar reference.
//
//   ./distance_bench --num_db=8192 --dims=1,2,3,7,16,31,64,128,257 --reps=50

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <ailego/math/distance.h>
#include <gflags/gflags.h>

DEFINE_int32(num_db, 8192, "database vectors scanned per pass");
DEFINE_int32(reps, 50, "timed repetitions per dim");
DEFINE_string(dims, "1,2,3,7,16,31,64,128,257",
              "comma-separated vector dims to benchmark");
DEFINE_double(tolerance, 1e-4,
              "max relative diff allowed between SIMD and scalar");

namespace {

// Plain scalar reference -- the ground truth the SIMD kernel must match.
float ScalarSquaredEuclidean(const float *a, const float *b, size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

std::vector<int> ParseDims(const std::string &spec) {
  std::vector<int> dims;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      dims.push_back(std::stoi(item));
    }
  }
  return dims;
}

}  // namespace

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage(
      "Micro-benchmark: FP32 SquaredEuclidean SIMD vs scalar reference");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::mt19937 rng(12345);  // fixed seed: reproducible inputs across runs
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  bool ok = true;
  std::printf("%6s %12s %12s %9s\n", "dim", "scalar(ms)", "simd(ms)",
              "speedup");
  for (int dim : ParseDims(FLAGS_dims)) {
    const size_t n = static_cast<size_t>(FLAGS_num_db);
    std::vector<float> db(n * dim);
    std::vector<float> query(dim);
    for (auto &v : db) v = dist(rng);
    for (auto &v : query) v = dist(rng);

    // Correctness gate: every SIMD result must match the scalar reference.
    double max_rel = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const float *dbi = db.data() + i * dim;
      float ref = ScalarSquaredEuclidean(dbi, query.data(), dim);
      float got =
          zvec::ailego::Distance::SquaredEuclidean(dbi, query.data(), dim);
      double denom = std::max(1e-6f, std::fabs(ref));
      max_rel = std::max(max_rel, std::fabs(got - ref) / denom);
    }
    if (max_rel > FLAGS_tolerance) {
      std::printf("dim %d: VALIDATION FAILED (max relative diff %.3e > %.1e)\n",
                  dim, max_rel, FLAGS_tolerance);
      ok = false;
      continue;
    }

    volatile float sink = 0.0f;  // keep the loops from being optimized away

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < FLAGS_reps; ++r)
      for (size_t i = 0; i < n; ++i)
        sink += ScalarSquaredEuclidean(db.data() + i * dim, query.data(), dim);
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < FLAGS_reps; ++r)
      for (size_t i = 0; i < n; ++i)
        sink += zvec::ailego::Distance::SquaredEuclidean(db.data() + i * dim,
                                                         query.data(), dim);
    auto t2 = std::chrono::steady_clock::now();
    (void)sink;

    double scalar_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double simd_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double speedup = simd_ms > 0.0 ? scalar_ms / simd_ms : 0.0;
    std::printf("%6d %12.3f %12.3f %8.2fx\n", dim, scalar_ms, simd_ms, speedup);
  }

  gflags::ShutDownCommandLineFlags();
  return ok ? 0 : 1;
}
