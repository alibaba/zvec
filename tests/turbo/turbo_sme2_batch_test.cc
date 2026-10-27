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

// Copyright 2026-present the gemv_bench contributors
// SPDX-License-Identifier: Apache-2.0

#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <zvec/turbo/turbo.h>
#if ZVEC_TURBO_HAVE_SME2
#include "distance/sme2/fp32/cosine.h"
#include "distance/sme2/fp32/inner_product.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__APPLE__) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
#if ZVEC_TURBO_HAVE_SME2
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}

// SVCR's architectural encoding works with a baseline arm64 assembler, so the
// test itself needs no SME target flag. Only call after the runtime SME2 gate.
uint64_t svcr() {
#if defined(__APPLE__) && defined(__aarch64__)
  uint64_t state;
  __asm__ volatile("mrs %0, S3_3_C4_C2_2" : "=r"(state) : : "memory");
  return state;
#else
  return 0;
#endif
}

void invoke(const void *const *pointers, const void *query, size_t n,
            size_t dim, float *output, bool cosine = false) {
  const uint64_t before = svcr();
  require((before & 3) == 0,
          "test caller unexpectedly owns streaming/ZA state");
  auto kernel = cosine ? zvec::turbo::sme2::cosine_fp32_batch_distance
                       : zvec::turbo::sme2::inner_product_fp32_batch_distance;
  kernel(pointers, query, n, dim, output, nullptr);
  const uint64_t after = svcr();
  require((after & 3) == (before & 3),
          "SME2 batch failed to restore the normal non-streaming ABI");
#if defined(__APPLE__) && defined(__aarch64__)
  // Normal SIMD code must work after returning from the locally streaming
  // kernel. Volatile inputs keep this from becoming a constant expression.
  volatile float values[4] = {1.25f, -2.f, 4.f, 8.f};
  const float plain[4] = {values[0], values[1], values[2], values[3]};
  const auto doubled = vmulq_n_f32(vld1q_f32(plain), 2.f);
  require(vaddvq_f32(doubled) == 22.5f, "normal NEON failed after SME2 call");
#endif
}

void fill(float *query, const std::vector<float *> &rows, size_t dim) {
  uint32_t state = 0xdecafbad;
  auto next = [&] {
    state = state * 1664525u + 1013904223u;
    return (int((state >> 11) % 2049) - 1024) / 997.f;
  };
  for (size_t d = 0; d < dim; ++d) query[d] = next();
  for (size_t r = 0; r < rows.size(); ++r)
    for (size_t d = 0; d < dim; ++d) {
      float value = next();
      if (r % 7 == 0) value = query[d];
      if (r % 7 == 1) value = -query[d];
      if (r % 7 == 2) value = 0.f;
      if (r % 7 == 3) value = (d % 2 ? -1.f : 1.f) * 1024.f;
      if (r % 7 == 4) value = std::ldexp(value, int(d % 9) - 4);
      rows[r][d] = value;
    }
}

void check_output(const float *query, const std::vector<float *> &rows,
                  size_t dim, const float *output, const std::string &label,
                  bool cosine = false) {
  for (size_t r = 0; r < rows.size(); ++r) {
    double dot = 0, magnitude = 0;
    for (size_t d = 0; d < dim; ++d) {
      const double product = double(rows[r][d]) * double(query[d]);
      dot += product;
      magnitude += std::abs(product);
    }
    // The FP64 oracle is independent of the kernel's tile/reduction order.
    // Use absolute-product magnitude to give cancellation the same error
    // budget as other inputs, while keeping missing/duplicated lanes visible.
    const double tolerance = 2e-6 + 3e-6 * magnitude;
    require(std::isfinite(output[r]) &&
                std::abs(double(output[r]) + dot - (cosine ? 1.0 : 0.0)) <=
                    tolerance,
            label + " row=" + std::to_string(r) +
                " actual=" + std::to_string(output[r]) +
                " expected=" + std::to_string((cosine ? 1.0 : 0.0) - dot));
  }
}

size_t check_padded(const float *query, const std::vector<float *> &rows,
                    size_t dim, const std::string &layout) {
  const size_t n = rows.size();
  std::vector<const void *> pointers(rows.begin(), rows.end());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> output(n + 2, nan);
  const std::string label =
      layout + " n=" + std::to_string(n) + " dim=" + std::to_string(dim);
  // Repeated calls must not accumulate into stale ZA/output state.
  for (size_t repeat = 0; repeat < 4; ++repeat) {
    std::fill(output.begin(), output.end(), nan);
    invoke(n ? pointers.data() : nullptr, n && dim ? query : nullptr, n, dim,
           output.data() + 1, repeat % 2 != 0);
    require(std::isnan(output.front()) && std::isnan(output.back()),
            label + " overwrote output canary");
    check_output(query, rows, dim, output.data() + 1, label, repeat % 2 != 0);
  }
  return 4;
}

size_t test_padded_and_disjoint() {
  size_t calls = 0;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (size_t n :
       {0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128})
    for (size_t dim : {0,  1,  3,   4,   7,   15,  16,  17,  31,  32,   33,  63,
                       64, 65, 127, 128, 129, 255, 256, 257, 511, 1024, 1025})
      for (size_t offset : {0, 1, 3, 7}) {
        // Legal FP32 alignment with no extra SIMD alignment guarantee.
        // Poison padding detects lanes that are accidentally accumulated.
        std::vector<float> query(dim + 32, nan);
        const size_t stride = dim + 19;
        std::vector<float> matrix(n * stride + 32, nan);
        std::vector<float *> rows(n);
        for (size_t r = 0; r < n; ++r)
          rows[r] = matrix.data() + offset + r * stride;
        fill(query.data() + offset, rows, dim);
        calls += check_padded(query.data() + offset, rows, dim, "padded");

        // Independently allocated, variably offset rows invalidate any
        // hidden stride assumption. Reverse and rotate the address order.
        std::vector<std::vector<float>> separate(n);
        for (size_t r = 0; r < n; ++r) {
          separate[r].assign(dim + 32, nan);
          rows[n - 1 - r] = separate[r].data() + (offset + r) % 16;
        }
        if (n > 1) std::rotate(rows.begin(), rows.begin() + n / 3, rows.end());
        fill(query.data() + offset, rows, dim);
        calls += check_padded(query.data() + offset, rows, dim, "disjoint");
      }
  return calls;
}

#if defined(__unix__) || defined(__APPLE__)
template <class T>
class GuardedBuffer {
 public:
  GuardedBuffer(size_t elements, bool at_end) {
    const long page_size = sysconf(_SC_PAGESIZE);
    require(page_size > 0, "cannot determine page size");
    const size_t page = size_t(page_size);
    const size_t data_bytes =
        std::max(page, ((elements * sizeof(T) + page - 1) / page) * page);
    bytes_ = data_bytes + 2 * page;
    memory_ = mmap(nullptr, bytes_, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    require(memory_ != MAP_FAILED, "guard-page mmap failed");
    begin_ = static_cast<unsigned char *>(memory_) + page;
    data_bytes_ = data_bytes;
    if (mprotect(begin_, data_bytes, PROT_READ | PROT_WRITE) != 0) {
      munmap(memory_, bytes_);
      throw std::runtime_error("guard-page mprotect failed");
    }
    data_ = reinterpret_cast<T *>(
        at_end ? begin_ + data_bytes - elements * sizeof(T) : begin_);
  }
  ~GuardedBuffer() {
    munmap(memory_, bytes_);
  }
  GuardedBuffer(const GuardedBuffer &) = delete;
  GuardedBuffer &operator=(const GuardedBuffer &) = delete;
  T *data() {
    return data_;
  }
  void read_only() {
    require(mprotect(begin_, data_bytes_, PROT_READ) == 0,
            "guard-page input protection failed");
  }

 private:
  void *memory_ = nullptr;
  unsigned char *begin_ = nullptr;
  size_t data_bytes_ = 0;
  size_t bytes_ = 0;
  T *data_ = nullptr;
};

size_t test_guard_pages() {
  size_t calls = 0;
  for (size_t n : {1, 3, 4, 15, 16, 17, 31, 32, 33})
    for (size_t dim : {1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1024})
      for (bool at_end : {false, true}) {
        GuardedBuffer<float> query(dim, at_end), output(n, at_end);
        GuardedBuffer<const void *> pointers(n, at_end);
        std::vector<std::unique_ptr<GuardedBuffer<float>>> allocations;
        std::vector<float *> rows(n);
        for (size_t r = 0; r < n; ++r) {
          allocations.emplace_back(
              std::make_unique<GuardedBuffer<float>>(dim, at_end));
          rows[n - 1 - r] = allocations.back()->data();
        }
        fill(query.data(), rows, dim);
        for (size_t r = 0; r < n; ++r) pointers.data()[r] = rows[r];
        query.read_only();
        pointers.read_only();
        for (auto &allocation : allocations) allocation->read_only();
        for (bool cosine : {false, true}) {
          invoke(pointers.data(), query.data(), n, dim, output.data(), cosine);
          check_output(query.data(), rows, dim, output.data(),
                       "guard-page n=" + std::to_string(n) +
                           " dim=" + std::to_string(dim),
                       cosine);
          ++calls;
        }
      }
  return calls;
}
#endif

size_t test_zero_work() {
  // No-row calls must not touch query, rows or output, even at nonzero dim.
  invoke(nullptr, nullptr, 0, 0, nullptr);
  invoke(nullptr, nullptr, 0, 257, nullptr);
  // Empty dot products have no input payload and must still write all zeros.
  std::vector<const void *> rows(33, nullptr);
  std::vector<float> output(33, std::numeric_limits<float>::quiet_NaN());
  invoke(rows.data(), nullptr, rows.size(), 0, output.data());
  for (float value : output)
    require(value == 0.f, "zero dimension is not zero");
  invoke(nullptr, nullptr, 0, 257, nullptr, true);
  invoke(rows.data(), nullptr, rows.size(), 0, output.data(), true);
  for (float value : output) require(value == 1.f, "empty cosine is not one");
  return 5;
}
#endif  // ZVEC_TURBO_HAVE_SME2
}  // namespace

TEST(Sme2Batch, PaddedAndDisjointRows) {
#if ZVEC_TURBO_HAVE_SME2
  if (!zvec::ailego::internal::CpuFeatures::SME2()) {
    GTEST_SKIP() << "SME2 is unavailable on this CPU/OS";
  }
  EXPECT_NO_THROW(test_zero_work());
  EXPECT_NO_THROW(test_padded_and_disjoint());
#else
  GTEST_SKIP() << "SME2 kernel was not compiled";
#endif
}

TEST(Sme2Batch, GuardPages) {
#if ZVEC_TURBO_HAVE_SME2 && (defined(__unix__) || defined(__APPLE__))
  if (!zvec::ailego::internal::CpuFeatures::SME2()) {
    GTEST_SKIP() << "SME2 is unavailable on this CPU/OS";
  }
  EXPECT_NO_THROW(test_guard_pages());
#else
  GTEST_SKIP() << "SME2 guard-page checks are unavailable in this build";
#endif
}

TEST(Sme2Batch, PublicDispatchAndFallback) {
  using namespace zvec::turbo;
  const bool available =
      ZVEC_TURBO_HAVE_SME2 && zvec::ailego::internal::CpuFeatures::SME2();
  for (MetricType metric : {MetricType::kInnerProduct, MetricType::kCosine}) {
    const auto explicit_sme2 = get_distance_kernels(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kSME2);
    EXPECT_EQ(static_cast<bool>(explicit_sme2.batch), available);
    EXPECT_EQ(static_cast<bool>(explicit_sme2.dist), available);
    EXPECT_FALSE(explicit_sme2.preprocess);
    for (CpuArchType arch :
         {CpuArchType::kAuto, CpuArchType::kNEON, CpuArchType::kSME2}) {
      auto batch = get_batch_distance_func(metric, DataType::kFp32,
                                           QuantizeType::kFp32, arch);
      if (arch == CpuArchType::kAuto) ASSERT_TRUE(batch);
      if (!batch) continue;
      for (size_t count : {1, 16, 33, 128}) {
        for (size_t dim : {17, 64, 256, 257}) {
          std::vector<float> query(dim + 1);
          std::vector<std::vector<float>> rows(count);
          std::vector<const void *> pointers(count);
          std::vector<float> output(count);
          for (size_t d = 0; d < dim; ++d) {
            query[d + 1] = static_cast<float>(int(d % 23) - 11) / 17.f;
          }
          for (size_t r = 0; r < count; ++r) {
            rows[r].resize(dim + 1);
            for (size_t d = 0; d < dim; ++d) {
              rows[r][d + 1] =
                  static_cast<float>(int((d + r) % 31) - 15) / 13.f;
            }
            pointers[count - 1 - r] = rows[r].data() + 1;
          }
          batch(pointers.data(), query.data() + 1, count, dim, output.data(),
                nullptr);
          for (size_t r = 0; r < count; ++r) {
            double expected = metric == MetricType::kCosine ? 1.0 : 0.0;
            double magnitude = 0;
            const auto *row = static_cast<const float *>(pointers[r]);
            for (size_t d = 0; d < dim; ++d) {
              const double product = double(row[d]) * double(query[d + 1]);
              expected -= product;
              magnitude += std::abs(product);
            }
            EXPECT_NEAR(expected, output[r], 2e-6 + 3e-6 * magnitude)
                << "arch=" << int(arch) << " n=" << count << " dim=" << dim;
          }
        }
      }
    }
  }
  // Unsupported metric/type combinations retain their established backends.
  EXPECT_FALSE(get_batch_distance_func(MetricType::kSquaredEuclidean,
                                       DataType::kFp32, QuantizeType::kFp32,
                                       CpuArchType::kSME2));
  EXPECT_FALSE(get_batch_distance_func(MetricType::kInnerProduct,
                                       DataType::kFp16, QuantizeType::kFp16,
                                       CpuArchType::kSME2));
  EXPECT_TRUE(get_batch_distance_func(MetricType::kSquaredEuclidean,
                                      DataType::kFp32, QuantizeType::kFp32));
}

TEST(Sme2Batch, RuntimeDisabledUsesExistingBackend) {
  using namespace zvec::turbo;
  auto &flags = zvec::ailego::internal::CpuFeatures::static_flags_;
  struct RestoreFlag {
    bool &flag;
    bool saved;
    ~RestoreFlag() {
      flag = saved;
    }
  } restore{flags.SME2, flags.SME2};
  // Disable only: a test must never invent hardware support on an older CPU.
  flags.SME2 = false;
  using RawBatchFn = void (*)(const void *const *, const void *, size_t, size_t,
                              float *, const void *const *);
  for (MetricType metric : {MetricType::kInnerProduct, MetricType::kCosine}) {
    const auto unavailable = get_distance_kernels(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kSME2);
    EXPECT_FALSE(unavailable.batch);
    EXPECT_FALSE(unavailable.dist);
    const auto automatic = get_distance_kernels(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kAuto);
    ASSERT_TRUE(automatic.batch);
    ASSERT_TRUE(automatic.dist);
    if (flags.NEON) {
      const auto neon = get_distance_kernels(
          metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kNEON);
      ASSERT_TRUE(neon.batch);
      const auto *actual = automatic.batch.target<RawBatchFn>();
      const auto *expected = neon.batch.target<RawBatchFn>();
      ASSERT_NE(actual, nullptr);
      ASSERT_NE(expected, nullptr);
      EXPECT_EQ(*actual, *expected);
    }
    // Exercise the fallback after disabling SME2, not only its lookup.
    const float query[4] = {1.f, 2.f, 3.f, 4.f};
    const void *rows[1] = {query};
    float output = 0.f;
    automatic.batch(rows, query, 1, 4, &output, nullptr);
    EXPECT_FLOAT_EQ(output, metric == MetricType::kCosine ? -29.f : -30.f);
  }
}
