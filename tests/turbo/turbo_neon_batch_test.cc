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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include "distance/neon/fp16/cosine.h"
#include "distance/neon/fp16/inner_product.h"
#include "distance/neon/fp16/squared_euclidean.h"
#include "distance/neon/fp32/cosine.h"
#include "distance/neon/fp32/inner_product.h"
#include "distance/neon/fp32/squared_euclidean.h"
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using Batch = void (*)(const void *const *, const void *, size_t, size_t,
                       float *, const void *const *);
enum class Metric { InnerProduct, Cosine, SquaredL2 };

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}

template <class T>
T store(float value) {
  if constexpr (std::is_same_v<T, float>)
    return value;
  else
    return zvec::ailego::FloatHelper::ToFP16(value);
}
template <class T>
float load(T value) {
  if constexpr (std::is_same_v<T, float>)
    return value;
  else
    return zvec::ailego::FloatHelper::ToFP32(value);
}
template <class T>
Batch kernel(Metric metric) {
  using namespace zvec::turbo::neon;
  if constexpr (std::is_same_v<T, float>) {
    if (metric == Metric::InnerProduct)
      return inner_product_fp32_batch_distance;
    if (metric == Metric::Cosine) return cosine_fp32_batch_distance;
    return squared_euclidean_fp32_batch_distance;
  } else {
    if (metric == Metric::InnerProduct)
      return inner_product_fp16_batch_distance;
    if (metric == Metric::Cosine) return cosine_fp16_batch_distance;
    return squared_euclidean_fp16_batch_distance;
  }
}

struct BatchCase {
  Metric metric;
  const char *label;
  Batch run;
};
template <class T>
std::vector<BatchCase> kernel_cases() {
  std::vector<BatchCase> result;
  for (Metric metric :
       {Metric::InnerProduct, Metric::Cosine, Metric::SquaredL2})
    result.push_back({metric, "production", kernel<T>(metric)});

  return result;
}

// Generate actual stored inputs, including identical, zero, opposite and
// cancellation-rich rows. Cosine's native contract is exactly 1 - dot; it does
// not renormalize inputs or read an extra field.
template <class T>
void fill(T *query, const std::vector<T *> &rows, size_t dim) {
  uint32_t state = 0x52bcef19;
  auto next = [&] {
    state = state * 1664525u + 1013904223u;
    return (static_cast<int>((state >> 12) % 2049) - 1024) / 997.f;
  };
  for (size_t d = 0; d < dim; ++d) query[d] = store<T>(next());
  for (size_t r = 0; r < rows.size(); ++r)
    for (size_t d = 0; d < dim; ++d) {
      const float value = r % 7 == 0   ? load(query[d])
                          : r % 7 == 1 ? 0.f
                          : r % 7 == 2 ? -load(query[d])
                                       : next();
      rows[r][d] = store<T>(value);
    }
}

template <class T>
size_t check(const T *query, const std::vector<T *> &rows, size_t dim,
             const std::string &layout) {
  const size_t n = rows.size();
  std::vector<const void *> pointers(rows.begin(), rows.end());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto cases = kernel_cases<T>();
  for (const auto &test : cases) {
    const Metric metric = test.metric;
    std::vector<float> output(n + 2, nan);
    // A zero-row call must not read query or the pointer table.
    test.run(n ? pointers.data() : nullptr, n ? query : nullptr, n, dim,
             output.data() + 1, nullptr);
    const std::string label =
        (std::is_same_v<T, float> ? "fp32 " : "fp16 ") + layout +
        " n=" + std::to_string(n) + " dim=" + std::to_string(dim) +
        " metric=" + std::to_string(static_cast<int>(metric)) +
        " variant=" + test.label;
    require(std::isnan(output.front()) && std::isnan(output.back()),
            label + " overwrote an output canary");
    for (size_t r = 0; r < n; ++r) {
      double dot = 0, squared_l2 = 0, absolute_products = 0;
      for (size_t d = 0; d < dim; ++d) {
        const double a = load(rows[r][d]), q = load(query[d]);
        dot += a * q;
        absolute_products += std::abs(a * q);
        squared_l2 += (a - q) * (a - q);
      }
      const double expected = metric == Metric::SquaredL2 ? squared_l2
                              : metric == Metric::Cosine  ? 1 - dot
                                                          : -dot;
      const double magnitude =
          metric == Metric::SquaredL2 ? squared_l2 : absolute_products;
      // Bound error by the magnitudes accumulated, including cancellation.
      // The FP16 oracle uses decoded stored values and still accumulates FP64.
      const double tolerance = 2e-6 + 3e-6 * magnitude;
      require(std::isfinite(output[r + 1]) &&
                  std::abs(output[r + 1] - expected) <= tolerance,
              label + " mismatch at row " + std::to_string(r) +
                  ": actual=" + std::to_string(output[r + 1]) +
                  " expected=" + std::to_string(expected));
    }
  }
  return cases.size();
}

template <class T>
size_t test_padded_and_disjoint() {
  size_t checks = 0;
  const T nan = store<T>(std::numeric_limits<float>::quiet_NaN());
  for (size_t n : {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33})
    for (size_t dim :
         {0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 63, 65, 127, 256, 257, 1024}) {
      // Offset by one element: legal scalar alignment, no SIMD alignment
      // guarantee. NaN padding exposes accidentally included tail lanes.
      std::vector<T> query(dim + 17, nan);
      const size_t stride = dim + 5;
      std::vector<T> matrix(n * stride + 17, nan);
      std::vector<T *> rows(n);
      for (size_t r = 0; r < n; ++r) rows[r] = matrix.data() + 1 + r * stride;
      fill(query.data() + 1, rows, dim);
      checks += check(query.data() + 1, rows, dim, "padded");

      // Separate allocations and reverse pointer order. Direct batch kernels
      // accept arbitrary pointers without a common-stride requirement.
      std::vector<std::vector<T>> separate(n);
      for (size_t r = 0; r < n; ++r) {
        separate[r].assign(dim + 17 + r % 5, nan);
        rows[n - 1 - r] = separate[r].data() + 1 + r % 5;
      }
      fill(query.data() + 1, rows, dim);
      checks += check(query.data() + 1, rows, dim, "disjoint");
    }
  return checks;
}

#if defined(__unix__) || defined(__APPLE__)
// Put the last logical element immediately before an inaccessible page, so a
// vector tail overread fails even when the unused lanes do not affect output.
template <class T>
class GuardedBuffer {
 public:
  explicit GuardedBuffer(size_t elements) {
    const long page_size = sysconf(_SC_PAGESIZE);
    require(page_size > 0, "cannot determine page size");
    const size_t page = static_cast<size_t>(page_size);
    const size_t data_bytes = ((elements * sizeof(T) + page - 1) / page) * page;
    bytes_ = data_bytes + page;
    memory_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    require(memory_ != MAP_FAILED, "guard-page mmap failed");
    auto *end = static_cast<unsigned char *>(memory_) + data_bytes;
    if (mprotect(end, page, PROT_NONE) != 0) {
      munmap(memory_, bytes_);
      throw std::runtime_error("guard-page mprotect failed");
    }
    data_ = reinterpret_cast<T *>(end - elements * sizeof(T));
  }
  ~GuardedBuffer() {
    munmap(memory_, bytes_);
  }
  GuardedBuffer(const GuardedBuffer &) = delete;
  GuardedBuffer &operator=(const GuardedBuffer &) = delete;
  T *data() {
    return data_;
  }

 private:
  void *memory_ = nullptr;
  size_t bytes_ = 0;
  T *data_ = nullptr;
};

template <class T>
size_t test_guard_pages() {
  size_t checks = 0;
  for (size_t n : {1, 3, 4, 7, 8, 9, 17, 33})
    for (size_t dim : {1, 3, 4, 7, 8, 9, 15, 16, 17, 65, 257}) {
      GuardedBuffer<T> query(dim);
      std::vector<std::unique_ptr<GuardedBuffer<T>>> storage;
      std::vector<T *> rows(n);
      for (size_t r = 0; r < n; ++r) {
        storage.emplace_back(std::make_unique<GuardedBuffer<T>>(dim));
        rows[n - 1 - r] = storage.back()->data();
      }
      fill(query.data(), rows, dim);
      checks += check(query.data(), rows, dim, "guard-page");
    }
  return checks;
}
#endif
}  // namespace

TEST(NeonBatch, Fp32PaddedAndDisjointRows) {
  EXPECT_NO_THROW(test_padded_and_disjoint<float>());
}

TEST(NeonBatch, Fp16PaddedAndDisjointRows) {
  EXPECT_NO_THROW(test_padded_and_disjoint<uint16_t>());
}

#if defined(__unix__) || defined(__APPLE__)
TEST(NeonBatch, Fp32GuardPages) {
  EXPECT_NO_THROW(test_guard_pages<float>());
}

TEST(NeonBatch, Fp16GuardPages) {
  EXPECT_NO_THROW(test_guard_pages<uint16_t>());
}
#endif

TEST(NeonBatch, EmptyInputs) {
  for (const auto &test : kernel_cases<float>()) {
    test.run(nullptr, nullptr, 0, 257, nullptr, nullptr);
    std::vector<const void *> rows(33, nullptr);
    std::vector<float> output(rows.size(), -100.f);
    test.run(rows.data(), nullptr, rows.size(), 0, output.data(), nullptr);
    for (float value : output) {
      EXPECT_EQ(value, test.metric == Metric::Cosine ? 1.f : 0.f);
    }
  }
  for (const auto &test : kernel_cases<uint16_t>()) {
    test.run(nullptr, nullptr, 0, 257, nullptr, nullptr);
    std::vector<const void *> rows(33, nullptr);
    std::vector<float> output(rows.size(), -100.f);
    test.run(rows.data(), nullptr, rows.size(), 0, output.data(), nullptr);
    for (float value : output) {
      EXPECT_EQ(value, test.metric == Metric::Cosine ? 1.f : 0.f);
    }
  }
}
