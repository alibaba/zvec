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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/internal/platform.h>
#include "distance/neon/fp32/cosine.h"
#include "distance/neon/fp32/inner_product.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace zvec::turbo {
namespace {

constexpr float kSentinel = 1234567.0f;
constexpr std::array<size_t, 22> kBatchSizes = {
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 49, 50,
    63, 64, 65, 100, 101};

std::vector<size_t> TestDimensions() {
  std::vector<size_t> dimensions;
  for (size_t dim = 0; dim <= 65; ++dim) {
    dimensions.push_back(dim);
  }
  for (size_t center : {128, 256, 512, 1024, 1536}) {
    dimensions.push_back(center - 1);
    dimensions.push_back(center);
    dimensions.push_back(center + 1);
  }
  return dimensions;
}

struct DotOracle {
  double dot = 0.0;
  double absolute_products = 0.0;
};

DotOracle ReferenceDot(const float *a, const float *b, size_t dim) {
  DotOracle result;
  for (size_t i = 0; i < dim; ++i) {
    const double product = static_cast<double>(a[i]) * b[i];
    result.dot += product;
    result.absolute_products += std::abs(product);
  }
  return result;
}

// Reassociation and FMA change the final rounding. Scale the allowance by the
// sum of absolute products, so cancellation does not create a false failure.
// The bound also covers the scalar fallback's longer accumulation chain.
double ErrorBound(const DotOracle &reference, size_t dim) {
  const double epsilon = std::numeric_limits<float>::epsilon();
  const double gamma = (dim + 4) * epsilon / (1.0 - (dim + 4) * epsilon);
  return gamma * reference.absolute_products + 4 * epsilon;
}

void ExpectDistances(const float *a, const float *b, size_t dim) {
  const auto reference = ReferenceDot(a, b, dim);
  std::array<float, 3> result = {kSentinel, kSentinel, kSentinel};
  neon::inner_product_fp32_distance(a, b, dim, &result[1]);
  EXPECT_NEAR(-reference.dot, result[1], ErrorBound(reference, dim));
  EXPECT_EQ(kSentinel, result.front());
  EXPECT_EQ(kSentinel, result.back());
  neon::cosine_fp32_distance(a, b, dim, &result[1]);
  EXPECT_NEAR(1.0 - reference.dot, result[1], ErrorBound(reference, dim));
  EXPECT_EQ(kSentinel, result.front());
  EXPECT_EQ(kSentinel, result.back());
}

TEST(Fp32NeonKernel, SingleMatchesDoubleOracleAtAllRemainders) {
  std::mt19937 generator(0x98ab13);
  std::uniform_real_distribution<float> random(-2.0f, 2.0f);
  for (size_t dim : TestDimensions()) {
    SCOPED_TRACE(dim);
    // Offset from a known 16-byte boundary to exercise unaligned NEON loads
    // while retaining the float alignment required by the API.
    std::vector<float> a_storage(dim + 8);
    std::vector<float> b_storage(dim + 8);
    const auto offset = [](const float *ptr) {
      return (16 - reinterpret_cast<uintptr_t>(ptr) % 16) / sizeof(float) + 1;
    };
    float *a = a_storage.data() + offset(a_storage.data());
    float *b = b_storage.data() + offset(b_storage.data());
    ASSERT_EQ(4u, reinterpret_cast<uintptr_t>(a) % 16);
    ASSERT_EQ(4u, reinterpret_cast<uintptr_t>(b) % 16);
    for (size_t trial = 0; trial < 4; ++trial) {
      for (size_t i = 0; i < dim; ++i) {
        a[i] = random(generator);
        b[i] = random(generator);
      }
      ExpectDistances(a, b, dim);
    }
    std::fill(a, a + dim, 0.0f);
    ExpectDistances(a, b, dim);
    std::fill(a, a + dim, -0.5f);
    std::fill(b, b + dim, 0.25f);
    ExpectDistances(a, b, dim);
    // Large exactly representable products cancel across adjacent lanes.
    for (size_t i = 0; i < dim; ++i) {
      a[i] = i % 4 < 2 ? 1024.0f : 0.03125f;
      b[i] = i % 2 ? -1024.0f : 1024.0f;
    }
    ExpectDistances(a, b, dim);
  }
}

TEST(Fp32NeonKernel, BatchMatchesDoubleOracleAndWritesOnlyRequestedOutputs) {
  std::mt19937 generator(0x471230);
  std::uniform_real_distribution<float> random(-1.0f, 1.0f);
  for (size_t dim : TestDimensions()) {
    SCOPED_TRACE(dim);
    std::vector<float> query(dim + 1);
    for (float &value : query) {
      value = random(generator);
    }
    std::vector<std::vector<float>> storage(kBatchSizes.back(),
                                            std::vector<float>(dim + 1));
    std::vector<const void *> vectors;
    for (auto &vector : storage) {
      for (float &value : vector) {
        value = random(generator);
      }
      vectors.push_back(vector.data() + 1);
    }
    // Include zero and negative vectors in the same four-vector batch.
    std::fill(storage[0].begin(), storage[0].end(), 0.0f);
    for (size_t i = 0; i < query.size(); ++i) {
      storage[1][i] = -query[i];
    }
    for (size_t n : kBatchSizes) {
      SCOPED_TRACE(n);
      std::vector<float> distances(kBatchSizes.back() + 2, kSentinel);
      neon::inner_product_fp32_batch_distance(
          vectors.data(), query.data() + 1, n, dim, distances.data() + 1,
          nullptr);
      for (size_t i = 0; i < n; ++i) {
        const auto reference = ReferenceDot(
            static_cast<const float *>(vectors[i]), query.data() + 1, dim);
        EXPECT_NEAR(-reference.dot, distances[i + 1], ErrorBound(reference, dim));
      }
      EXPECT_EQ(kSentinel, distances.front());
      for (size_t i = n + 1; i < distances.size(); ++i) {
        EXPECT_EQ(kSentinel, distances[i]);
      }
      neon::cosine_fp32_batch_distance(vectors.data(), query.data() + 1, n,
                                      dim, distances.data() + 1, nullptr);
      for (size_t i = 0; i < n; ++i) {
        const auto reference = ReferenceDot(
            static_cast<const float *>(vectors[i]), query.data() + 1, dim);
        EXPECT_NEAR(1.0 - reference.dot, distances[i + 1],
                    ErrorBound(reference, dim));
      }
      EXPECT_EQ(kSentinel, distances.front());
      for (size_t i = n + 1; i < distances.size(); ++i) {
        EXPECT_EQ(kSentinel, distances[i]);
      }
    }
  }
}

TEST(Fp32NeonKernel, EmptyBatchDoesNotDereferencePointers) {
  neon::inner_product_fp32_batch_distance(nullptr, nullptr, 0, 256, nullptr,
                                         nullptr);
  neon::cosine_fp32_batch_distance(nullptr, nullptr, 0, 256, nullptr, nullptr);
}

TEST(Fp32NeonKernel, BatchCancellationAndEveryFloatAlignmentMatchDoubleOracle) {
  constexpr size_t count = 17;
  for (size_t dim : TestDimensions()) {
    SCOPED_TRACE(dim);
    for (size_t query_offset = 0; query_offset < 4; ++query_offset) {
      SCOPED_TRACE(query_offset);
      std::vector<float> query_storage(dim + 4);
      float *query = query_storage.data() + query_offset;
      std::array<std::vector<float>, count> storage;
      std::array<const void *, count> vectors;
      for (size_t d = 0; d < dim; ++d) {
        query[d] = d % 2 ? -1024.0f : 1024.0f;
      }
      for (size_t i = 0; i < count; ++i) {
        storage[i].resize(dim + 4);
        float *vector = storage[i].data() + (i % 4);
        vectors[i] = vector;
        for (size_t d = 0; d < dim; ++d) {
          vector[d] = d % 4 < 2 ? 1024.0f : 0.03125f;
        }
      }
      for (size_t n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}) {
        SCOPED_TRACE(n);
        std::array<float, count + 2> distances;
        distances.fill(kSentinel);
        neon::inner_product_fp32_batch_distance(
            vectors.data(), query, n, dim, distances.data() + 1, nullptr);
        for (size_t i = 0; i < n; ++i) {
          const auto oracle = ReferenceDot(
              static_cast<const float *>(vectors[i]), query, dim);
          EXPECT_NEAR(-oracle.dot, distances[i + 1], ErrorBound(oracle, dim));
        }
        EXPECT_EQ(kSentinel, distances.front());
        for (size_t i = n + 1; i < distances.size(); ++i) {
          EXPECT_EQ(kSentinel, distances[i]);
        }
      }
    }
  }
}

#if defined(__unix__) || defined(__APPLE__)

// Put the final element immediately before an inaccessible page. This detects
// overreads that ordinary std::vector capacity and ASan padding can hide.
template <typename T>
class GuardedBuffer {
 public:
  explicit GuardedBuffer(size_t count) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }
    const size_t page = static_cast<size_t>(page_size);
    const size_t writable = ((count * sizeof(T) + page - 1) / page + 1) * page;
    length_ = writable + page;
    mapping_ = mmap(nullptr, length_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_ == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }
    auto *end = static_cast<char *>(mapping_) + writable;
    if (mprotect(end, page, PROT_NONE) != 0) {
      munmap(mapping_, length_);
      throw std::runtime_error("mprotect failed");
    }
    data_ = reinterpret_cast<T *>(end) - count;
  }

  ~GuardedBuffer() {
    munmap(mapping_, length_);
  }

  GuardedBuffer(const GuardedBuffer &) = delete;
  GuardedBuffer &operator=(const GuardedBuffer &) = delete;

  T *data() {
    return data_;
  }

 private:
  void *mapping_ = MAP_FAILED;
  size_t length_ = 0;
  T *data_ = nullptr;
};

TEST(Fp32NeonKernel, GuardPagesProtectInputsPointerArrayAndOutputs) {
  for (size_t dim : TestDimensions()) {
    SCOPED_TRACE(dim);
    GuardedBuffer<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      query.data()[j] = static_cast<float>(static_cast<int>(j % 11) - 5) / 16;
    }
    std::vector<std::unique_ptr<GuardedBuffer<float>>> storage;
    for (size_t i = 0; i < kBatchSizes.back(); ++i) {
      auto vector = std::make_unique<GuardedBuffer<float>>(dim);
      for (size_t j = 0; j < dim; ++j) {
        vector->data()[j] =
            static_cast<float>(static_cast<int>((j + i) % 13) - 6) / 16;
      }
      ExpectDistances(vector->data(), query.data(), dim);
      storage.push_back(std::move(vector));
    }
    for (size_t n : kBatchSizes) {
      SCOPED_TRACE(n);
      GuardedBuffer<const void *> vectors(n);
      GuardedBuffer<float> distances(n + 1);
      distances.data()[0] = kSentinel;
      for (size_t i = 0; i < n; ++i) {
        vectors.data()[i] = storage[i]->data();
      }
      neon::inner_product_fp32_batch_distance(
          vectors.data(), query.data(), n, dim, distances.data() + 1, nullptr);
      for (size_t i = 0; i < n; ++i) {
        const auto reference =
            ReferenceDot(storage[i]->data(), query.data(), dim);
        EXPECT_NEAR(-reference.dot, distances.data()[i + 1],
                    ErrorBound(reference, dim));
      }
      EXPECT_EQ(kSentinel, distances.data()[0]);
      neon::cosine_fp32_batch_distance(vectors.data(), query.data(), n, dim,
                                      distances.data() + 1, nullptr);
      for (size_t i = 0; i < n; ++i) {
        const auto reference =
            ReferenceDot(storage[i]->data(), query.data(), dim);
        EXPECT_NEAR(1.0 - reference.dot, distances.data()[i + 1],
                    ErrorBound(reference, dim));
      }
      EXPECT_EQ(kSentinel, distances.data()[0]);
    }
  }
}

#endif

}  // namespace
}  // namespace zvec::turbo
