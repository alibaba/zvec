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
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/internal/platform.h>
#include "distance/neon/fp32/inner_product.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace zvec::turbo {
namespace {

constexpr float kGuard = 9876.0f;
constexpr size_t kCacheLineSize = AILEGO_CACHE_LINE_SIZE;
constexpr size_t kFloatsPerCacheLine = kCacheLineSize / sizeof(float);
constexpr std::array<size_t, 23> kBatchSizes = {0,  1,  2,  3,  4,  5,   7,  8,
                                                9,  15, 16, 17, 23, 24,  25, 31,
                                                32, 33, 49, 50, 65, 100, 101};

float ReferenceInnerProduct(const void *data, const float *query, size_t dim) {
  const auto *vector = static_cast<const float *>(data);
  double dot = 0.0;
  for (size_t d = 0; d < dim; ++d) {
    dot += static_cast<double>(vector[d]) * query[d];
  }
  return static_cast<float>(-dot);
}

float *AlignedData(std::vector<float> &storage, size_t offset_floats) {
  const auto address = reinterpret_cast<uintptr_t>(storage.data());
  const size_t padding = (kCacheLineSize - address % kCacheLineSize) %
                         kCacheLineSize / sizeof(float);
  return storage.data() + padding + offset_floats;
}

void CheckBatch(size_t dim, size_t count, size_t offset_floats) {
  SCOPED_TRACE(testing::Message()
               << "dim=" << dim << ", count=" << count
               << ", alignment=" << offset_floats * sizeof(float));
  std::vector<float> query_storage(dim + 2 * kFloatsPerCacheLine, kGuard);
  float *query =
      AlignedData(query_storage, (offset_floats + 13) % kFloatsPerCacheLine);
  for (size_t d = 0; d < dim; ++d) {
    query[d] = static_cast<float>(static_cast<int>(d % 13) - 6) / 16.0f;
  }
  std::vector<std::vector<float>> storage(
      count, std::vector<float>(dim + 2 * kFloatsPerCacheLine, kGuard));
  std::vector<const void *> vectors(count);
  for (size_t i = 0; i < count; ++i) {
    float *vector =
        AlignedData(storage[i], (offset_floats + i * 7) % kFloatsPerCacheLine);
    vectors[i] = vector;
    for (size_t d = 0; d < dim; ++d) {
      vector[d] =
          static_cast<float>(static_cast<int>((d + i * 3) % 17) - 8) / 32.0f;
    }
  }

  std::vector<float> actual(count + 2, kGuard);
  neon::inner_product_fp32_batch_distance(vectors.data(), query, count, dim,
                                          actual.data() + 1, nullptr);
  EXPECT_EQ(kGuard, actual.front());
  EXPECT_EQ(kGuard, actual.back());
  for (size_t i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(ReferenceInnerProduct(vectors[i], query, dim),
                    actual[i + 1])
        << "index=" << i;
  }
}

TEST(Fp32PrefetchKernel, EmptyBatchDoesNotAccessPointers) {
  for (size_t dim : {0, 255, 256, 257}) {
    neon::inner_product_fp32_batch_distance(nullptr, nullptr, 0, dim, nullptr,
                                            nullptr);
  }
}

TEST(Fp32PrefetchKernel, DimensionsAndBatchTailsMatchReference) {
  for (size_t dim : {0, 1, 2, 3, 4, 7, 8, 31, 32, 33, 255, 256, 257, 512}) {
    for (size_t count : kBatchSizes) {
      CheckBatch(dim, count, 1);
    }
  }
}

TEST(Fp32PrefetchKernel, EveryFloatAlignmentAcrossCacheLineMatchesReference) {
  for (size_t offset = 0; offset < kFloatsPerCacheLine; ++offset) {
    for (size_t count : {23, 24, 25, 33, 65}) {
      CheckBatch(256, count, offset);
    }
  }
}

TEST(Fp32PrefetchKernel, SequentialReversedAndShuffledRowsMatchReference) {
  constexpr size_t dim = 256;
  std::vector<float> query(dim, 0.125f);
  std::mt19937 random(20261022);
  for (size_t count : kBatchSizes) {
    for (size_t padding : {size_t{0}, size_t{1}, kFloatsPerCacheLine}) {
      SCOPED_TRACE(testing::Message()
                   << "count=" << count << ", padding=" << padding);
      const size_t stride = dim + padding;
      std::vector<float> rows(count * stride, kGuard);
      std::vector<const void *> vectors(count);
      for (size_t i = 0; i < count; ++i) {
        vectors[i] = rows.data() + i * stride;
        for (size_t d = 0; d < dim; ++d) {
          rows[i * stride + d] =
              static_cast<float>(static_cast<int>((i + d) % 19) - 9) / 16;
        }
      }
      for (size_t order = 0; order < 3; ++order) {
        SCOPED_TRACE(order);
        if (order == 1) std::reverse(vectors.begin(), vectors.end());
        if (order == 2) std::shuffle(vectors.begin(), vectors.end(), random);
        std::vector<float> actual(count + 2, kGuard);
        neon::inner_product_fp32_batch_distance(vectors.data(), query.data(),
                                                count, dim, actual.data() + 1,
                                                nullptr);
        EXPECT_EQ(kGuard, actual.front());
        EXPECT_EQ(kGuard, actual.back());
        for (size_t i = 0; i < count; ++i) {
          EXPECT_FLOAT_EQ(ReferenceInnerProduct(vectors[i], query.data(), dim),
                          actual[i + 1]);
        }
      }
    }
  }
}

#if defined(__unix__) || defined(__APPLE__)
template <typename T>
class GuardedBuffer {
 public:
  explicit GuardedBuffer(size_t count) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || count * sizeof(T) > static_cast<size_t>(page_size)) {
      throw std::runtime_error("Invalid guarded buffer size");
    }
    bytes_ = static_cast<size_t>(page_size) * 2;
    mapping_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_ == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }
    char *guard = static_cast<char *>(mapping_) + page_size;
    if (mprotect(guard, static_cast<size_t>(page_size), PROT_NONE) != 0) {
      munmap(mapping_, bytes_);
      throw std::runtime_error("mprotect failed");
    }
    data_ = reinterpret_cast<T *>(guard - count * sizeof(T));
  }

  ~GuardedBuffer() {
    munmap(mapping_, bytes_);
  }

  GuardedBuffer(const GuardedBuffer &) = delete;
  GuardedBuffer &operator=(const GuardedBuffer &) = delete;

  T *data() const {
    return data_;
  }

 private:
  void *mapping_;
  size_t bytes_;
  T *data_;
};

TEST(Fp32PrefetchKernel, GuardPagesProtectInputsPointerArrayAndOutput) {
  for (size_t dim : {0, 255, 256, 257}) {
    SCOPED_TRACE(dim);
    GuardedBuffer<float> query(dim);
    for (size_t d = 0; d < dim; ++d) {
      query.data()[d] = static_cast<float>(static_cast<int>(d % 11) - 5) / 16;
    }
    std::vector<std::unique_ptr<GuardedBuffer<float>>> storage;
    for (size_t i = 0; i < kBatchSizes.back(); ++i) {
      auto vector = std::make_unique<GuardedBuffer<float>>(dim);
      for (size_t d = 0; d < dim; ++d) {
        vector->data()[d] =
            static_cast<float>(static_cast<int>((d + i) % 13) - 6) / 16;
      }
      storage.push_back(std::move(vector));
    }
    for (size_t count : kBatchSizes) {
      SCOPED_TRACE(count);
      GuardedBuffer<const void *> vectors(count);
      GuardedBuffer<float> actual(count + 1);
      for (size_t i = 0; i < count; ++i) {
        vectors.data()[i] = storage[i]->data();
      }
      actual.data()[0] = kGuard;
      std::fill(actual.data() + 1, actual.data() + count + 1, kGuard);
      neon::inner_product_fp32_batch_distance(
          vectors.data(), query.data(), count, dim, actual.data() + 1, nullptr);
      EXPECT_EQ(kGuard, actual.data()[0]);
      for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(
            ReferenceInnerProduct(vectors.data()[i], query.data(), dim),
            actual.data()[i + 1]);
      }
    }
  }
}
#endif

}  // namespace
}  // namespace zvec::turbo
