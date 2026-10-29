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
#include <random>
#include <vector>
#include <gtest/gtest.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>
#include "distance/neon/fp16/cosine.h"
#include "distance/neon/fp16/inner_product.h"
#include "distance/neon/fp16/squared_euclidean.h"
#include "distance/neon/integer/squared_euclidean.h"
#include "distance/neon/pq_quantizer_int4/pq_distance.h"
#include "distance/neon/pq_quantizer_int8/pq_distance.h"
#include "distance/neon/record_quantized_int4/cosine.h"
#include "distance/neon/record_quantized_int4/inner_product.h"
#include "distance/neon/record_quantized_int4/squared_euclidean.h"
#include "distance/neon/record_quantized_int8/cosine.h"
#include "distance/neon/record_quantized_int8/inner_product.h"
#include "distance/neon/record_quantized_int8/squared_euclidean.h"

namespace zvec::turbo {
namespace {

using SingleFn = void (*)(const void *, const void *, size_t, float *);
using BatchFn = void (*)(const void *const *, const void *, size_t, size_t,
                         float *, const void *const *);
constexpr size_t kCounts[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32, 33};
constexpr float kSentinel = 12345.0f;
std::vector<size_t> Dimensions() {
  std::vector<size_t> dims;
  for (size_t d = 0; d <= 65; ++d) dims.push_back(d);
  for (size_t d : {127, 128, 129, 768, 769, 1536, 4097}) dims.push_back(d);
  return dims;
}

TEST(NeonBatch, PlainFp16PreservesFp32Arithmetic) {
  const SingleFn singles[] = {neon::squared_euclidean_fp16_distance,
                              neon::inner_product_fp16_distance,
                              neon::cosine_fp16_distance};
  const BatchFn batches[] = {neon::squared_euclidean_fp16_batch_distance,
                             neon::inner_product_fp16_batch_distance,
                             neon::cosine_fp16_batch_distance};
  for (auto batch : batches) batch(nullptr, nullptr, 0, 768, nullptr, nullptr);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-1.0f, 1.0f);
  for (size_t dim : Dimensions()) {
    SCOPED_TRACE(dim);
    // Offset by a half element and end at the last element; sanitizers catch
    // over-reads even on remainder dimensions with unaligned SIMD addresses.
    std::vector<std::vector<uint16_t>> storage(34,
                                               std::vector<uint16_t>(dim + 1));
    for (auto &row : storage) {
      for (auto &value : row) value = ailego::FloatHelper::ToFP16(random(rng));
    }
    const auto *query = storage[0].data() + 1;
    std::vector<const void *> rows;
    for (size_t i = 1; i < storage.size(); ++i)
      rows.push_back(storage[i].data() + 1);
    for (size_t n : kCounts)
      for (size_t m = 0; m < 3; ++m) {
        SCOPED_TRACE(testing::Message() << "n=" << n << " metric=" << m);
        std::vector<float> out(n + 2, kSentinel);
        batches[m](rows.data(), query, n, dim, out.data() + 1, nullptr);
        EXPECT_EQ(kSentinel, out.front());
        EXPECT_EQ(kSentinel, out.back());
        for (size_t r = 0; r < n; ++r) {
          float single;
          singles[m](rows[r], query, dim, &single);
          EXPECT_FLOAT_EQ(single, out[r + 1]);
          double reference = 0;
          for (size_t d = 0; d < dim; ++d) {
            const double a = ailego::FloatHelper::ToFP32(storage[r + 1][d + 1]);
            const double b = ailego::FloatHelper::ToFP32(query[d]);
            reference += m == 0 ? (a - b) * (a - b) : a * b;
          }
          if (m != 0) reference = (m == 2 ? 1.0 : 0.0) - reference;
          EXPECT_NEAR(reference, out[r + 1],
                      2e-5 * std::max(1.0, std::abs(reference)));
        }
      }
  }
  // Multiplying in FP16 would overflow even before summation.
  for (size_t dim : {1, 3, 4, 7, 8, 9, 31, 32, 33, 4097}) {
    std::vector<uint16_t> high(dim, ailego::FloatHelper::ToFP16(256.0f));
    std::vector<uint16_t> zero(dim, 0);
    const void *rows[] = {high.data(), high.data(), high.data(), high.data(),
                          high.data()};
    float out[5];
    batches[0](rows, zero.data(), 5, dim, out, nullptr);
    for (float value : out) EXPECT_FLOAT_EQ(float(dim) * 65536.0f, value);
    batches[1](rows, high.data(), 5, dim, out, nullptr);
    for (float value : out) EXPECT_FLOAT_EQ(-float(dim) * 65536.0f, value);
  }
}

TEST(NeonBatch, RecordPackedUnalignedTailsAndRemainders) {
  const SingleFn singles[2][3] = {
      {neon::squared_euclidean_int4_distance, neon::inner_product_int4_distance,
       neon::cosine_int4_distance},
      {neon::squared_euclidean_int8_distance, neon::inner_product_int8_distance,
       neon::cosine_int8_distance}};
  const BatchFn batches[2][3] = {{neon::squared_euclidean_int4_batch_distance,
                                  neon::inner_product_int4_batch_distance,
                                  neon::cosine_int4_batch_distance},
                                 {neon::squared_euclidean_int8_batch_distance,
                                  neon::inner_product_int8_batch_distance,
                                  neon::cosine_int8_batch_distance}};
  for (const auto &group : batches)
    for (auto batch : group) batch(nullptr, nullptr, 0, 768, nullptr, nullptr);
  std::mt19937 rng(17);
  for (size_t type = 0; type < 2; ++type) {
    const size_t bits = type == 0 ? 4 : 8;
    for (size_t dim : Dimensions()) {
      if (dim == 0 || (bits == 4 && dim % 2)) continue;
      SCOPED_TRACE(testing::Message() << "bits=" << bits << " dim=" << dim);
      const size_t bytes = dim * bits / 8;
      for (size_t m = 0; m < 3; ++m) {
        const size_t tail = bits == 4 ? (m == 2 ? 20 : 16) : (m == 2 ? 24 : 20);
        const size_t encoded_dim = dim + tail * 8 / bits;
        std::vector<std::vector<uint8_t>> storage(
            34, std::vector<uint8_t>(1 + bytes + tail));
        std::vector<std::vector<double>> decoded(34, std::vector<double>(dim));
        for (size_t r = 0; r < storage.size(); ++r) {
          auto *row = storage[r].data() + 1;
          const float scale = float(r % 3 + 1) * 0.125f;
          const float bias = float(int(r % 5) - 2) * 0.25f;
          double sum = 0, square_sum = 0;
          for (size_t d = 0; d < dim; ++d) {
            const int v = int(rng() % (1 << bits)) - (1 << (bits - 1));
            if (bits == 8)
              row[d] = static_cast<uint8_t>(v);
            else
              row[d / 2] |= uint8_t((v & 15) << ((d % 2) * 4));
            sum += v;
            square_sum += v * v;
            decoded[r][d] = scale * v + bias;
          }
          float params[] = {scale, bias, static_cast<float>(sum),
                            static_cast<float>(square_sum)};
          std::memcpy(row + bytes, params, sizeof(params));
        }
        std::vector<const void *> rows;
        for (size_t r = 1; r < storage.size(); ++r)
          rows.push_back(storage[r].data() + 1);
        for (size_t n : kCounts) {
          std::vector<float> out(n + 2, kSentinel);
          batches[type][m](rows.data(), storage[0].data() + 1, n, encoded_dim,
                           out.data() + 1, nullptr);
          EXPECT_EQ(kSentinel, out.front());
          EXPECT_EQ(kSentinel, out.back());
          for (size_t r = 0; r < n; ++r) {
            float single;
            singles[type][m](rows[r], storage[0].data() + 1, encoded_dim,
                             &single);
            EXPECT_FLOAT_EQ(single, out[r + 1]);
            double reference = 0;
            for (size_t d = 0; d < dim; ++d) {
              const double a = decoded[r + 1][d], b = decoded[0][d];
              reference += m == 0 ? (a - b) * (a - b) : -a * b;
            }
            // Record cosine's offset to 1-cos belongs to the quantizer.
            EXPECT_NEAR(reference, out[r + 1],
                        2e-5 * std::max(1.0, std::abs(reference)));
          }
        }
      }
    }
  }
}

TEST(NeonBatch, IntegerL2LayoutsAndWideSums) {
  const BatchFn batches[] = {
      neon::squared_euclidean_raw_uint8_batch_distance,
      neon::uniform_squared_euclidean_uint7_batch_distance,
      neon::uniform_squared_euclidean_uint4_batch_distance,
      neon::uniform_squared_euclidean_uint8_batch_distance};
  const SingleFn singles[] = {neon::squared_euclidean_raw_uint8_distance,
                              neon::uniform_squared_euclidean_uint7_distance,
                              neon::uniform_squared_euclidean_uint4_distance,
                              neon::uniform_squared_euclidean_uint8_distance};
  for (auto batch : batches) batch(nullptr, nullptr, 0, 768, nullptr, nullptr);
  auto dims = Dimensions();
  // Full-range uint8 L2 exceeds both INT32_MAX and UINT32_MAX here.
  dims.push_back(65537);
  dims.push_back(131073);
  for (size_t dim : dims)
    for (size_t type = 0; type < 4; ++type) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " type=" << type);
      const size_t extra = type == 3 ? 4 : 0;
      std::vector<uint8_t> q(1 + dim + extra, 0), a(q.size(), 0);
      auto *query = q.data() + 1;
      auto *row = a.data() + 1;
      for (size_t d = 0; d < dim; ++d) {
        if (type == 3) {
          query[d] = 128;
          row[d] = 127;
        } else {
          query[d] = 0;
          row[d] = type == 1 ? 127 : 255;
        }
      }
      const void *rows[33];
      for (auto &ptr : rows) ptr = row;
      float single;
      singles[type](row, query, dim + extra, &single);
      const uint64_t per_byte = type == 2 ? 450 : type == 1 ? 16129 : 65025;
      const float reference = static_cast<float>(uint64_t(dim) * per_byte);
      EXPECT_FLOAT_EQ(reference, single);
      if (type == 3)
        neon::uniform_squared_euclidean_uint8_query_preprocess(query,
                                                               dim + extra);
      for (size_t n : kCounts) {
        std::vector<float> out(n + 2, kSentinel);
        batches[type](rows, query, n, dim + extra, out.data() + 1, nullptr);
        EXPECT_EQ(kSentinel, out.front());
        EXPECT_EQ(kSentinel, out.back());
        for (size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(reference, out[i + 1]);
      }
    }
}

TEST(NeonBatch, MixedIntegerCodesAndQueryPreprocess) {
  const BatchFn batches[] = {
      neon::squared_euclidean_raw_uint8_batch_distance,
      neon::uniform_squared_euclidean_uint7_batch_distance,
      neon::uniform_squared_euclidean_uint4_batch_distance,
      neon::uniform_squared_euclidean_uint8_batch_distance};
  std::mt19937 random(42);
  for (size_t dim : Dimensions()) {
    for (size_t type = 0; type < 4; ++type) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " type=" << type);
      const size_t extra = type == 3 ? 4 : 0;
      std::vector<std::vector<uint8_t>> data(
          18, std::vector<uint8_t>(1 + dim + extra));
      for (auto &row : data)
        for (size_t d = 0; d < dim; ++d)
          row[d + 1] = uint8_t(random() & (type == 1 ? 127 : 255));
      auto *query = data[0].data() + 1;
      std::vector<const void *> rows;
      std::vector<float> reference;
      for (size_t r = 1; r < data.size(); ++r) {
        auto *row = data[r].data() + 1;
        rows.push_back(row);
        int64_t total = 0;
        for (size_t d = 0; d < dim; ++d) {
          if (type == 2) {
            const int lo = int(row[d] & 15) - int(query[d] & 15);
            const int hi = int(row[d] >> 4) - int(query[d] >> 4);
            total += lo * lo + hi * hi;
          } else {
            const int diff = type == 3 ? int(static_cast<int8_t>(row[d])) -
                                             int(static_cast<int8_t>(query[d]))
                                       : int(row[d]) - int(query[d]);
            total += diff * diff;
          }
        }
        reference.push_back(static_cast<float>(total));
      }
      if (type == 3) {
        uint32_t norm = 0, sum = 0;
        for (size_t d = 0; d < dim; ++d) {
          const uint32_t value = query[d] ^ uint8_t { 128 };
          norm += value * value;
          sum += value;
        }
        std::memcpy(query + dim, &norm, sizeof(norm));
        neon::uniform_squared_euclidean_uint8_query_preprocess(query,
                                                               dim + extra);
        int32_t correction;
        std::memcpy(&correction, query + dim, sizeof(correction));
        EXPECT_EQ(int64_t(norm) - 256 * int64_t(sum), correction);
      }
      for (size_t n : {0, 1, 2, 3, 4, 5, 6, 7, 17}) {
        std::vector<float> out(n + 2, kSentinel);
        batches[type](rows.data(), query, n, dim + extra, out.data() + 1,
                      nullptr);
        EXPECT_EQ(kSentinel, out.front());
        EXPECT_EQ(kSentinel, out.back());
        for (size_t r = 0; r < n; ++r)
          EXPECT_FLOAT_EQ(reference[r], out[r + 1]);
      }
    }
  }
}

TEST(NeonBatch, PqSdcUsesCandidateMajorAsymmetricTables) {
  for (size_t bits : {4, 8}) {
    const auto batch = bits == 4 ? neon::pq_sdc_int4_batch_distance_neon
                                 : neon::pq_sdc_int8_batch_distance_neon;
    const auto single = bits == 4 ? neon::pq_sdc_int4_distance_neon
                                  : neon::pq_sdc_int8_distance_neon;
    batch(nullptr, nullptr, nullptr, 0, 128, nullptr);
    const size_t centroids = size_t{1} << bits;
    for (size_t chunks :
         {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
      SCOPED_TRACE(testing::Message()
                   << "bits=" << bits << " chunks=" << chunks);
      std::vector<float> table(chunks * centroids * centroids);
      // Deliberately non-symmetric: swapping query and candidate must fail.
      for (size_t m = 0; m < chunks; ++m)
        for (size_t a = 0; a < centroids; ++a)
          for (size_t b = 0; b < centroids; ++b) {
            table[(m * centroids + a) * centroids + b] =
                float(int((m * 17 + a * 13 + b * 3) % 257) - 128) * 0.0625f;
          }
      const size_t bytes = (chunks * bits + 7) / 8;
      std::vector<std::vector<uint8_t>> codes(34,
                                              std::vector<uint8_t>(1 + bytes));
      for (size_t r = 0; r < codes.size(); ++r)
        for (size_t d = 0; d < bytes; ++d)
          codes[r][d + 1] = uint8_t(r * 37 + d * 19);
      std::vector<const void *> rows;
      for (size_t r = 1; r < codes.size(); ++r)
        rows.push_back(codes[r].data() + 1);
      auto code_at = [bits](const uint8_t *p, size_t m) -> size_t {
        return bits == 8 ? p[m] : (p[m / 2] >> ((m % 2) * 4)) & 15;
      };
      for (size_t n : kCounts) {
        std::vector<float> out(n + 2, kSentinel);
        batch(rows.data(), codes[0].data() + 1, table.data(), n, chunks,
              out.data() + 1);
        EXPECT_EQ(kSentinel, out.front());
        EXPECT_EQ(kSentinel, out.back());
        for (size_t r = 0; r < n; ++r) {
          float expected;
          single(rows[r], codes[0].data() + 1, table.data(), chunks, &expected);
          EXPECT_FLOAT_EQ(expected, out[r + 1]);
          double reference = 0;
          for (size_t m = 0; m < chunks; ++m)
            reference +=
                table[(m * centroids + code_at(codes[r + 1].data() + 1, m)) *
                          centroids +
                      code_at(codes[0].data() + 1, m)];
          EXPECT_DOUBLE_EQ(reference, out[r + 1]);
        }
      }
    }
  }
}

TEST(NeonBatch, AutoDispatchIncludesRawUniformAndSdc) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "ARM dispatch requires AArch64";
#else
  struct Case {
    DataType type;
    QuantizeType quant;
    BatchFn batch;
  };
  const Case cases[] = {{DataType::kUint8, QuantizeType::kRaw,
                         neon::squared_euclidean_raw_uint8_batch_distance},
                        {DataType::kInt8, QuantizeType::kUniform,
                         neon::uniform_squared_euclidean_uint7_batch_distance},
                        {DataType::kInt4, QuantizeType::kUniformUint4,
                         neon::uniform_squared_euclidean_uint4_batch_distance},
                        {DataType::kInt8, QuantizeType::kUniformUint8,
                         neon::uniform_squared_euclidean_uint8_batch_distance},
                        {DataType::kFp16, QuantizeType::kRaw,
                         neon::squared_euclidean_fp16_batch_distance},
                        {DataType::kInt8, QuantizeType::kRecord,
                         neon::squared_euclidean_int8_batch_distance},
                        {DataType::kInt4, QuantizeType::kRecord,
                         neon::squared_euclidean_int4_batch_distance}};
  for (auto arch : {CpuArchType::kAuto, CpuArchType::kNEON}) {
    for (const auto &entry : cases) {
      const auto k = get_distance_kernels(MetricType::kSquaredEuclidean,
                                          entry.type, entry.quant, arch);
      ASSERT_TRUE(k.dist);
      ASSERT_TRUE(k.batch);
      const auto *target = k.batch.target<BatchFn>();
      ASSERT_NE(nullptr, target);
      EXPECT_EQ(entry.batch, *target);
      EXPECT_EQ(entry.quant == QuantizeType::kUniformUint8
                    ? neon::uniform_squared_euclidean_uint8_query_preprocess
                    : nullptr,
                k.preprocess);
    }
    const auto raw_cosine = get_distance_kernels(
        MetricType::kCosine, DataType::kFp16, QuantizeType::kRaw, arch);
    const auto *cosine = raw_cosine.batch.target<BatchFn>();
    ASSERT_NE(nullptr, cosine);
    EXPECT_EQ(&neon::cosine_fp16_batch_distance, *cosine);
    const auto pq4 = get_pq_kernels(DataType::kInt4, QuantizeType::kPQ, arch);
    const auto pq8 = get_pq_kernels(DataType::kInt8, QuantizeType::kPQ, arch);
    EXPECT_EQ(&neon::pq_sdc_int4_batch_distance_neon,
              pq4.batch_symmetric_distance);
    EXPECT_EQ(&neon::pq_sdc_int8_batch_distance_neon,
              pq8.batch_symmetric_distance);
    EXPECT_EQ(nullptr,
              get_pq_kernels(DataType::kInt4, QuantizeType::kPQFast, arch)
                  .batch_symmetric_distance);
  }
#endif
}


#if defined(__unix__) || defined(__APPLE__)
// Put the final byte against an inaccessible page so even a one-byte SIMD
// overread, padded candidate pointer, or output overrun fails immediately.
class GuardedBuffer {
 public:
  explicit GuardedBuffer(size_t bytes)
      : page_(static_cast<size_t>(sysconf(_SC_PAGESIZE))) {
    mapping_ = mmap(nullptr, page_ * 2, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping_ == MAP_FAILED) return;
    data_ = static_cast<uint8_t *>(mapping_) + page_ - bytes;
    if (mprotect(static_cast<uint8_t *>(mapping_) + page_, page_, PROT_NONE) !=
        0)
      data_ = nullptr;
  }
  ~GuardedBuffer() {
    if (mapping_ != MAP_FAILED) munmap(mapping_, page_ * 2);
  }
  void *data() const {
    return data_;
  }

 private:
  size_t page_;
  void *mapping_ = MAP_FAILED;
  void *data_ = nullptr;
};

TEST(NeonBatch, ExactBuffersAtGuardPages) {
  struct Case {
    BatchFn batch;
    size_t bits;
    size_t tail;
    bool even;
  };
  const Case cases[] = {
      {neon::squared_euclidean_fp16_batch_distance, 16, 0, false},
      {neon::inner_product_fp16_batch_distance, 16, 0, false},
      {neon::cosine_fp16_batch_distance, 16, 0, false},
      {neon::squared_euclidean_int8_batch_distance, 8, 20, false},
      {neon::inner_product_int8_batch_distance, 8, 20, false},
      {neon::cosine_int8_batch_distance, 8, 24, false},
      {neon::squared_euclidean_int4_batch_distance, 4, 16, true},
      {neon::inner_product_int4_batch_distance, 4, 16, true},
      {neon::cosine_int4_batch_distance, 4, 20, true},
      {neon::squared_euclidean_raw_uint8_batch_distance, 8, 0, false},
      {neon::uniform_squared_euclidean_uint7_batch_distance, 8, 0, false},
      // Uniform uint4 takes byte dimensions, unlike Record int4.
      {neon::uniform_squared_euclidean_uint4_batch_distance, 8, 0, false},
      {neon::uniform_squared_euclidean_uint8_batch_distance, 8, 4, false}};
  for (size_t dim = 1; dim <= 65; ++dim) {
    for (const auto &c : cases) {
      if (c.even && dim % 2) continue;
      const size_t bytes = dim * c.bits / 8 + c.tail;
      GuardedBuffer query(bytes), row(bytes);
      ASSERT_NE(nullptr, query.data());
      ASSERT_NE(nullptr, row.data());
      std::memset(query.data(), 0, bytes);
      std::memset(row.data(), 0, bytes);
      const size_t encoded_dim = dim + c.tail * 8 / c.bits;
      if (c.tail == 4)
        neon::uniform_squared_euclidean_uint8_query_preprocess(query.data(),
                                                               encoded_dim);
      for (size_t n = 1; n <= 9; ++n) {
        GuardedBuffer pointers(n * sizeof(void *)), output(n * sizeof(float));
        ASSERT_NE(nullptr, pointers.data());
        ASSERT_NE(nullptr, output.data());
        auto **rows = static_cast<const void **>(pointers.data());
        for (size_t r = 0; r < n; ++r) rows[r] = row.data();
        c.batch(rows, query.data(), n, encoded_dim,
                static_cast<float *>(output.data()), nullptr);
      }
    }
    for (size_t bits : {4, 8}) {
      const auto batch = bits == 4 ? neon::pq_sdc_int4_batch_distance_neon
                                   : neon::pq_sdc_int8_batch_distance_neon;
      const size_t bytes = (dim * bits + 7) / 8;
      GuardedBuffer query(bytes), row(bytes);
      ASSERT_NE(nullptr, query.data());
      ASSERT_NE(nullptr, row.data());
      std::memset(query.data(), 0, bytes);
      std::memset(row.data(), 0, bytes);
      std::vector<float> table(dim * (size_t{1} << (bits * 2)), 0.0f);
      for (size_t n = 1; n <= 9; ++n) {
        GuardedBuffer pointers(n * sizeof(void *)), output(n * sizeof(float));
        ASSERT_NE(nullptr, pointers.data());
        ASSERT_NE(nullptr, output.data());
        auto **rows = static_cast<const void **>(pointers.data());
        for (size_t r = 0; r < n; ++r) rows[r] = row.data();
        batch(rows, query.data(), table.data(), n, dim,
              static_cast<float *>(output.data()));
      }
    }
  }
}
#endif

}  // namespace
}  // namespace zvec::turbo
