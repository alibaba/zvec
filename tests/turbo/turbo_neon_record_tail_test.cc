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
#include <cstring>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "distance/neon/record_quantized_int8/cosine.h"
#include "distance/neon/record_quantized_int8/inner_product.h"
#include "distance/neon/record_quantized_int8/squared_euclidean.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace zvec::turbo {
namespace {

// Exact-sized, potentially unaligned records ending at an inaccessible page.
class RecordBuffer {
 public:
  explicit RecordBuffer(size_t bytes) {
#if defined(__unix__) || defined(__APPLE__)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;
    const size_t page = static_cast<size_t>(page_size);
    const size_t readable = (bytes + page - 1) / page * page;
    size_ = readable + page;
    mapping_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping_ == MAP_FAILED) return;
    auto *end = static_cast<int8_t *>(mapping_) + readable;
    if (mprotect(end, page, PROT_NONE) == 0) data_ = end - bytes;
#else
    storage_.resize(bytes);
    data_ = storage_.data();
#endif
  }

  ~RecordBuffer() {
#if defined(__unix__) || defined(__APPLE__)
    if (mapping_ != MAP_FAILED) munmap(mapping_, size_);
#endif
  }

  int8_t *data() const {
    return data_;
  }

  RecordBuffer(const RecordBuffer &) = delete;
  RecordBuffer &operator=(const RecordBuffer &) = delete;

 private:
  int8_t *data_ = nullptr;
#if defined(__unix__) || defined(__APPLE__)
  void *mapping_ = MAP_FAILED;
  size_t size_ = 0;
#else
  std::vector<int8_t> storage_;
#endif
};

TEST(NeonRecordTail, ScalarAndSimdTailsMatchWideReference) {
  using Batch = void (*)(const void *const *, const void *, size_t, size_t,
                         float *, const void *const *);
  const Batch batches[] = {neon::squared_euclidean_int8_batch_distance,
                           neon::inner_product_int8_batch_distance,
                           neon::cosine_int8_batch_distance};
  std::vector<size_t> dimensions;
  for (size_t prefix : {0, 16, 128}) {
    for (size_t tail = 0; tail < 16; ++tail) {
      if (prefix + tail) dimensions.push_back(prefix + tail);
    }
  }
  // Exceed INT32_MAX before adding either a scalar or SIMD remainder.
  for (size_t dim : {131073, 131075, 131076}) dimensions.push_back(dim);

  for (size_t metric = 0; metric < 3; ++metric) {
    batches[metric](nullptr, nullptr, 0, 41, nullptr, nullptr);
    for (size_t dim : dimensions) {
      const size_t bytes = dim + (metric == 2 ? 24 : 20);
      std::array<std::unique_ptr<RecordBuffer>, 34> records;
      for (auto &record : records) {
        record = std::make_unique<RecordBuffer>(bytes);
        ASSERT_NE(nullptr, record->data());
      }
      for (int pattern = 0; pattern < 3; ++pattern) {
        SCOPED_TRACE(testing::Message() << "metric=" << metric << " dim=" << dim
                                        << " pattern=" << pattern);
        std::array<float, 34> norms;
        for (size_t r = 0; r < records.size(); ++r) {
          auto *record = records[r]->data();
          std::memset(record, 0, bytes);
          int64_t sum = 0, square = 0;
          for (size_t d = 0; d < dim; ++d) {
            const int code = pattern == 0 ? -128
                             : pattern == 1
                                 ? ((r + d) % 2 ? 127 : -128)
                                 : int((r * 31 + d * 37) % 256) - 128;
            record[d] = static_cast<int8_t>(code);
            sum += code;
            square += code * code;
          }
          norms[r] = static_cast<float>(square);
          const float metadata[] = {1.0f, 0.0f, float(sum), norms[r]};
          std::memcpy(record + dim, metadata, sizeof(metadata));
          const int32_t code_sum = static_cast<int32_t>(sum);
          std::memcpy(record + dim + sizeof(metadata), &code_sum,
                      sizeof(code_sum));
        }

        std::array<const void *, 33> rows;
        std::array<float, 33> expected;
        const auto *query = records[0]->data();
        for (size_t r = 0; r < rows.size(); ++r) {
          const auto *row = records[r + 1]->data();
          rows[r] = row;
          int64_t dot = 0;
          for (size_t d = 0; d < dim; ++d) dot += int64_t(row[d]) * query[d];
          const float ip = static_cast<float>(dot);
          expected[r] = metric == 0 ? norms[r + 1] + norms[0] - 2 * ip : -ip;
        }
        for (size_t count : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32, 33}) {
          SCOPED_TRACE(testing::Message() << "count=" << count);
          constexpr float sentinel = 12345.0f;
          std::array<float, 35> out;
          out.fill(sentinel);
          batches[metric](rows.data(), query, count, bytes, out.data() + 1,
                          nullptr);
          EXPECT_EQ(sentinel, out.front());
          for (size_t r = 0; r < count; ++r) {
            EXPECT_FLOAT_EQ(expected[r], out[r + 1]) << "row=" << r;
          }
          for (size_t r = count + 1; r < out.size(); ++r) {
            EXPECT_EQ(sentinel, out[r]);
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace zvec::turbo
