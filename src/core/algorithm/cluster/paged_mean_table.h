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
#pragma once

#include <limits>
#include "utility/paged_build_buffer.h"
#include "vector_mean.h"

namespace zvec::core {

// One exact accumulator per (worker, centroid), without a heap matrix per
// worker. Arithmetic/packed/binary means retain their original precision.
// Workers own disjoint rows; reduction starts only after the workers join.
class PagedMeanTable {
 public:
  int init(size_t rows, const VectorMean &prototype) {
    const size_t bytes = prototype.state_size();
    if (!bytes || rows > std::numeric_limits<size_t>::max() / bytes)
      return IndexError_Overflow;
    int ret = data_.init(rows * bytes, 64 * 1024);
    if (ret != 0) return ret;
    row_bytes_ = bytes;
    // A reserved BufferStorage range has no readable data until written.
    // Initialize it in bounded chunks, just as the heap accumulator is zeroed.
    const std::vector<uint8_t> zeros(std::min<size_t>(64 * 1024, data_.size()),
                                     0);
    for (size_t offset = 0; offset < data_.size();) {
      const size_t n = std::min(zeros.size(), data_.size() - offset);
      ret = data_.write(offset, zeros.data(), n);
      if (ret != 0) return ret;
      offset += n;
    }
    return 0;
  }

  int read(size_t row, VectorMean *mean, std::vector<uint8_t> *scratch) const {
    if (!mean || !scratch || mean->state_size() != row_bytes_ || !row_bytes_ ||
        row >= data_.size() / row_bytes_)
      return IndexError_InvalidArgument;
    scratch->resize(row_bytes_);
    int ret = data_.read(row * row_bytes_, scratch->data(), row_bytes_);
    if (ret == 0) mean->restore_state(scratch->data());
    return ret;
  }

  int write(size_t row, const VectorMean &mean, std::vector<uint8_t> *scratch) {
    if (!scratch || mean.state_size() != row_bytes_ || !row_bytes_ ||
        row >= data_.size() / row_bytes_)
      return IndexError_InvalidArgument;
    scratch->resize(row_bytes_);
    mean.save_state(scratch->data());
    return data_.write(row * row_bytes_, scratch->data(), row_bytes_);
  }

 private:
  PagedBuildBuffer data_;
  size_t row_bytes_{0};
};

}  // namespace zvec::core
