// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/core/framework/index_cluster.h>
#include "utility/appendable_build_buffer.h"

namespace zvec::core {
// All levels share one file; keeping a child centroid must not retain its
// training matrix or create a writable page table per centroid/bucket.
class PagedCentroids : public IndexCluster::FeatureStorage {
 public:
  int append(const void *data, size_t bytes, IndexCluster::Centroid *row) try {
    if (!row || (!data && bytes)) return IndexError_InvalidArgument;
    const size_t offset = size_;
    if (bytes > std::numeric_limits<size_t>::max() - offset)
      return IndexError_Overflow;
    int ret = 0;
    if (!data_.size() && offset + bytes <= 64 * 1024) {
      small_.resize(offset + bytes);
      std::memcpy(small_.data() + offset, data, bytes);
    } else if (!data_.size()) {
      AppendableBuildBuffer candidate;
      ret = candidate.init(offset + bytes);
      if (ret == 0) ret = candidate.write(0, small_.data(), small_.size());
      if (ret == 0) ret = candidate.write(offset, data, bytes);
      if (ret != 0) return ret;
      data_ = std::move(candidate);
      decltype(small_)().swap(small_);
    } else {
      ret = data_.resize(offset + bytes);
      if (ret == 0) ret = data_.write(offset, data, bytes);
    }
    if (ret == 0) {
      row->set_feature_storage(owner_.lock(), offset, bytes);
      size_ += bytes;
    }
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }
  static std::shared_ptr<PagedCentroids> Create() {
    auto result = std::make_shared<PagedCentroids>();
    result->owner_ = result;
    return result;
  }
  int read(size_t offset, void *out, size_t bytes) const override {
    if (offset > size_ || bytes > size_ - offset || (!out && bytes))
      return IndexError_InvalidArgument;
    if (!data_.size()) {
      if (bytes) std::memcpy(out, small_.data() + offset, bytes);
      return 0;
    }
    return data_.read(offset, out, bytes);
  }

 private:
  std::weak_ptr<PagedCentroids> owner_;
  AppendableBuildBuffer data_;
  std::vector<uint8_t, ailego::BudgetedAllocator<uint8_t>> small_;
  size_t size_{0};
};
}  // namespace zvec::core
