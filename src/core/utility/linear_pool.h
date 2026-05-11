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
// ===========================================================================
// Acknowledgement
// ---------------
// The LinearPool implementation in this file (and the accompanying Neighbor /
// Bitset helpers) is adapted from the pyglass project, with modifications
// (e.g. a BlockHeap-compatible reset()/push_block() interface and the use of
// MemoryHelper for huge-page-backed allocation):
//
//     pyglass — Graph Library for Approximate Similarity Search
//     https://github.com/zilliztech/pyglass
//
// pyglass is distributed under the MIT License. The original copyright notice
// and permission notice are reproduced below as required by that license:
//
//     MIT License
//
//     Copyright (c) 2023 zh Wang
//
//     Permission is hereby granted, free of charge, to any person obtaining a
//     copy of this software and associated documentation files (the
//     "Software"), to deal in the Software without restriction, including
//     without limitation the rights to use, copy, modify, merge, publish,
//     distribute, sublicense, and/or sell copies of the Software, and to
//     permit persons to whom the Software is furnished to do so, subject to
//     the following conditions:
//
//     The above copyright notice and this permission notice shall be included
//     in all copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//     TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ailego/utility/memory_helper.h>

namespace zvec {
namespace core {

namespace linear_pool_impl {

template <typename dist_t = float>
struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }

  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};

template <typename Block = uint64_t>
struct Bitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int32_t nb = 0;
  int nbytes = 0;
  Block *data = nullptr;

  Bitset() = default;

  explicit Bitset(int n)
      : nb(n),
        nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((Block *)zvec::ailego::MemoryHelper::AllocateAligned(nbytes)) {}

  friend void swap(Bitset &lhs, Bitset &rhs) {
    using std::swap;
    swap(lhs.nb, rhs.nb);
    swap(lhs.nbytes, rhs.nbytes);
    swap(lhs.data, rhs.data);
  }

  Bitset(const Bitset &) = delete;

  Bitset(Bitset &&rhs) {
    swap(*this, rhs);
  }

  Bitset &operator=(const Bitset &) = delete;

  Bitset &operator=(Bitset &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  ~Bitset() {
    if (data) {
      zvec::ailego::MemoryHelper::FreeAligned(data, nbytes);
    }
  }

  void reset(int32_t n) {
    if (n != nb) {
      *this = Bitset(n);
    } else {
      memset(data, 0, nbytes);
    }
  }

  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }

  bool get(int i) const {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }
};

}  // namespace linear_pool_impl

template <typename dist_t, typename BitsetType = linear_pool_impl::Bitset<>>
struct LinearPool {
  using dist_type = dist_t;

  LinearPool() = default;

  LinearPool(int n, int ef, int capacity)
      : nb(n), ef_(ef), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  friend void swap(LinearPool &lhs, LinearPool &rhs) {
    using std::swap;
    swap(lhs.nb, rhs.nb);
    swap(lhs.size_, rhs.size_);
    swap(lhs.cur_, rhs.cur_);
    swap(lhs.ef_, rhs.ef_);
    swap(lhs.capacity_, rhs.capacity_);
    swap(lhs.data_, rhs.data_);
    swap(lhs.vis, rhs.vis);
  }

  LinearPool(const LinearPool &) = delete;

  LinearPool(LinearPool &&rhs) {
    swap(*this, rhs);
  }

  LinearPool &operator=(const LinearPool &) = delete;

  LinearPool &operator=(LinearPool &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  // reset() mirrors BlockHeap::reset(n, capacity, block_size): `n` is the
  // visit-set capacity (number of nodes), `capacity` is the retained top-k size
  // (stored in both ef_ and capacity_ here), and the third argument is the
  // per-batch push upper bound, used only by BlockHeap.  LinearPool keeps a
  // sorted array whose capacity equals `capacity` and therefore ignores the
  // `block_size` hint.  The signature is kept identical to BlockHeap so
  // templated greedy-search bodies can call either pool uniformly.
  void reset(int32_t n, int32_t capacity, int32_t /*block_size_ignored*/) {
    nb = n;
    size_ = cur_ = 0;
    ef_ = capacity;
    capacity_ = capacity;
    if (data_.size() < static_cast<size_t>(capacity + 1)) {
      data_.resize(capacity + 1);
    }
    vis.reset(n);
  }

  ailego_force_inline int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
    // int len = size_;
    // int loc = 0;
    // while (len > 1) {
    //   int half = len / 2;
    //   loc += (dist > data_[loc + half - 1].distance) * half;
    //   len -= half;
    // }
    // return loc;
  }

  // Block-insert interface matching BlockHeap::push_block: insert each
  // (node, distance) pair via the one-by-one sorted insert().  Used by the
  // templated greedy_search helpers so that LinearPool can be plugged in
  // when AVX2 is unavailable.
  void push_block(const float *distances, const uint32_t *nodes,
                  int32_t block_size) {
    for (int32_t i = 0; i < block_size; ++i) {
      insert(static_cast<int>(nodes[i]), static_cast<dist_t>(distances[i]));
    }
  }

  ailego_force_inline bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(linear_pool_impl::Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const {
    return cur_ < size_ && cur_ < ef_;
  }
  int id(int i) const {
    return get_id(data_[i].id);
  }
  dist_type dist(int i) const {
    return data_[i].distance;
  }
  int size() const {
    return size_;
  }
  int capacity() const {
    return capacity_;
  }

  void set_visited(int32_t u) {
    vis.set(u);
  }
  bool check_visited(int32_t u) const {
    return vis.get(u);
  }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const {
    return id & kMask;
  }
  void set_checked(int &id) {
    id |= 1 << 31;
  }
  bool is_checked(int id) const {
    return id >> 31 & 1;
  }

  void to_sorted(int32_t *ids, float *scores, int32_t length) const {
    for (int32_t i = 0; i < length; ++i) {
      ids[i] = id(i);
      if (scores) {
        scores[i] = dist(i);
      }
    }
  }

  int nb, size_ = 0, cur_ = 0, ef_, capacity_;
  std::vector<linear_pool_impl::Neighbor<dist_t>> data_;
  BitsetType vis;
};

// Copy a single-heap pool's (LinearPool/BlockHeap) distance-sorted retained
// results into a topk heap. Both the pool and the topk heap are template
// parameters so this helper is independent of any per-algorithm TopkHeap alias
// and can be shared by Vamana and HNSW greedy search.
template <typename PoolType, typename TopkType>
void copy_pool_to_topk(const PoolType &pool, TopkType &topk) {
  const int32_t n = static_cast<int32_t>(pool.size());
  for (int32_t i = 0; i < n; ++i) {
    topk.emplace(pool.id(i), pool.dist(i));
  }
}

}  // namespace core
}  // namespace zvec
