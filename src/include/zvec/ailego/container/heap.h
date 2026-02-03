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

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace zvec {
namespace ailego {

/*! Heap Adapter
 */
template <typename T, typename TCompare = std::less<T>,
          typename TBase = std::vector<T>>
class Heap : public TBase {
 public:
  //! Constructor
  Heap(void)
      : TBase(), limit_(std::numeric_limits<size_t>::max()), compare_() {}

  //! Constructor
  template <typename... Args>
  Heap(size_t max, Args &&...args)
      : TBase(),
        limit_(std::max<size_t>(max, 1u)),
        compare_(std::forward<Args>(args)...) {
    TBase::reserve(limit_);
  }

  //! Constructor
  Heap(const Heap &rhs)
      : TBase(rhs), limit_(rhs.limit_), compare_(rhs.compare_) {}

  //! Constructor
  Heap(Heap &&rhs)
      : TBase(std::move(rhs)),
        limit_(rhs.limit_),
        compare_(std::move(rhs.compare_)) {}

  //! Constructor
  Heap(const TBase &rhs)
      : TBase(rhs), limit_(std::numeric_limits<size_t>::max()), compare_() {
    std::make_heap(TBase::begin(), TBase::end(), compare_);
  }

  //! Constructor
  Heap(TBase &&rhs)
      : TBase(std::move(rhs)),
        limit_(std::numeric_limits<size_t>::max()),
        compare_() {
    std::make_heap(TBase::begin(), TBase::end(), compare_);
  }

  //! Assignment
  Heap &operator=(const Heap &rhs) {
    TBase::operator=(static_cast<const TBase &>(rhs));
    limit_ = rhs.limit_;
    compare_ = rhs.compare_;
    return *this;
  }

  //! Assignment
  Heap &operator=(Heap &&rhs) {
    TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
    limit_ = rhs.limit_;
    compare_ = std::move(rhs.compare_);
    return *this;
  }

  //! Exchange the content
  void swap(Heap &rhs) {
    TBase::swap(static_cast<TBase &>(rhs));
    std::swap(limit_, rhs.limit_);
    std::swap(compare_, rhs.compare_);
  }

  //! Pop the front element
  void pop(void) {
    if (TBase::size() > 1) {
      auto last = TBase::end() - 1;
      this->replace_heap(TBase::begin(), last, std::move(*last));
    }
    TBase::pop_back();
  }

  //! Insert a new element into the heap
  template <class... TArgs>
  void emplace(TArgs &&...args) {
    if (this->full()) {
      typename std::remove_reference<T>::type val(std::forward<TArgs>(args)...);

      auto first = TBase::begin();
      if (compare_(val, *first)) {
        this->replace_heap(first, TBase::end(), std::move(val));
      }
    } else {
      TBase::emplace_back(std::forward<TArgs>(args)...);
      std::push_heap(TBase::begin(), TBase::end(), compare_);
    }
  }

  //! Insert a new element into the heap
  void push(const T &val) {
    if (this->full()) {
      auto first = TBase::begin();
      if (compare_(val, *first)) {
        this->replace_heap(first, TBase::end(), val);
      }
    } else {
      TBase::push_back(val);
      std::push_heap(TBase::begin(), TBase::end(), compare_);
    }
  }

  //! Insert a new element into the heap
  void push(T &&val) {
    if (this->full()) {
      auto first = TBase::begin();
      if (compare_(val, *first)) {
        this->replace_heap(first, TBase::end(), std::move(val));
      }
    } else {
      TBase::push_back(std::move(val));
      std::push_heap(TBase::begin(), TBase::end(), compare_);
    }
  }

  //! Retrieve the limit of heap
  size_t limit(void) const {
    return limit_;
  }

  //! Limit the heap with max size
  void limit(size_t max) {
    limit_ = std::max<size_t>(max, 1u);
    TBase::reserve(limit_);
  }

  //! Unlimit the size of heap
  void unlimit(void) {
    limit_ = std::numeric_limits<size_t>::max();
  }

  //! Check whether the heap is full
  bool full(void) const {
    return (TBase::size() == limit_);
  }

  //! Update the heap
  void update(void) {
    std::make_heap(TBase::begin(), TBase::end(), compare_);
    while (limit_ < TBase::size()) {
      this->pop();
    }
  }

  //! Sort the elements in the heap
  void sort(void) {
    std::sort(TBase::begin(), TBase::end(), compare_);
  }

 protected:
  //! Replace the top element of heap
  template <typename TRandomIterator, typename TValue>
  void replace_heap(TRandomIterator first, TRandomIterator last, TValue &&val) {
    using _DistanceType =
        typename std::iterator_traits<TRandomIterator>::difference_type;

    _DistanceType hole = 0;
    _DistanceType count = _DistanceType(last - first);

    if (count > 1) {
      _DistanceType child = (hole << 1) + 1;

      while (child < count) {
        _DistanceType right_child = child + 1;

        if (right_child < count &&
            compare_(*(first + child), *(first + right_child))) {
          child = right_child;
        }
        if (!compare_(val, *(first + child))) {
          break;
        }
        *(first + hole) = std::move(*(first + child));
        hole = child;
        child = (hole << 1) + 1;
      }
    }
    *(first + hole) = std::forward<TValue>(val);
  }

 private:
  size_t limit_;
  TCompare compare_;
};

/*! Key Value Heap Comparer
 */
template <typename TKey, typename TValue, typename TCompare = std::less<TValue>>
struct KeyValueHeapComparer {
  //! Function call
  bool operator()(const std::pair<TKey, TValue> &lhs,
                  const std::pair<TKey, TValue> &rhs) const {
    return compare_(lhs.second, rhs.second);
  }

 private:
  TCompare compare_;
};

/*! Key Value Heap
 */
template <typename TKey, typename TValue, typename TCompare = std::less<TValue>>
using KeyValueHeap =
    Heap<std::pair<TKey, TValue>, KeyValueHeapComparer<TKey, TValue, TCompare>>;

}  // namespace ailego
}  // namespace zvec