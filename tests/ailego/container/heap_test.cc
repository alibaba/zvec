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

#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/container/heap.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(Heap, General) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 100);

  {
    ailego::Heap<float> heap;

    for (size_t i = 0; i < 12; ++i) {
      heap.emplace(dist(gen));
    }
    EXPECT_EQ(12u, heap.size());
    EXPECT_FALSE(heap.full());

    for (auto it : heap) {
      std::cout << it << " ";
    }
    std::cout << std::endl;

    ailego::Heap<float> heap1(std::move(heap));
    EXPECT_TRUE(heap.empty());
    EXPECT_FALSE(heap1.empty());
    for (size_t i = 0; i < 12; ++i) {
      heap1.pop();
    }
    EXPECT_TRUE(heap1.empty());
  }

  {
    ailego::Heap<float> heap(12);

    for (size_t i = 0; i < 200; ++i) {
      heap.push(dist(gen));
    }
    EXPECT_EQ(12u, heap.size());
    EXPECT_TRUE(std::is_heap(heap.begin(), heap.end()));
    EXPECT_TRUE(heap.full());

    ailego::Heap<float> heap2(heap);
    for (auto it : heap2) {
      std::cout << it << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 12; ++i) {
      heap2.pop();
    }
    EXPECT_TRUE(heap2.empty());
    EXPECT_FALSE(heap.empty());
  }

  {
    ailego::Heap<float> heap(12);
    ailego::Heap<float> heap1;
    ailego::Heap<float> heap2;

    for (size_t i = 0; i < 50; ++i) {
      heap.emplace(dist(gen));
    }

    EXPECT_NE(heap1.limit(), heap.limit());
    EXPECT_FALSE(heap.empty());
    EXPECT_TRUE(heap1.empty());
    heap1 = heap;

    EXPECT_FALSE(heap.empty());
    EXPECT_FALSE(heap1.empty());
    EXPECT_EQ(heap1.limit(), heap.limit());

    heap2 = std::move(heap);
    EXPECT_TRUE(heap.empty());
    EXPECT_FALSE(heap2.empty());
    EXPECT_EQ(heap2.limit(), heap.limit());
  }

  {
    ailego::Heap<float> heap(12);
    ailego::Heap<float> heap1;

    for (size_t i = 0; i < 50; ++i) {
      heap.emplace(dist(gen));
    }

    heap.swap(heap1);
    EXPECT_FALSE(heap1.empty());
    EXPECT_TRUE(heap.empty());
  }

  {
    ailego::Heap<float> heap(32);

    for (size_t i = 0; i < 200; ++i) {
      heap.emplace(dist(gen));
    }
    EXPECT_EQ(32u, heap.size());
    EXPECT_TRUE(std::is_heap(heap.begin(), heap.end()));

    heap.limit(55);
    for (size_t i = 0; i < 100; ++i) {
      heap.emplace(dist(gen));
    }
    EXPECT_TRUE(std::is_heap(heap.begin(), heap.end()));
    EXPECT_EQ(55u, heap.size());
    EXPECT_TRUE(heap.full());
  }
}

TEST(Heap, Make) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 100);

  std::vector<float> raw_data;
  for (size_t i = 0; i < 200; ++i) {
    raw_data.push_back(dist(gen));
  }

  ailego::Heap<float> heap(raw_data);
  EXPECT_FALSE(raw_data.empty());
  EXPECT_EQ(heap.front(), *std::max_element(raw_data.begin(), raw_data.end()));

  ailego::Heap<float> heap1(std::move(raw_data));
  EXPECT_TRUE(raw_data.empty());
  EXPECT_EQ(heap1.front(), *std::max_element(heap.begin(), heap.end()));

  raw_data = std::move(heap);
  EXPECT_FALSE(raw_data.empty());
  EXPECT_TRUE(heap.empty());
}

TEST(Heap, Sort) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 100);

  std::vector<float> raw_data;
  for (size_t i = 0; i < 200; ++i) {
    raw_data.push_back(dist(gen));
  }

  ailego::Heap<float> heap(raw_data);
  EXPECT_EQ(heap.front(), *std::max_element(raw_data.begin(), raw_data.end()));

  heap.sort();
  EXPECT_EQ(heap.front(), *std::min_element(raw_data.begin(), raw_data.end()));

  heap.limit(50);
  EXPECT_EQ(200u, heap.size());
  heap.update();
  EXPECT_EQ(50u, heap.size());
  EXPECT_EQ(heap.front(), *std::max_element(heap.begin(), heap.end()));

  heap.sort();
  EXPECT_EQ(heap.front(), *std::min_element(raw_data.begin(), raw_data.end()));
}

struct HeapValue {
  HeapValue(void) : score(0.0f) {
    std::cout << "HeapValue(void)" << std::endl;
  }

  HeapValue(float val) : score(val) {
    std::cout << "HeapValue(float)" << std::endl;
  }

  HeapValue(const HeapValue &rhs) : score(rhs.score) {
    std::cout << "HeapValue(const HeapValue &)" << std::endl;
  }

  HeapValue(HeapValue &&rhs) : score(rhs.score) {
    std::cout << "HeapValue(HeapValue &&)" << std::endl;
  }

  //! Less than
  bool operator<(const HeapValue &rhs) const {
    return (this->score < rhs.score);
  }

  //! Greater than
  bool operator>(const HeapValue &rhs) const {
    return (this->score > rhs.score);
  }

  //! Assignment
  HeapValue &operator=(const HeapValue &rhs) {
    std::cout << "operator=(const HeapValue &)" << std::endl;
    score = rhs.score;
    return *this;
  }

  //! Assignment
  HeapValue &operator=(HeapValue &&rhs) {
    std::cout << "operator=(HeapValue &&)" << std::endl;
    score = rhs.score;
    return *this;
  }

  float score;
};

TEST(Heap, Constructor) {
  ailego::Heap<HeapValue> heap(2);
  heap.push(HeapValue(2.0f));
  heap.emplace(1.0f);

  HeapValue val;
  heap.push(val);

  heap.pop();
  EXPECT_EQ(1u, heap.size());
  heap.pop();
  EXPECT_EQ(0u, heap.size());
  // heap.pop(); // disallowed
}

template <typename T, class TAllocator = std::allocator<T>>
class HeapVector {
 public:
  typedef size_t size_type;
  typedef typename std::remove_reference<T>::type value_type;
  typedef TAllocator allocator_type;

  //! Constructor
  HeapVector(void) : begin_(nullptr), end_(nullptr), capacity_(0u), alloc_() {}

  //! Constructor
  HeapVector(const HeapVector &rhs)
      : begin_(nullptr), end_(nullptr), capacity_(0u), alloc_() {
    size_type count = rhs.size();
    if (count) {
      this->expand(count);

      end_ = begin_ + count;
      for (value_type *iter = begin_, *src = rhs.begin_; iter != end_;
           ++iter, ++src) {
        iter->value_type(*src);
      }
    }
  }

  //! Constructor
  HeapVector(HeapVector &&rhs)
      : begin_(rhs.begin_), end_(rhs.end_), capacity_(rhs.capacity_), alloc_() {
    rhs.begin_ = nullptr;
    rhs.end_ = nullptr;
    rhs.capacity_ = 0u;
  }

  //! Destructor
  ~HeapVector(void) {
    if (capacity_) {
      for (value_type *iter = begin_; iter != end_; ++iter) {
        iter->~value_type();
      }
      alloc_.deallocate(begin_, capacity_);
    }
  }

  //! Assignment
  HeapVector &operator=(const HeapVector &rhs) {
    this->clear();

    size_type count = rhs.size();
    if (capacity_ < count) {
      this->expand(count);
    }

    if (count) {
      end_ = begin_ + count;
      for (value_type *iter = begin_, *src = rhs.begin_; iter != end_;
           ++iter, ++src) {
        iter->value_type(*src);
      }
    }
    return *this;
  }

  //! Assignment
  HeapVector &operator=(HeapVector &&rhs) {
    this->clear();
    begin_ = rhs.begin_;
    end_ = rhs.end_;
    capacity_ = rhs.capacity_;
    rhs.begin_ = nullptr;
    rhs.end_ = nullptr;
    rhs.capacity_ = 0u;
    return *this;
  }

  //! Clear the vector
  void clear(void) {
    for (value_type *iter = begin_; iter != end_; ++iter) {
      iter->~value_type();
    }
    end_ = begin_;
  }

  //! Retrieve the begin iterator
  value_type *begin(void) {
    return begin_;
  }

  //! Retrieve the begin iterator
  const value_type *begin(void) const {
    return begin_;
  }

  //! Retrieve the end iterator
  value_type *end(void) {
    return end_;
  }

  //! Retrieve the end iterator
  const value_type *end(void) const {
    return end_;
  }

  //! Retrieve the front element
  value_type &front(void) {
    return *begin_;
  }

  //! Retrieve the front element
  const value_type &front(void) const {
    return *begin_;
  }

  //! Retrieve the back element
  value_type &back(void) {
    return *(end_ - 1);
  }

  //! Retrieve the back element
  const value_type &back(void) const {
    return *(end_ - 1);
  }

  //! Retrieve count of elements in vector
  size_type size(void) const {
    return (end_ - begin_);
  }

  //! Retrieve capacity of vector
  size_type capacity(void) const {
    return capacity_;
  }

  //! Check whether the heap is empty
  bool empty(void) const {
    return (begin_ == end_);
  }

  //! Request a change in capacity
  void reserve(size_type n) {
    if (capacity_ < n) {
      this->expand(n);
    }
  }

  void push_back(const value_type &val) {
    size_type count = this->size();

    if (count == capacity_) {
      this->expand(count + 1);
    }
    // (end_++)->value_type(val);
    *(end_++) = val;
  }

  void push_back(value_type &&val) {
    size_type count = this->size();

    if (count == capacity_) {
      this->expand(count + 1);
    }
    // (end_++)->value_type(std::move(val));
    *(end_++) = std::move(val);
  }

  void pop_back(void) {
    (--end_)->~value_type();
  }

 protected:
  //! Find the number which is upper power of 2
  static inline size_type clp2(size_type n) {
    n = n - 1;
    n = n | (n >> 1);
    n = n | (n >> 2);
    n = n | (n >> 4);
    n = n | (n >> 8);
    n = n | (n >> 16);
    // n = n | (n >> 32);
    return (n + 1);
  }

  //! Expand the buffer
  void expand(size_type need) {
    need = clp2(need);
    value_type *buf = alloc_.allocate(need);
    size_type count = this->size();

    if (count) {
      memcpy(buf, begin_, sizeof(value_type) * count);
    }
    alloc_.deallocate(begin_, capacity_);
    begin_ = buf;
    end_ = buf + count;
    capacity_ = need;
  }

 private:
  //! Members
  value_type *begin_;
  value_type *end_;
  size_type capacity_;
  allocator_type alloc_;
};

TEST(Heap, Becnhmark) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 100);

  std::vector<float> raw_data;
  for (size_t i = 0; i < 1000000; ++i) {
    raw_data.push_back(dist(gen));
  }

  ailego::Heap<float> heap1(100);
  ailego::Heap<float, std::less<float>, HeapVector<float>> heap2(100);

  ailego::ElapsedTime stamp;
  stamp.reset();
  for (uint32_t i = 0; i < raw_data.size(); ++i) {
    heap1.emplace(raw_data[i]);
  }
  std::cout << "Heap 1: " << stamp.milli_seconds() << " ms" << std::endl;
  EXPECT_EQ(100u, heap1.size());

  stamp.reset();
  for (uint32_t i = 0; i < raw_data.size(); ++i) {
    heap2.push(raw_data[i]);
  }
  std::cout << "Heap 2: " << stamp.milli_seconds() << " ms" << std::endl;
  EXPECT_EQ(100u, heap2.size());
}
