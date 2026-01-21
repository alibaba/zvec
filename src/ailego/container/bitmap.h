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
#include <vector>
#include <ailego/utility/bitset_helper.h>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

/*! Fixed Bitset
 */
template <size_t N, typename = typename std::enable_if<N % 32 == 0>::type>
class FixedBitset {
 public:
  enum { MAX_SIZE = N };

  //! Constructor
  FixedBitset(void) {
    memset(array_, 0, sizeof(array_));
  }

  //! Constructor
  FixedBitset(const FixedBitset &rhs) {
    memcpy(array_, rhs.array_, sizeof(array_));
  }

  //! Destructor
  ~FixedBitset(void) {}

  //! Assignment
  FixedBitset &operator=(const FixedBitset &rhs) {
    memcpy(array_, rhs.array_, sizeof(array_));
    return *this;
  }

  //! Retrieve data pointer
  uint32_t *data(void) {
    return reinterpret_cast<uint32_t *>(array_);
  }

  //! Retrieve data pointer
  const uint32_t *data(void) const {
    return reinterpret_cast<const uint32_t *>(array_);
  }

  //! Retrieve count of bits in set
  constexpr size_t size(void) const {
    return MAX_SIZE;
  }

  // ！Clear the bitset
  void clear(void) {
    memset(array_, 0, sizeof(array_));
  }

  //! Test a bit in bitset
  bool test(size_t num) const {
    ailego_assert_with(N > num, "overflow argument");
    return ((array_[num >> 5] & (1u << (num & 0x1f))) != 0);
  }

  //! Set a bit in bitset
  void set(size_t num) {
    ailego_assert_with(N > num, "overflow argument");
    uint32_t mask = (1u << (num & 0x1f));
    array_[num >> 5] |= mask;
  }

  //! Clear a bit in bitset
  void reset(size_t num) {
    ailego_assert_with(N > num, "overflow argument");
    uint32_t mask = (1u << (num & 0x1f));
    array_[num >> 5] &= ~mask;
  }

  //! Toggle a bit in bitset
  void flip(size_t num) {
    ailego_assert_with(N > num, "overflow argument");
    uint32_t mask = (1u << (num & 0x1f));
    array_[num >> 5] ^= mask;
  }

  //! Perform binary AND
  void bitwise_and(const FixedBitset &rhs) {
    BitsetHelper::BitwiseAnd(array_, rhs.array_, ((N + 0x1f) >> 5));
  }

  //! Perform binary AND NOT
  void bitwise_andnot(const FixedBitset &rhs) {
    BitsetHelper::BitwiseAndnot(array_, rhs.array_, ((N + 0x1f) >> 5));
  }

  //! Perform binary OR
  void bitwise_or(const FixedBitset &rhs) {
    BitsetHelper::BitwiseOr(array_, rhs.array_, ((N + 0x1f) >> 5));
  }

  //! Perform binary XOR
  void bitwise_xor(const FixedBitset &rhs) {
    BitsetHelper::BitwiseXor(array_, rhs.array_, ((N + 0x1f) >> 5));
  }

  //! Perform binary NOT
  void bitwise_not(void) {
    BitsetHelper::BitwiseNot(array_, ((N + 0x1f) >> 5));
  }

  //! Check if all bits are set to true
  bool test_all(void) const {
    return BitsetHelper::TestAll(array_, ((N + 0x1f) >> 5));
  }

  //! Check if any bits are set to true
  bool test_any(void) const {
    return BitsetHelper::TestAny(array_, ((N + 0x1f) >> 5));
  }

  //! Check if none of the bits are set to true
  bool test_none(void) const {
    return BitsetHelper::TestNone(array_, ((N + 0x1f) >> 5));
  }

  //! Compute the cardinality of a bitset
  size_t cardinality(void) const {
    return BitsetHelper::Cardinality(array_, ((N + 0x1f) >> 5));
  }

  //! Extract the bitset to an array
  void extract(size_t base, std::vector<size_t> *out) const {
    const uint32_t *iter = array_;
    const uint32_t *last = array_ + ((N + 0x1f) >> 5);

    for (; iter != last; ++iter) {
      uint32_t w = *iter;

      while (w != 0) {
        uint32_t c = ailego_ctz32(w);
        w &= ~(1u << c);
        out->push_back(base + c);
      }
      base += 32u;
    }
  }

  //! Extract the bitset to an array
  void extract(std::vector<size_t> *out) const {
    this->extract(0, out);
  }

  //! Compute the AND cardinality between two bitsets
  static size_t BitwiseAndCardinality(const FixedBitset &lhs,
                                      const FixedBitset &rhs) {
    return BitsetHelper::BitwiseAndCardinality(lhs.array_, rhs.array_,
                                               ((N + 0x1f) >> 5));
  }

  //! Compute the ANDNOT cardinality between two bitsets
  static size_t BitwiseAndnotCardinality(const FixedBitset &lhs,
                                         const FixedBitset &rhs) {
    return BitsetHelper::BitwiseAndnotCardinality(lhs.array_, rhs.array_,
                                                  ((N + 0x1f) >> 5));
  }

  //! Compute the XOR cardinality between two bitsets
  static size_t BitwiseXorCardinality(const FixedBitset &lhs,
                                      const FixedBitset &rhs) {
    return BitsetHelper::BitwiseXorCardinality(lhs.array_, rhs.array_,
                                               ((N + 0x1f) >> 5));
  }

  //! Compute the OR cardinality between two bitsets
  static size_t BitwiseOrCardinality(const FixedBitset &lhs,
                                     const FixedBitset &rhs) {
    return BitsetHelper::BitwiseOrCardinality(lhs.array_, rhs.array_,
                                              ((N + 0x1f) >> 5));
  }

  //! Convert a array pointer to bitset pointer
  static FixedBitset *Cast(uint32_t *arr) {
    return reinterpret_cast<FixedBitset<N> *>(arr);
  }

  //! Convert a array pointer to bitset pointer
  static const FixedBitset *Cast(const uint32_t *arr) {
    return reinterpret_cast<const FixedBitset<N> *>(arr);
  }

  //! Convert a array pointer to bitset pointer
  static FixedBitset *Cast(uint64_t *arr) {
    return reinterpret_cast<FixedBitset<N> *>(arr);
  }

  //! Convert a array pointer to bitset pointer
  static const FixedBitset *Cast(const uint64_t *arr) {
    return reinterpret_cast<const FixedBitset<N> *>(arr);
  }

 private:
  uint32_t array_[(N + 0x1f) >> 5];
};

/*! Fixed Bitset (Special)
 */
template <>
class FixedBitset<0> {
 public:
  enum { MAX_SIZE = 0 };

  //! Retrieve max size of bitset
  constexpr size_t size(void) const {
    return MAX_SIZE;
  }
};

/*! Bitset
 */
class Bitset {
 public:
  //! Constructor
  Bitset(void) : array_() {}

  //! Constructor
  Bitset(size_t bits) : array_((bits + 0x1f) >> 5) {}

  //! Constructor
  Bitset(const Bitset &rhs) : array_(rhs.array_) {}

  //! Constructor
  Bitset(Bitset &&rhs) : array_(std::move(rhs.array_)) {}

  //! Destructor
  ~Bitset(void) {}

  //! Assignment
  Bitset &operator=(const Bitset &rhs) {
    array_ = rhs.array_;
    return *this;
  }

  //! Assignment
  Bitset &operator=(Bitset &&rhs) {
    array_ = std::move(rhs.array_);
    return *this;
  }

  //! Retrieve data pointer
  uint32_t *data(void) {
    return array_.data();
  }

  //! Retrieve data pointer
  const uint32_t *data(void) const {
    return array_.data();
  }

  //! Retrieve count of bits in set
  size_t size(void) const {
    return (array_.size() << 5);
  }

  //! Resize the bitset
  void resize(size_t bits) {
    array_.resize((bits + 0x1f) >> 5);
  }

  // ！Clear the bitset
  void clear(void) {
    array_.clear();
  }

  //! Test a bit in bitset
  bool test(size_t num) const {
    ailego_assert_with(this->size() > num, "overflow argument");
    return ((array_[num >> 5] & (1u << (num & 0x1f))) != 0);
  }

  //! Set a bit in bitset
  void set(size_t num) {
    ailego_assert_with(this->size() > num, "overflow argument");
    uint32_t mask = (1u << (num & 0x1f));
    array_[num >> 5] |= mask;
  }

  //! Clear a bit in bitset
  void reset(size_t num) {
    ailego_assert_with(this->size() > num, "overflow argument");
    uint32_t mask = (1u << (num & 0x1f));
    array_[num >> 5] &= ~mask;
  }

  //! Toggle a bit in bitset
  void flip(size_t num) {
    ailego_assert_with(this->size() > num, "overflow argument");
    uint32_t mask = (1u << (num & 0x1f));
    array_[num >> 5] ^= mask;
  }

  //! Perform binary AND
  void bitwise_and(const Bitset &rhs) {
    BitsetHelper::BitwiseAnd(array_.data(), rhs.array_.data(),
                             std::min(array_.size(), rhs.array_.size()));
  }

  //! Perform binary AND NOT
  void bitwise_andnot(const Bitset &rhs) {
    BitsetHelper::BitwiseAndnot(array_.data(), rhs.array_.data(),
                                std::min(array_.size(), rhs.array_.size()));
  }

  //! Perform binary OR
  void bitwise_or(const Bitset &rhs) {
    BitsetHelper::BitwiseOr(array_.data(), rhs.array_.data(),
                            std::min(array_.size(), rhs.array_.size()));
  }

  //! Perform binary XOR
  void bitwise_xor(const Bitset &rhs) {
    BitsetHelper::BitwiseXor(array_.data(), rhs.array_.data(),
                             std::min(array_.size(), rhs.array_.size()));
  }

  //! Perform binary NOT
  void bitwise_not(void) {
    BitsetHelper::BitwiseNot(array_.data(), array_.size());
  }

  //! Check if all bits are set to true
  bool test_all(void) const {
    return BitsetHelper::TestAll(array_.data(), array_.size());
  }

  //! Check if any bits are set to true
  bool test_any(void) const {
    return BitsetHelper::TestAny(array_.data(), array_.size());
  }

  //! Check if none of the bits are set to true
  bool test_none(void) const {
    return BitsetHelper::TestNone(array_.data(), array_.size());
  }

  //! Compute the cardinality of a bitset
  size_t cardinality(void) const {
    return BitsetHelper::Cardinality(array_.data(), array_.size());
  }

  //! Extract the bitset to an array
  void extract(size_t base, std::vector<size_t> *out) const {
    const uint32_t *iter = array_.data();
    const uint32_t *last = array_.data() + array_.size();

    for (; iter != last; ++iter) {
      uint32_t w = *iter;

      while (w != 0) {
        uint32_t c = ailego_ctz32(w);
        w &= ~(1u << c);
        out->push_back(base + c);
      }
      base += 32u;
    }
  }

  //! Extract the bitset to an array
  void extract(std::vector<size_t> *out) const {
    this->extract(0, out);
  }

  //! Compute the AND cardinality between two bitsets
  static size_t BitwiseAndCardinality(const Bitset &lhs, const Bitset &rhs);

  //! Compute the ANDNOT cardinality between two bitsets
  static size_t BitwiseAndnotCardinality(const Bitset &lhs, const Bitset &rhs);

  //! Compute the XOR cardinality between two bitsets
  static size_t BitwiseXorCardinality(const Bitset &lhs, const Bitset &rhs);

  //! Compute the OR cardinality between two bitsets
  static size_t BitwiseOrCardinality(const Bitset &lhs, const Bitset &rhs);

 private:
  std::vector<uint32_t> array_;
};

/*! Bitmap
 */
class Bitmap {
 public:
  typedef FixedBitset<65536u> Bucket;

  //! Constructor
  Bitmap(void) : array_() {}

  //! Constructor
  Bitmap(const Bitmap &rhs) {
    this->copy(rhs);
  }

  //! Destructor
  ~Bitmap(void) {
    this->clear();
  }

  //! Assignment
  Bitmap &operator=(const Bitmap &rhs) {
    this->copy(rhs);
    return *this;
  }

  //! Retrieve bucket size of bitmap
  size_t bucket_size(void) const {
    return array_.size();
  }

  // ！Clear the bitmap
  void clear(void);

  //! Remove the none buckets
  void shrink_to_fit(void);

  //! Test a bit in bitmap
  bool test(size_t num) const;

  //! Set a bit in bitmap
  void set(size_t num);

  //! Reset a bit in bitmap
  void reset(size_t num);

  //! Toggle a bit in bitmap
  void flip(size_t num);

  //! Perform binary AND
  void bitwise_and(const Bitmap &rhs);

  //! Perform binary AND NOT
  void bitwise_andnot(const Bitmap &rhs);

  //! Perform binary OR
  void bitwise_or(const Bitmap &rhs);

  //! Perform binary XOR
  void bitwise_xor(const Bitmap &rhs);

  //! Perform binary NOT (It will expand the whole map)
  void bitwise_not(void);

  //! Check if all bits are set to true
  bool test_all(void) const;

  //! Check if any bits are set to true
  bool test_any(void) const;

  //! Check if none of the bits are set to true
  bool test_none(void) const;

  //! Compute the cardinality of a bitmap
  size_t cardinality(void) const;

  //! Extract the bitmap to an array
  void extract(size_t base, std::vector<size_t> *out) const;

  //! Extract the bitmap to an array
  void extract(std::vector<size_t> *out) const {
    this->extract(0, out);
  }

  //! Compute the AND cardinality between two bitmaps
  static size_t BitwiseAndCardinality(const Bitmap &lhs, const Bitmap &rhs);

  //! Compute the ANDNOT cardinality between two bitmaps
  static size_t BitwiseAndnotCardinality(const Bitmap &lhs, const Bitmap &rhs);

  //! Compute the XOR cardinality between two bitmaps
  static size_t BitwiseXorCardinality(const Bitmap &lhs, const Bitmap &rhs);

  //! Compute the OR cardinality between two bitmaps
  static size_t BitwiseOrCardinality(const Bitmap &lhs, const Bitmap &rhs);

 protected:
  //! Copy the content from another bitmap
  void copy(const Bitmap &rhs);

 private:
  std::vector<Bucket *> array_;
};

}  // namespace ailego
}  // namespace zvec