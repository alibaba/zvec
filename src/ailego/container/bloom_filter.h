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

#include <cmath>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {

/*! Bloom Filter Calculator
 */
struct BloomFilterCalculator {
  /**
   *  \brief          Calculate probability of false positives
   *  \param n        Number of items in the filter
   *  \param m        Number of bits in the filter
   *  \param k        Number of hash functions
   *  \return         Probability of false positives
   */
  static double Probability(size_t n, size_t m, size_t k) {
    return std::pow(1.0 - std::exp(-((double)k / (double)m * (double)n)), k);
  }

  /**
   *  \brief          Calculate number of items in the filter
   *  \param m        Number of bits in the filter
   *  \param k        Number of hash functions
   *  \param p        Probability of false positives
   *  \return         Number of items in the filter
   */
  static size_t NumberOfItems(size_t m, size_t k, double p) {
    return (size_t)std::ceil(
        -((double)m / (double)k *
          std::log(1.0 - std::exp(std::log(p) / (double)k))));
  }

  /**
   *  \brief          Calculate number of bits in the filter
   *  \param n        Number of items in the filter
   *  \param p        Probability of false positives
   *  \return         Number of bits in the filter
   */
  static size_t NumberOfBits(size_t n, double p) {
    return (size_t)std::ceil((double)n * std::log(p) /
                             std::log(1.0 / std::pow(2.0, std::log(2.0))));
  }

  /**
   *  \brief          Calculate number of bits in the filter
   *  \param n        Number of items in the filter
   *  \param k        Number of hash functions
   *  \param p        Probability of false positives
   *  \return         Number of bits in the filter
   */
  static size_t NumberOfBits(size_t n, size_t k, double p) {
    return (size_t)std::ceil(-((double)k * (double)n /
                               std::log(1.0 - std::pow(p, 1.0 / (double)k))));
  }

  /**
   *  \brief          Calculate number of bytes in the filter
   *  \param n        Number of items in the filter
   *  \param p        Probability of false positives
   *  \return         Number of bytes in the filter
   */
  static size_t NumberOfBytes(size_t n, double p) {
    return ((NumberOfBits(n, p) + 7) >> 3);
  }

  /**
   *  \brief          Calculate number of bits in the filter
   *  \param n        Number of items in the filter
   *  \param k        Number of hash functions
   *  \param p        Probability of false positives
   *  \return         Number of bits in the filter
   */
  static size_t NumberOfBytes(size_t n, size_t k, double p) {
    return ((NumberOfBits(n, k, p) + 7) >> 3);
  }

  /**
   *  \brief          Calculate number of hash functions
   *  \param n        Number of items in the filter
   *  \param m        Number of bits in the filter
   *  \return         Number of hash functions
   */
  static size_t NumberOfHash(size_t n, size_t m) {
    return (size_t)std::round((double)m / (double)n * std::log(2.0));
  }
};

/*! Bloom Filter
 */
template <size_t K>
class BloomFilter {
 public:
  //! Constructor
  BloomFilter(void) {}

  //! Constructor
  BloomFilter(size_t n, double p) {
    if (n > 0 && p > 0.0 && p < 1.0) {
      capacity_ = n;
      bits_count_ = BloomFilterCalculator::NumberOfBits(n, K, p);
      bits_count_ = ((bits_count_ + 31) >> 5) << 5;
      probability_ = BloomFilterCalculator::Probability(n, bits_count_, K);
      bitset_ = new uint32_t[bits_count_ >> 5];
      memset(bitset_, 0, (bits_count_ >> 3));
    }
  }

  //! Constructor
  BloomFilter(BloomFilter &&rhs)
      : bitset_(rhs.bitset_),
        bits_count_(rhs.bits_count_),
        capacity_(rhs.capacity_),
        count_(rhs.count_),
        probability_(rhs.probability_) {
    rhs.bitset_ = nullptr;
    rhs.bits_count_ = 0u;
    rhs.capacity_ = 0u;
    rhs.count_ = 0u;
    rhs.probability_ = 0u;
  }

  //! Destructor
  ~BloomFilter(void) {
    delete[] bitset_;
  }

  //! Test if the filter is valid
  bool is_valid(void) const {
    return (bitset_ != nullptr);
  }

  //! Reset the bloom filter
  bool reset(size_t n, double p) {
    if (n <= 0 || p <= 0.0 || p >= 1.0) {
      return false;
    }
    delete[] bitset_;
    capacity_ = n;
    count_ = 0u;
    bits_count_ = BloomFilterCalculator::NumberOfBits(n, K, p);
    bits_count_ = ((bits_count_ + 31) >> 5) << 5;
    probability_ = BloomFilterCalculator::Probability(n, bits_count_, K);
    bitset_ = new (std::nothrow) uint32_t[bits_count_ >> 5];
    if (!bitset_) {
      return false;
    }
    memset(bitset_, 0, (bits_count_ >> 3));
    return true;
  }

  //! Clear the bloom filter
  void clear(void) {
    if (bitset_) {
      memset(bitset_, 0, (bits_count_ >> 3));
      count_ = 0u;
    }
  }

  //! Insert a item into bloom filter
  template <typename... TArgs,
            typename = typename std::enable_if<
                Conjunction<std::is_integral<TArgs>...>::value &&
                sizeof...(TArgs) == K>::type>
  bool insert(TArgs... vals) {
    if (count_ >= capacity_) {
      return false;
    }
    this->set_bits(vals...);
    ++count_;
    return true;
  }

  //! Force insert a item into bloom filter
  template <typename... TArgs,
            typename = typename std::enable_if<
                Conjunction<std::is_integral<TArgs>...>::value &&
                sizeof...(TArgs) == K>::type>
  void force_insert(TArgs... vals) {
    this->set_bits(vals...);
    ++count_;
  }

  //! Insert a item into bloom filter
  template <typename... TArgs,
            typename = typename std::enable_if<
                Conjunction<std::is_integral<TArgs>...>::value &&
                sizeof...(TArgs) == K>::type>
  bool has(TArgs... vals) const {
    return this->test_bits(vals...);
  }

  //! Retrieve count of bits in bloom filter
  size_t bits_count(void) const {
    return bits_count_;
  }

  //! Retrieve capacity of bloom filter
  size_t capacity(void) const {
    return capacity_;
  }

  //! Retrieve count of items in bloom filter
  size_t count(void) const {
    return count_;
  }

  //! Retrieve probability of false positives
  double probability(void) const {
    return probability_;
  }

 protected:
  //! Disable them
  BloomFilter(const BloomFilter &) = delete;
  BloomFilter &operator=(const BloomFilter &) = delete;

  //! Set bits in bloom filter
  template <typename TArg>
  void set_bits(TArg val) {
    size_t num = static_cast<size_t>(val) % bits_count_;
    bitset_[num >> 5] |= (1u << (num & 0x1f));
  }

  //! Set bits in bloom filter
  template <typename TArg, typename... TArgs>
  void set_bits(TArg val, TArgs... vals) {
    this->set_bits(val);
    this->set_bits(vals...);
  }

  //! Test bits in bloom filter
  template <typename TArg>
  bool test_bits(TArg val) const {
    size_t num = static_cast<size_t>(val) % bits_count_;
    return ((bitset_[num >> 5] & (1u << (num & 0x1f))) != 0);
  }

  //! Test bits in bloom filter
  template <typename TArg, typename... TArgs>
  bool test_bits(TArg val, TArgs... vals) const {
    if (!this->test_bits(val)) {
      return false;
    }
    return this->test_bits(vals...);
  }

 private:
  uint32_t *bitset_{nullptr};
  size_t bits_count_{0u};
  size_t capacity_{0u};
  size_t count_{0u};
  double probability_{0.0};
};

/*! Bloom Filter (Special)
 */
template <>
struct BloomFilter<0> {};

}  // namespace ailego
}  // namespace zvec