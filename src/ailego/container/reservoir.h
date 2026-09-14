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

#include <random>
#include <vector>

namespace zvec {
namespace ailego {

/*! Sampling Reservoir
 */
template <typename T, typename Allocator = std::allocator<T>>
class Reservoir {
 public:
  //! Constructor
  Reservoir(size_t cnt)
      : samples_(cnt), total_(0), mt_(std::random_device()()), pool_() {
    pool_.reserve(samples_);
  }

  //! Constructor
  Reservoir(const Reservoir &rhs)
      : samples_(rhs.samples_),
        total_(rhs.total_),
        mt_(std::random_device()()),
        pool_(rhs.pool_) {}

  //! Constructor
  Reservoir(Reservoir &&rhs)
      : samples_(rhs.samples_),
        total_(rhs.total_),
        mt_(std::random_device()()),
        pool_(std::move(rhs.pool_)) {}

  //! Destructor
  ~Reservoir(void) = default;

  //! Assignment
  Reservoir &operator=(const Reservoir &rhs) {
    samples_ = rhs.samples_;
    total_ = rhs.total_;
    pool_ = rhs.pool_;
    return *this;
  }

  //! Assignment
  Reservoir &operator=(Reservoir &&rhs) {
    samples_ = rhs.samples_;
    total_ = rhs.total_;
    pool_ = std::move(rhs.pool_);
    return *this;
  }

  //! Retrieve pool of reservoir
  std::vector<T, Allocator> *mutable_pool(void) {
    return &pool_;
  }

  //! Retrieve pool of reservoir
  const std::vector<T, Allocator> &pool(void) const {
    return pool_;
  }

  //! Retrieve count of samples
  size_t samples(void) const {
    return samples_;
  }

  //! Retrieve total count of filling
  size_t total(void) const {
    return total_;
  }

  //! Reset the reservoir
  void reset(void) {
    total_ = 0;
    pool_.clear();
    pool_.reserve(samples_);
  }

  //! Fill the reservoir
  void fill(const T &item) {
    if (samples_ > 0) {
      if (pool_.size() >= samples_) {
        std::uniform_int_distribution<size_t> dt(0, total_);
        size_t i = dt(mt_);

        if (i < samples_) {
          pool_[i] = item;
        }
      } else {
        pool_.push_back(item);
      }
    }
    ++total_;
  }

  //! Fill the reservoir
  void fill(T &&item) {
    if (samples_ > 0) {
      if (pool_.size() >= samples_) {
        std::uniform_int_distribution<size_t> dt(0, total_);
        size_t i = dt(mt_);

        if (i < samples_) {
          pool_[i] = std::move(item);
        }
      } else {
        pool_.push_back(std::move(item));
      }
    }
    ++total_;
  }

  //! Disable them
  Reservoir(void) = delete;

 private:
  //! Members
  size_t samples_;
  size_t total_;
  std::mt19937 mt_;
  std::vector<T, Allocator> pool_;
};

}  // namespace ailego
}  // namespace zvec