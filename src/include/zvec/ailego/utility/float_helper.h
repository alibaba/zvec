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

#include <cstddef>
#include <cstdint>

namespace zvec {
namespace ailego {

/*! Float Helper
 */
struct FloatHelper {
  //! Convert FP16 to FP32
  static float ToFP32(uint16_t val);

  //! Convert FP16 to FP32 (array)
  static void ToFP32(const uint16_t *arr, size_t size, float *out);

  //! Convert FP16 to FP32 with normalization (array)
  static void ToFP32(const uint16_t *arr, size_t size, float norm, float *out);

  //! Convert FP32 to FP16
  static uint16_t ToFP16(float val);

  //! Convert FP32 to FP16 (array)
  static void ToFP16(const float *arr, size_t size, uint16_t *out);

  //! Convert FP32 to FP16 with normalization (array)
  static void ToFP16(const float *arr, size_t size, float norm, uint16_t *out);

  //! Convert FP16 to FP32 with normalization
  static inline float ToFP32(uint16_t val, float norm) {
    return (FloatHelper::ToFP32(val) / norm);
  }

  //! Convert FP32 to FP16 with normalization
  static inline uint16_t ToFP16(float val, float norm) {
    return FloatHelper::ToFP16(val / norm);
  }
};

#if !defined(__aarch64__)
/*! Half-Precision Floating Point
 */
class Float16 {
 public:
  //! Constructor
  Float16(void) : value_(0) {}

  //! Constructor
  Float16(float val) : value_(FloatHelper::ToFP16(val)) {}

  //! Constructor
  Float16(double val) : value_(FloatHelper::ToFP16(static_cast<float>(val))) {}

  //! Assigment
  Float16 &operator=(float val) {
    this->value_ = FloatHelper::ToFP16(val);
    return *this;
  }

  //! Assigment
  Float16 &operator+=(float val) {
    this->value_ = FloatHelper::ToFP16(FloatHelper::ToFP32(this->value_) + val);
    return *this;
  }

  //! Assigment
  Float16 &operator-=(float val) {
    this->value_ = FloatHelper::ToFP16(FloatHelper::ToFP32(this->value_) - val);
    return *this;
  }

  //! Assigment
  Float16 &operator*=(float val) {
    this->value_ = FloatHelper::ToFP16(FloatHelper::ToFP32(this->value_) * val);
    return *this;
  }

  //! Assigment
  Float16 &operator/=(float val) {
    this->value_ = FloatHelper::ToFP16(FloatHelper::ToFP32(this->value_) / val);
    return *this;
  }

  //! Retrieve value in FP32
  operator float() const {
    return FloatHelper::ToFP32(this->value_);
  }

  //! Equal operator
  bool operator==(const Float16 &rhs) const {
    return this->value_ == rhs.value_;
  }

  //! No equal operator
  bool operator!=(const Float16 &rhs) const {
    return this->value_ != rhs.value_;
  }

  //! Less than operator
  bool operator<(const Float16 &rhs) const {
    return FloatHelper::ToFP32(this->value_) < FloatHelper::ToFP32(rhs.value_);
  }

  //! Less than or equal operator
  bool operator<=(const Float16 &rhs) const {
    return FloatHelper::ToFP32(this->value_) <= FloatHelper::ToFP32(rhs.value_);
  }

  //! Greater than operator
  bool operator>(const Float16 &rhs) const {
    return FloatHelper::ToFP32(this->value_) > FloatHelper::ToFP32(rhs.value_);
  }

  //! Greater than or equal operator
  bool operator>=(const Float16 &rhs) const {
    return FloatHelper::ToFP32(this->value_) >= FloatHelper::ToFP32(rhs.value_);
  }

  //! Calculate the absolute value
  static inline Float16 Absolute(const Float16 &x) {
    Float16 abs;
    abs.value_ = static_cast<uint16_t>(x.value_ & 0x7fff);
    return abs;
  }

 private:
  uint16_t value_;
};
#else
/*! Half-Precision Floating Point
 */
class Float16 {
 public:
  //! Constructor
  Float16(void) : value_(0) {}

  //! Constructor
  Float16(__fp16 val) : value_(val) {}

  //! Assigment
  Float16 &operator=(__fp16 val) {
    this->value_ = val;
    return *this;
  }

  //! Assigment
  Float16 &operator+=(__fp16 val) {
    this->value_ = this->value_ + val;
    return *this;
  }

  //! Assigment
  Float16 &operator-=(__fp16 val) {
    this->value_ = this->value_ - val;
    return *this;
  }

  //! Assigment
  Float16 &operator*=(__fp16 val) {
    this->value_ = this->value_ * val;
    return *this;
  }

  //! Assigment
  Float16 &operator/=(__fp16 val) {
    this->value_ = this->value_ / val;
    return *this;
  }

  //! Retrieve value in FP16
  operator __fp16() const {
    return this->value_;
  }

  //! Equal operator
  bool operator==(const Float16 &rhs) const {
    return this->value_ == rhs.value_;
  }

  //! No equal operator
  bool operator!=(const Float16 &rhs) const {
    return this->value_ != rhs.value_;
  }

  //! Less than operator
  bool operator<(const Float16 &rhs) const {
    return this->value_ < rhs.value_;
  }

  //! Less than or equal operator
  bool operator<=(const Float16 &rhs) const {
    return this->value_ <= rhs.value_;
  }

  //! Greater than operator
  bool operator>(const Float16 &rhs) const {
    return this->value_ > rhs.value_;
  }

  //! Greater than or equal operator
  bool operator>=(const Float16 &rhs) const {
    return this->value_ >= rhs.value_;
  }

  //! Calculate the absolute value
  static inline Float16 Absolute(const Float16 &x) {
    Float16 abs(x.value_);
    uint16_t *p = reinterpret_cast<uint16_t *>(&abs.value_);
    *p &= 0x7fff;
    return abs;
  }

 private:
  __fp16 value_;
};
#endif

// Check size of Float16
static_assert(sizeof(Float16) == 2, "Float16 must be aligned with 2 bytes");

}  // namespace ailego
}  // namespace zvec
