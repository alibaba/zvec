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

#include <zvec/ailego/container/hypercube.h>

namespace zvec {
namespace ailego {

//! Trying compatible with T
#define _TRYING_COMPATIBLE(cube, T, out)                                     \
  if (cube->compatible<T>())                                                 \
  return (                                                                   \
      *out = static_cast<typename std::remove_pointer<decltype(out)>::type>( \
          cube->unsafe_cast<T>()),                                           \
      true)

//! Trying compatible with T (Boolean)
#define _TRYING_COMPATIBLE_BOOL(cube, T, out) \
  if (cube->compatible<T>()) return (*out = !!cube->unsafe_cast<T>(), true)

//! Trying compatible with T (String)
#define _TRYING_COMPATIBLE_STRING(cube, T, out) \
  if (cube->compatible<T>())                    \
  return (out->assign(std::to_string(cube->unsafe_cast<T>())), true)

//! Trying convert string
#define _TRYING_CONVERT_STRING(cube, out)                                      \
  if (cube->compatible<std::string>())                                         \
  return (*out = Params::StringCast<std::remove_pointer<decltype(out)>::type>( \
              cube->unsafe_cast<std::string>()),                               \
          true)

/*! Index Params
 */
class Params {
 public:
  //! Constructor
  Params(void) : hypercube_() {}

  //! Constructor
  Params(const Params &rhs) : hypercube_(rhs.hypercube_) {}

  //! Constructor
  Params(Params &&rhs) : hypercube_() {
    hypercube_.swap(rhs.hypercube_);
  }

  //! Destructor
  ~Params(void) {}

  //! Assignment
  Params &operator=(const Params &rhs) {
    hypercube_ = rhs.hypercube_;
    return *this;
  }

  //! Assignment
  Params &operator=(Params &&rhs) {
    hypercube_.swap(rhs.hypercube_);
    return *this;
  }

  //! Overloaded operator []
  ailego::Cube &operator[](const std::string &key) {
    return hypercube_[key];
  }

  //! Overloaded operator []
  ailego::Cube &operator[](std::string &&key) {
    return hypercube_[std::move(key)];
  }

  //! Test if the element is exist
  bool has(const std::string &key) const {
    return hypercube_.has(key);
  }

  //! Test if the map is empty
  bool empty(void) const {
    return hypercube_.empty();
  }

  //! Clear the map
  void clear(void) {
    hypercube_.clear();
  }

  //! Erase the pair via a key
  bool erase(const std::string &key) {
    return hypercube_.erase(key);
  }

  //! Merge another index params
  void merge(const Params &rhs) {
    hypercube_.merge(rhs.hypercube_);
  }

  //! Merge another index params
  void merge(Params &&rhs) {
    hypercube_.merge(std::move(rhs.hypercube_));
  }

  //! Set the value of key in T
  template <typename T>
  bool insert(const std::string &key, T &&val) {
    return hypercube_.insert<T>(key, std::forward<T>(val));
  }

  //! Set the value of key in T
  template <typename T>
  bool insert(std::string &&key, T &&val) {
    return hypercube_.insert<T>(std::forward<std::string>(key),
                                std::forward<T>(val));
  }

  //! Set the value of key in T
  template <typename T>
  void set(const std::string &key, T &&val) {
    hypercube_.insert_or_assign<T>(key, std::forward<T>(val));
  }

  //! Set the value of key in T
  template <typename T>
  void set(std::string &&key, T &&val) {
    hypercube_.insert_or_assign<T>(std::forward<std::string>(key),
                                   std::forward<T>(val));
  }

  //! Retrieve the value in boolean
  bool get(const std::string &key, bool *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE_BOOL(cube, char, out);
      _TRYING_COMPATIBLE_BOOL(cube, unsigned char, out);
      _TRYING_COMPATIBLE_BOOL(cube, signed char, out);
      _TRYING_COMPATIBLE_BOOL(cube, short int, out);
      _TRYING_COMPATIBLE_BOOL(cube, unsigned short int, out);
      _TRYING_COMPATIBLE_BOOL(cube, int, out);
      _TRYING_COMPATIBLE_BOOL(cube, unsigned int, out);
      _TRYING_COMPATIBLE_BOOL(cube, long int, out);
      _TRYING_COMPATIBLE_BOOL(cube, unsigned long int, out);
      _TRYING_COMPATIBLE_BOOL(cube, long long int, out);
      _TRYING_COMPATIBLE_BOOL(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE_BOOL(cube, float, out);
      _TRYING_COMPATIBLE_BOOL(cube, double, out);
      _TRYING_COMPATIBLE_BOOL(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'char'
  bool get(const std::string &key, char *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'unsigned char'
  bool get(const std::string &key, unsigned char *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'signed char'
  bool get(const std::string &key, signed char *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'short int'
  bool get(const std::string &key, short int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'unsigned short int'
  bool get(const std::string &key, unsigned short int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'int'
  bool get(const std::string &key, int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'unsigned int'
  bool get(const std::string &key, unsigned int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'long int'
  bool get(const std::string &key, long int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'unsigned long int'
  bool get(const std::string &key, unsigned long int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'long long int'
  bool get(const std::string &key, long long int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'unsigned long long int'
  bool get(const std::string &key, unsigned long long int *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'float'
  bool get(const std::string &key, float *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'double'
  bool get(const std::string &key, double *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in 'long double'
  bool get(const std::string &key, long double *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, long double, out);
      _TRYING_COMPATIBLE(cube, double, out);
      _TRYING_COMPATIBLE(cube, float, out);
      _TRYING_COMPATIBLE(cube, long long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE(cube, long int, out);
      _TRYING_COMPATIBLE(cube, unsigned long int, out);
      _TRYING_COMPATIBLE(cube, int, out);
      _TRYING_COMPATIBLE(cube, unsigned int, out);
      _TRYING_COMPATIBLE(cube, short int, out);
      _TRYING_COMPATIBLE(cube, unsigned short int, out);
      _TRYING_COMPATIBLE(cube, char, out);
      _TRYING_COMPATIBLE(cube, unsigned char, out);
      _TRYING_COMPATIBLE(cube, signed char, out);
      _TRYING_COMPATIBLE(cube, bool, out);
      _TRYING_CONVERT_STRING(cube, out);
    }
    return false;
  }

  //! Retrieve the value in string
  bool get(const std::string &key, std::string *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, std::string, out);
      _TRYING_COMPATIBLE_STRING(cube, bool, out);
      _TRYING_COMPATIBLE_STRING(cube, char, out);
      _TRYING_COMPATIBLE_STRING(cube, unsigned char, out);
      _TRYING_COMPATIBLE_STRING(cube, signed char, out);
      _TRYING_COMPATIBLE_STRING(cube, short int, out);
      _TRYING_COMPATIBLE_STRING(cube, unsigned short int, out);
      _TRYING_COMPATIBLE_STRING(cube, int, out);
      _TRYING_COMPATIBLE_STRING(cube, unsigned int, out);
      _TRYING_COMPATIBLE_STRING(cube, long int, out);
      _TRYING_COMPATIBLE_STRING(cube, unsigned long int, out);
      _TRYING_COMPATIBLE_STRING(cube, long long int, out);
      _TRYING_COMPATIBLE_STRING(cube, unsigned long long int, out);
      _TRYING_COMPATIBLE_STRING(cube, float, out);
      _TRYING_COMPATIBLE_STRING(cube, double, out);
      _TRYING_COMPATIBLE_STRING(cube, long double, out);
    }
    return false;
  }

  //! Retrieve the value in T
  template <typename T>
  bool get(const std::string &key, T *out) const {
    const ailego::Cube *cube = hypercube_.get(key);
    if (cube) {
      _TRYING_COMPATIBLE(cube, T, out);
    }
    return false;
  }

  //! Retrieve the value in boolean
  bool get_as_bool(const std::string &key) const {
    bool result = false;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in int8
  int8_t get_as_int8(const std::string &key) const {
    int8_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in int16
  int16_t get_as_int16(const std::string &key) const {
    int16_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in int32
  int32_t get_as_int32(const std::string &key) const {
    int32_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in int64
  int64_t get_as_int64(const std::string &key) const {
    int64_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in uint8
  uint8_t get_as_uint8(const std::string &key) const {
    uint8_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in uint16
  uint16_t get_as_uint16(const std::string &key) const {
    uint16_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in uint32
  uint32_t get_as_uint32(const std::string &key) const {
    uint32_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in uint64
  uint64_t get_as_uint64(const std::string &key) const {
    uint64_t result = 0;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in float
  float get_as_float(const std::string &key) const {
    float result = 0.0f;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in double
  double get_as_double(const std::string &key) const {
    double result = 0.0f;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the value in string
  std::string get_as_string(const std::string &key) const {
    std::string result;
    this->get(key, &result);
    return result;
  }

  //! Retrieve the debug string
  std::string debug_string(void) const {
    std::string str;
    SerializeToBuffer(*this, &str);
    return str;
  }

  //! Retrieve the map of parameters
  const ailego::Hypercube &hypercube(void) const {
    return hypercube_;
  }

  //! Retrieve the map of parameters
  ailego::Hypercube *mutable_hypercube(void) {
    return &hypercube_;
  }

  //! Parse parameters from buffer (Json format)
  static bool ParseFromBuffer(const std::string &buf, Params *params);

  //! Parse parameters from OS environment
  static void ParseFromEnvironment(Params *params);

  //! Serialize parameters into buffer
  static void SerializeToBuffer(const Params &params, std::string *buf);

 protected:
  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, float>::value, T>::type {
    return std::strtof(str.c_str(), nullptr);
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, double>::value, T>::type {
    return std::strtod(str.c_str(), nullptr);
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, long double>::value, T>::type {
    return std::strtold(str.c_str(), nullptr);
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, char>::value, T>::type {
    return static_cast<char>(std::strtol(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, signed char>::value, T>::type {
    return static_cast<signed char>(std::strtol(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, unsigned char>::value, T>::type {
    return static_cast<unsigned char>(std::strtoul(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, short int>::value, T>::type {
    return static_cast<short int>(std::strtol(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, int>::value, T>::type {
    return static_cast<int>(std::strtol(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, long int>::value, T>::type {
    return static_cast<long int>(std::strtol(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, long long int>::value, T>::type {
    return static_cast<long long int>(std::strtoll(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, unsigned short int>::value,
                              T>::type {
    return static_cast<unsigned short int>(
        std::strtoul(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, unsigned int>::value, T>::type {
    return static_cast<unsigned int>(std::strtoul(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, unsigned long int>::value,
                              T>::type {
    return static_cast<unsigned long int>(
        std::strtoul(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, unsigned long long int>::value,
                              T>::type {
    return static_cast<unsigned long long int>(
        std::strtoull(str.c_str(), nullptr, 0));
  }

  //! Convert string type to another type
  template <typename T>
  static auto StringCast(const std::string &str) ->
      typename std::enable_if<std::is_same<T, bool>::value, T>::type {
    if (str.empty()) {
      return false;
    }
    char c = str[0];
    if (c == 'Y' || c == 'T' || c == 'y' || c == 't') {
      return true;
    }
    return !!std::strtof(str.c_str(), nullptr);
  }

 private:
  ailego::Hypercube hypercube_;
};

#undef _TRYING_COMPATIBLE
#undef _TRYING_COMPATIBLE
#undef _TRYING_COMPATIBLE_BOOL
#undef _TRYING_COMPATIBLE_STRING
#undef _TRYING_CONVERT_STRING

}  // namespace ailego
}  // namespace zvec