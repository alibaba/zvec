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

#include <cfloat>
#include <cstring>
#include <stdexcept>
#include <string>
#include "mod_json.h"

namespace zvec {
namespace ailego {

/*! JSON String
 */
class JsonString {
 public:
  typedef mod_json_size_t size_type;
  typedef mod_json_ssize_t ssize_type;
  typedef mod_json_float_t float_type;
  typedef mod_json_integer_t integer_type;

  //! Constructor
  JsonString(void) : str_(0) {}

  //! Constructor
  JsonString(const JsonString &rhs) : str_(0) {
    if (rhs.str_) {
      str_ = mod_json_string_grab(rhs.str_);
    }
  }

#if __cplusplus >= 201103L
  //! Constructor
  JsonString(JsonString &&rhs) : str_(rhs.str_) {
    rhs.str_ = 0;
  }
#endif

  //! Constructor
  JsonString(const char *cstr) {
    str_ = cstr ? mod_json_string_set(cstr, (mod_json_size_t)std::strlen(cstr))
                : 0;
  }

  //! Constructor
  JsonString(const char *cstr, size_type len) {
    str_ = mod_json_string_set(cstr, len);
  }

  //! Constructor
  JsonString(const std::string &str) {
    str_ = mod_json_string_set(str.c_str(), (mod_json_size_t)str.size());
  }

  //! Destructor
  ~JsonString(void) {
    mod_json_string_unset(str_);
  }

  //! Assign new contents to the string, replacing its current content
  JsonString &operator=(const JsonString &rhs) {
    this->assign(rhs);
    return *this;
  }

#if __cplusplus >= 201103L
  //! Assign new contents to the string, replacing its current content
  JsonString &operator=(JsonString &&rhs) {
    this->assign(std::move(rhs));
    return *this;
  }
#endif

  //! Assign new contents to the string, replacing its current content
  JsonString &operator=(const char *cstr) {
    this->assign(cstr);
    return *this;
  }

  //! Assign new contents to the string, replacing its current content
  JsonString &operator=(const std::string &rhs) {
    this->assign(rhs);
    return *this;
  }

  //! Append a JSON string
  JsonString &operator+=(const JsonString &str) {
    this->append(str);
    return *this;
  }

  //! Append a c-style string
  JsonString &operator+=(const char *cstr) {
    this->append(cstr);
    return *this;
  }

  //! Append a character to string
  JsonString &operator+=(char c) {
    this->append(c);
    return *this;
  }

  //! Equality
  bool operator==(const JsonString &rhs) const {
    return (mod_json_string_compare(str_, rhs.str_) == 0);
  }

  //! No equality
  bool operator!=(const JsonString &rhs) const {
    return !(*this == rhs);
  }

  //! Retrieve the character at index n
  char &operator[](size_type n) {
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonString::operator[]");
    }
    return *(str_->first + n);
  }

  //! Retrieve the character at index n
  const char &operator[](size_type n) const {
    return *(str_->first + n);
  }

  //! Retrieve non-zero if the string is valid
  bool is_valid(void) const {
    return (str_ != (mod_json_string_t *)0);
  }

  //! Retrieve non-zero if the string is empty
  bool empty(void) const {
    return mod_json_string_empty(str_);
  }

  //! Assign a JSON string
  void assign(const JsonString &rhs) {
    mod_json_string_unset(str_);
    str_ = rhs.str_ ? mod_json_string_grab(rhs.str_) : 0;
  }

#if __cplusplus >= 201103L
  //! Assign a JSON string
  void assign(JsonString &&rhs) {
    mod_json_string_unset(str_);
    str_ = rhs.str_;
    rhs.str_ = 0;
  }
#endif

  //! Assign a c-style string
  void assign(const char *cstr) {
    if (cstr) {
      if (!copy_on_write() ||
          mod_json_string_assign(str_, cstr,
                                 (mod_json_size_t)std::strlen(cstr)) != 0) {
        throw std::runtime_error("JsonString::assign");
      }
    }
  }

  //! Assign a c-style string
  void assign(const char *cstr, size_type len) {
    if (!copy_on_write() || mod_json_string_assign(str_, cstr, len) != 0) {
      throw std::runtime_error("JsonString::assign");
    }
  }

  //! Assign a STL-style string
  void assign(const std::string &str) {
    if (!copy_on_write() ||
        mod_json_string_assign(str_, str.c_str(),
                               (mod_json_size_t)str.size()) != 0) {
      throw std::runtime_error("JsonString::assign");
    }
  }

  //! Append a JSON string
  void append(const JsonString &str) {
    if (str.str_) {
      if (!copy_on_write() || mod_json_string_add(str_, str.str_) != 0) {
        throw std::runtime_error("JsonString::append");
      }
    }
  }

  //! Append a c-style string
  void append(const char *cstr) {
    if (cstr) {
      if (!copy_on_write() ||
          mod_json_string_append(str_, cstr,
                                 (mod_json_size_t)std::strlen(cstr)) != 0) {
        throw std::runtime_error("JsonString::append");
      }
    }
  }

  //! Append a c-style string
  void append(const char *cstr, size_type len) {
    if (!copy_on_write() || mod_json_string_append(str_, cstr, len) != 0) {
      throw std::runtime_error("JsonString::append");
    }
  }

  //! Append a STL-style string
  void append(const std::string &str) {
    if (!copy_on_write() ||
        mod_json_string_append(str_, str.c_str(),
                               (mod_json_size_t)str.size()) != 0) {
      throw std::runtime_error("JsonString::append");
    }
  }

  //! Append a character to string
  void append(char c) {
    if (!copy_on_write() || mod_json_string_append(str_, &c, 1) != 0) {
      throw std::runtime_error("JsonString::append");
    }
  }

  //! Retrieve the character at index n
  char &at(size_type n) {
    if (this->size() <= n) {
      throw std::out_of_range("JsonString::at");
    }
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonString::at");
    }
    return *(str_->first + n);
  }

  //! Retrieve the character at index n
  const char &at(size_type n) const {
    if (this->size() <= n) {
      throw std::out_of_range("JsonString::at");
    }
    return *(str_->first + n);
  }

  //! Request a change in capacity
  void reserve(size_type n) {
    if (!copy_on_write() || mod_json_string_reserve(str_, n) != 0) {
      throw std::runtime_error("JsonString::reserve");
    }
  }

  //! Clear the JSON string
  void clear(void) {
    mod_json_string_unset(str_);
    str_ = 0;
  }

  //! Exchange the content with another JSON string
  void swap(JsonString &rhs) {
    mod_json_string_t *str = str_;
    str_ = rhs.str_;
    rhs.str_ = str;
  }

  //! Retrieve the data pointer
  char *data(void) {
    return mod_json_string_data(str_);
  }

  //! Retrieve the data pointer
  const char *data(void) const {
    return mod_json_string_data(str_);
  }

  //! Retrieve HASH of a JSON string
  size_type hash(void) const {
    return mod_json_string_hash(str_);
  }

  //! Compare two JSON strings (case sensitive)
  int compare(const JsonString &rhs) const {
    return mod_json_string_compare(str_, rhs.str_);
  }

  //! Compare two strings (case sensitive)
  int compare(const char *cstr) const {
    const char *self = this->c_str();
    if (self && cstr) {
      return std::strcmp(self, cstr);
    }

    // particular case
    if (!self && cstr) {
      return -1;
    } else if (self && !cstr) {
      return 1;
    }
    return 0;
  }

  // Encode a JSON string
  JsonString encode(void) const {
    JsonString ret;
    ret.str_ = mod_json_string_encode(str_);
    return ret;
  }

  // Decode a JSON string
  JsonString decode(void) const {
    JsonString ret;
    ret.str_ = mod_json_string_decode(str_);
    return ret;
  }

  //! Retrieve the capacity of string
  size_type capacity(void) const {
    return mod_json_string_capacity(str_);
  }

  //! Retrieve the length of string
  size_type size(void) const {
    return mod_json_string_length(str_);
  }

  //! Retrieve the length of string
  size_type length(void) const {
    return mod_json_string_length(str_);
  }

  //! Retrieve refer-counter of string
  ssize_type refer(void) const {
    return mod_json_string_refer(str_);
  }

  //! Retrieve the c-style string
  const char *c_str(void) const {
    return mod_json_string_cstr(str_);
  }

  //! Convert string to float
  float_type as_float(void) const {
    return mod_json_string_float(str_);
  }

  //! Convert string to integer
  integer_type as_integer(void) const {
    return mod_json_string_integer(str_);
  }

  //! Retrieve string as a STL string
  std::string as_stl_string(void) const {
    if (!this->empty()) {
      return std::string(this->data(), this->size());
    }
    return std::string();
  }

 protected:
  //! Clone the string for writing
  bool copy_on_write(void) {
    if (str_) {
      if (mod_json_string_is_shared(str_)) {
        mod_json_string_put(str_);
        str_ = mod_json_string_clone(str_);
      }
    } else {
      str_ = mod_json_string_set("", 0);
    }
    return (str_ != 0);
  }

  //! Clone the value and leak it
  bool copy_and_leak(void) {
    if (copy_on_write()) {
      mod_json_string_set_leaked(str_);
      return true;
    }
    return false;
  }

 private:
  mod_json_string_t *str_;
};

class JsonArray;
class JsonObject;

/*! JSON Value
 */
class JsonValue {
 public:
  typedef mod_json_size_t size_type;
  typedef mod_json_ssize_t ssize_type;
  typedef mod_json_float_t float_type;
  typedef mod_json_integer_t integer_type;

  //! Constructor
  JsonValue(void) : val_(0) {}

  //! Constructor
  explicit JsonValue(const bool &val) {
    val_ = mod_json_value_set_boolean((mod_json_boolean_t)val);
  }

  //! Constructor
  explicit JsonValue(const signed char &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const char &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const short int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const long int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const long long int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const float &val) {
    val_ = mod_json_value_set_float((mod_json_float_t)val);
  }

  //! Constructor
  explicit JsonValue(const double &val) {
    val_ = mod_json_value_set_float((mod_json_float_t)val);
  }

  //! Constructor
  explicit JsonValue(const long double &val) {
    val_ = mod_json_value_set_float((mod_json_float_t)val);
  }

  //! Constructor
  explicit JsonValue(const unsigned char &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const unsigned short int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const unsigned int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const unsigned long int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  explicit JsonValue(const unsigned long long int &val) {
    val_ = mod_json_value_set_integer((mod_json_integer_t)val);
  }

  //! Constructor
  JsonValue(const JsonString &val) {
    val_ = mod_json_value_set_string(*(mod_json_string_t **)&val);
  }

  //! Constructor
  JsonValue(const char *val) {
    val_ = mod_json_value_set_buffer(
        val, val ? (mod_json_size_t)std::strlen(val) : 0);
  }

  //! Constructor
  JsonValue(const char *val, size_type len) {
    val_ = mod_json_value_set_buffer(val, len);
  }

  //! Constructor
  JsonValue(const std::string &val) {
    val_ = mod_json_value_set_buffer(val.data(), (mod_json_size_t)val.size());
  }

  //! Constructor
  JsonValue(const JsonArray &val) {
    val_ = mod_json_value_set_array(*(mod_json_array_t **)&val);
  }

  //! Constructor
  JsonValue(const JsonObject &val) {
    val_ = mod_json_value_set_object(*(mod_json_object_t **)&val);
  }

  //! Constructor
  JsonValue(const JsonValue &rhs) : val_(0) {
    if (rhs.val_) {
      val_ = mod_json_value_grab(rhs.val_);
    }
  }

#if __cplusplus >= 201103L
  //! Constructor
  JsonValue(JsonValue &&rhs) : val_(rhs.val_) {
    rhs.val_ = 0;
  }
#endif

  //! Destructor
  ~JsonValue(void) {
    mod_json_value_unset(val_);
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const JsonValue &rhs) {
    this->assign(rhs);
    return *this;
  }

#if __cplusplus >= 201103L
  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(JsonValue &&rhs) {
    this->assign(std::move(rhs));
    return *this;
  }
#endif

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const bool &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const signed char &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const char &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const short int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const long int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const long long int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const float &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const double &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const long double &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const unsigned char &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const unsigned short int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const unsigned int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const unsigned long int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const unsigned long long int &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const JsonString &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const char *val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const std::string &val) {
    this->assign(val);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const JsonArray &arr) {
    this->assign(arr);
    return *this;
  }

  //! Assign new contents to the value, replacing its current content
  JsonValue &operator=(const JsonObject &obj) {
    this->assign(obj);
    return *this;
  }

  //! Equality
  bool operator==(const JsonValue &rhs) const {
    return mod_json_value_is_equal(val_, rhs.val_);
  }

  //! No equality
  bool operator!=(const JsonValue &rhs) const {
    return !(*this == rhs);
  }

  //! Treat self value as object by force, retrieving value of a key
  JsonValue &operator[](const char *key) {
    return this->get_value(key);
  }

  //! Retrieve a reference of value by a key
  JsonValue operator[](const char *key) const {
    return this->get_value(key);
  }

  //! Treat self value as object by force, retrieving value of a key
  JsonValue &operator[](const JsonString &key) {
    return this->get_value(key.c_str());
  }

  //! Retrieve a reference of value by a key
  JsonValue operator[](const JsonString &key) const {
    return this->get_value(key.c_str());
  }

  //! Treat self value as object by force, retrieving value of a key
  JsonValue &operator[](const std::string &key) {
    return this->get_value(key.c_str());
  }

  //! Retrieve a reference of value by a key
  JsonValue operator[](const std::string &key) const {
    return this->get_value(key.c_str());
  }

  //! Treat self value as array by force, retrieving value at index n
  JsonValue &operator[](size_type n) {
    return this->get_value(n);
  }

  //! Retrieve a reference of value at index n
  JsonValue operator[](size_type n) const {
    return this->get_value(n);
  }

  //! Retrieve non-zero if the value is valid
  bool is_valid(void) const {
    return (val_ != (mod_json_value_t *)0);
  }

  //! Retrieve non-zero if the value is a object
  bool is_object(void) const {
    return mod_json_value_is_object(val_);
  }

  //! Retrieve non-zero if the value is an array
  bool is_array(void) const {
    return mod_json_value_is_array(val_);
  }

  //! Retrieve non-zero if the value is a string
  bool is_string(void) const {
    return mod_json_value_is_string(val_);
  }

  //! Retrieve non-zero if the value is null
  bool is_null(void) const {
    return mod_json_value_is_null(val_);
  }

  //! Retrieve non-zero if the value is a float
  bool is_float(void) const {
    return mod_json_value_is_float(val_);
  }

  //! Retrieve non-zero if the value is an integer
  bool is_integer(void) const {
    return mod_json_value_is_integer(val_);
  }

  //! Retrieve non-zero if the value is a boolean
  bool is_boolean(void) const {
    return mod_json_value_is_boolean(val_);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const JsonValue &rhs) {
    mod_json_value_unset(val_);
    val_ = rhs.val_ ? mod_json_value_grab(rhs.val_) : 0;
  }

#if __cplusplus >= 201103L
  //! Assign new contents to the value, replacing its current content
  void assign(JsonValue &&rhs) {
    mod_json_value_unset(val_);
    val_ = rhs.val_;
    rhs.val_ = 0;
  }
#endif

  //! Assign new contents to the value, replacing its current content
  void assign(const bool &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_boolean(val_, (mod_json_boolean_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const signed char &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const char &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const short int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const long int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const long long int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const float &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_float(val_, (mod_json_float_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const double &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_float(val_, (mod_json_float_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const long double &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_float(val_, (mod_json_float_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const unsigned char &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const unsigned short int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const unsigned int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const unsigned long int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const unsigned long long int &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_integer(val_, (mod_json_integer_t)val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const JsonString &val) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_string(val_, *(mod_json_string_t **)&val);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const char *val) {
    JsonString str(val);
    if (!str.is_valid() || !copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_string(val_, *(mod_json_string_t **)&str);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const char *val, size_type len) {
    JsonString str(val, len);
    if (!str.is_valid() || !copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_string(val_, *(mod_json_string_t **)&str);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const std::string &val) {
    JsonString str(val);
    if (!str.is_valid() || !copy_on_write()) {
      throw std::runtime_error("JsonValue::assign");
    }
    mod_json_value_assign_string(val_, *(mod_json_string_t **)&str);
  }

  //! Assign new contents to the value, replacing its current content
  void assign(const JsonArray &arr);

  //! Assign new contents to the value, replacing its current content
  void assign(const JsonObject &obj);

  //! Retrieve refer-counter of JSON value
  ssize_type refer(void) const {
    return mod_json_value_refer(val_);
  }

  //! Retrieve value as JSON format string
  JsonString as_json_string(void) const {
    mod_json_string_t *tmp = mod_json_dump(val_);
    JsonString ret = *reinterpret_cast<JsonString *>(&tmp);
    if (tmp) {
      mod_json_string_unset(tmp);
    }
    return ret;
  }

  //! Retrieve value as a STL string
  std::string as_stl_string(void) const {
    if (is_string()) {
      return to_string().as_stl_string();
    }
    return std::string();
  }

  //! Retrieve value as JSON string
  const JsonString &as_string(void) const {
    if (!is_string()) {
      throw std::logic_error("JsonValue::as_string");
    }
    return to_string();
  }

  //! Retrieve value as c-style string
  const char *as_c_string(void) const {
    return mod_json_value_cstring(val_);
  }

  //! Retrieve value as JSON string
  JsonString &as_string(void) {
    if (!is_string()) {
      throw std::logic_error("JsonValue::as_string");
    }
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonValue::as_string");
    }
    return to_string();
  }

  //! Retrieve value as JSON array
  const JsonArray &as_array(void) const {
    if (!is_array()) {
      throw std::logic_error("JsonValue::as_array");
    }
    return to_array();
  }

  //! Retrieve value as JSON array
  JsonArray &as_array(void) {
    if (!is_array()) {
      throw std::logic_error("JsonValue::as_array");
    }
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonValue::as_array");
    }
    return to_array();
  }

  //! Retrieve value as JSON object
  const JsonObject &as_object(void) const {
    if (!is_object()) {
      throw std::logic_error("JsonValue::as_object");
    }
    return to_object();
  }

  //! Retrieve value as JSON object
  JsonObject &as_object(void) {
    if (!is_object()) {
      throw std::logic_error("JsonValue::as_object");
    }
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonValue::as_object");
    }
    return to_object();
  }

  //! Retrieve value as float
  float_type as_float(void) const {
    return mod_json_value_float(val_);
  }

  //! Retrieve value as integer
  integer_type as_integer(void) const {
    return mod_json_value_integer(val_);
  }

  //! Retrieve value as boolean
  bool as_bool(void) const {
    return mod_json_value_boolean(val_);
  }

  //! Exchange the content with another JSON value
  void swap(JsonValue &rhs) {
    mod_json_value_t *val = val_;
    val_ = rhs.val_;
    rhs.val_ = val;
  }

  //! Merge another JSON value
  void merge(const JsonValue &rhs) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonValue::merge");
    }
    mod_json_value_merge(val_, rhs.val_);
  }

  //! Parse a sting as a JSON value
  bool parse(const char *str) {
    mod_json_token_t *tok = mod_json_token_create(NULL);

    if (tok) {
      mod_json_value_t *jval = mod_json_parse(tok, str);

      mod_json_token_destroy(tok);
      if (jval) {
        *this = *reinterpret_cast<JsonValue *>(&jval);
        mod_json_value_unset(jval);
        return is_valid();
      }
    }
    return false;
  }

  //! Parse a sting as a JSON value
  bool parse(const JsonString &str) {
    return this->parse(str.c_str());
  }

  //! Parse a sting as a JSON value
  bool parse(const std::string &str) {
    return this->parse(str.c_str());
  }

 protected:
  //! Clone the value for writing
  bool copy_on_write(void) {
    if (val_) {
      if (mod_json_value_is_shared(val_)) {
        mod_json_value_put(val_);
        val_ = mod_json_value_clone(val_);
      }
    } else {
      val_ = mod_json_value_set_null();
    }
    return (val_ != 0);
  }

  //! Clone the value and leak it
  bool copy_and_leak(void) {
    if (copy_on_write()) {
      mod_json_value_set_leaked(val_);
      return true;
    }
    return false;
  }

  //! Convert value to JSON object
  JsonObject &to_object(void);

  //! Convert value to JSON object
  const JsonObject &to_object(void) const;

  //! Convert value to JSON array
  JsonArray &to_array(void);

  //! Convert value to JSON array
  const JsonArray &to_array(void) const;

  //! Convert value to JSON string
  JsonString &to_string(void);

  //! Convert value to JSON string
  const JsonString &to_string(void) const;

  //! Treat self value as object by force, retrieving value of a key
  JsonValue &get_value(const char *key);

  //! Retrieve a reference of value by a key
  JsonValue get_value(const char *key) const;

  //! Treat self value as array by force, retrieving value at index n
  JsonValue &get_value(size_type n);

  //! Retrieve a reference of value at index n
  JsonValue get_value(size_type n) const;

  //! Set the new array to the value, replacing its current content
  void set_value(const JsonArray &val);

  //! Set the new object to the value, replacing its current content
  void set_value(const JsonObject &val);

 private:
  mod_json_value_t *val_;
};

/*! JSON Array
 */
class JsonArray {
 public:
  typedef mod_json_size_t size_type;
  typedef mod_json_ssize_t ssize_type;

  class iterator;
  class const_iterator;
  class reverse_iterator;
  class const_reverse_iterator;

  /*! Const iterator of JSON Array
   */
  class const_iterator {
   public:
    //! Constructor
    const_iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const const_iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const const_iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    const_iterator &operator++() {
      ++iter_;
      return *this;
    }

    //! Increment (Suffix)
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    const_iterator &operator--() {
      --iter_;
      return *this;
    }

    //! Decrement (Suffix)
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    const JsonValue &operator*() const {
      return *reinterpret_cast<const JsonValue *>(iter_);
    }

    //! Structure dereference (eg. iter->)
    const JsonValue *operator->() const {
      return reinterpret_cast<const JsonValue *>(iter_);
    }

    //! Retrieve as const reverse iterator
    operator const_reverse_iterator() const {
      return const_reverse_iterator(iter_);
    }

   protected:
    friend class JsonArray;
    friend class JsonArray::iterator;
    friend class JsonArray::reverse_iterator;
    friend class JsonArray::const_reverse_iterator;

    //! Constructor for friends
    const_iterator(mod_json_value_t *const *iter) : iter_(iter) {}

   private:
    mod_json_value_t *const *iter_;
  };

  /*! iterator of JSON Array
   */
  class iterator {
   public:
    //! Constructor
    iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    iterator &operator++() {
      ++iter_;
      return *this;
    }

    //! Increment (Suffix)
    iterator operator++(int) {
      iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    iterator &operator--() {
      --iter_;
      return *this;
    }

    //! Decrement (Suffix)
    iterator operator--(int) {
      iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    JsonValue &operator*() const {
      return *reinterpret_cast<JsonValue *>(iter_);
    }

    //! Structure dereference (eg. iter->)
    JsonValue *operator->() const {
      return reinterpret_cast<JsonValue *>(iter_);
    }

    //! Retrieve as const iterator
    operator const_iterator() const {
      return const_iterator(iter_);
    }

    //! Retrieve as reverse iterator
    operator reverse_iterator() const {
      return reverse_iterator(iter_);
    }

    //! Retrieve as const reverse iterator
    operator const_reverse_iterator() const {
      return const_reverse_iterator(iter_);
    }

   protected:
    friend class JsonArray;
    friend class JsonArray::reverse_iterator;

    //! Constructor for friends
    iterator(mod_json_value_t **iter) : iter_(iter) {}

   private:
    mod_json_value_t **iter_;
  };

  /*! Const Reverse iterator of JSON Array
   */
  class const_reverse_iterator {
   public:
    //! Constructor
    const_reverse_iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const const_reverse_iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const const_reverse_iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    const_reverse_iterator &operator++() {
      --iter_;
      return *this;
    }

    //! Increment (Suffix)
    const_reverse_iterator operator++(int) {
      const_reverse_iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    const_reverse_iterator &operator--() {
      ++iter_;
      return *this;
    }

    //! Decrement (Suffix)
    const_reverse_iterator operator--(int) {
      const_reverse_iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    const JsonValue &operator*() const {
      return *reinterpret_cast<const JsonValue *>(iter_);
    }

    //! Structure dereference (eg. iter->)
    const JsonValue *operator->() const {
      return reinterpret_cast<const JsonValue *>(iter_);
    }

    //! Retrieve as const iterator
    operator const_iterator() const {
      return const_iterator(iter_);
    }

   protected:
    friend class JsonArray;
    friend class JsonArray::iterator;
    friend class JsonArray::const_iterator;
    friend class JsonArray::reverse_iterator;

    //! Constructor for friends
    const_reverse_iterator(mod_json_value_t *const *iter) : iter_(iter) {}

   private:
    mod_json_value_t *const *iter_;
  };

  /*! Reverse iterator of JSON Array
   */
  class reverse_iterator {
   public:
    //! Constructor
    reverse_iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const reverse_iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const reverse_iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    reverse_iterator &operator++() {
      --iter_;
      return *this;
    }

    //! Increment (Suffix)
    reverse_iterator operator++(int) {
      reverse_iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    reverse_iterator &operator--() {
      ++iter_;
      return *this;
    }

    //! Decrement (Suffix)
    reverse_iterator operator--(int) {
      reverse_iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    JsonValue &operator*() const {
      return *reinterpret_cast<JsonValue *>(iter_);
    }

    //! Structure dereference (eg. iter->)
    JsonValue *operator->() const {
      return reinterpret_cast<JsonValue *>(iter_);
    }

    //! Retrieve as iterator
    operator iterator() const {
      return iterator(iter_);
    }

    //! Retrieve as const iterator
    operator const_iterator() const {
      return const_iterator(iter_);
    }

    //! Retrieve as const reverse iterator
    operator const_reverse_iterator() const {
      return const_reverse_iterator(iter_);
    }

   protected:
    friend class JsonArray;
    friend class JsonArray::iterator;

    //! Constructor for friends
    reverse_iterator(mod_json_value_t **iter) : iter_(iter) {}

   private:
    mod_json_value_t **iter_;
  };

  //! Constructor
  JsonArray(void) : arr_(0) {}

  //! Constructor
  JsonArray(const JsonArray &rhs) : arr_(0) {
    if (rhs.arr_) {
      arr_ = mod_json_array_grab(rhs.arr_);
    }
  }

#if __cplusplus >= 201103L
  //! Constructor
  JsonArray(JsonArray &&rhs) : arr_(rhs.arr_) {
    rhs.arr_ = 0;
  }
#endif

  //! Destructor
  ~JsonArray(void) {
    mod_json_array_unset(arr_);
  }

  //! Assign new contents to the array, replacing its current content
  JsonArray &operator=(const JsonArray &rhs) {
    this->assign(rhs);
    return *this;
  }

#if __cplusplus >= 201103L
  //! Assign new contents to the array, replacing its current content
  JsonArray &operator=(JsonArray &&rhs) {
    this->assign(std::move(rhs));
    return *this;
  }
#endif

  //! Equality
  bool operator==(const JsonArray &rhs) const {
    return mod_json_array_is_equal(arr_, rhs.arr_);
  }

  //! No equality
  bool operator!=(const JsonArray &rhs) const {
    return !(*this == rhs);
  }

  //! Retrieve the value at index n, if no one exists, throw an exception.
  JsonValue &operator[](size_type n) {
    return this->at(n);
  }

  //! Retrieve the value at index n, if no one exists, return a null value.
  JsonValue operator[](size_type n) const {
    return ((n < this->size()) ? this->get_value(n) : JsonValue());
  }

  //! Retrieve non-zero if the array is valid
  bool is_valid(void) const {
    return (arr_ != (mod_json_array_t *)0);
  }

  //! Retrieve non-zero if the array is empty
  bool empty(void) const {
    return mod_json_array_empty(arr_);
  }

  //! Retrieve the size of JSON array
  size_type size(void) const {
    return mod_json_array_count(arr_);
  }

  //! Retrieve the capacity of JSON array
  size_type capacity(void) const {
    return mod_json_array_capacity(arr_);
  }

  //! Retrieve refer-counter of JSON array
  ssize_type refer(void) const {
    return mod_json_array_refer(arr_);
  }

  //! Assign new contents to the array, replacing its current content
  void assign(const JsonArray &rhs) {
    mod_json_array_unset(arr_);
    arr_ = rhs.arr_ ? mod_json_array_grab(rhs.arr_) : 0;
  }

#if __cplusplus >= 201103L
  //! Assign new contents to the array, replacing its current content
  void assign(JsonArray &&rhs) {
    mod_json_array_unset(arr_);
    arr_ = rhs.arr_;
    rhs.arr_ = 0;
  }
#endif

  //! Request a change in capacity
  void reserve(size_type n) {
    if (!copy_on_write() || mod_json_array_reserve(arr_, n) != 0) {
      throw std::runtime_error("JsonArray::reserve");
    }
  }

  //! Reverse the order of the elements
  void reverse(void) {
    if (arr_ && copy_on_write()) {
      mod_json_array_reverse(arr_);
    }
  }

  //! Push a value to array
  void push(const JsonValue &val) {
    JsonValue tmp(val);

    if (!copy_on_write() ||
        mod_json_array_push(arr_, *((mod_json_value_t **)&tmp)) != 0) {
      throw std::runtime_error("JsonArray::push");
    }
  }

  //! Pop the last element from array
  void pop(void) {
    if (arr_) {
      if (!copy_on_write()) {
        throw std::runtime_error("JsonArray::pop");
      }
      mod_json_array_pop(arr_);
    }
  }

  //! Remove the first element of array
  void shift(void) {
    if (arr_) {
      if (!copy_on_write()) {
        throw std::runtime_error("JsonArray::shift");
      }
      mod_json_array_shift(arr_);
    }
  }

  //! Retrieve the value at index n
  JsonValue &at(size_type n) {
    if (this->size() <= n) {
      throw std::out_of_range("JsonArray::at");
    }
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonArray::at");
    }
    return this->get_value(n);
  }

  //! Retrieve the value at index n
  const JsonValue &at(size_type n) const {
    if (this->size() <= n) {
      throw std::out_of_range("JsonArray::at");
    }
    return this->get_value(n);
  }

  //! Retrieve a reference to the first element
  JsonValue &front(void) {
    if (this->size() <= 0) {
      throw std::out_of_range("JsonArray::front");
    }
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonArray::front");
    }
    return this->get_value(0);
  }

  //! Retrieve a reference to the first element
  const JsonValue &front(void) const {
    if (this->size() <= 0) {
      throw std::out_of_range("JsonArray::front");
    }
    return this->get_value(0);
  }

  //! Retrieve a reference to the last element
  JsonValue &back(void) {
    if (this->size() <= 0) {
      throw std::out_of_range("JsonArray::back");
    }
    if (!copy_and_leak()) {
      throw std::runtime_error("JsonArray::back");
    }
    return this->get_value(this->size() - 1);
  }

  //! Retrieve a reference to the last element
  const JsonValue &back(void) const {
    if (this->size() <= 0) {
      throw std::out_of_range("JsonArray::back");
    }
    return this->get_value(this->size() - 1);
  }

  //! Clear the JSON array
  void clear(void) {
    mod_json_array_unset(arr_);
    arr_ = 0;
  }

  //! Exchange the content with another JSON array
  void swap(JsonArray &rhs) {
    mod_json_array_t *arr = arr_;
    arr_ = rhs.arr_;
    rhs.arr_ = arr;
  }

  //! Merge another JSON array
  void merge(const JsonArray &rhs) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonArray::merge");
    }
    mod_json_array_merge(arr_, rhs.arr_);
  }

  //! Resize a JSON array so that it contains n elements
  void resize(size_type n, const JsonValue &val = JsonValue()) {
    if (!copy_on_write() ||
        mod_json_array_resize(arr_, n, *((mod_json_value_t **)&val)) != 0) {
      throw std::runtime_error("JsonArray::resize");
    }
  }

  //! Retrieve an iterator pointing to the first element
  iterator begin(void) {
    if (copy_and_leak()) {
      return iterator(mod_json_array_begin(arr_));
    }
    return iterator();
  }

  //! Retrieve a const iterator pointing to the first element
  const_iterator begin(void) const {
    if (arr_) {
      return const_iterator(mod_json_array_begin(arr_));
    }
    return const_iterator();
  }

  //! Retrieve a const iterator pointing to the first element
  const_iterator cbegin(void) const {
    if (arr_) {
      return const_iterator(mod_json_array_begin(arr_));
    }
    return const_iterator();
  }

  //! Retrieve a reverse iterator pointing to the last element
  reverse_iterator rbegin(void) {
    if (copy_and_leak()) {
      return reverse_iterator(mod_json_array_rbegin(arr_));
    }
    return reverse_iterator();
  }

  //! Retrieve a const reverse iterator pointing to the last element
  const_reverse_iterator rbegin(void) const {
    if (arr_) {
      return const_reverse_iterator(mod_json_array_rbegin(arr_));
    }
    return const_reverse_iterator();
  }

  //! Retrieve a const reverse iterator pointing to the last element
  const_reverse_iterator crbegin(void) const {
    if (arr_) {
      return const_reverse_iterator(mod_json_array_rbegin(arr_));
    }
    return const_reverse_iterator();
  }

  //! Retrieve an iterator pointing to the past-the-end element
  iterator end(void) {
    if (copy_and_leak()) {
      return iterator(mod_json_array_end(arr_));
    }
    return iterator();
  }

  //! Retrieve a const iterator pointing to the past-the-end element
  const_iterator end(void) const {
    if (arr_) {
      return const_iterator(mod_json_array_end(arr_));
    }
    return const_iterator();
  }

  //! Retrieve a const iterator pointing to the past-the-end element
  const_iterator cend(void) const {
    if (arr_) {
      return const_iterator(mod_json_array_end(arr_));
    }
    return const_iterator();
  }

  //! Retrieve a reverse pointing to the past-the-end element
  reverse_iterator rend(void) {
    if (copy_and_leak()) {
      return reverse_iterator(mod_json_array_rend(arr_));
    }
    return reverse_iterator();
  }

  //! Retrieve a const reverse pointing to the past-the-end element
  const_reverse_iterator rend(void) const {
    if (arr_) {
      return const_reverse_iterator(mod_json_array_rend(arr_));
    }
    return const_reverse_iterator();
  }

  //! Retrieve a const reverse pointing to the past-the-end element
  const_reverse_iterator crend(void) const {
    if (arr_) {
      return const_reverse_iterator(mod_json_array_rend(arr_));
    }
    return const_reverse_iterator();
  }

 protected:
  //! Clone the array for writing
  bool copy_on_write(void) {
    if (arr_) {
      if (mod_json_array_is_shared(arr_)) {
        mod_json_array_put(arr_);
        arr_ = mod_json_array_clone(arr_);
      }
    } else {
      arr_ = mod_json_array_set_default();
    }
    return (arr_ != 0);
  }

  //! Clone the array and leak it
  bool copy_and_leak(void) {
    if (copy_on_write()) {
      mod_json_array_set_leaked(arr_);
      return true;
    }
    return false;
  }

  //! Retrieve the value at index n
  JsonValue &get_value(size_type n) {
    return *reinterpret_cast<JsonValue *>(arr_->first + n);
  }

  //! Retrieve the value at index n
  const JsonValue &get_value(size_type n) const {
    return *reinterpret_cast<JsonValue *>(arr_->first + n);
  }

 private:
  mod_json_array_t *arr_;
};

/*! JSON Pair
 */
class JsonPair {
 public:
  //! Constructor
  JsonPair(void) : pair_(0) {}

  //! Retrieve non-zero if the pair is valid
  bool is_valid(void) const {
    return (pair_ != (mod_json_pair_t *)0);
  }

  //! Retrieve the key of pair
  const JsonString &key(void) const {
    return *reinterpret_cast<JsonString *>(&pair_->key);
  }

  //! Retrieve the value of pair
  JsonValue &value(void) {
    return *reinterpret_cast<JsonValue *>(&pair_->val);
  }

  //! Retrieve the value of pair
  const JsonValue &value(void) const {
    return *reinterpret_cast<JsonValue *>(&pair_->val);
  }

 protected:
  friend class JsonObject;

  //! Constructor for friends
  JsonPair(mod_json_pair_t *pair) : pair_(pair) {}

  //! Constructor for friends
  JsonPair(const JsonPair &rhs) : pair_(rhs.pair_) {}

 private:
  mod_json_pair_t *pair_;
};

/*! JSON Object
 */
class JsonObject {
 public:
  typedef mod_json_size_t size_type;
  typedef mod_json_ssize_t ssize_type;

  class iterator;
  class const_iterator;
  class reverse_iterator;
  class const_reverse_iterator;

  /*! Const iterator of JSON Object
   */
  class const_iterator {
   public:
    //! Constructor
    const_iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const const_iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const const_iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    const_iterator &operator++() {
      ++iter_;
      return *this;
    }

    //! Increment (Suffix)
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    const_iterator &operator--() {
      --iter_;
      return *this;
    }

    //! Decrement (Suffix)
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    const JsonPair &operator*() const {
      return *reinterpret_cast<const JsonPair *>(&iter_);
    }

    //! Structure dereference (eg. iter->)
    const JsonPair *operator->() const {
      return reinterpret_cast<const JsonPair *>(&iter_);
    }

    //! Retrieve as const reverse iterator
    operator const_reverse_iterator() const {
      return const_reverse_iterator(iter_);
    }

   protected:
    friend class JsonObject;
    friend class JsonObject::iterator;
    friend class JsonObject::reverse_iterator;
    friend class JsonObject::const_reverse_iterator;

    //! Constructor for friends
    const_iterator(const mod_json_pair_t *iter) : iter_(iter) {}

   private:
    const mod_json_pair_t *iter_;
  };

  /*! iterator of JSON Object
   */
  class iterator {
   public:
    //! Constructor
    iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    iterator &operator++() {
      ++iter_;
      return *this;
    }

    //! Increment (Suffix)
    iterator operator++(int) {
      iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    iterator &operator--() {
      --iter_;
      return *this;
    }

    //! Decrement (Suffix)
    iterator operator--(int) {
      iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    JsonPair &operator*() const {
      return *reinterpret_cast<JsonPair *>((mod_json_pair_t **)&iter_);
    }

    //! Structure dereference (eg. iter->)
    JsonPair *operator->() const {
      return reinterpret_cast<JsonPair *>((mod_json_pair_t **)&iter_);
    }

    //! Retrieve as const iterator
    operator const_iterator() const {
      return const_iterator(iter_);
    }

    //! Retrieve as reverse iterator
    operator reverse_iterator() const {
      return reverse_iterator(iter_);
    }

    //! Retrieve as const reverse iterator
    operator const_reverse_iterator() const {
      return const_reverse_iterator(iter_);
    }

   protected:
    friend class JsonObject;
    friend class JsonObject::reverse_iterator;

    //! Constructor for friends
    iterator(mod_json_pair_t *iter) : iter_(iter) {}

   private:
    mod_json_pair_t *iter_;
  };

  /*! Const Reverse iterator of JSON Object
   */
  class const_reverse_iterator {
   public:
    //! Constructor
    const_reverse_iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const const_reverse_iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const const_reverse_iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    const_reverse_iterator &operator++() {
      --iter_;
      return *this;
    }

    //! Increment (Suffix)
    const_reverse_iterator operator++(int) {
      const_reverse_iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    const_reverse_iterator &operator--() {
      ++iter_;
      return *this;
    }

    //! Decrement (Suffix)
    const_reverse_iterator operator--(int) {
      const_reverse_iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    const JsonPair &operator*() const {
      return *reinterpret_cast<const JsonPair *>(&iter_);
    }

    //! Structure dereference (eg. iter->)
    const JsonPair *operator->() const {
      return reinterpret_cast<const JsonPair *>(&iter_);
    }

    //! Retrieve as const iterator
    operator const_iterator() const {
      return const_iterator(iter_);
    }

   protected:
    friend class JsonObject;
    friend class JsonObject::iterator;
    friend class JsonObject::const_iterator;
    friend class JsonObject::reverse_iterator;

    //! Constructor for friends
    const_reverse_iterator(const mod_json_pair_t *iter) : iter_(iter) {}

   private:
    const mod_json_pair_t *iter_;
  };

  /*! iterator of JSON Object
   */
  class reverse_iterator {
   public:
    //! Constructor
    reverse_iterator(void) : iter_(0) {}

    //! Equality
    bool operator==(const reverse_iterator &rhs) const {
      return (iter_ == rhs.iter_);
    }

    //! No equality
    bool operator!=(const reverse_iterator &rhs) const {
      return (iter_ != rhs.iter_);
    }

    //! Increment (Prefix)
    reverse_iterator &operator++() {
      --iter_;
      return *this;
    }

    //! Increment (Suffix)
    reverse_iterator operator++(int) {
      reverse_iterator tmp = *this;
      --iter_;
      return tmp;
    }

    //! Decrement (Prefix)
    reverse_iterator &operator--() {
      ++iter_;
      return *this;
    }

    //! Decrement (Suffix)
    reverse_iterator operator--(int) {
      reverse_iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    //! Indirection (eg. *iter)
    JsonPair &operator*() const {
      return *reinterpret_cast<JsonPair *>((mod_json_pair_t **)&iter_);
    }

    //! Structure dereference (eg. iter->)
    JsonPair *operator->() const {
      return reinterpret_cast<JsonPair *>((mod_json_pair_t **)&iter_);
    }

    //! Retrieve as iterator
    operator iterator() const {
      return iterator(iter_);
    }

    //! Retrieve as const iterator
    operator const_iterator() const {
      return const_iterator(iter_);
    }

    //! Retrieve as const reverse iterator
    operator const_reverse_iterator() const {
      return const_reverse_iterator(iter_);
    }

   protected:
    friend class JsonObject;
    friend class JsonArray::iterator;

    //! Constructor for friends
    reverse_iterator(mod_json_pair_t *iter) : iter_(iter) {}

   private:
    mod_json_pair_t *iter_;
  };

  //! Constructor
  JsonObject(void) : obj_(0) {}

  //! Constructor
  JsonObject(const JsonObject &rhs) : obj_(0) {
    if (rhs.obj_) {
      obj_ = mod_json_object_grab(rhs.obj_);
    }
  }

#if __cplusplus >= 201103L
  //! Constructor
  JsonObject(JsonObject &&rhs) : obj_(rhs.obj_) {
    rhs.obj_ = 0;
  }
#endif

  //! Destructor
  ~JsonObject(void) {
    mod_json_object_unset(obj_);
  }

  //! Assign new contents to the object, replacing its current content
  JsonObject &operator=(const JsonObject &rhs) {
    this->assign(rhs);
    return *this;
  }

#if __cplusplus >= 201103L
  //! Assign new contents to the object, replacing its current content
  JsonObject &operator=(JsonObject &&rhs) {
    this->assign(std::move(rhs));
    return *this;
  }
#endif

  //! Equality
  bool operator==(const JsonObject &rhs) const {
    return mod_json_object_is_equal(obj_, rhs.obj_);
  }

  //! No equality
  bool operator!=(const JsonObject &rhs) const {
    return !(*this == rhs);
  }

  //! Retrieve the value of a key, if no one exists, create a new one.
  JsonValue &operator[](const char *key) {
    if (!key) {
      throw std::invalid_argument("JsonObject::operator[]");
    }

    if (!copy_and_leak()) {
      throw std::runtime_error("JsonObject::operator[]");
    }

    JsonPair pair(mod_json_object_touch(obj_, key));
    if (!pair.is_valid()) {
      throw std::runtime_error("JsonObject::operator[]");
    }
    return pair.value();
  }

  //! Retrieve the value of a key, if no one exists, return a null value.
  JsonValue operator[](const char *key) const {
    if (!key) {
      throw std::invalid_argument("JsonObject::operator[]");
    }

    JsonPair pair(mod_json_object_find(obj_, key));
    return (pair.is_valid() ? pair.value() : JsonValue());
  }

  //! Retrieve the value of a key, if no one exists, create a new one.
  JsonValue &operator[](const JsonString &key) {
    return (*this)[key.c_str()];
  }

  //! Retrieve the value of a key, if no one exists, return a null value.
  JsonValue operator[](const JsonString &key) const {
    return (*this)[key.c_str()];
  }

  //! Retrieve non-zero if the object is valid
  bool is_valid(void) const {
    return (obj_ != (mod_json_object_t *)0);
  }

  //! Retrieve non-zero if the object is empty
  bool empty(void) const {
    return mod_json_object_empty(obj_);
  }

  //! Retrieve the size of JSON object
  size_type size(void) const {
    return mod_json_object_count(obj_);
  }

  //! Retrieve refer-counter of JSON object
  ssize_type refer(void) const {
    return mod_json_object_refer(obj_);
  }

  //! Assign new contents to the object, replacing its current content
  void assign(const JsonObject &rhs) {
    mod_json_object_unset(obj_);
    obj_ = rhs.obj_ ? mod_json_object_grab(rhs.obj_) : 0;
  }

#if __cplusplus >= 201103L
  //! Assign new contents to the object, replacing its current content
  void assign(JsonObject &&rhs) {
    mod_json_object_unset(obj_);
    obj_ = rhs.obj_;
    rhs.obj_ = 0;
  }
#endif

  //! Clear the JSON object
  void clear(void) {
    mod_json_object_unset(obj_);
    obj_ = 0;
  }

  //! Set the value of a key
  bool set(const JsonString &key, const JsonValue &val) {
    JsonValue tmp(val);
    if (!copy_on_write()) {
      throw std::runtime_error("JsonObject::set");
    }
    return (mod_json_object_insert(obj_, *(mod_json_string_t **)&key,
                                   *(mod_json_value_t **)&tmp) !=
            (mod_json_pair_t *)0);
  }

  //! Retrieve the value of a key
  bool get(const char *key, JsonValue *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = pair.value();
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, JsonString *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid() || !pair.value().is_string()) {
      return false;
    }
    *val = pair.value().as_string();
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, std::string *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid() || !pair.value().is_string()) {
      return false;
    }
    *val = pair.value().as_stl_string();
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, JsonArray *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid() || !pair.value().is_array()) {
      return false;
    }
    *val = pair.value().as_array();
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, JsonObject *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid() || !pair.value().is_object()) {
      return false;
    }
    *val = pair.value().as_object();
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, bool *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = pair.value().as_bool();
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, signed char *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<signed char>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, char *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<char>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, short int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<short int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, long int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<long int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, long long int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<long long int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, unsigned char *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<unsigned char>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, unsigned short int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<unsigned short int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, unsigned int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<unsigned int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, unsigned long int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<unsigned long int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, unsigned long long int *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<unsigned long long int>(pair.value().as_integer());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, float *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<float>(pair.value().as_float());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, double *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<double>(pair.value().as_float());
    return true;
  }

  //! Retrieve the value of a key
  bool get(const char *key, long double *val) const {
    const JsonPair pair(mod_json_object_find(obj_, key));
    if (!pair.is_valid()) {
      return false;
    }
    *val = static_cast<long double>(pair.value().as_float());
    return true;
  }

  //! Retrieve the value of a key
  template <typename T>
  bool get(const JsonString &key, T *val) const {
    return this->get(key.c_str(), val);
  }

  //! Retrieve the value of a key
  template <typename T>
  bool get(const std::string &key, T *val) const {
    return this->get(key.c_str(), val);
  }

  //! Delete a key-value pair from JSON object
  void unset(const char *key) {
    if (obj_ && key) {
      if (!copy_on_write()) {
        throw std::runtime_error("JsonObject::unset");
      }
      mod_json_object_erase(obj_, key);
    }
  }

  //! Retrieve non-zero if the key exists in JSON object
  bool has(const char *key) const {
    return (mod_json_object_find(obj_, key) != (mod_json_pair_t *)0);
  }

  //! Exchange the content with another JSON object
  void swap(JsonObject &rhs) {
    mod_json_object_t *obj = obj_;
    obj_ = rhs.obj_;
    rhs.obj_ = obj;
  }

  //! Merge another JSON object
  void merge(const JsonObject &rhs) {
    if (!copy_on_write()) {
      throw std::runtime_error("JsonObject::merge");
    }
    mod_json_object_merge(obj_, rhs.obj_);
  }

  //! Retrieve an iterator pointing to the first element
  iterator begin(void) {
    if (copy_and_leak()) {
      return iterator(mod_json_object_begin(obj_));
    }
    return iterator();
  }

  //! Retrieve a const iterator pointing to the first element
  const_iterator begin(void) const {
    if (obj_) {
      return const_iterator(mod_json_object_begin(obj_));
    }
    return const_iterator();
  }

  //! Retrieve a const iterator pointing to the first element
  const_iterator cbegin(void) const {
    if (obj_) {
      return const_iterator(mod_json_object_begin(obj_));
    }
    return const_iterator();
  }

  //! Retrieve a reverse iterator pointing to the last element
  reverse_iterator rbegin(void) {
    if (copy_and_leak()) {
      return reverse_iterator(mod_json_object_rbegin(obj_));
    }
    return reverse_iterator();
  }

  //! Retrieve a const reverse iterator pointing to the last element
  const_reverse_iterator rbegin(void) const {
    if (obj_) {
      return const_reverse_iterator(mod_json_object_rbegin(obj_));
    }
    return const_reverse_iterator();
  }

  //! Retrieve a const reverse iterator pointing to the last element
  const_reverse_iterator crbegin(void) const {
    if (obj_) {
      return const_reverse_iterator(mod_json_object_rbegin(obj_));
    }
    return const_reverse_iterator();
  }

  //! Retrieve an iterator pointing to the past-the-end element
  iterator end(void) {
    if (copy_and_leak()) {
      return iterator(mod_json_object_end(obj_));
    }
    return iterator();
  }

  //! Retrieve a const iterator pointing to the past-the-end element
  const_iterator end(void) const {
    if (obj_) {
      return const_iterator(mod_json_object_end(obj_));
    }
    return const_iterator();
  }

  //! Retrieve a const iterator pointing to the past-the-end element
  const_iterator cend(void) const {
    if (obj_) {
      return const_iterator(mod_json_object_end(obj_));
    }
    return const_iterator();
  }

  //! Retrieve a reverse pointing to the past-the-end element
  reverse_iterator rend(void) {
    if (copy_and_leak()) {
      return reverse_iterator(mod_json_object_rend(obj_));
    }
    return reverse_iterator();
  }

  //! Retrieve a const reverse pointing to the past-the-end element
  const_reverse_iterator rend(void) const {
    if (obj_) {
      return const_reverse_iterator(mod_json_object_rend(obj_));
    }
    return const_reverse_iterator();
  }

  //! Retrieve a const reverse pointing to the past-the-end element
  const_reverse_iterator crend(void) const {
    if (obj_) {
      return const_reverse_iterator(mod_json_object_rend(obj_));
    }
    return const_reverse_iterator();
  }

 protected:
  //! Clone the object for writing
  bool copy_on_write(void) {
    if (obj_) {
      if (mod_json_object_is_shared(obj_)) {
        mod_json_object_put(obj_);
        obj_ = mod_json_object_clone(obj_);
      }
    } else {
      obj_ = mod_json_object_set_default();
    }
    return (obj_ != 0);
  }

  //! Clone the object and leak it
  bool copy_and_leak(void) {
    if (copy_on_write()) {
      mod_json_object_set_leaked(obj_);
      return true;
    }
    return false;
  }

 private:
  mod_json_object_t *obj_;
};

//! Assign new contents to the value, replacing its current content
inline void JsonValue::assign(const JsonArray &arr) {
  this->set_value(arr);
}

//! Assign new contents to the value, replacing its current content
inline void JsonValue::assign(const JsonObject &obj) {
  this->set_value(obj);
}

//! Convert value to JSON object
inline JsonObject &JsonValue::to_object(void) {
  return *reinterpret_cast<JsonObject *>(&val_->data.c_obj);
}

//! Convert value to JSON object
inline const JsonObject &JsonValue::to_object(void) const {
  return *reinterpret_cast<JsonObject *>(&val_->data.c_obj);
}

//! Convert value to JSON array
inline JsonArray &JsonValue::to_array(void) {
  return *reinterpret_cast<JsonArray *>(&val_->data.c_arr);
}

//! Convert value to JSON array
inline const JsonArray &JsonValue::to_array(void) const {
  return *reinterpret_cast<JsonArray *>(&val_->data.c_arr);
}

//! Convert value to JSON string
inline JsonString &JsonValue::to_string(void) {
  return *reinterpret_cast<JsonString *>(&val_->data.c_str);
}

//! Convert value to JSON string
inline const JsonString &JsonValue::to_string(void) const {
  return *reinterpret_cast<JsonString *>(&val_->data.c_str);
}

//! Treat self value as object by force, retrieving value of a key
inline JsonValue &JsonValue::get_value(const char *key) {
  if (!is_object()) {
    *this = JsonObject();
  }
  if (!copy_and_leak()) {
    throw std::runtime_error("JsonValue::get_value");
  }
  return (to_object())[key];
}

//! Retrieve a reference of value by a key
inline JsonValue JsonValue::get_value(const char *key) const {
  return (is_object() ? (to_object())[key] : JsonValue());
}

//! Treat self value as array by force, retrieving value at index n
inline JsonValue &JsonValue::get_value(size_type n) {
  if (!is_array()) {
    throw std::logic_error("JsonValue::get_value");
  }
  if (!copy_and_leak()) {
    throw std::runtime_error("JsonValue::get_value");
  }
  return (to_array())[n];
}

//! Retrieve a reference of value at index n
inline JsonValue JsonValue::get_value(size_type n) const {
  return (is_array() ? (to_array())[n] : JsonValue());
}

//! Set the new array to the value, replacing its current content
inline void JsonValue::set_value(const JsonArray &val) {
  if (!copy_on_write()) {
    throw std::runtime_error("JsonValue::set_value");
  }
  mod_json_value_assign_array(val_, *(mod_json_array_t **)&val);
}

//! Set the new object to the value, replacing its current content
inline void JsonValue::set_value(const JsonObject &val) {
  if (!copy_on_write()) {
    throw std::runtime_error("JsonValue::set_value");
  }
  mod_json_value_assign_object(val_, *(mod_json_object_t **)&val);
}

/*! JSON Parser
 */
class JsonParser {
 public:
  typedef mod_json_size_t size_type;

  //! Constructor
  JsonParser(void)
      : state_(mod_json_state_null), error_(mod_json_error_null), context_(0) {
    option_.options = 0;
    option_.object_depth = 0;
    option_.array_depth = 0;
  }

  //! Destructor
  ~JsonParser(void) {}

  //! Set the max object depth
  void set_object_depth(size_type depth) {
    option_.object_depth = depth;
  }

  //! Set the max array depth
  void set_array_depth(size_type depth) {
    option_.array_depth = depth;
  }

  //! Enable/Disable comments
  void set_comment(bool enable = true) {
    if (enable) {
      option_.options |= MOD_JSON_COMMENT;
    } else {
      option_.options &= ~MOD_JSON_COMMENT;
    }
  }

  //! Enable/Disable loose strings
  void set_unstrict(bool enable = true) {
    if (enable) {
      option_.options |= MOD_JSON_UNSTRICT;
    } else {
      option_.options &= ~MOD_JSON_UNSTRICT;
    }
  }

  //! Enable/Disable simple format
  void set_simple(bool enable = true) {
    if (enable) {
      option_.options |= MOD_JSON_SIMPLE;
    } else {
      option_.options &= ~MOD_JSON_SIMPLE;
    }
  }

  //! Enable/Disable single quotes support
  void set_squote(bool enable = true) {
    if (enable) {
      option_.options |= MOD_JSON_SQUOTE;
    } else {
      option_.options &= ~MOD_JSON_SQUOTE;
    }
  }

  //! Convert a sting to a JSON value
  bool parse(const char *str, JsonValue *out) {
    mod_json_token_t *tok;

    state_ = mod_json_state_null;
    error_ = mod_json_error_null;
    context_ = str;

    tok = mod_json_token_create(&option_);
    if (tok) {
      mod_json_value_t *jval;

      jval = mod_json_parse(tok, str);

      /* save information of token */
      state_ = mod_json_token_state(tok);
      error_ = mod_json_token_error(tok);
      context_ = mod_json_token_context(tok);
      mod_json_token_destroy(tok);

      if (jval) {
        *out = *reinterpret_cast<JsonValue *>(&jval);
        mod_json_value_unset(jval);

        return out->is_valid();
      }
    }
    return false;
  }

  //! Retrieve the error code of parser
  int error(void) const {
    return (int)error_;
  }

  //! Retrieve the state code of parser
  int state(void) const {
    return (int)state_;
  }

  //! Retrieve the context of parser
  const char *context(void) const {
    return context_;
  }

 private:
  mod_json_option_t option_;
  mod_json_state_t state_;
  mod_json_error_t error_;
  mod_json_cchar_t *context_;
};

/*! JSON Dumper
 */
class JsonDumper {
 public:
  //! Constructor
  JsonDumper(void) : str_() {}

  //! Destructor
  ~JsonDumper(void) {}

  //! Dump a JSON value to string
  bool dump(const JsonValue &val) {
    mod_json_string_t *str;

    str = mod_json_dump(*((mod_json_value_t **)&val));
    str_ = *reinterpret_cast<JsonString *>(&str);
    if (str) {
      mod_json_string_unset(str);
      return true;
    }
    return false;
  }

  //! Retrieve result of dumper
  JsonString &result(void) {
    return str_;
  }

  //! Retrieve result of dumper
  const JsonString &result(void) const {
    return str_;
  }

 private:
  JsonString str_;
};

//! Equality
static inline bool operator==(const ailego::JsonString &lhs, const char *rhs) {
  const char *self = lhs.c_str();
  if (self == rhs) {
    return true;
  }

  if (self && rhs) {
    return (std::strcmp(self, rhs) == 0);
  }
  return false;
}

//! Equality
static inline bool operator==(const char *lhs, const ailego::JsonString &rhs) {
  return (rhs == lhs);
}

//! Equality
static inline bool operator==(const ailego::JsonString &lhs,
                              const std::string &rhs) {
  std::size_t ls = lhs.size();
  std::size_t rs = rhs.size();
  if (ls == 0 && rs == 0) {
    return true;
  }

  if (ls == rs) {
    const char *ld = lhs.data();
    const char *rd = rhs.data();

    if (ld && rd) {
      return (std::memcmp(ld, rd, ls) == 0);
    }
  }
  return false;
}

//! Equality
static inline bool operator==(const std::string &lhs, const JsonString &rhs) {
  return (rhs == lhs);
}

//! Equality
static inline bool operator==(const JsonString &lhs, const JsonValue &rhs) {
  return (rhs.is_string() ? lhs == rhs.as_string() : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const JsonString &rhs) {
  return (lhs.is_string() ? lhs.as_string() == rhs : false);
}

//! Equality
static inline bool operator==(const JsonArray &lhs, const JsonValue &rhs) {
  return (rhs.is_array() ? lhs == rhs.as_array() : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const JsonArray &rhs) {
  return (lhs.is_array() ? lhs.as_array() == rhs : false);
}

//! Equality
static inline bool operator==(const JsonObject &lhs, const JsonValue &rhs) {
  return (rhs.is_object() ? lhs == rhs.as_object() : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const JsonObject &rhs) {
  return (lhs.is_object() ? lhs.as_object() == rhs : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const bool &rhs) {
  return (lhs.is_boolean() ? lhs.as_bool() == rhs : false);
}

//! Equality
static inline bool operator==(const bool &lhs, const JsonValue &rhs) {
  return (rhs.is_boolean() ? lhs == rhs.as_bool() : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const signed char &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const signed char &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const char &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const char &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const short int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const short int &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const int &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const long int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const long int &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const long long int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const long long int &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const float &rhs) {
  if (lhs.is_float()) {
    double diff = static_cast<double>(lhs.as_float() - rhs);
    return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
  }
  return false;
}

//! Equality
static inline bool operator==(const float &lhs, const JsonValue &rhs) {
  if (rhs.is_float()) {
    double diff = static_cast<double>(rhs.as_float() - lhs);
    return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
  }
  return false;
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const double &rhs) {
  if (lhs.is_float()) {
    double diff = static_cast<double>(lhs.as_float() - rhs);
    return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
  }
  return false;
}

//! Equality
static inline bool operator==(const double &lhs, const JsonValue &rhs) {
  if (rhs.is_float()) {
    double diff = static_cast<double>(rhs.as_float() - lhs);
    return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
  }
  return false;
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const long double &rhs) {
  if (lhs.is_float()) {
    double diff = static_cast<double>(lhs.as_float() - rhs);
    return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
  }
  return false;
}

//! Equality
static inline bool operator==(const long double &lhs, const JsonValue &rhs) {
  if (rhs.is_float()) {
    double diff = static_cast<double>(rhs.as_float() - lhs);
    return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
  }
  return false;
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const unsigned char &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const unsigned char &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs,
                              const unsigned short int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const unsigned short int &lhs,
                              const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const unsigned int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const unsigned int &lhs, const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs,
                              const unsigned long int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const unsigned long int &lhs,
                              const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs,
                              const unsigned long long int &rhs) {
  return (lhs.is_integer()
              ? lhs.as_integer() == static_cast<JsonValue::integer_type>(rhs)
              : false);
}

//! Equality
static inline bool operator==(const unsigned long long int &lhs,
                              const JsonValue &rhs) {
  return (rhs.is_integer()
              ? static_cast<JsonValue::integer_type>(lhs) == rhs.as_integer()
              : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const char *rhs) {
  return (lhs.is_string() ? lhs.as_string() == rhs : false);
}

//! Equality
static inline bool operator==(const char *lhs, const JsonValue &rhs) {
  return (rhs.is_string() ? lhs == rhs.as_string() : false);
}

//! Equality
static inline bool operator==(const JsonValue &lhs, const std::string &rhs) {
  return (lhs.is_string() ? lhs.as_string() == rhs : false);
}

//! Equality
static inline bool operator==(const std::string &lhs, const JsonValue &rhs) {
  return (rhs.is_string() ? lhs == rhs.as_string() : false);
}

//! No equality
static inline bool operator!=(const JsonString &lhs, const char *rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const char *lhs, const JsonString &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonString &lhs, const std::string &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const std::string &lhs, const JsonString &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonString &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const JsonString &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonArray &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const JsonArray &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonObject &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const JsonObject &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const bool &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const bool &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const signed char &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const signed char &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const char &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const char &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const short int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const short int &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const int &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const long int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const long int &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const long long int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const long long int &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const float &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const float &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const double &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const double &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const long double &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const long double &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const unsigned char &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned char &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs,
                              const unsigned short int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned short int &lhs,
                              const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const unsigned int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned int &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs,
                              const unsigned long int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned long int &lhs,
                              const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs,
                              const unsigned long long int &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned long long int &lhs,
                              const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const char *rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const char *lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const JsonValue &lhs, const std::string &rhs) {
  return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const std::string &lhs, const JsonValue &rhs) {
  return !(lhs == rhs);
}

}  // namespace ailego
}  // namespace zvec
