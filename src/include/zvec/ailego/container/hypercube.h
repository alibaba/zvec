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

#include <map>
#include <string>
#include <zvec/ailego/container/cube.h>

namespace zvec {
namespace ailego {

/*! Hypercube
 */
class Hypercube {
 public:
  //! Constructor
  Hypercube(void) : cubes_() {}

  //! Constructor
  Hypercube(const Hypercube &rhs) : cubes_(rhs.cubes_) {}

  //! Constructor
  Hypercube(Hypercube &&rhs) : cubes_() {
    cubes_.swap(rhs.cubes_);
  }

  //! Destructor
  ~Hypercube(void) {}

  //! Assignment
  Hypercube &operator=(const Hypercube &rhs) {
    cubes_ = rhs.cubes_;
    return *this;
  }

  //! Assignment
  Hypercube &operator=(Hypercube &&rhs) {
    cubes_ = std::move(rhs.cubes_);
    return *this;
  }

  //! Overloaded operator []
  Cube &operator[](const std::string &key) {
    return cubes_[key];
  }

  //! Overloaded operator []
  Cube &operator[](std::string &&key) {
    return cubes_[std::forward<std::string>(key)];
  }

  //! Test if the element is exist
  bool has(const std::string &key) const {
    return (cubes_.find(key) != cubes_.end());
  }

  //! Test if the hyper cube is empty
  bool empty(void) const {
    return cubes_.empty();
  }

  //! Insert a key-value pair into map
  bool insert(const std::string &key, Cube &&val) {
    return cubes_.emplace(key, std::forward<Cube>(val)).second;
  }

  //! Insert a key-value pair into map
  bool insert(std::string &&key, Cube &&val) {
    return cubes_
        .emplace(std::forward<std::string>(key), std::forward<Cube>(val))
        .second;
  }

  //! Insert a key-value pair into map
  template <typename T>
  bool insert(const std::string &key, T &&val) {
    return cubes_.emplace(key, Cube(std::forward<T>(val))).second;
  }

  //! Insert a key-value pair into map
  template <typename T>
  bool insert(std::string &&key, T &&val) {
    return cubes_
        .emplace(std::forward<std::string>(key), Cube(std::forward<T>(val)))
        .second;
  }

  //! Insert or assign a key-value pair to map
  void insert_or_assign(const std::string &key, Cube &&val) {
    auto it = cubes_.lower_bound(key);
    if (it != cubes_.end() && it->first == key) {
      it->second = std::forward<Cube>(val);
    } else {
      cubes_.emplace_hint(it, key, std::forward<Cube>(val));
    }
  }

  //! Insert or assign a key-value pair to map
  void insert_or_assign(std::string &&key, Cube &&val) {
    auto it = cubes_.lower_bound(key);
    if (it != cubes_.end() && it->first == key) {
      it->second = std::forward<Cube>(val);
    } else {
      cubes_.emplace_hint(it, std::forward<std::string>(key),
                          std::forward<Cube>(val));
    }
  }

  //! Insert or assign a key-value pair to map
  template <typename T>
  void insert_or_assign(const std::string &key, T &&val) {
    auto it = cubes_.lower_bound(key);
    if (it != cubes_.end() && it->first == key) {
      it->second = Cube(std::forward<T>(val));
    } else {
      cubes_.emplace_hint(it, key, Cube(std::forward<T>(val)));
    }
  }

  //! Insert or assign a key-value pair to map
  template <typename T>
  void insert_or_assign(std::string &&key, T &&val) {
    auto it = cubes_.lower_bound(key);
    if (it != cubes_.end() && it->first == key) {
      it->second = Cube(std::forward<T>(val));
    } else {
      cubes_.emplace_hint(it, std::forward<std::string>(key),
                          Cube(std::forward<T>(val)));
    }
  }

  //! Clear the map
  void clear(void) {
    cubes_.clear();
  }

  //! Swap the map
  void swap(Hypercube &rhs) {
    cubes_.swap(rhs.cubes_);
  }

  //! Erase the pair via a key
  bool erase(const std::string &key) {
    auto iter = cubes_.find(key);
    if (iter != cubes_.end()) {
      cubes_.erase(iter);
      return true;
    }
    return false;
  }

  //! Retrieve the value via a key
  bool get(const std::string &key, Cube *out) const {
    auto iter = cubes_.find(key);
    if (iter != cubes_.end()) {
      *out = iter->second;
      return true;
    }
    return false;
  }

  //! Retrieve the value via a key
  Cube *get(const std::string &key) {
    auto iter = cubes_.find(key);
    if (iter != cubes_.end()) {
      return &iter->second;
    }
    return nullptr;
  }

  //! Retrieve the value via a key
  const Cube *get(const std::string &key) const {
    auto iter = cubes_.find(key);
    if (iter != cubes_.end()) {
      return &iter->second;
    }
    return nullptr;
  }

  //! Retrieve the value via a key
  template <typename T>
  bool get(const std::string &key, T *out) const {
    auto iter = cubes_.find(key);
    if (iter != cubes_.end()) {
      if (iter->second.compatible<T>()) {
        *out = iter->second.unsafe_cast<T>();
        return true;
      }
    }
    return false;
  }

  //! Retrieve the value via a key
  template <typename T>
  T &get(const std::string &key, T &def) {
    auto iter = cubes_.find(key);
    if (iter != cubes_.end()) {
      if (iter->second.compatible<T>()) {
        return iter->second.unsafe_cast<T>();
      }
    }
    return def;
  }

  //! Retrieve the value via a key
  template <typename T>
  const T &get(const std::string &key, const T &def) const {
    auto iter = cubes_.find(key);
    if (iter != cubes_.end()) {
      if (iter->second.compatible<T>()) {
        return iter->second.unsafe_cast<T>();
      }
    }
    return def;
  }

  //! Merge another hyper cube
  void merge(const Hypercube &rhs) {
    for (const auto &it : rhs.cubes_) {
      auto iter = cubes_.find(it.first);
      if (iter != cubes_.end()) {
        iter->second = it.second;
      } else {
        cubes_.emplace(it.first, it.second);
      }
    }
  }

  //! Merge another hyper cube
  void merge(Hypercube &&rhs) {
    for (auto &it : rhs.cubes_) {
      auto iter = cubes_.find(it.first);
      if (iter != cubes_.end()) {
        iter->second = std::move(it.second);
      } else {
        cubes_.emplace(std::move(it.first), std::move(it.second));
      }
    }
  }

  //! Retrieve the cubes
  const std::map<std::string, Cube> &cubes(void) const {
    return cubes_;
  }

  //! Retrieve the cubes
  std::map<std::string, Cube> *mutable_cubes(void) {
    return &cubes_;
  }

 private:
  std::map<std::string, Cube> cubes_;
};

}  // namespace ailego
}  // namespace zvec