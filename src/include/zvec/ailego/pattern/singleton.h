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

#include <type_traits>

namespace zvec {
namespace ailego {

/*! Singleton (C++11)
 */
template <typename T>
class Singleton {
 public:
  using ObjectType = typename std::remove_reference<T>::type;

  //! Retrieve instance of object
  static ObjectType &Instance(void) noexcept(
      std::is_nothrow_constructible<ObjectType>::value) {
    // Since it's a static variable, if the class has already been created,
    // it won't be created again. And it is thread-safe in C++11.
    static ObjectType obj;
    return obj;
  }

 protected:
  //! Constructor (Allow inheritance)
  Singleton(void) {}

 private:
  //! Disable them
  Singleton(Singleton const &) = delete;
  Singleton(Singleton &&) = delete;
  Singleton &operator=(Singleton const &) = delete;
  Singleton &operator=(Singleton &&) = delete;
};

}  // namespace ailego
}  // namespace zvec
