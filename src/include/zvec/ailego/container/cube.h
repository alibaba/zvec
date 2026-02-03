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
#include <string>
#include <typeinfo>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {
namespace internal {

/*! Cube Policy
 */
struct CubePolicy {
  //! Destructor
  virtual ~CubePolicy(void) {}

  //! Assign `src` to `dst`
  virtual void assign(const void *src, void **dst) = 0;

  //! Cleanup value
  virtual void cleanup(void **val) = 0;

  //! Clone value of `src` to `dst`
  virtual void clone(void *const *src, void **dst) = 0;

  //! Move `src` to `dst`
  virtual void move(void *src, void **dst) = 0;

  //! Retrieve size
  virtual size_t size(void) const = 0;

  //! Retrieve type information
  virtual const std::type_info &type(void) const = 0;

  //! Retrieve value
  virtual void *value(void **src) = 0;

  //! Retrieve value
  virtual const void *value(void *const *src) const = 0;
};

/*! Small Cube Policy
 */
template <typename T>
struct SmallCubePolicy : public CubePolicy {
  //! Assign `src` to `dst`
  void assign(const void *src, void **dst) {
    new (dst) T(*reinterpret_cast<const T *>(src));
  }

  //! Cleanup value
  void cleanup(void **val) {
    reinterpret_cast<T *>(val)->~T();
  }

  //! Clone value of `src` to `dst`
  void clone(void *const *src, void **dst) {
    new (dst) T(*reinterpret_cast<const T *>(src));
  }

  //! Move `src` to `dst`
  void move(void *src, void **dst) {
    new (dst) T(std::move(*reinterpret_cast<T *>(src)));
  }

  //! Retrieve size
  size_t size(void) const {
    return sizeof(T);
  }

  //! Retrieve type information
  const std::type_info &type(void) const {
    return typeid(T);
  }

  //! Retrieve value
  void *value(void **src) {
    return reinterpret_cast<void *>(src);
  }

  //! Retrieve value
  const void *value(void *const *src) const {
    return reinterpret_cast<const void *>(src);
  }
};

/*! Large Cube Policy
 */
template <typename T>
struct LargeCubePolicy : public CubePolicy {
  //! Assign `src` to `dst`
  void assign(const void *src, void **dst) {
    *dst = new T(*reinterpret_cast<const T *>(src));
  }

  //! Cleanup value
  void cleanup(void **val) {
    delete (reinterpret_cast<T *>(*val));
  }

  //! Clone value of `src` to `dst`
  void clone(void *const *src, void **dst) {
    *dst = new T(**reinterpret_cast<T *const *>(src));
  }

  //! Move `src` to `dst`
  void move(void *src, void **dst) {
    *dst = new T(std::move(*reinterpret_cast<T *>(src)));
  }

  //! Retrieve size
  size_t size(void) const {
    return sizeof(T);
  }

  //! Retrieve type information
  const std::type_info &type(void) const {
    return typeid(T);
  }

  //! Retrieve value
  void *value(void **src) {
    return *src;
  }

  //! Retrieve value
  const void *value(void *const *src) const {
    return *src;
  }
};

/*! Policy Selector
 */
template <typename T, typename = void>
struct PolicySelector {
  typedef LargeCubePolicy<T> Type;
};

/*! Policy Selector
 */
template <typename T>
struct PolicySelector<
    T, typename std::enable_if<sizeof(T) <= sizeof(void *)>::type> {
  typedef SmallCubePolicy<T> Type;
};

}  // namespace internal

/*! Cube class
 */
class Cube {
 public:
  //! Constructor
  Cube(void) : policy_(Cube::Policy<Cube::EmptyPolicy>()), object_(nullptr) {}

  //! Constructor
  template <typename T>
  Cube(const T &rhs) : policy_(Cube::Policy<T>()), object_(nullptr) {
    policy_->assign(&rhs, &object_);
  }

  //! Constructor
  template <typename T, typename = typename std::enable_if<
                            !std::is_same<Cube &, T>::value &&
                            !std::is_same<T &, T>::value>::type>
  Cube(T &&rhs) : policy_(Cube::Policy<T>()), object_(nullptr) {
    policy_->move(&rhs, &object_);
  }

  //! Constructor
  Cube(const char *str)
      : policy_(Cube::Policy<std::string>()), object_(nullptr) {
    std::string rhs(str);
    policy_->move(&rhs, &object_);
  }

  //! Constructor
  Cube(char str[]) : policy_(Cube::Policy<std::string>()), object_(nullptr) {
    std::string rhs(str);
    policy_->move(&rhs, &object_);
  }

  //! Constructor
  Cube(const Cube &rhs) : policy_(rhs.policy_), object_(nullptr) {
    policy_->clone(&rhs.object_, &object_);
  }

  //! Constructor
  Cube(Cube &&rhs) : policy_(rhs.policy_), object_(rhs.object_) {
    rhs.policy_ = Cube::Policy<Cube::EmptyPolicy>();
    rhs.object_ = nullptr;
  }

  //! Destructor
  ~Cube(void) {
    policy_->cleanup(&object_);
  }

  //! Assignment
  template <typename T>
  Cube &operator=(const T &rhs) {
    this->assign(rhs);
    return *this;
  }

  //! Assignment
  template <typename T, typename = typename std::enable_if<
                            !std::is_same<Cube &, T>::value &&
                            !std::is_same<T &, T>::value>::type>
  Cube &operator=(T &&rhs) {
    this->assign(std::forward<T>(rhs));
    return *this;
  }

  //! Assignment
  Cube &operator=(const Cube &rhs) {
    this->assign(rhs);
    return *this;
  }

  //! Assignment
  Cube &operator=(Cube &&rhs) {
    this->assign(std::forward<Cube>(rhs));
    return *this;
  }

  //! Assignment
  Cube &operator=(const char *str) {
    this->assign(str);
    return *this;
  }

  //! Assignment
  Cube &operator=(char str[]) {
    this->assign(str);
    return *this;
  }

  //! Retrieve object in original type
  template <typename T>
  operator T &() {
    return this->cast<T>();
  }

  //! Retrieve object in original type
  template <typename T>
  operator const T &() const {
    return this->cast<T>();
  }

  //! Assign content
  template <typename T>
  void assign(const T &rhs) {
    policy_->cleanup(&object_);
    policy_ = Cube::Policy<T>();
    policy_->assign(&rhs, &object_);
  }

  //! Assign content
  template <typename T, typename = typename std::enable_if<
                            !std::is_same<Cube &, T>::value &&
                            !std::is_same<T &, T>::value>::type>
  void assign(T &&rhs) {
    policy_->cleanup(&object_);
    policy_ = Cube::Policy<T>();
    policy_->move(&rhs, &object_);
  }

  //! Assign content from another Cube
  void assign(const Cube &rhs) {
    policy_->cleanup(&object_);
    policy_ = rhs.policy_;
    policy_->clone(&rhs.object_, &object_);
  }

  //! Assign content from another Cube
  void assign(Cube &&rhs) {
    if (this != &rhs) {
      policy_->cleanup(&object_);
      policy_ = rhs.policy_;
      object_ = rhs.object_;
      rhs.policy_ = Cube::Policy<Cube::EmptyPolicy>();
      rhs.object_ = nullptr;
    }
  }

  //! Assign content
  void assign(const char *str) {
    policy_->cleanup(&object_);
    policy_ = Cube::Policy<std::string>();
    std::string rhs(str);
    policy_->move(&rhs, &object_);
  }

  //! Assign content
  void assign(char str[]) {
    policy_->cleanup(&object_);
    policy_ = Cube::Policy<std::string>();
    std::string rhs(str);
    policy_->move(&rhs, &object_);
  }

  //! Swap the content with another Cube
  Cube &swap(Cube &rhs) {
    std::swap(policy_, rhs.policy_);
    std::swap(object_, rhs.object_);
    return *this;
  }

  //! Cast to the original type
  template <typename T>
  T &cast(void) {
    if (policy_ != Cube::Policy<T>()) {
      throw std::bad_cast();
    }
    return *reinterpret_cast<T *>(policy_->value(&object_));
  }

  //! Cast to the original type
  template <typename T>
  const T &cast(void) const {
    if (policy_ != Cube::Policy<T>()) {
      throw std::bad_cast();
    }
    return *reinterpret_cast<const T *>(policy_->value(&object_));
  }

  //! Cast to the original type (unsafe)
  template <typename T>
  T &unsafe_cast(void) {
    return *reinterpret_cast<T *>(policy_->value(&object_));
  }

  //! Cast to the original type (unsafe)
  template <typename T>
  const T &unsafe_cast(void) const {
    return *reinterpret_cast<const T *>(policy_->value(&object_));
  }

  //! Test if the Cube is empty
  bool empty(void) const {
    return (policy_ == Cube::Policy<Cube::EmptyPolicy>());
  }

  //! Reset Cube allocated memory
  void reset(void) {
    policy_->cleanup(&object_);
    policy_ = Cube::Policy<Cube::EmptyPolicy>();
    object_ = nullptr;
  }

  //! Test if the Cube is compatible with another one
  bool compatible(const Cube &rhs) const {
    return (policy_ == rhs.policy_ || policy_->type() == rhs.policy_->type());
  }

  //! Test if the Cube is compatible with another one
  template <typename T>
  bool compatible(void) const {
    return (policy_ == Cube::Policy<T>() ||
            policy_->type() == Cube::Policy<T>()->type());
  }

  //! Retrieve size
  size_t size(void) const {
    return (!this->empty() ? policy_->size() : 0u);
  }

  //! Retrieve type information
  const std::type_info &type(void) const {
    return (!this->empty() ? policy_->type() : typeid(void));
  }

 protected:
  /*! Empty Policy
   */
  struct EmptyPolicy {};

  //! Make a static policy object
  template <typename T>
  static internal::CubePolicy *MakePolicy(void) {
    static typename internal::PolicySelector<T>::Type policy;
    return (&policy);
  }

  //! Retrieve a static policy object
  template <typename T>
  static internal::CubePolicy *Policy(void) {
    return MakePolicy<typename UnderlyingType<T>::type>();
  }

 private:
  //! Members
  internal::CubePolicy *policy_;
  void *object_;
};

}  // namespace ailego
}  // namespace zvec