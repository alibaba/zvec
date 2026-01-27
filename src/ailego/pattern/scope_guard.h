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

#include <zvec/ailego/pattern/closure.h>

namespace zvec {
namespace ailego {

/*! Scope Guard Implementation
 */
template <typename T, typename TFunc>
class ScopeGuardImpl {
 public:
  using Object = CallbackObject<T>;
  using Functor = CallbackFunctor<TFunc>;

  //! Constructor
  ScopeGuardImpl(ScopeGuardImpl &&rhs)
      : obj_(rhs.obj_),
        impl_(std::move(rhs.impl_)),
        tuple_(std::move(rhs.tuple_)) {
    rhs.obj_ = nullptr;
  }

  //! Constructor
  template <typename... TArgs>
  ScopeGuardImpl(typename Object::Type *obj, const typename Functor::Type &impl,
                 TArgs &&...args)
      : obj_(obj), impl_(impl), tuple_(std::forward<TArgs>(args)...) {}

  //! Constructor
  template <typename... TArgs>
  ScopeGuardImpl(typename Object::Type *obj, typename Functor::Type &&impl,
                 TArgs &&...args)
      : obj_(obj),
        impl_(std::move(impl)),
        tuple_(std::forward<TArgs>(args)...) {}

  // Destructor
  ~ScopeGuardImpl(void) {
    if (obj_) {
      Functor::Run(obj_, impl_, tuple_);
    }
  }

 protected:
  //! Disable them
  ScopeGuardImpl(void) = delete;
  ScopeGuardImpl(const ScopeGuardImpl &) = delete;
  ScopeGuardImpl &operator=(const ScopeGuardImpl &) = delete;

 private:
  //! Members
  typename Object::Type *obj_;
  typename Functor::Type impl_;
  typename Functor::TupleType tuple_;
};

/*! Scope Guard Implementation (void, TFunc)
 */
template <typename TFunc>
class ScopeGuardImpl<void, TFunc> {
 public:
  //! Callback Functor Type
  using Functor = CallbackFunctor<TFunc>;

  //! Constructor
  ScopeGuardImpl(ScopeGuardImpl &&rhs)
      : impl_(std::move(rhs.impl_)),
        tuple_(std::move(rhs.tuple_)),
        valid_(rhs.valid_) {
    rhs.valid_ = false;
  }

  //! Constructor
  template <typename... TArgs>
  ScopeGuardImpl(const typename Functor::Type &impl, TArgs &&...args)
      : impl_(impl), tuple_(std::forward<TArgs>(args)...), valid_(true) {}

  //! Constructor
  template <typename... TArgs>
  ScopeGuardImpl(typename Functor::Type &&impl, TArgs &&...args)
      : impl_(std::move(impl)),
        tuple_(std::forward<TArgs>(args)...),
        valid_(true) {}

  // Destructor
  ~ScopeGuardImpl(void) {
    if (valid_) {
      Functor::Run(impl_, tuple_);
    }
  }

 protected:
  //! Disable them
  ScopeGuardImpl(void) = delete;
  ScopeGuardImpl(const ScopeGuardImpl &) = delete;
  ScopeGuardImpl &operator=(const ScopeGuardImpl &) = delete;

 private:
  //! Members
  typename Functor::Type impl_;
  typename Functor::TupleType tuple_;
  bool valid_;
};

/*! Scope Guard
 */
struct ScopeGuard {
  //! Make a scope guard object (member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static inline auto Make(T *obj, R (T::*impl)(TParams...), TArgs &&...args)
      -> ScopeGuardImpl<T, typename CallbackTraits<decltype(impl)>::Type> {
    return ScopeGuardImpl<T, typename CallbackTraits<decltype(impl)>::Type>(
        obj, impl, std::forward<TArgs>(args)...);
  }

  //! Make a scope guard object (constable member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static inline auto Make(const T *obj, R (T::*impl)(TParams...) const,
                          TArgs &&...args)
      -> ScopeGuardImpl<const T,
                        typename CallbackTraits<decltype(impl)>::Type> {
    return ScopeGuardImpl<const T,
                          typename CallbackTraits<decltype(impl)>::Type>(
        obj, impl, std::forward<TArgs>(args)...);
  }

  //! Make a scope guard object (volatile member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static inline auto Make(volatile T *obj, R (T::*impl)(TParams...) volatile,
                          TArgs &&...args)
      -> ScopeGuardImpl<volatile T,
                        typename CallbackTraits<decltype(impl)>::Type> {
    return ScopeGuardImpl<volatile T,
                          typename CallbackTraits<decltype(impl)>::Type>(
        obj, impl, std::forward<TArgs>(args)...);
  }

  //! Make a scope guard object (constable volatile member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static inline auto Make(const volatile T *obj,
                          R (T::*impl)(TParams...) const volatile,
                          TArgs &&...args)
      -> ScopeGuardImpl<const volatile T,
                        typename CallbackTraits<decltype(impl)>::Type> {
    return ScopeGuardImpl<const volatile T,
                          typename CallbackTraits<decltype(impl)>::Type>(
        obj, impl, std::forward<TArgs>(args)...);
  }

  //! Make a scope guard object (function)
  template <
      typename TFunc, typename... TArgs,
      typename = typename std::enable_if<CallbackValidator<TFunc>::Value>::type>
  static inline auto Make(TFunc &&impl, TArgs &&...args)
      -> ScopeGuardImpl<void, typename CallbackTraits<TFunc>::Type> {
    return ScopeGuardImpl<void, typename CallbackTraits<TFunc>::Type>(
        std::forward<TFunc>(impl), std::forward<TArgs>(args)...);
  }
};

}  // namespace ailego
}  // namespace zvec
