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

#include <memory>
#include <tuple>
#include <type_traits>

namespace zvec {
namespace ailego {

/*! Callback Validator (declaration)
 */
template <typename TFunc>
struct CallbackValidator;

/*! Callback Validator (function pointer)
 */
template <typename R, typename... TParams>
struct CallbackValidator<R (*)(TParams...)> {
  enum { Value = true };
};

/*! Callback Validator (function)
 */
template <typename R, typename... TParams>
struct CallbackValidator<R(TParams...)> : CallbackValidator<R (*)(TParams...)> {
};

/*! Callback Validator (member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackValidator<R (T::*)(TParams...)>
    : CallbackValidator<R (*)(TParams...)> {};

/*! Callback Validator (constable member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackValidator<R (T::*)(TParams...) const>
    : CallbackValidator<R (*)(TParams...)> {};

/*! Callback Validator (volatile member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackValidator<R (T::*)(TParams...) volatile>
    : CallbackValidator<R (*)(TParams...)> {};

/*! Callback Validator (constable volatile member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackValidator<R (T::*)(TParams...) const volatile>
    : CallbackValidator<R (*)(TParams...)> {};

/*! Callback Validator
 */
template <typename TFunc>
struct CallbackValidator {
 protected:
  using FalseType = long;
  using TrueType = char;

  //! Check if the class contains operator()
  template <typename T>
  static TrueType &Validate(decltype(&T::operator()));

  //! Check if the class contains operator()
  template <typename T>
  static FalseType &Validate(...);

 public:
  enum { Value = (sizeof(Validate<TFunc>(nullptr)) == sizeof(TrueType)) };
};

/*! Callback Validator (left reference)
 */
template <typename TFunc>
struct CallbackValidator<TFunc &> : CallbackValidator<TFunc> {};

/*! Callback Validator (right reference)
 */
template <typename TFunc>
struct CallbackValidator<TFunc &&> : CallbackValidator<TFunc> {};

/*! Callback Traits (declaration)
 */
template <typename TFunc>
struct CallbackTraits;

/*! Callback Traits (function pointer)
 */
template <typename R, typename... TParams>
struct CallbackTraits<R (*)(TParams...)> {
  using Type = R (*)(TParams...);
  using ResultType = R;
  using TupleType = std::tuple<typename std::decay<TParams>::type...>;

  //! Callback Traits Parameter
  template <size_t N>
  struct Parameter {
    using Type = typename std::tuple_element<N, std::tuple<TParams...>>::type;
  };

  //! Number of parameters
  enum { Arity = sizeof...(TParams) };
};

/*! Callback Traits (function)
 */
template <typename R, typename... TParams>
struct CallbackTraits<R(TParams...)> : CallbackTraits<R (*)(TParams...)> {
  using Type = R (*)(TParams...);
};

/*! Callback Traits (member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackTraits<R (T::*)(TParams...)>
    : CallbackTraits<R (*)(TParams...)> {
  using Type = R (T::*)(TParams...);
};

/*! Callback Traits (constable member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackTraits<R (T::*)(TParams...) const>
    : CallbackTraits<R (*)(TParams...)> {
  using Type = R (T::*)(TParams...) const;
};

/*! Callback Traits (volatile member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackTraits<R (T::*)(TParams...) volatile>
    : CallbackTraits<R (*)(TParams...)> {
  using Type = R (T::*)(TParams...) volatile;
};

/*! Callback Traits (constable volatile member function pointer)
 */
template <typename T, typename R, typename... TParams>
struct CallbackTraits<R (T::*)(TParams...) const volatile>
    : CallbackTraits<R (*)(TParams...)> {
  using Type = R (T::*)(TParams...) const volatile;
};

/*! Callback Traits
 */
template <typename TFunc>
struct CallbackTraits : CallbackTraits<decltype(&TFunc::operator())> {
  using Type = TFunc;
};

/*! Callback Traits (left reference)
 */
template <typename TFunc>
struct CallbackTraits<TFunc &> : CallbackTraits<TFunc> {};

/*! Callback Traits (right reference)
 */
template <typename TFunc>
struct CallbackTraits<TFunc &&> : CallbackTraits<TFunc> {};

/*! Callback Functor
 */
template <typename TFunc>
struct CallbackFunctor {
  using Traits = CallbackTraits<TFunc>;
  using Type = typename Traits::Type;
  using ResultType = typename Traits::ResultType;
  using TupleType = typename Traits::TupleType;

  //! Tuple Index Maker
  template <size_t N, size_t... I>
  struct TupleIndexMaker : TupleIndexMaker<N - 1, N - 1, I...> {};

  //! Tuple Index
  template <size_t...>
  struct TupleIndex {};

  //! Tuple Index Maker (special)
  template <size_t... I>
  struct TupleIndexMaker<0, I...> {
    using Type = TupleIndex<I...>;
  };

  //! Run the callback function
  template <size_t... I>
  static ResultType Run(Type &impl, TupleType &tuple, TupleIndex<I...>) {
    return (impl)(std::forward<typename Traits::template Parameter<I>::Type>(
        std::get<I>(tuple))...);
  }

  //! Run the callback member function
  template <typename T, size_t... I>
  static ResultType Run(T *obj, Type &impl, TupleType &tuple,
                        TupleIndex<I...>) {
    return (obj->*impl)(
        std::forward<typename Traits::template Parameter<I>::Type>(
            std::get<I>(tuple))...);
  }

  //! Run the callback function
  static ResultType Run(Type &impl, TupleType &tuple) {
    return Run(impl, tuple, typename TupleIndexMaker<Traits::Arity>::Type());
  }

  //! Run the callback member function
  template <typename T>
  static ResultType Run(T *obj, Type &impl, TupleType &tuple) {
    return Run(obj, impl, tuple,
               typename TupleIndexMaker<Traits::Arity>::Type());
  }
};

/*! Callback Object
 */
template <typename T>
struct CallbackObject {
  using Type = typename std::remove_reference<T>::type;
};

/*! Callback (declaration)
 */
template <typename R>
class Callback;

/*! Callback (void)
 */
template <>
class Callback<void> {
 public:
  using Pointer = std::shared_ptr<Callback<void>>;

  //! Destructor
  virtual ~Callback(void) {}

  //! Function call
  void operator()(void) {
    this->run();
  }

  //! Run the callback function
  virtual void run(void) = 0;

  //! Create callback closure (member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static typename Callback<R>::Pointer New(T *obj, R (T::*impl)(TParams...),
                                           TArgs &&...args);

  //! Create callback closure (constable member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static typename Callback<R>::Pointer New(const T *obj,
                                           R (T::*impl)(TParams...) const,
                                           TArgs &&...args);

  //! Create callback closure (volatile member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static typename Callback<R>::Pointer New(volatile T *obj,
                                           R (T::*impl)(TParams...) volatile,
                                           TArgs &&...args);

  //! Create callback closure (constable volatile member function pointer)
  template <typename T, typename R, typename... TParams, typename... TArgs>
  static typename Callback<R>::Pointer New(const volatile T *obj,
                                           R (T::*impl)(TParams...)
                                               const volatile,
                                           TArgs &&...args);

  //! Create callback closure (function)
  template <
      typename TFunc, typename... TArgs,
      typename = typename std::enable_if<CallbackValidator<TFunc>::Value>::type>
  static typename Callback<typename CallbackTraits<TFunc>::ResultType>::Pointer
  New(TFunc &&impl, TArgs &&...args);
};

/*! Callback
 */
template <typename R>
class Callback : public Callback<void> {
 public:
  using Pointer = std::shared_ptr<Callback<R>>;
  using Callback<void>::run;

  //! Function call
  void operator()(void) {
    this->run();
  }

  //! Function call with return
  void operator()(R *r) {
    this->run(r);
  }

  //! Run the callback function
  virtual void run(R *) = 0;

 protected:
  //! Constructor
  Callback(void) {};
};

/*! Callback Implementation
 */
template <typename T, typename R, typename TFunc>
class CallbackImpl : public Callback<R> {
 public:
  using Object = CallbackObject<T>;
  using Functor = CallbackFunctor<TFunc>;

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(typename Object::Type *obj, const typename Functor::Type &impl,
               TArgs &&...args)
      : obj_(obj), impl_(impl), tuple_(std::forward<TArgs>(args)...) {}

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(typename Object::Type *obj, typename Functor::Type &&impl,
               TArgs &&...args)
      : obj_(obj),
        impl_(std::move(impl)),
        tuple_(std::forward<TArgs>(args)...) {}

  //! Run the callback function
  void run(void) override {
    Functor::Run(obj_, impl_, tuple_);
  }

  //! Run the callback function
  void run(R *r) override {
    *r = Functor::Run(obj_, impl_, tuple_);
  }

 protected:
  //! Disable them
  CallbackImpl(void) = delete;
  CallbackImpl(const CallbackImpl &) = delete;
  CallbackImpl(CallbackImpl &&) = delete;
  CallbackImpl &operator=(const CallbackImpl &) = delete;

 private:
  typename Object::Type *obj_;
  typename Functor::Type impl_;
  typename Functor::TupleType tuple_;
};

/*! Callback Implementation
 */
template <typename T, typename TFunc>
class CallbackImpl<T, void, TFunc> : public Callback<void> {
 public:
  using Object = CallbackObject<T>;
  using Functor = CallbackFunctor<TFunc>;

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(typename Object::Type *obj, const typename Functor::Type &impl,
               TArgs &&...args)
      : obj_(obj), impl_(impl), tuple_(std::forward<TArgs>(args)...) {}

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(typename Object::Type *obj, typename Functor::Type &&impl,
               TArgs &&...args)
      : obj_(obj),
        impl_(std::move(impl)),
        tuple_(std::forward<TArgs>(args)...) {}

  //! Run the callback function
  void run(void) override {
    Functor::Run(obj_, impl_, tuple_);
  }

 protected:
  //! Disable them
  CallbackImpl(void) = delete;
  CallbackImpl(const CallbackImpl &) = delete;
  CallbackImpl(CallbackImpl &&) = delete;
  CallbackImpl &operator=(const CallbackImpl &) = delete;

 private:
  typename Object::Type *obj_;
  typename Functor::Type impl_;
  typename Functor::TupleType tuple_;
};

/*! Callback Implementation
 */
template <typename R, typename TFunc>
class CallbackImpl<void, R, TFunc> : public Callback<R> {
 public:
  using Functor = CallbackFunctor<TFunc>;

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(const typename Functor::Type &impl, TArgs &&...args)
      : impl_(impl), tuple_(std::forward<TArgs>(args)...) {}

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(typename Functor::Type &&impl, TArgs &&...args)
      : impl_(std::move(impl)), tuple_(std::forward<TArgs>(args)...) {}

  //! Run the callback function
  void run(void) override {
    Functor::Run(impl_, tuple_);
  }

  //! Run the callback function
  void run(R *r) override {
    *r = Functor::Run(impl_, tuple_);
  }

 protected:
  //! Disable them
  CallbackImpl(void) = delete;
  CallbackImpl(const CallbackImpl &) = delete;
  CallbackImpl(CallbackImpl &&) = delete;
  CallbackImpl &operator=(const CallbackImpl &) = delete;

 private:
  typename Functor::Type impl_;
  typename Functor::TupleType tuple_;
};

/*! Callback Implementation
 */
template <typename TFunc>
class CallbackImpl<void, void, TFunc> : public Callback<void> {
 public:
  using Functor = CallbackFunctor<TFunc>;

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(const typename Functor::Type &impl, TArgs &&...args)
      : impl_(impl), tuple_(std::forward<TArgs>(args)...) {}

  //! Constructor
  template <typename... TArgs>
  CallbackImpl(typename Functor::Type &&impl, TArgs &&...args)
      : impl_(std::move(impl)), tuple_(std::forward<TArgs>(args)...) {}

  //! Run the callback function
  void run(void) override {
    Functor::Run(impl_, tuple_);
  }

 protected:
  //! Disable them
  CallbackImpl(void) = delete;
  CallbackImpl(const CallbackImpl &) = delete;
  CallbackImpl(CallbackImpl &&) = delete;
  CallbackImpl &operator=(const CallbackImpl &) = delete;

 private:
  typename Functor::Type impl_;
  typename Functor::TupleType tuple_;
};

//! Create callback closure (member function pointer)
template <typename T, typename R, typename... TParams, typename... TArgs>
typename Callback<R>::Pointer Callback<void>::New(T *obj,
                                                  R (T::*impl)(TParams...),
                                                  TArgs &&...args) {
  return std::make_shared<CallbackImpl<T, R, decltype(impl)>>(
      obj, impl, std::forward<TArgs>(args)...);
}

//! Create callback closure (constable member function pointer)
template <typename T, typename R, typename... TParams, typename... TArgs>
typename Callback<R>::Pointer Callback<void>::New(const T *obj,
                                                  R (T::*impl)(TParams...)
                                                      const,
                                                  TArgs &&...args) {
  return std::make_shared<CallbackImpl<const T, R, decltype(impl)>>(
      obj, impl, std::forward<TArgs>(args)...);
}

//! Create callback closure (volatile member function pointer)
template <typename T, typename R, typename... TParams, typename... TArgs>
typename Callback<R>::Pointer Callback<void>::New(
    volatile T *obj, R (T::*impl)(TParams...) volatile, TArgs &&...args) {
  return std::make_shared<CallbackImpl<volatile T, R, decltype(impl)>>(
      obj, impl, std::forward<TArgs>(args)...);
}

//! Create callback closure (constable volatile member function pointer)
template <typename T, typename R, typename... TParams, typename... TArgs>
typename Callback<R>::Pointer Callback<void>::New(const volatile T *obj,
                                                  R (T::*impl)(TParams...)
                                                      const volatile,
                                                  TArgs &&...args) {
  return std::make_shared<CallbackImpl<const volatile T, R, decltype(impl)>>(
      obj, impl, std::forward<TArgs>(args)...);
}

//! Create callback closure (function)
template <typename TFunc, typename... TArgs, typename>
typename Callback<typename CallbackTraits<TFunc>::ResultType>::Pointer
Callback<void>::New(TFunc &&impl, TArgs &&...args) {
  return std::make_shared<CallbackImpl<
      void, typename CallbackTraits<TFunc>::ResultType, decltype(impl)>>(
      std::forward<TFunc>(impl), std::forward<TArgs>(args)...);
}

//! Callback Handler
template <typename R>
using CallbackHandler = typename Callback<R>::Pointer;

//! Closure
using Closure = Callback<void>;

//! Closure Handler
using ClosureHandler = Closure::Pointer;

}  // namespace ailego
}  // namespace zvec
