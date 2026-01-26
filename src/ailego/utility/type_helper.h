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

#include <functional>
#include <type_traits>
#include <zvec/ailego/utility/float_helper.h>

namespace zvec {
namespace ailego {

//! Determines if a type is an arithmetic type (includes Float16)
template <typename T>
struct IsArithmetic
    : std::integral_constant<bool, std::is_arithmetic<T>::value ||
                                       std::is_same<T, Float16>::value> {};

//! Determines if a type is a signed arithmetic type (includes Float16)
template <typename T>
struct IsSignedArithmetic
    : std::integral_constant<bool, std::is_signed<T>::value ||
                                       std::is_same<T, Float16>::value> {};

//! Determines if a type is a unsigned arithmetic type (includes Float16)
template <typename T>
struct IsUnsignedArithmetic
    : std::integral_constant<bool, std::is_unsigned<T>::value> {};

//! Determines if a type is a floating-point type (includes Float16)
template <typename T>
struct IsFloatingPoint
    : std::integral_constant<bool, std::is_floating_point<T>::value ||
                                       std::is_same<T, Float16>::value> {};

#if __GNUC__ >= 5 || defined(_MSC_VER) || defined(__clang__)
template <typename T>
using IsTriviallyCopyable = std::is_trivially_copyable<T>;
#else
template <typename T>
using IsTriviallyCopyable = std::has_trivial_copy_constructor<T>;
#endif

#if __cplusplus >= 201703L  // C++17

//! Determines if a type can be invoked with the specified argument types
template <typename TFunc, typename... TArgs>
using IsInvocable = std::is_invocable<TFunc, TArgs...>;

//! Determines if a type can be invoked with the specified argument types
template <typename R, typename TFunc, typename... TArgs>
using IsInvocableWithResult = std::is_invocable_r<R, TFunc, TArgs...>;

#else
//! Determines if a type can be invoked with the specified argument types
template <typename TFunc, typename... TArgs>
struct IsInvocable
    : std::is_constructible<std::function<void(TArgs...)>,
                            std::reference_wrapper<
                                typename std::remove_reference<TFunc>::type> > {
};

//! Determines if a type can be invoked with the specified argument types
template <typename R, typename TFunc, typename... TArgs>
struct IsInvocableWithResult
    : std::is_constructible<std::function<R(TArgs...)>,
                            std::reference_wrapper<
                                typename std::remove_reference<TFunc>::type> > {
};
#endif

//! Fixed underlying_type used with conditional
template <typename T, bool = std::is_enum<T>::value>
struct UnderlyingType {
  typedef typename std::remove_cv<T>::type type;
};

//! Fixed underlying_type used with conditional
template <typename T>
struct UnderlyingType<T, true> {
  typedef typename std::underlying_type<T>::type type;
};

#if __cplusplus >= 201703L  // C++17

//! Variadic logical AND metafunction
template <typename... TConds>
using Conjunction = std::conjunction<TConds...>;

//! Variadic logical OR metafunction
template <typename... TConds>
using Disjunction = std::disjunction<TConds...>;

#else
//! Variadic logical AND metafunction
template <typename... TConds>
struct Conjunction : std::true_type {};

//! Variadic logical AND metafunction
template <typename TCond, typename... TConds>
struct Conjunction<TCond, TConds...>
    : std::conditional<TCond::value, Conjunction<TConds...>,
                       std::false_type>::type {};

//! Variadic logical OR metafunction
template <typename... TConds>
struct Disjunction : std::false_type {};

//! Variadic logical OR metafunction
template <typename TCond, typename... TConds>
struct Disjunction<TCond, TConds...>
    : std::conditional<TCond::value, std::true_type,
                       Disjunction<TConds...> >::type {};
#endif

}  // namespace ailego
}  // namespace zvec
