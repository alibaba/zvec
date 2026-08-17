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

#if defined(_WIN32) || defined(__CYGWIN__)
#define ZVEC_HELPER_DLL_EXPORT __declspec(dllexport)
#define ZVEC_HELPER_DLL_IMPORT __declspec(dllimport)
#else
#if defined(__GNUC__) && __GNUC__ >= 4
#define ZVEC_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define ZVEC_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#else
#define ZVEC_HELPER_DLL_EXPORT
#define ZVEC_HELPER_DLL_IMPORT
#endif
#endif

#if defined(ZVEC_DB_BUILD_SHARED)
#define ZVEC_API ZVEC_HELPER_DLL_EXPORT
#elif defined(ZVEC_DB_USE_SHARED)
#define ZVEC_API ZVEC_HELPER_DLL_IMPORT
#else
#define ZVEC_API
#endif

#if defined(ZVEC_CORE_BUILD_SHARED)
#define ZVEC_CORE_API ZVEC_HELPER_DLL_EXPORT
#elif defined(ZVEC_CORE_USE_SHARED)
#define ZVEC_CORE_API ZVEC_HELPER_DLL_IMPORT
#else
#define ZVEC_CORE_API
#endif

#if defined(ZVEC_AILEGO_BUILD_SHARED)
#define ZVEC_AILEGO_API ZVEC_HELPER_DLL_EXPORT
#elif defined(ZVEC_AILEGO_USE_SHARED)
#define ZVEC_AILEGO_API ZVEC_HELPER_DLL_IMPORT
#else
#define ZVEC_AILEGO_API
#endif

#if defined(ZVEC_TURBO_BUILD_SHARED)
#define ZVEC_TURBO_API ZVEC_HELPER_DLL_EXPORT
#elif defined(ZVEC_TURBO_USE_SHARED)
#define ZVEC_TURBO_API ZVEC_HELPER_DLL_IMPORT
#else
#define ZVEC_TURBO_API
#endif
