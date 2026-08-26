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

// Unified FP16 SIMD availability macros shared by all fp16 distance kernels.
//
// x86: FP16 kernels rely on F16C conversions. MSVC never defines __F16C__;
// its /arch:AVX2 (and above) flags imply F16C intrinsics are usable.
#if defined(__AVX2__) && (defined(__F16C__) || defined(_MSC_VER))
#define ZVEC_TURBO_FP16_AVX2 1
#else
#define ZVEC_TURBO_FP16_AVX2 0
#endif

#if defined(__AVX512F__) && (defined(__F16C__) || defined(_MSC_VER))
#define ZVEC_TURBO_FP16_AVX512 1
#else
#define ZVEC_TURBO_FP16_AVX512 0
#endif

// AArch64: besides NEON itself, native half-precision vector arithmetic
// (vfmaq_f16 etc.) additionally requires the compiler to advertise
// __ARM_FEATURE_FP16_VECTOR_ARITHMETIC (e.g. -march=armv8.2-a+fp16).
#if defined(__aarch64__) && defined(__ARM_NEON) && \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define ZVEC_TURBO_FP16_NEON 1
#else
#define ZVEC_TURBO_FP16_NEON 0
#endif
