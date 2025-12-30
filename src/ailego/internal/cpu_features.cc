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

#include "cpu_features.h"

#if !defined(_MSC_VER) && !defined(__ARM_ARCH)
#include <cpuid.h>
#endif

namespace zvec {
namespace ailego {
namespace internal {

//
// REFER: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/
//        tree/arch/x86/include/asm/cpufeatures.h
//        https://software.intel.com/sites/default/files/managed/c5/15/
//        architecture-instruction-set-extensions-programming-reference.pdf
//

CpuFeatures::CpuFlags CpuFeatures::flags_;

#if defined(_MSC_VER)
CpuFeatures::CpuFlags::CpuFlags(void)
    : L1_ECX(0), L1_EDX(0), L7_EBX(0), L7_ECX(0), L7_EDX(0) {
  int l1[4] = {0, 0, 0, 0};
  int l7[4] = {0, 0, 0, 0};

  __cpuidex(l1, 1, 0);
  __cpuidex(l7, 7, 0);
  L1_ECX = l1[2];
  L1_EDX = l1[3];
  L7_EBX = l7[1];
  L7_ECX = l7[2];
  L7_EDX = l7[3];
}
#elif !defined(__ARM_ARCH)
CpuFeatures::CpuFlags::CpuFlags(void)
    : L1_ECX(0), L1_EDX(0), L7_EBX(0), L7_ECX(0), L7_EDX(0) {
  uint32_t eax, ebx, ecx, edx;

  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    L1_ECX = ecx;
    L1_EDX = edx;
  }
  if (__get_cpuid_max(0, NULL) >= 7) {
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    L7_EBX = ebx;
    L7_ECX = ecx;
    L7_EDX = edx;
  }
}
#else
CpuFeatures::CpuFlags::CpuFlags(void)
    : L1_ECX(0), L1_EDX(0), L7_EBX(0), L7_ECX(0), L7_EDX(0) {}
#endif

//! 16-bit FP conversions
bool CpuFeatures::F16C(void) {
  return !!(flags_.L1_ECX & (1u << 29));
}

//! Multimedia Extensions
bool CpuFeatures::MMX(void) {
  return !!(flags_.L1_EDX & (1u << 23));
}

//! Streaming SIMD Extensions
bool CpuFeatures::SSE(void) {
  return !!(flags_.L1_EDX & (1u << 25));
}

//! Streaming SIMD Extensions 2
bool CpuFeatures::SSE2(void) {
  return !!(flags_.L1_EDX & (1u << 26));
}

//! Streaming SIMD Extensions 3
bool CpuFeatures::SSE3(void) {
  return !!(flags_.L1_ECX & (1u << 0));
}

//! Supplemental Streaming SIMD Extensions 3
bool CpuFeatures::SSSE3(void) {
  return !!(flags_.L1_ECX & (1u << 9));
}

//! Streaming SIMD Extensions 4.1
bool CpuFeatures::SSE4_1(void) {
  return !!(flags_.L1_ECX & (1u << 19));
}

//! Streaming SIMD Extensions 4.2
bool CpuFeatures::SSE4_2(void) {
  return !!(flags_.L1_ECX & (1u << 20));
}

//! Advanced Vector Extensions
bool CpuFeatures::AVX(void) {
  return !!(flags_.L1_ECX & (1u << 28));
}

//! Advanced Vector Extensions 2
bool CpuFeatures::AVX2(void) {
  return !!(flags_.L7_EBX & (1u << 5));
}

//! AVX-512 Foundation
bool CpuFeatures::AVX512F(void) {
  return !!(flags_.L7_EBX & (1u << 16));
}

//! AVX-512 DQ (Double/Quad granular) Instructions
bool CpuFeatures::AVX512DQ(void) {
  return !!(flags_.L7_EBX & (1u << 17));
}

//! AVX-512 Prefetch
bool CpuFeatures::AVX512PF(void) {
  return !!(flags_.L7_EBX & (1u << 26));
}

//! AVX-512 Exponential and Reciprocal
bool CpuFeatures::AVX512ER(void) {
  return !!(flags_.L7_EBX & (1u << 27));
}

//! AVX-512 Conflict Detection
bool CpuFeatures::AVX512CD(void) {
  return !!(flags_.L7_EBX & (1u << 28));
}

//! AVX-512 BW (Byte/Word granular) Instructions
bool CpuFeatures::AVX512BW(void) {
  return !!(flags_.L7_EBX & (1u << 30));
}

//! AVX-512 VL (128/256 Vector Length) Extensions
bool CpuFeatures::AVX512VL(void) {
  return !!(flags_.L7_EBX & (1u << 31));
}

//! AVX-512 Integer Fused Multiply-Add instructions
bool CpuFeatures::AVX512_IFMA(void) {
  return !!(flags_.L7_EBX & (1u << 21));
}

//! AVX512 Vector Bit Manipulation instructions
bool CpuFeatures::AVX512_VBMI(void) {
  return !!(flags_.L7_ECX & (1u << 1));
}

//! Additional AVX512 Vector Bit Manipulation Instructions
bool CpuFeatures::AVX512_VBMI2(void) {
  return !!(flags_.L7_ECX & (1u << 6));
}

//! Vector Neural Network Instructions
bool CpuFeatures::AVX512_VNNI(void) {
  return !!(flags_.L7_ECX & (1u << 11));
}

//! Support for VPOPCNT[B,W] and VPSHUF-BITQMB instructions
bool CpuFeatures::AVX512_BITALG(void) {
  return !!(flags_.L7_ECX & (1u << 12));
}

//! POPCNT for vectors of DW/QW
bool CpuFeatures::AVX512_VPOPCNTDQ(void) {
  return !!(flags_.L7_ECX & (1u << 14));
}

//! AVX-512 Neural Network Instructions
bool CpuFeatures::AVX512_4VNNIW(void) {
  return !!(flags_.L7_EDX & (1u << 2));
}

//! AVX-512 Multiply Accumulation Single precision
bool CpuFeatures::AVX512_4FMAPS(void) {
  return !!(flags_.L7_EDX & (1u << 3));
}

//! AVX-512 FP16 instructions
bool CpuFeatures::AVX512_FP16(void) {
  return !!(flags_.L7_EDX & (1u << 23));
}

//! CMPXCHG8 instruction
bool CpuFeatures::CX8(void) {
  return !!(flags_.L1_EDX & (1u << 8));
}

//! CMPXCHG16B instruction
bool CpuFeatures::CX16(void) {
  return !!(flags_.L1_ECX & (1u << 13));
}

//! PCLMULQDQ instruction
bool CpuFeatures::PCLMULQDQ(void) {
  return !!(flags_.L1_ECX & (1u << 1));
}

//! Carry-Less Multiplication Double Quadword
bool CpuFeatures::VPCLMULQDQ(void) {
  return !!(flags_.L7_ECX & (1u << 10));
}

//! CMOV instructions (plus FCMOVcc, FCOMI with FPU)
bool CpuFeatures::CMOV(void) {
  return !!(flags_.L1_EDX & (1u << 15));
}

//! MOVBE instruction
bool CpuFeatures::MOVBE(void) {
  return !!(flags_.L1_ECX & (1u << 22));
}

//! Enhanced REP MOVSB/STOSB instructions
bool CpuFeatures::ERMS(void) {
  return !!(flags_.L7_EBX & (1u << 9));
}

//! POPCNT instruction
bool CpuFeatures::POPCNT(void) {
  return !!(flags_.L1_ECX & (1u << 23));
}

//! XSAVE/XRSTOR/XSETBV/XGETBV instructions
bool CpuFeatures::XSAVE(void) {
  return !!(flags_.L1_ECX & (1u << 26));
}

//! Fused multiply-add
bool CpuFeatures::FMA(void) {
  return !!(flags_.L1_ECX & (1u << 12));
}

//! ADCX and ADOX instructions
bool CpuFeatures::ADX(void) {
  return !!(flags_.L7_EBX & (1u << 19));
}

//! Galois Field New Instructions
bool CpuFeatures::GFNI(void) {
  return !!(flags_.L7_ECX & (1u << 8));
}

//! AES instructions
bool CpuFeatures::AES(void) {
  return !!(flags_.L1_ECX & (1u << 25));
}

//! Vector AES
bool CpuFeatures::VAES(void) {
  return !!(flags_.L7_ECX & (1u << 9));
}

//! RDSEED instruction
bool CpuFeatures::RDSEED(void) {
  return !!(flags_.L7_EBX & (1u << 18));
}

//! RDRAND instruction
bool CpuFeatures::RDRAND(void) {
  return !!(flags_.L1_ECX & (1u << 30));
}

//! SHA1/SHA256 Instruction Extensions
bool CpuFeatures::SHA(void) {
  return !!(flags_.L7_EBX & (1u << 29));
}

//! 1st group bit manipulation extensions
bool CpuFeatures::BMI1(void) {
  return !!(flags_.L7_EBX & (1u << 3));
}

//! 2nd group bit manipulation extensions
bool CpuFeatures::BMI2(void) {
  return !!(flags_.L7_EBX & (1u << 8));
}

//! CLFLUSH instruction
bool CpuFeatures::CLFLUSH(void) {
  return !!(flags_.L1_EDX & (1u << 19));
}

//! CLFLUSHOPT instruction
bool CpuFeatures::CLFLUSHOPT(void) {
  return !!(flags_.L7_EBX & (1u << 23));
}

//! CLWB instruction
bool CpuFeatures::CLWB(void) {
  return !!(flags_.L7_EBX & (1u << 24));
}

//! RDPID instruction
bool CpuFeatures::RDPID(void) {
  return !!(flags_.L7_ECX & (1u << 22));
}

//! Onboard FPU
bool CpuFeatures::FPU(void) {
  return !!(flags_.L1_EDX & (1u << 0));
}

//! Hyper-Threading
bool CpuFeatures::HT(void) {
  return !!(flags_.L1_EDX & (1u << 28));
}

//! Hardware virtualization
bool CpuFeatures::VMX(void) {
  return !!(flags_.L1_ECX & (1u << 5));
}

// ！Running on a hypervisor
bool CpuFeatures::HYPERVISOR(void) {
  return !!(flags_.L1_ECX & (1u << 31));
}

const char *CpuFeatures::Intrinsics(void) {
  return ""
#if defined(__ARM_NEON)
         "Neon"
#if defined(__ARM_FEATURE_CRC32)
         "+CRC"
#endif
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
         "+FP16"
#endif
#elif defined(__AVX512F__)
         "AVX512F"
#if defined(__AVX512VL__)
         "+AVX512VL"
#endif
#if defined(__AVX512BW__)
         "+AVX512BW"
#endif
#if defined(__AVX512DQ__)
         "+AVX512DQ"
#endif
#if defined(__AVX512CD__)
         "+AVX512CD"
#endif
#if defined(__AVX512ER__)
         "+AVX512ER"
#endif
#if defined(__AVX512PF__)
         "+AVX512PF"
#endif
#if defined(__AVX512IFMA__)
         "+AVX512IFMA"
#endif
#if defined(__AVX512VBMI__)
         "+AVX512VBMI"
#endif
#if defined(__AVX512VBMI2__)
         "+AVX512VBMI2"
#endif
#if defined(__AVX512VNNI__)
         "+AVX512VNNI"
#endif
#if defined(__AVX512BITALG__)
         "+AVX512BITALG"
#endif
#if defined(__AVX512VPOPCNTDQ__)
         "+AVX512VPOPCNTDQ"
#endif
#if defined(__AVX512FP16__)
         "+AVX512FP16"
#endif
#elif defined(__AVX2__)
         "AVX2"
#elif defined(__AVX__)
         "AVX"
#elif defined(__SSE4_2__)
         "SSE4.2"
#elif defined(__SSE4_1__)
         "SSE4.1"
#elif defined(__SSSE3__)
         "SSSE3"
#elif defined(__SSE3__)
         "SSE3"
#elif defined(__SSE2__)
         "SSE2"
#elif defined(__SSE__)
         "SSE"
#elif defined(__MMX__)
         "MMX"
#endif
#if defined(__FMA__)
         "+FMA"
#endif
#if defined(__BMI2__)
         "+BMI2"
#elif defined(__BMI__)
         "+BMI"
#endif
#if defined(__F16C__)
         "+F16C"
#endif
      ;
}

CpuFeatures::StaticFlags CpuFeatures::static_flags_;
}  // namespace internal
}  // namespace ailego
}  // namespace zvec