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

#include "platform.h"

namespace zvec {
namespace ailego {
namespace internal {

/*! Cpu Features
 */
class CpuFeatures {
 public:
  //! 16-bit FP conversions
  static bool F16C(void);

  //! Multimedia Extensions
  static bool MMX(void);

  //! Streaming SIMD Extensions
  static bool SSE(void);

  //! Streaming SIMD Extensions 2
  static bool SSE2(void);

  //! Streaming SIMD Extensions 3
  static bool SSE3(void);

  //! Supplemental Streaming SIMD Extensions 3
  static bool SSSE3(void);

  //! Streaming SIMD Extensions 4.1
  static bool SSE4_1(void);

  //! Streaming SIMD Extensions 4.2
  static bool SSE4_2(void);

  //! Advanced Vector Extensions
  static bool AVX(void);

  //! Advanced Vector Extensions 2
  static bool AVX2(void);

  //! AVX-512 Foundation
  static bool AVX512F(void);

  //! AVX-512 DQ (Double/Quad granular) Instructions
  static bool AVX512DQ(void);

  //! AVX-512 Prefetch
  static bool AVX512PF(void);

  //! AVX-512 Exponential and Reciprocal
  static bool AVX512ER(void);

  //! AVX-512 Conflict Detection
  static bool AVX512CD(void);

  //! AVX-512 BW (Byte/Word granular) Instructions
  static bool AVX512BW(void);

  //! AVX-512 VL (128/256 Vector Length) Extensions
  static bool AVX512VL(void);

  //! AVX-512 Integer Fused Multiply-Add instructions
  static bool AVX512_IFMA(void);

  //! AVX512 Vector Bit Manipulation instructions
  static bool AVX512_VBMI(void);

  //! Additional AVX512 Vector Bit Manipulation Instructions
  static bool AVX512_VBMI2(void);

  //! Vector Neural Network Instructions
  static bool AVX512_VNNI(void);

  //! Support for VPOPCNT[B,W] and VPSHUF-BITQMB instructions
  static bool AVX512_BITALG(void);

  //! POPCNT for vectors of DW/QW
  static bool AVX512_VPOPCNTDQ(void);

  //! AVX-512 Neural Network Instructions
  static bool AVX512_4VNNIW(void);

  //! AVX-512 Multiply Accumulation Single precision
  static bool AVX512_4FMAPS(void);

  //! AVX-512 FP16 instructions
  static bool AVX512_FP16(void);

  //! CMPXCHG8 instruction
  static bool CX8(void);

  //! CMPXCHG16B instruction
  static bool CX16(void);

  //! PCLMULQDQ instruction
  static bool PCLMULQDQ(void);

  //! Carry-Less Multiplication Double Quadword
  static bool VPCLMULQDQ(void);

  //! CMOV instructions (plus FCMOVcc, FCOMI with FPU)
  static bool CMOV(void);

  //! MOVBE instruction
  static bool MOVBE(void);

  //! Enhanced REP MOVSB/STOSB instructions
  static bool ERMS(void);

  //! POPCNT instruction
  static bool POPCNT(void);

  //! XSAVE/XRSTOR/XSETBV/XGETBV instructions
  static bool XSAVE(void);

  //! Fused multiply-add
  static bool FMA(void);

  //! ADCX and ADOX instructions
  static bool ADX(void);

  //! Galois Field New Instructions
  static bool GFNI(void);

  //! AES instructions
  static bool AES(void);

  //! Vector AES
  static bool VAES(void);

  //! RDSEED instruction
  static bool RDSEED(void);

  //! RDRAND instruction
  static bool RDRAND(void);

  //! SHA1/SHA256 Instruction Extensions
  static bool SHA(void);

  //! 1st group bit manipulation extensions
  static bool BMI1(void);

  //! 2nd group bit manipulation extensions
  static bool BMI2(void);

  //! CLFLUSH instruction
  static bool CLFLUSH(void);

  //! CLFLUSHOPT instruction
  static bool CLFLUSHOPT(void);

  //! CLWB instruction
  static bool CLWB(void);

  //! RDPID instruction
  static bool RDPID(void);

  //! Onboard FPU
  static bool FPU(void);

  //! Hyper-Threading
  static bool HT(void);

  //! Hardware virtualization
  static bool VMX(void);

  // ！Running on a hypervisor
  static bool HYPERVISOR(void);

  //! Intrinsics of compiling
  static const char *Intrinsics(void);

 private:
  struct CpuFlags {
    //! Constructor
    CpuFlags(void);

    //! Members
    uint32_t L1_ECX;
    uint32_t L1_EDX;
    uint32_t L7_EBX;
    uint32_t L7_ECX;
    uint32_t L7_EDX;
  };

  //! Static Members
  static CpuFlags flags_;

 public:
  struct StaticFlags {
    //! 16-bit FP conversions
    bool F16C = CpuFeatures::F16C();

    //! Multimedia Extensions
    bool MMX = CpuFeatures::MMX();

    //! Streaming SIMD Extensions
    bool SSE = CpuFeatures::SSE();

    //! Streaming SIMD Extensions 2
    bool SSE2 = CpuFeatures::SSE2();

    //! Streaming SIMD Extensions 3
    bool SSE3 = CpuFeatures::SSE3();

    //! Supplemental Streaming SIMD Extensions 3
    bool SSSE3 = CpuFeatures::SSSE3();

    //! Streaming SIMD Extensions 4.1
    bool SSE4_1 = CpuFeatures::SSE4_1();

    //! Streaming SIMD Extensions 4.2
    bool SSE4_2 = CpuFeatures::SSE4_2();

    //! Advanced Vector Extensions
    bool AVX = CpuFeatures::AVX();

    //! Advanced Vector Extensions 2
    bool AVX2 = CpuFeatures::AVX2();

    //! AVX-512 Foundation
    bool AVX512F = CpuFeatures::AVX512F();

    //! AVX-512 DQ (Double/Quad granular) Instructions
    bool AVX512DQ = CpuFeatures::AVX512DQ();

    //! AVX-512 Prefetch
    bool AVX512PF = CpuFeatures::AVX512PF();

    //! AVX-512 Exponential and Reciprocal
    bool AVX512ER = CpuFeatures::AVX512ER();

    //! AVX-512 Conflict Detection
    bool AVX512CD = CpuFeatures::AVX512CD();

    //! AVX-512 BW (Byte/Word granular) Instructions
    bool AVX512BW = CpuFeatures::AVX512BW();

    //! AVX-512 VL (128/256 Vector Length) Extensions
    bool AVX512VL = CpuFeatures::AVX512VL();

    //! AVX-512 Integer Fused Multiply-Add instructions
    bool AVX512_IFMA = CpuFeatures::AVX512_IFMA();

    //! AVX512 Vector Bit Manipulation instructions
    bool AVX512_VBMI = CpuFeatures::AVX512_VBMI();

    //! Additional AVX512 Vector Bit Manipulation Instructions
    bool AVX512_VBMI2 = CpuFeatures::AVX512_VBMI2();

    //! Vector Neural Network Instructions
    bool AVX512_VNNI = CpuFeatures::AVX512_VNNI();

    //! Support for VPOPCNT[B,W] and VPSHUF-BITQMB instructions
    bool AVX512_BITALG = CpuFeatures::AVX512_BITALG();

    //! POPCNT for vectors of DW/QW
    bool AVX512_VPOPCNTDQ = CpuFeatures::AVX512_VPOPCNTDQ();

    //! AVX-512 Neural Network Instructions
    bool AVX512_4VNNIW = CpuFeatures::AVX512_4VNNIW();

    //! AVX-512 Multiply Accumulation Single precision
    bool AVX512_4FMAPS = CpuFeatures::AVX512_4FMAPS();

    //! AVX-512 FP16 instructions
    bool AVX512_FP16 = CpuFeatures::AVX512_FP16();

    //! CMPXCHG8 instruction
    bool CX8 = CpuFeatures::CX8();

    //! CMPXCHG16B instruction
    bool CX16 = CpuFeatures::CX16();

    //! PCLMULQDQ instruction
    bool PCLMULQDQ = CpuFeatures::PCLMULQDQ();

    //! Carry-Less Multiplication Double Quadword
    bool VPCLMULQDQ = CpuFeatures::VPCLMULQDQ();

    //! CMOV instructions (plus FCMOVcc, FCOMI with FPU)
    bool CMOV = CpuFeatures::CMOV();

    //! MOVBE instruction
    bool MOVBE = CpuFeatures::MOVBE();

    //! Enhanced REP MOVSB/STOSB instructions
    bool ERMS = CpuFeatures::ERMS();

    //! POPCNT instruction
    bool POPCNT = CpuFeatures::POPCNT();

    //! XSAVE/XRSTOR/XSETBV/XGETBV instructions
    bool XSAVE = CpuFeatures::XSAVE();

    //! Fused multiply-add
    bool FMA = CpuFeatures::FMA();

    //! ADCX and ADOX instructions
    bool ADX = CpuFeatures::ADX();

    //! Galois Field New Instructions
    bool GFNI = CpuFeatures::GFNI();

    //! AES instructions
    bool AES = CpuFeatures::AES();

    //! Vector AES
    bool VAES = CpuFeatures::VAES();

    //! RDSEED instruction
    bool RDSEED = CpuFeatures::RDSEED();

    //! RDRAND instruction
    bool RDRAND = CpuFeatures::RDRAND();

    //! SHA1/SHA256 Instruction Extensions
    bool SHA = CpuFeatures::SHA();

    //! 1st group bit manipulation extensions
    bool BMI1 = CpuFeatures::BMI1();

    //! 2nd group bit manipulation extensions
    bool BMI2 = CpuFeatures::BMI2();

    //! CLFLUSH instruction
    bool CLFLUSH = CpuFeatures::CLFLUSH();

    //! CLFLUSHOPT instruction
    bool CLFLUSHOPT = CpuFeatures::CLFLUSHOPT();

    //! CLWB instruction
    bool CLWB = CpuFeatures::CLWB();

    //! RDPID instruction
    bool RDPID = CpuFeatures::RDPID();

    //! Onboard FPU
    bool FPU = CpuFeatures::FPU();

    //! Hyper-Threading
    bool HT = CpuFeatures::HT();

    //! Hardware virtualization
    bool VMX = CpuFeatures::VMX();

    // ！Running on a hypervisor
    bool HYPERVISOR = CpuFeatures::HYPERVISOR();
  };
  static StaticFlags static_flags_;
};

}  // namespace internal
}  // namespace ailego
}  // namespace zvec
