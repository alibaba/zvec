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

#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>

using namespace zvec::ailego::internal;

TEST(CpuFeatures, General) {
  std::cout << "* Intrinsics:       " << CpuFeatures::Intrinsics() << std::endl;
  std::cout << "* F16C:             " << CpuFeatures::F16C() << std::endl;
  std::cout << "* SSE:              " << CpuFeatures::SSE() << std::endl;
  std::cout << "* SSE2:             " << CpuFeatures::SSE2() << std::endl;
  std::cout << "* SSE3:             " << CpuFeatures::SSE3() << std::endl;
  std::cout << "* SSSE3:            " << CpuFeatures::SSSE3() << std::endl;
  std::cout << "* SSE4_1:           " << CpuFeatures::SSE4_1() << std::endl;
  std::cout << "* SSE4_2:           " << CpuFeatures::SSE4_2() << std::endl;
  std::cout << "* AVX:              " << CpuFeatures::AVX() << std::endl;
  std::cout << "* AVX2:             " << CpuFeatures::AVX2() << std::endl;
  std::cout << "* AVX512F:          " << CpuFeatures::AVX512F() << std::endl;
  std::cout << "* AVX512DQ:         " << CpuFeatures::AVX512DQ() << std::endl;
  std::cout << "* AVX512PF:         " << CpuFeatures::AVX512PF() << std::endl;
  std::cout << "* AVX512ER:         " << CpuFeatures::AVX512ER() << std::endl;
  std::cout << "* AVX512CD:         " << CpuFeatures::AVX512CD() << std::endl;
  std::cout << "* AVX512BW:         " << CpuFeatures::AVX512BW() << std::endl;
  std::cout << "* AVX512VL:         " << CpuFeatures::AVX512VL() << std::endl;
  std::cout << "* AVX512_IFMA:      " << CpuFeatures::AVX512_IFMA()
            << std::endl;
  std::cout << "* AVX512_VBMI:      " << CpuFeatures::AVX512_VBMI()
            << std::endl;
  std::cout << "* AVX512_VBMI2:     " << CpuFeatures::AVX512_VBMI2()
            << std::endl;
  std::cout << "* AVX512_VNNI:      " << CpuFeatures::AVX512_VNNI()
            << std::endl;
  std::cout << "* AVX512_BITALG:    " << CpuFeatures::AVX512_BITALG()
            << std::endl;
  std::cout << "* AVX512_VPOPCNTDQ: " << CpuFeatures::AVX512_VPOPCNTDQ()
            << std::endl;
  std::cout << "* AVX512_4VNNIW:    " << CpuFeatures::AVX512_4VNNIW()
            << std::endl;
  std::cout << "* AVX512_4FMAPS:    " << CpuFeatures::AVX512_4FMAPS()
            << std::endl;
  std::cout << "* AVX512_FP16:      " << CpuFeatures::AVX512_FP16()
            << std::endl;
  std::cout << "* CX8:              " << CpuFeatures::CX8() << std::endl;
  std::cout << "* CX16:             " << CpuFeatures::CX16() << std::endl;
  std::cout << "* PCLMULQDQ:        " << CpuFeatures::PCLMULQDQ() << std::endl;
  std::cout << "* VPCLMULQDQ:       " << CpuFeatures::VPCLMULQDQ() << std::endl;
  std::cout << "* CMOV:             " << CpuFeatures::CMOV() << std::endl;
  std::cout << "* MOVBE:            " << CpuFeatures::MOVBE() << std::endl;
  std::cout << "* ERMS:             " << CpuFeatures::ERMS() << std::endl;
  std::cout << "* POPCNT:           " << CpuFeatures::POPCNT() << std::endl;
  std::cout << "* XSAVE:            " << CpuFeatures::XSAVE() << std::endl;
  std::cout << "* FMA:              " << CpuFeatures::FMA() << std::endl;
  std::cout << "* ADX:              " << CpuFeatures::ADX() << std::endl;
  std::cout << "* GFNI:             " << CpuFeatures::GFNI() << std::endl;
  std::cout << "* AES:              " << CpuFeatures::AES() << std::endl;
  std::cout << "* VAES:             " << CpuFeatures::VAES() << std::endl;
  std::cout << "* RDSEED:           " << CpuFeatures::RDSEED() << std::endl;
  std::cout << "* RDRAND:           " << CpuFeatures::RDRAND() << std::endl;
  std::cout << "* SHA:              " << CpuFeatures::SHA() << std::endl;
  std::cout << "* BMI1:             " << CpuFeatures::BMI1() << std::endl;
  std::cout << "* BMI2:             " << CpuFeatures::BMI2() << std::endl;
  std::cout << "* CLFLUSH:          " << CpuFeatures::CLFLUSH() << std::endl;
  std::cout << "* CLFLUSHOPT:       " << CpuFeatures::CLFLUSHOPT() << std::endl;
  std::cout << "* CLWB:             " << CpuFeatures::CLWB() << std::endl;
  std::cout << "* RDPID:            " << CpuFeatures::RDPID() << std::endl;
  std::cout << "* FPU:              " << CpuFeatures::FPU() << std::endl;
  std::cout << "* HT:               " << CpuFeatures::HT() << std::endl;
  std::cout << "* VMX:              " << CpuFeatures::VMX() << std::endl;
  std::cout << "* HYPERVISOR:       " << CpuFeatures::HYPERVISOR() << std::endl;

// #if defined(__AVX512VBMI2__)
//     EXPECT_TRUE(CpuFeatures::AVX512VBMI2());
// #endif
// #if defined(__AVX512VBMI__)
//     EXPECT_TRUE(CpuFeatures::AVX512VBMI());
// #endif
// #if defined(__AVX512VL__)
//     EXPECT_TRUE(CpuFeatures::AVX512VL());
// #endif
// #if defined(__AVX512BW__)
//     EXPECT_TRUE(CpuFeatures::AVX512BW());
// #endif
// #if defined(__AVX512CD__)
//     EXPECT_TRUE(CpuFeatures::AVX512CD());
// #endif
// #if defined(__AVX512ER__)
//     EXPECT_TRUE(CpuFeatures::AVX512ER());
// #endif
// #if defined(__AVX512PF__)
//     EXPECT_TRUE(CpuFeatures::AVX512PF());
// #endif
// #if defined(__AVX512IFMA__)
//     EXPECT_TRUE(CpuFeatures::AVX512IFMA());
// #endif
// #if defined(__AVX512DQ__)
//     EXPECT_TRUE(CpuFeatures::AVX512DQ());
// #endif
// #if defined(__AVX512F__)
//     EXPECT_TRUE(CpuFeatures::AVX512F());
// #endif
#if defined(__AVX2__)
  EXPECT_TRUE(CpuFeatures::AVX2());
  EXPECT_TRUE(CpuFeatures::AVX());
  EXPECT_TRUE(CpuFeatures::SSE4_2());
  EXPECT_TRUE(CpuFeatures::SSE4_1());
  EXPECT_TRUE(CpuFeatures::SSSE3());
  EXPECT_TRUE(CpuFeatures::SSE3());
  EXPECT_TRUE(CpuFeatures::SSE2());
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
#if defined(__AVX__)
  EXPECT_TRUE(CpuFeatures::AVX());
  EXPECT_TRUE(CpuFeatures::SSE4_2());
  EXPECT_TRUE(CpuFeatures::SSE4_1());
  EXPECT_TRUE(CpuFeatures::SSSE3());
  EXPECT_TRUE(CpuFeatures::SSE3());
  EXPECT_TRUE(CpuFeatures::SSE2());
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
#if defined(__SSE4_2__)
  EXPECT_TRUE(CpuFeatures::SSE4_2());
  EXPECT_TRUE(CpuFeatures::SSE4_1());
  EXPECT_TRUE(CpuFeatures::SSSE3());
  EXPECT_TRUE(CpuFeatures::SSE3());
  EXPECT_TRUE(CpuFeatures::SSE2());
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
  EXPECT_TRUE(CpuFeatures::POPCNT());
#endif
#if defined(__SSE4_1__)
  EXPECT_TRUE(CpuFeatures::SSE4_1());
  EXPECT_TRUE(CpuFeatures::SSSE3());
  EXPECT_TRUE(CpuFeatures::SSE3());
  EXPECT_TRUE(CpuFeatures::SSE2());
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
#if defined(__SSSE3__)
  EXPECT_TRUE(CpuFeatures::SSSE3());
  EXPECT_TRUE(CpuFeatures::SSE3());
  EXPECT_TRUE(CpuFeatures::SSE2());
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
#if defined(__SSE3__)
  EXPECT_TRUE(CpuFeatures::SSE3());
  EXPECT_TRUE(CpuFeatures::SSE2());
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
#if defined(__SSE2__)
  EXPECT_TRUE(CpuFeatures::SSE2());
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
#if defined(__SSE__)
  EXPECT_TRUE(CpuFeatures::SSE());
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
#if defined(__MMX__)
  EXPECT_TRUE(CpuFeatures::MMX());
#endif
}


TEST(CpuFeatures, Static) {
  std::cout << "* F16C:             " << CpuFeatures::static_flags_.F16C
            << std::endl;
  std::cout << "* SSE:              " << CpuFeatures::static_flags_.SSE
            << std::endl;
  std::cout << "* SSE2:             " << CpuFeatures::static_flags_.SSE2
            << std::endl;
  std::cout << "* SSE3:             " << CpuFeatures::static_flags_.SSE3
            << std::endl;
  std::cout << "* SSSE3:            " << CpuFeatures::static_flags_.SSSE3
            << std::endl;
  std::cout << "* SSE4_1:           " << CpuFeatures::static_flags_.SSE4_1
            << std::endl;
  std::cout << "* SSE4_2:           " << CpuFeatures::static_flags_.SSE4_2
            << std::endl;
  std::cout << "* AVX:              " << CpuFeatures::static_flags_.AVX
            << std::endl;
  std::cout << "* AVX2:             " << CpuFeatures::static_flags_.AVX2
            << std::endl;
  std::cout << "* AVX512F:          " << CpuFeatures::static_flags_.AVX512F
            << std::endl;
  std::cout << "* AVX512DQ:         " << CpuFeatures::static_flags_.AVX512DQ
            << std::endl;
  std::cout << "* AVX512PF:         " << CpuFeatures::static_flags_.AVX512PF
            << std::endl;
  std::cout << "* AVX512ER:         " << CpuFeatures::static_flags_.AVX512ER
            << std::endl;
  std::cout << "* AVX512CD:         " << CpuFeatures::static_flags_.AVX512CD
            << std::endl;
  std::cout << "* AVX512BW:         " << CpuFeatures::static_flags_.AVX512BW
            << std::endl;
  std::cout << "* AVX512VL:         " << CpuFeatures::static_flags_.AVX512VL
            << std::endl;
  std::cout << "* AVX512_IFMA:      " << CpuFeatures::static_flags_.AVX512_IFMA
            << std::endl;
  std::cout << "* AVX512_VBMI:      " << CpuFeatures::static_flags_.AVX512_VBMI
            << std::endl;
  std::cout << "* AVX512_VBMI2:     " << CpuFeatures::static_flags_.AVX512_VBMI2
            << std::endl;
  std::cout << "* AVX512_VNNI:      " << CpuFeatures::static_flags_.AVX512_VNNI
            << std::endl;
  std::cout << "* AVX512_BITALG:    "
            << CpuFeatures::static_flags_.AVX512_BITALG << std::endl;
  std::cout << "* AVX512_VPOPCNTDQ: "
            << CpuFeatures::static_flags_.AVX512_VPOPCNTDQ << std::endl;
  std::cout << "* AVX512_4VNNIW:    "
            << CpuFeatures::static_flags_.AVX512_4VNNIW << std::endl;
  std::cout << "* AVX512_4FMAPS:    "
            << CpuFeatures::static_flags_.AVX512_4FMAPS << std::endl;
  std::cout << "* AVX512_FP16:      " << CpuFeatures::static_flags_.AVX512_FP16
            << std::endl;
  std::cout << "* CX8:              " << CpuFeatures::static_flags_.CX8
            << std::endl;
  std::cout << "* CX16:             " << CpuFeatures::static_flags_.CX16
            << std::endl;
  std::cout << "* PCLMULQDQ:        " << CpuFeatures::static_flags_.PCLMULQDQ
            << std::endl;
  std::cout << "* VPCLMULQDQ:       " << CpuFeatures::static_flags_.VPCLMULQDQ
            << std::endl;
  std::cout << "* CMOV:             " << CpuFeatures::static_flags_.CMOV
            << std::endl;
  std::cout << "* MOVBE:            " << CpuFeatures::static_flags_.MOVBE
            << std::endl;
  std::cout << "* ERMS:             " << CpuFeatures::static_flags_.ERMS
            << std::endl;
  std::cout << "* POPCNT:           " << CpuFeatures::static_flags_.POPCNT
            << std::endl;
  std::cout << "* XSAVE:            " << CpuFeatures::static_flags_.XSAVE
            << std::endl;
  std::cout << "* FMA:              " << CpuFeatures::static_flags_.FMA
            << std::endl;
  std::cout << "* ADX:              " << CpuFeatures::static_flags_.ADX
            << std::endl;
  std::cout << "* GFNI:             " << CpuFeatures::static_flags_.GFNI
            << std::endl;
  std::cout << "* AES:              " << CpuFeatures::static_flags_.AES
            << std::endl;
  std::cout << "* VAES:             " << CpuFeatures::static_flags_.VAES
            << std::endl;
  std::cout << "* RDSEED:           " << CpuFeatures::static_flags_.RDSEED
            << std::endl;
  std::cout << "* RDRAND:           " << CpuFeatures::static_flags_.RDRAND
            << std::endl;
  std::cout << "* SHA:              " << CpuFeatures::static_flags_.SHA
            << std::endl;
  std::cout << "* BMI1:             " << CpuFeatures::static_flags_.BMI1
            << std::endl;
  std::cout << "* BMI2:             " << CpuFeatures::static_flags_.BMI2
            << std::endl;
  std::cout << "* CLFLUSH:          " << CpuFeatures::static_flags_.CLFLUSH
            << std::endl;
  std::cout << "* CLFLUSHOPT:       " << CpuFeatures::static_flags_.CLFLUSHOPT
            << std::endl;
  std::cout << "* CLWB:             " << CpuFeatures::static_flags_.CLWB
            << std::endl;
  std::cout << "* RDPID:            " << CpuFeatures::static_flags_.RDPID
            << std::endl;
  std::cout << "* FPU:              " << CpuFeatures::static_flags_.FPU
            << std::endl;
  std::cout << "* HT:               " << CpuFeatures::static_flags_.HT
            << std::endl;
  std::cout << "* VMX:              " << CpuFeatures::static_flags_.VMX
            << std::endl;
  std::cout << "* HYPERVISOR:       " << CpuFeatures::static_flags_.HYPERVISOR
            << std::endl;
}