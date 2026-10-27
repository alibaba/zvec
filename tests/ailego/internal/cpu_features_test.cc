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

namespace zvec {
namespace ailego {
namespace internal {

// Decode synthetic CPU/OS combinations without executing their instructions.
class CpuFeaturesTestPeer {
 public:
  CpuFeaturesTestPeer(uint32_t leaf1_ecx, uint32_t leaf7_ebx,
                      uint32_t extended_ecx, uint64_t xcr0,
                      bool darwin_avx512 = false)
      : saved_(CpuFeatures::flags_) {
    CpuFeatures::flags_.L1_ECX = leaf1_ecx;
    CpuFeatures::flags_.L7_EBX = leaf7_ebx;
    CpuFeatures::flags_.L80000001_ECX = extended_ecx;
    CpuFeatures::flags_.XCR0 = xcr0;
    CpuFeatures::flags_.DARWIN_AVX512 = darwin_avx512;
  }

  ~CpuFeaturesTestPeer() {
    CpuFeatures::flags_ = saved_;
  }

  CpuFeaturesTestPeer(const CpuFeaturesTestPeer &) = delete;
  CpuFeaturesTestPeer &operator=(const CpuFeaturesTestPeer &) = delete;

 private:
  CpuFeatures::CpuFlags saved_;
};

}  // namespace internal
}  // namespace ailego
}  // namespace zvec

using namespace zvec::ailego::internal;

TEST(CpuFeatures, VectorStateRequiresCpuAndOsXsave) {
  constexpr uint32_t kXsave = 1u << 26;
  constexpr uint32_t kOsxsave = 1u << 27;
  constexpr uint32_t kAvx = 1u << 28;
  constexpr uint32_t kAvx2AndAvx512 = (1u << 5) | (1u << 16);

  for (uint32_t xsave_flags : {0u, kXsave, kOsxsave, kXsave | kOsxsave}) {
    SCOPED_TRACE(xsave_flags);
    const CpuFeaturesTestPeer cpu(xsave_flags | kAvx, kAvx2AndAvx512, 0, 0xe6u);
    const bool enabled = xsave_flags == (kXsave | kOsxsave);
    EXPECT_EQ(CpuFeatures::OSXSAVE(), (xsave_flags & kOsxsave) != 0);
    EXPECT_EQ(CpuFeatures::AVX_STATE(), enabled);
    EXPECT_EQ(CpuFeatures::AVX512_STATE(), enabled);
    // Hardware features keep their previous CPUID semantics when the OS cannot
    // save vector registers. Callers combine these bits with the state flags.
    EXPECT_TRUE(CpuFeatures::AVX());
    EXPECT_TRUE(CpuFeatures::AVX2());
    EXPECT_TRUE(CpuFeatures::AVX512F());
  }
}

TEST(CpuFeatures, DarwinLazyAvx512StillRequiresUsableAvxState) {
  constexpr uint32_t kXsaveAndOsxsave = (1u << 26) | (1u << 27);
  for (uint64_t xcr0 : {0u, 2u, 4u, 6u}) {
    const CpuFeaturesTestPeer cpu(kXsaveAndOsxsave, 0, 0, xcr0, true);
    EXPECT_EQ(CpuFeatures::AVX512_STATE(), xcr0 == 6u);
  }
  {
    const CpuFeaturesTestPeer cpu(kXsaveAndOsxsave, 0, 0, 6u, false);
    EXPECT_FALSE(CpuFeatures::AVX512_STATE());
  }
  {
    const CpuFeaturesTestPeer cpu(0, 0, 0, 6u, true);
    EXPECT_FALSE(CpuFeatures::AVX512_STATE());
  }
}

TEST(CpuFeatures, VectorStateRequiresEveryRegisterComponent) {
  constexpr uint32_t kXsaveAndOsxsave = (1u << 26) | (1u << 27);
  struct TestCase {
    uint64_t xcr0;
    bool avx;
    bool avx512;
  };
  const TestCase cases[] = {
      {0, false, false},     // No enabled vector state.
      {0xe4, false, false},  // Missing XMM.
      {0xe2, false, false},  // Missing YMM.
      {0xc6, true, false},   // Missing opmask.
      {0xa6, true, false},   // Missing ZMM high halves.
      {0x66, true, false},   // Missing upper ZMM registers.
      {0x6, true, false},    // AVX-only OS support.
      {0xe6, true, true},    // All required vector state.
      {0x1e7, true, true},   // Other enabled state does not prevent AVX-512.
      {1ull << 32, false, false},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.xcr0);
    const CpuFeaturesTestPeer cpu(kXsaveAndOsxsave, 0, 0, test.xcr0);
    EXPECT_EQ(CpuFeatures::AVX_STATE(), test.avx);
    EXPECT_EQ(CpuFeatures::AVX512_STATE(), test.avx512);
  }
}

TEST(CpuFeatures, LzcntUsesExtendedLeaf) {
  {
    // BMI1 has a different CPUID location and does not imply LZCNT.
    const CpuFeaturesTestPeer cpu(1u << 5, (1u << 3) | (1u << 5), 0, 0);
    EXPECT_TRUE(CpuFeatures::BMI1());
    EXPECT_FALSE(CpuFeatures::LZCNT());
  }
  {
    const CpuFeaturesTestPeer cpu(0, 0, 1u << 5, 0);
    EXPECT_TRUE(CpuFeatures::LZCNT());
    EXPECT_FALSE(CpuFeatures::BMI1());
    EXPECT_FALSE(CpuFeatures::AVX2());
  }
}

TEST(CpuFeatures, CachedVectorState) {
  const auto &flags = CpuFeatures::static_flags_;
  EXPECT_EQ(flags.OSXSAVE, CpuFeatures::OSXSAVE());
  EXPECT_EQ(flags.AVX_STATE, CpuFeatures::AVX_STATE());
  EXPECT_EQ(flags.AVX512_STATE, CpuFeatures::AVX512_STATE());
  EXPECT_EQ(flags.LZCNT, CpuFeatures::LZCNT());
  if (flags.AVX_STATE) {
    EXPECT_TRUE(flags.XSAVE);
    EXPECT_TRUE(flags.OSXSAVE);
  }
  if (flags.AVX512_STATE) {
    EXPECT_TRUE(flags.AVX_STATE);
  }
#if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_IX86) && \
    !defined(_M_X64)
  EXPECT_FALSE(flags.OSXSAVE);
  EXPECT_FALSE(flags.AVX_STATE);
  EXPECT_FALSE(flags.AVX512_STATE);
  EXPECT_FALSE(flags.LZCNT);
#endif
}

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
