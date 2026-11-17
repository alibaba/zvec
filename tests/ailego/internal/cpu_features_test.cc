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

namespace {
constexpr uint32_t kSse42 = 1u << 20;
constexpr uint32_t kXsave = 1u << 26;
constexpr uint32_t kOsxsave = 1u << 27;
constexpr uint32_t kAvx = 1u << 28;
constexpr uint32_t kAvx2 = 1u << 5;
constexpr uint32_t kAvx512f = 1u << 16;
constexpr uint32_t kLeaf1 = kSse42 | kXsave | kOsxsave | kAvx;
constexpr uint32_t kLeaf7 = kAvx2 | kAvx512f;

void ExpectNoAvx(const CpuFeatures::X86Features &features) {
  EXPECT_FALSE(features.avx);
  EXPECT_FALSE(features.avx2);
  EXPECT_FALSE(features.avx512f);
}
}  // namespace

TEST(CpuFeatures, UsableX86BaselineAndHardwareBits) {
  const auto baseline = CpuFeatures::DecodeUsableX86(0, 0, 0);
  EXPECT_FALSE(baseline.sse42);
  ExpectNoAvx(baseline);
  const auto sse = CpuFeatures::DecodeUsableX86(kSse42, 0, 0);
  EXPECT_TRUE(sse.sse42);
  ExpectNoAvx(sse);

  const auto all = CpuFeatures::DecodeUsableX86(kLeaf1, kLeaf7, 0xe6);
  EXPECT_TRUE(all.sse42);
  EXPECT_TRUE(all.avx);
  EXPECT_TRUE(all.avx2);
  EXPECT_TRUE(all.avx512f);
  const auto avx = CpuFeatures::DecodeUsableX86(kLeaf1, 0, 0xe6);
  EXPECT_TRUE(avx.avx);
  EXPECT_FALSE(avx.avx2);
  EXPECT_FALSE(avx.avx512f);
  // Report individual hardware features, not a compiled kernel's ISA bundle.
  const auto independent =
      CpuFeatures::DecodeUsableX86(kLeaf1 & ~kSse42, kAvx512f, 0xe6);
  EXPECT_FALSE(independent.sse42);
  EXPECT_TRUE(independent.avx);
  EXPECT_FALSE(independent.avx2);
  EXPECT_TRUE(independent.avx512f);
}

TEST(CpuFeatures, UsableX86RequiresAvxAndBothXsaveFlags) {
  for (const auto missing : {kXsave, kOsxsave, kAvx}) {
    SCOPED_TRACE(missing);
    for (bool lazy : {false, true}) {
      const auto features =
          CpuFeatures::DecodeUsableX86(kLeaf1 & ~missing, kLeaf7, 0xe6, lazy);
      EXPECT_TRUE(features.sse42);
      ExpectNoAvx(features);
    }
  }
}

TEST(CpuFeatures, UsableX86RequiresXmmAndYmmState) {
  for (uint64_t state : {uint64_t{0}, uint64_t{0xe6} & ~uint64_t{2},
                         uint64_t{0xe6} & ~uint64_t{4}, uint64_t{0xe6} << 32}) {
    SCOPED_TRACE(state);
    for (bool lazy : {false, true}) {
      const auto features =
          CpuFeatures::DecodeUsableX86(kLeaf1, kLeaf7, state, lazy);
      EXPECT_TRUE(features.sse42);
      ExpectNoAvx(features);
    }
  }
}

TEST(CpuFeatures, UsableX86RequiresCompleteZmmState) {
  for (uint64_t state :
       {uint64_t{6}, uint64_t{0xe6} & ~uint64_t{0x20},
        uint64_t{0xe6} & ~uint64_t{0x40}, uint64_t{0xe6} & ~uint64_t{0x80}}) {
    SCOPED_TRACE(state);
    const auto features = CpuFeatures::DecodeUsableX86(kLeaf1, kLeaf7, state);
    EXPECT_TRUE(features.avx);
    EXPECT_TRUE(features.avx2);
    EXPECT_FALSE(features.avx512f);
  }
}

TEST(CpuFeatures, UsableX86DarwinLazyStateNeedsHardwareSupport) {
  const auto lazy = CpuFeatures::DecodeUsableX86(kLeaf1, kLeaf7, 6, true);
  EXPECT_TRUE(lazy.avx);
  EXPECT_TRUE(lazy.avx2);
  EXPECT_TRUE(lazy.avx512f);
  const auto absent = CpuFeatures::DecodeUsableX86(kLeaf1, kAvx2, 6, true);
  EXPECT_TRUE(absent.avx2);
  EXPECT_FALSE(absent.avx512f);
}

TEST(CpuFeatures, UsableX86SnapshotDoesNotChangeRawFlags) {
  const auto &raw = CpuFeatures::static_flags_;
  const auto usable = CpuFeatures::UsableX86();
  EXPECT_EQ(raw.SSE4_2, CpuFeatures::SSE4_2());
  EXPECT_EQ(raw.AVX, CpuFeatures::AVX());
  EXPECT_EQ(raw.AVX2, CpuFeatures::AVX2());
  EXPECT_EQ(raw.AVX512F, CpuFeatures::AVX512F());
  EXPECT_EQ(usable.sse42, raw.SSE4_2);
  EXPECT_FALSE(usable.avx && !raw.AVX);
  EXPECT_FALSE(usable.avx2 && (!usable.avx || !raw.AVX2));
  EXPECT_FALSE(usable.avx512f && (!usable.avx || !raw.AVX512F));
  const auto again = CpuFeatures::UsableX86();
  EXPECT_EQ(usable.sse42, again.sse42);
  EXPECT_EQ(usable.avx, again.avx);
  EXPECT_EQ(usable.avx2, again.avx2);
  EXPECT_EQ(usable.avx512f, again.avx512f);
#if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_X64) && \
    !defined(_M_IX86)
  EXPECT_FALSE(usable.sse42);
  ExpectNoAvx(usable);
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