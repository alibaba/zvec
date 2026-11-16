#include <cstdio>
#include <cstring>
#include <ailego/internal/cpu_features.h>
#include <zvec/turbo/turbo.h>

int main(int argc, char **argv) {
  using namespace zvec::turbo;
  const auto &f = zvec::ailego::internal::CpuFeatures::static_flags_;
  bool ready = true, ready512 = true;
  for (auto metric :
       {MetricType::kSquaredEuclidean, MetricType::kInnerProduct}) {
    for (auto type : {DataType::kFp32, DataType::kFp16}) {
      auto avx2 = get_interleaved_many_to_many_distance_func(
          metric, type, 32, CpuArchType::kAVX2);
      auto avx512 = get_interleaved_many_to_many_distance_func(
          metric, type, 32, CpuArchType::kAVX512);
      auto automatic =
          get_interleaved_many_to_many_distance_func(metric, type, 32);
      std::printf("metric=%d dtype=%d AVX2=%d AUTO=%d\n", int(metric),
                  int(type), bool(avx2), bool(automatic));
      ready = ready && bool(avx2) && bool(automatic);
      const bool selected =
          avx512 && automatic.target_type() == avx512.target_type();
      std::printf("metric=%d dtype=%d AVX512=%d AUTO_IS_AVX512=%d\n",
                  int(metric), int(type), bool(avx512), selected);
      ready512 = ready512 && bool(avx512) && selected;
    }
  }
  std::printf("CPU AVX2=%d FMA=%d F16C=%d NEON=%d; joint_avx2_ready=%d\n",
              f.AVX2, f.FMA, f.F16C, f.NEON, ready);
  std::printf("CPU AVX512F=%d; joint_avx512_ready=%d\n", f.AVX512F, ready512);
  if (argc > 1 && std::strcmp(argv[1], "--require-avx512") == 0)
    return ready512 ? 0 : 2;
  return argc > 1 && std::strcmp(argv[1], "--require-avx2") == 0 && !ready ? 2
                                                                           : 0;
}
