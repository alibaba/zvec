#include <cstdio>
#include <cstring>
#include <ailego/internal/cpu_features.h>
#include <zvec/turbo/turbo.h>

int main(int argc, char **argv) {
  using namespace zvec::turbo;
  const auto &f = zvec::ailego::internal::CpuFeatures::static_flags_;
  bool ready = true;
  for (auto metric :
       {MetricType::kSquaredEuclidean, MetricType::kInnerProduct}) {
    for (auto type : {DataType::kFp32, DataType::kFp16}) {
      auto avx2 = get_interleaved_many_to_many_distance_func(
          metric, type, 32, CpuArchType::kAVX2);
      auto automatic =
          get_interleaved_many_to_many_distance_func(metric, type, 32);
      std::printf("metric=%d dtype=%d AVX2=%d AUTO=%d\n", int(metric),
                  int(type), bool(avx2), bool(automatic));
      ready = ready && bool(avx2) && bool(automatic);
    }
  }
  std::printf("CPU AVX2=%d FMA=%d F16C=%d NEON=%d; joint_avx2_ready=%d\n",
              f.AVX2, f.FMA, f.F16C, f.NEON, ready);
  return argc > 1 && std::strcmp(argv[1], "--require-avx2") == 0 && !ready ? 2
                                                                           : 0;
}
