#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#ifdef LEGACY
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/inner_product_matrix.h>
namespace kernels = zvec::ailego;
#else
#include <turbo/distance/matrix.h>
namespace kernels = zvec::turbo;
#endif
using zvec::ailego::Float16;
volatile float consumed = 0;
template <typename T, size_t M, size_t N, bool IP>
__attribute__((noinline)) void kernel(const T *m, const T *q, size_t dim, float *out) {
  if constexpr (IP) kernels::MinusInnerProductMatrix<T, M, N>::Compute(m,q,dim,out);
  else kernels::SquaredEuclideanDistanceMatrix<T, M, N>::Compute(m,q,dim,out);
}
template <typename T> float val(size_t row, size_t d, int salt) {
  int v = int((row * 13 + d * 7 + salt) % 16) - 8;
  if constexpr (std::is_integral_v<T>) return v;
  else return float(T(float(v) / 7));
}
template <typename T> std::vector<T> fill(size_t rows, size_t dim, int salt) {
  std::vector<T> a(rows * (std::is_same_v<T,uint8_t> ? dim/2 : dim));
  for(size_t i=0;i<rows;++i) for(size_t d=0;d<dim;++d) {
    float v=val<T>(i,d,salt);
    if constexpr(std::is_same_v<T,uint8_t>) a[(d/8*rows+i)*4+d%8/2] |= (int(v)&15) << (4*(d%2));
    else if constexpr(std::is_same_v<T,int8_t>) a[(d/4*rows+i)*4+d%4]=T(v);
    else a[d*rows+i]=T(v);
  }
  return a;
}
template <typename T,size_t M,size_t N,bool IP> void run(const char *type) {
  for(size_t dim:{128U,512U}) {
    auto m=fill<T>(M,dim,0),q=fill<T>(N,dim,3); std::vector<float> out(M*N);
    auto fn=&kernel<T,M,N,IP>;fn(m.data(),q.data(),dim,out.data());
    for(size_t j=0;j<N;++j) for(size_t i=0;i<M;++i) {
      double expected=0;
      for(size_t d=0;d<dim;++d){double a=val<T>(i,d,0),b=val<T>(j,d,3);expected+=IP?-a*b:(a-b)*(a-b);}
#ifdef LEGACY
      // Old FP16 pair kernels use reduced-precision arithmetic. This run is
      // the performance reference, not the correctness oracle for new code.
      constexpr bool validate = !(std::is_same_v<T, Float16> && M == 1);
#else
      constexpr bool validate = true;
#endif
      if(validate && std::abs(expected-out[j*M+i]) > std::max(0.01,std::abs(expected)*1e-4)) {
        std::fprintf(stderr,"FAIL %s %zu %zu %zu %d expected=%f actual=%f\n",type,M,N,dim,IP,expected,out[j*M+i]);std::exit(1);
      }
    }
    auto time=[&](size_t count){
      auto begin=BenchmarkClock::now();
      for(size_t i=0;i<count;++i) fn(m.data(),q.data(),dim,out.data());
      consumed=out.back();
      return std::chrono::duration<double,std::nano>(BenchmarkClock::now()-begin).count();
    };
    size_t count=1;
    while(time(count)<2000000 && count<(1U<<24))count*=2;
    std::vector<double> times;
    for(int round=0;round<5;++round)times.push_back(time(count)/count);
    std::sort(times.begin(),times.end());
    std::printf("%s,%s,%zu,%zu,%zu,%.3f\n",type,IP?"IP":"L2",M,N,dim,times[2]);
  }
}
template <typename T,size_t M,size_t N=1> void columns(const char *type) {
  run<T,M,N,false>(type);run<T,M,N,true>(type);
  if constexpr(N<M)columns<T,M,N*2>(type);
}
template <typename T> void shapes(const char *type) {
  columns<T,1>(type);columns<T,2>(type);columns<T,4>(type);
  columns<T,8>(type);columns<T,16>(type);columns<T,32>(type);
}
int main(){
  std::puts("type,metric,rows,columns,dim,ns_per_matrix");
  shapes<float>("FP32");shapes<Float16>("FP16");shapes<int8_t>("INT8");
  shapes<uint8_t>("INT4");shapes<double>("FP64");shapes<int16_t>("INT16");
  return 0;
}
