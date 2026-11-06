#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_framework.h>
#include <core/metric/metric_params.h>
using namespace zvec;
using namespace zvec::core;
volatile float consumed_adapter = 0;
int main() {
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_ERROR);
  std::puts("type,metric,rows,columns,dim,ns_per_matrix");
  for(auto type:{IndexMeta::DT_FP32,IndexMeta::DT_FP16,IndexMeta::DT_INT8,IndexMeta::DT_INT4})
  for(int mode=0;mode<3;++mode) {
    if(mode && type!=IndexMeta::DT_INT8 && type!=IndexMeta::DT_INT4)continue;
    for(size_t dim:{128U,512U}) for(size_t n:{1U,32U}) {
      const size_t rows=32, payload=type==IndexMeta::DT_INT4?dim/2:dim*IndexMeta::UnitSizeof(type);
      const size_t encoded_dim=dim+(mode?(type==IndexMeta::DT_INT4?32:20):0);
      IndexMeta meta(type,encoded_dim);ailego::Params params;
      const char *name=mode?"QuantizedInteger":"MipsSquaredEuclidean";
      if(mode)params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME,std::string(mode==1?"SquaredEuclidean":"InnerProduct"));
      auto metric=IndexFactory::CreateMetric(name);
      if(!metric || metric->init(meta,params)!=0)std::exit(2);
      const size_t bytes=meta.element_size(),unit=type==IndexMeta::DT_FP16?2:4;
      std::vector<uint8_t> m(rows*bytes),q(n*bytes),mr(rows*bytes),qr(n*bytes);
      auto fill=[&](std::vector<uint8_t>&packed,std::vector<uint8_t>&raw,size_t count,int salt){
        for(size_t i=0;i<count;++i){
          auto *p=raw.data()+i*bytes;float sum=0,sum2=0;
          for(size_t d=0;d<dim;++d){
            int value=int((i*13+d*7+salt)%16)-8;sum+=value;sum2+=value*value;
            if(type==IndexMeta::DT_FP32){float x=value/7.0f;std::memcpy(p+d*4,&x,4);}
            else if(type==IndexMeta::DT_FP16){ailego::Float16 x(value/7.0f);std::memcpy(p+d*2,&x,2);}
            else if(type==IndexMeta::DT_INT8)p[d]=uint8_t(value);
            else p[d/2]|=(value&15)<<(4*(d%2));
          }
          if(mode){
            float tail[5]={1,0,sum,sum2,0};
            std::memcpy(p+payload,tail,bytes-payload);
            const int32_t code_sum=int32_t(sum);
            if(type==IndexMeta::DT_INT8) std::memcpy(p+payload+16,&code_sum,4);
            else if(mode==2) std::memcpy(p+payload+12,&code_sum,4);
          }
          for(size_t d=0;d<bytes/unit;++d)std::memcpy(packed.data()+(d*count+i)*unit,p+d*unit,unit);
        }
      };
      fill(m,mr,rows,0);fill(q,qr,n,3);std::vector<float> out(rows*n);
      auto fn=metric->distance_matrix(rows,n);if(!fn)std::exit(3);
      fn(m.data(),q.data(),encoded_dim,out.data());
      auto pair=metric->distance();
      for(size_t j=0;j<n;++j)for(size_t i=0;i<rows;++i){float ref;pair(mr.data()+i*bytes,qr.data()+j*bytes,encoded_dim,&ref);
        float tolerance=1e-4f;
#ifdef LEGACY
        if(type==IndexMeta::DT_FP16)tolerance=1e-2f;
#endif
        if(!std::isfinite(out[j*rows+i]) || std::abs(ref-out[j*rows+i])>tolerance*std::max(1.0f,std::abs(ref))){std::fprintf(stderr,"matrix/pair mismatch type=%d mode=%d dim=%zu n=%zu ref=%f actual=%f\n",int(type),mode,dim,n,ref,out[j*rows+i]);std::exit(4);}
      }
      auto time=[&](size_t count){auto begin=BenchmarkClock::now();for(size_t i=0;i<count;++i)fn(m.data(),q.data(),encoded_dim,out.data());consumed_adapter=out.back();return std::chrono::duration<double,std::nano>(BenchmarkClock::now()-begin).count();};
      size_t count=1;while(time(count)<2000000 && count<(1U<<20))count*=2;
      std::vector<double> samples;for(int i=0;i<5;++i)samples.push_back(time(count)/count);std::sort(samples.begin(),samples.end());
      std::printf("%d,%s,%zu,%zu,%zu,%.3f\n",int(type),mode==0?"MIPS":mode==1?"RecordL2":"RecordIP",rows,n,dim,samples[2]);
    }
  }
}
