#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>
#include <sys/resource.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
using namespace zvec;
using namespace zvec::core;
using Clock = std::chrono::steady_clock;
static void check(int ret, const char* what) { if(ret) {std::cerr<<what<<": "<<ret<<std::endl; std::exit(2);} }
int main(int argc,char**argv) {
 if(argc!=9){std::cerr<<"bench class storage index queries gt threads seconds pool_mib\n";return 2;}
 ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_ERROR);
 std::string cls=argv[1],storage_cls=argv[2];int threads=std::stoi(argv[6]);double seconds=std::stod(argv[7]);
 check(ailego::MemoryLimitPool::get_instance().init(std::stoull(argv[8])*1024*1024),"pool");
 std::vector<std::vector<float>> queries;std::vector<std::unordered_set<uint64_t>> truth;
 std::ifstream qf(argv[4]),gf(argv[5]);std::string line;
 while(std::getline(qf,line)){auto p=line.find(';');auto e=line.find(';',p+1);std::istringstream ss(line.substr(p+1,e-p-1));std::vector<float> v;float x;while(ss>>x)v.push_back(x);if(v.size()!=768)return 3;queries.push_back(std::move(v));}
 while(std::getline(gf,line)){std::istringstream ss(line.substr(line.find(';')+1));uint64_t x;std::unordered_set<uint64_t> v;for(int i=0;i<10&&ss>>x;++i)v.insert(x);truth.push_back(std::move(v));}
 if(queries.size()!=1000||truth.size()!=queries.size())return 4;
 auto storage=IndexFactory::CreateStorage(storage_cls);if(!storage)return 5;
 check(storage->init(ailego::Params()),"storage init");check(storage->open(argv[3],false),"storage open");
 ailego::Params p;p.set("proxima.hnsw.streamer.ef",200U);p.set("proxima.hnsw.streamer.estimate_doc_count",100000U);p.set("proxima.hnsw.streamer.max_neighbor_count",16U);p.set("proxima.hnsw.streamer.l0_max_neighbor_count_multiplier",2U);p.set("proxima.vamana.streamer.ef",200U);p.set("proxima.vamana.streamer.max_degree",32U);p.set("proxima.vamana.streamer.search_list_size",100U);p.set("proxima.ivf.searcher.nprobe",32U);p.set("zvec.diskann.searcher.list_size",200U);
 IndexMeta meta;IndexRunner::Pointer runner;auto streamer=IndexFactory::CreateStreamer(cls);auto searcher=IndexFactory::CreateSearcher(cls);
 if(streamer){check(IndexHelper::DeserializeFromStorage(storage.get(),&meta),"meta");check(streamer->init(meta,p),"streamer init");check(streamer->open(storage),"streamer open");runner=streamer;}
 else if(searcher){check(searcher->init(p),"searcher init");check(searcher->load(storage,IndexMetric::Pointer()),"searcher load");runner=searcher;meta=searcher->meta();}else return 6;
 IndexQueryMeta qm(IndexMeta::DataType::DT_FP32,768);
 std::vector<std::string> query_features(queries.size());
 if(!meta.reformer_name().empty()){
  auto reformer=IndexFactory::CreateReformer(meta.reformer_name());if(!reformer)return 10;
  check(reformer->init(meta.reformer_params()),"reformer init");check(reformer->load(storage),"reformer load");
  IndexQueryMeta out_meta;for(size_t i=0;i<queries.size();++i)check(reformer->transform(queries[i].data(),qm,&query_features[i],&out_meta),"query transform");qm=out_meta;
 }else for(size_t i=0;i<queries.size();++i)query_features[i].assign(reinterpret_cast<const char*>(queries[i].data()),queries[i].size()*sizeof(float));
 std::vector<IndexContext::Pointer> contexts;for(int t=0;t<threads;++t){auto ctx=runner->create_context();if(!ctx)return 7;ctx->set_topk(10);contexts.push_back(std::move(ctx));}
 auto search=[&](size_t qi,IndexContext::Pointer&ctx){check(runner->search_impl(query_features[qi].data(),qm,ctx),"search");if(ctx->result().size()!=10){std::cerr<<"bad result size\n";std::exit(8);}};
 // One complete pass both validates accuracy and warms the index before timing.
 size_t validation_queries=queries.size();if(const char* n=std::getenv("BENCH_VALIDATION_QUERIES"))validation_queries=std::min(queries.size(),size_t(std::stoull(n)));
 std::ofstream result_file;if(const char* path=std::getenv("BENCH_RESULTS_PATH"))result_file.open(path);
 uint64_t hits=0,hash=1469598103934665603ULL;for(size_t i=0;i<validation_queries;++i){search(i,contexts[0]);if(result_file.is_open()){for(auto&d:contexts[0]->result())result_file<<d.key()<<" ";result_file<<"\n";}for(auto&d:contexts[0]->result()){hits+=truth[i].count(d.key());hash=(hash^d.key())*1099511628211ULL;}}
 std::vector<std::vector<double>> times(threads);std::vector<std::thread> workers;std::atomic<int> ready{0};std::atomic<bool> go{false};Clock::time_point deadline;
 for(int t=0;t<threads;++t)workers.emplace_back([&,t]{times[t].reserve(size_t(seconds*30000));ready.fetch_add(1);while(!go.load(std::memory_order_acquire))std::this_thread::yield();size_t i=t*137;while(Clock::now()<deadline){auto begin=Clock::now();search(i%queries.size(),contexts[t]);auto end=Clock::now();times[t].push_back(std::chrono::duration<double,std::micro>(end-begin).count());++i;}});
 while(ready.load()!=threads)std::this_thread::yield();auto begin=Clock::now();deadline=begin+std::chrono::milliseconds(int(seconds*1000));go.store(true,std::memory_order_release);for(auto&w:workers)w.join();double wall=std::chrono::duration<double>(Clock::now()-begin).count();
 std::vector<double> all;for(auto&v:times)all.insert(all.end(),v.begin(),v.end());std::sort(all.begin(),all.end());if(all.empty())return 9;double avg=std::accumulate(all.begin(),all.end(),0.0)/all.size();struct rusage usage;getrusage(RUSAGE_SELF,&usage);auto stats=ailego::MemoryLimitPool::get_instance().stats();
 std::cout<<std::fixed<<std::setprecision(6)<<"RESULT {\"algorithm\":\""<<cls<<"\",\"storage\":\""<<storage_cls<<"\",\"threads\":"<<threads<<",\"queries\":"<<all.size()<<",\"seconds\":"<<wall<<",\"qps\":"<<all.size()/wall<<",\"mean_us\":"<<avg<<",\"p50_us\":"<<all[all.size()/2]<<",\"p95_us\":"<<all[size_t(all.size()*0.95)]<<",\"p99_us\":"<<all[size_t(all.size()*0.99)]<<",\"recall10\":"<<hits/double(validation_queries*10)<<",\"validation_queries\":"<<validation_queries<<",\"result_hash\":\""<<hash<<"\",\"max_rss_mib\":"<<usage.ru_maxrss/1048576.0<<",\"pool_page_mib\":"<<stats.page_used/1048576.0<<"}"<<std::endl;
 return 0;
}
