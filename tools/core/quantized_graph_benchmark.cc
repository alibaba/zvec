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
#include <sys/resource.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>


namespace ci = zvec::core_interface;
using Clock = std::chrono::steady_clock;
constexpr int DIM = 768;
int TOPK = 10, M = 50, EFC = 500, DEGREE = 32, BUILD_THREADS = 8;
const size_t N =
    std::getenv("BENCH_N") ? std::stoul(std::getenv("BENCH_N")) : 1000000;
const std::string DATA = std::getenv("BENCH_DATA")
                             ? std::getenv("BENCH_DATA")
                             : "/home/xingrui/bench/dataset/cohere/1m/";
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
long rss_kib() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line))
    if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
  return 0;
}
long peak_kib() {
  rusage r{};
  getrusage(RUSAGE_SELF, &r);
  return r.ru_maxrss;
}
ci::HNSWIndexParam params(const std::string &method) {
  ci::HNSWIndexParam p(ci::MetricType::kCosine, DIM, M, EFC);
  p.data_type = ci::DataType::DT_FP32;
  p.use_id_map = true;
  p.quantized_graph_model = method == "hnsw" ? "" : method;
  return p;
}
class TunableHNSW : public ci::HNSWIndex {
 public:
  int initialize(const ci::HNSWIndexParam &p) {
    return init(p);
  }
};
ci::Index::Pointer make_index(const std::string &method) {
  auto index = std::make_shared<TunableHNSW>();
  check(index->initialize(params(method)) == 0, "initialize index");
  return index;
}
struct Queries {
  std::vector<std::vector<float>> vectors;
  std::vector<std::vector<uint64_t>> gt;
  Queries() {
    std::ifstream in(DATA + "cohere_test_vector_1m.1000.new.txt");
    std::string line;
    while (std::getline(in, line)) {
      auto sep = line.find(';');
      check(sep != std::string::npos, "query separator");
      check(std::stoull(line.substr(0, sep)) == vectors.size(),
            "query IDs must be sequential");
      std::istringstream row(line.substr(sep + 1));
      std::vector<float> v;
      float x;
      while (row >> x) {
        check(std::isfinite(x), "nonfinite query");
        v.push_back(x);
      }
      check(v.size() == DIM, "query dimension");
      vectors.push_back(std::move(v));
    }
    std::ifstream truth(DATA + "neighbors.txt");
    while (std::getline(truth, line)) {
      auto sep = line.find(';');
      check(sep != std::string::npos, "ground truth separator");
      check(std::stoull(line.substr(0, sep)) == gt.size(),
            "GT query IDs must be sequential");
      std::istringstream row(line.substr(sep + 1));
      std::vector<uint64_t> v;
      uint64_t x;
      while (row >> x) {
        check(x < N, "GT ID outside base range");
        v.push_back(x);
      }
      check(v.size() >= size_t(TOPK), "too few GT neighbors");
      check(std::unordered_set<uint64_t>(v.begin(), v.end()).size() == v.size(),
            "duplicate GT IDs");
      gt.push_back(std::move(v));
    }
    check(vectors.size() == 1000 && vectors.size() == gt.size(),
          "expected 1000 queries and GT rows");
  }
};
void search(const std::string &path, const std::string &method, int round,
            int duration, const std::string &baseprefix) {
  Queries q;
  size_t query_begin = std::getenv("BENCH_QUERY_BEGIN")
                           ? std::stoul(std::getenv("BENCH_QUERY_BEGIN"))
                           : 0;
  size_t query_end = std::getenv("BENCH_QUERY_END")
                         ? std::stoul(std::getenv("BENCH_QUERY_END"))
                         : q.vectors.size();
  check(query_begin < query_end && query_end <= q.vectors.size(),
        "invalid query split");
  std::vector<int> thread_counts{1, 8};
  if (std::getenv("BENCH_THREADS")) {
    thread_counts.clear();
    std::istringstream in(std::getenv("BENCH_THREADS"));
    std::string item;
    while (std::getline(in, item, ','))
      thread_counts.push_back(std::stoi(item));
  }
  auto start = Clock::now();
  auto index = make_index(method);
  check(bool(index), "create search index");
  ci::StorageOptions storage;
  storage.type = ci::StorageOptions::StorageType::kMMAP;
  storage.read_only = true;
  check(index->open(path, storage) == 0, "open index");
  check(index->get_doc_count() == N, "search index count");
  std::cout << "OPEN_RESULT algorithm=" << method
            << " seconds=" << seconds(start) << " rss_kib=" << rss_kib()
            << " peak_rss_kib=" << peak_kib() << std::endl;
  int rounds =
      std::getenv("BENCH_ROUNDS") ? std::stoi(std::getenv("BENCH_ROUNDS")) : 1;
  int initial_round = round;
  for (int rr = 0; rr < rounds; ++rr) {
    round = initial_round + rr;
    std::vector<int> ks{TOPK};
    if (std::getenv("BENCH_ALL_TOPK"))
      ks = N == 100000 ? std::vector<int>{10, 100}
                       : std::vector<int>{10, 50, 100};
    if (round % 2) std::reverse(ks.begin(), ks.end());
    for (int k : ks) {
      TOPK = k;
      std::string prefix =
          baseprefix + "_k" + std::to_string(k) + "_r" + std::to_string(round);
      std::ofstream ids(prefix + "_ids.csv");
      std::ofstream csv(prefix + ".csv");
      csv << "algorithm,topk,degree,round,ef,threads,queries,seconds,qps,mean_"
             "us,p50_us,p95_us,p99_us,recall,rss_kib,peak_rss_kib,minor_faults,"
             "major_faults\n";
      std::ofstream per_query(prefix + "_recall.csv");
      per_query << "ef,query_id,hits,returned\n";
      std::vector<size_t> order(query_end - query_begin);
      std::iota(order.begin(), order.end(), query_begin);
      std::mt19937 rng(20261021);
      std::shuffle(order.begin(), order.end(), rng);
      std::vector<int> efs{TOPK, 2048};
      bool tune = duration == 0;
      int low = TOPK - 1, high = 2048;
      double high_recall = 0;
      double target = std::getenv("BENCH_TARGET")
                          ? std::stod(std::getenv("BENCH_TARGET"))
                          : .90;
      if (!tune) {
        const char *grid = TOPK == 10
                               ? "10,20,30,50,100,200,400"
                               : (TOPK == 50 ? "50,100,150,250,500,1000"
                                             : "100,200,250,400,800,1600");
        if (N == 100000)
          grid = TOPK == 10 ? "10,20,40,80,160" : "100,200,400,800";
        if (std::getenv("BENCH_ALL_TOPK")) setenv("BENCH_EFS", grid, 1);
        check(std::getenv("BENCH_EFS") != nullptr, "BENCH_EFS required");
        efs.clear();
        std::istringstream input(std::getenv("BENCH_EFS"));
        std::string item;
        while (std::getline(input, item, ',')) efs.push_back(std::stoi(item));
        if (round % 2) std::reverse(efs.begin(), efs.end());
      }
      for (size_t ei = 0; ei < efs.size(); ++ei) {
        int ef = efs[ei];
        auto param = std::make_shared<ci::HNSWQueryParam>();
        param->ef_search = ef;
        param->topk = TOPK;
        size_t hits = 0;
        // Full-query warmup plus exact ID-based recall measurement.
        for (size_t qi = query_begin; qi < query_end; ++qi) {
          ci::SearchResult result;
          check(index->search({ci::DenseVector{q.vectors[qi].data()}}, param,
                              &result) == 0,
                "recall search");
          check(result.doc_list_.size() == size_t(TOPK), "short result");
          std::unordered_set<uint64_t> seen;
          size_t local_hits = 0;
          for (auto &doc : result.doc_list_) {
            check(seen.insert(doc.key()).second && doc.key() < N &&
                      std::isfinite(doc.score()),
                  "invalid search result");
            ids << ef << "," << qi << "," << doc.key() << "," << std::hexfloat
                << doc.score() << "\n";
            local_hits += std::find(q.gt[qi].begin(), q.gt[qi].begin() + TOPK,
                                    doc.key()) != q.gt[qi].begin() + TOPK;
          }
          hits += local_hits;
          per_query << ef << ',' << qi << ',' << local_hits << ','
                    << result.doc_list_.size() << '\n';
        }
        double recall = double(hits) / ((query_end - query_begin) * TOPK);
        std::cout << std::setprecision(10) << "RECALL_RESULT ef=" << ef
                  << " recall=" << recall << std::endl;
        if (tune) {
          if (recall >= target) {
            high = ef;
            high_recall = recall;
          } else
            low = ef;
          if (ei == 0 && recall >= target) {
            std::cout << "SELECTED ef=" << TOPK << " recall=" << recall
                      << " minimum_ef_equals_topk=true" << std::endl;
            break;
          }
          if (ei >= 1) {
            check(high_recall >= target,
                  "upper tuning bound did not reach 90% recall");
            if (high - low == 1) {
              std::cout << "SELECTED ef=" << high << " recall=" << high_recall
                        << " previous_ef=" << low << std::endl;
              break;
            }
            efs.push_back((low + high) / 2);
          }
          continue;
        }
        if (duration < 0) continue;
        for (int threads : thread_counts) {
          rusage usage_begin{};
          getrusage(RUSAGE_SELF, &usage_begin);
          std::mutex mutex;
          std::condition_variable cv;
          int ready = 0;
          bool go = false;
          Clock::time_point begin, deadline;
          std::atomic<int> errors{0};
          std::vector<std::vector<double>> latencies(threads);
          std::vector<Clock::time_point> endings(threads);
          std::vector<std::thread> workers;
          for (int t = 0; t < threads; ++t)
            workers.emplace_back([&, t] {
              auto local = std::make_shared<ci::HNSWQueryParam>(*param);
              auto &times = latencies[t];
              times.reserve(200000);
              // Initialize each worker's thread-local search context before
              // timing.
              for (size_t i = t; i < order.size(); i += threads) {
                ci::SearchResult result;
                if (index->search({ci::DenseVector{q.vectors[order[i]].data()}},
                                  local, &result) != 0)
                  ++errors;
              }
              {
                std::unique_lock<std::mutex> lock(mutex);
                ++ready;
                cv.notify_all();
                cv.wait(lock, [&] { return go; });
              }
              size_t i = size_t(t) * 127;
              while (Clock::now() < deadline) {
                auto qi = order[(i++) % order.size()];
                ci::SearchResult result;
                auto before = Clock::now();
                int rc = index->search({ci::DenseVector{q.vectors[qi].data()}},
                                       local, &result);
                auto after = Clock::now();
                if (rc != 0 || result.doc_list_.size() != size_t(TOPK))
                  ++errors;
                times.push_back(
                    std::chrono::duration<double, std::micro>(after - before)
                        .count());
              }
              endings[t] = Clock::now();
            });
          {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return ready == threads; });
            begin = Clock::now();
            deadline = begin + std::chrono::seconds(duration);
            go = true;
          }
          cv.notify_all();
          for (auto &w : workers) w.join();
          check(errors == 0, "timed search errors");
          double elapsed =
              std::chrono::duration<double>(
                  *std::max_element(endings.begin(), endings.end()) - begin)
                  .count();
          std::vector<double> all;
          for (auto &v : latencies) all.insert(all.end(), v.begin(), v.end());
          check(!all.empty(), "empty timings");
          double mean =
              std::accumulate(all.begin(), all.end(), 0.0) / all.size();
          std::sort(all.begin(), all.end());
          auto pct = [&](double p) {
            return all[std::min(all.size() - 1,
                                size_t(std::ceil(p * all.size()) - 1))];
          };
          rusage usage_end{};
          getrusage(RUSAGE_SELF, &usage_end);
          csv << std::setprecision(10)
              << (std::getenv("BENCH_LABEL") ? std::getenv("BENCH_LABEL")
                                             : method)
              << ',' << TOPK << ',' << (method != "hnsw" ? DEGREE : 0) << ','
              << round << ',' << ef << ',' << threads << ',' << all.size()
              << ',' << elapsed << ',' << all.size() / elapsed << ',' << mean
              << ',' << pct(.5) << ',' << pct(.95) << ',' << pct(.99) << ','
              << recall << ',' << rss_kib() << ',' << peak_kib() << ','
              << usage_end.ru_minflt - usage_begin.ru_minflt << ','
              << usage_end.ru_majflt - usage_begin.ru_majflt << std::endl;
          std::cout << "MEASURE ef=" << ef << " threads=" << threads
                    << " recall=" << recall << " qps=" << all.size() / elapsed
                    << std::endl;
        }
      }
    }
  }
  check(index->close() == 0, "close search index");
}

int main(int argc, char **argv) {
  try {
    zvec::ailego::LoggerBroker::SetLevel(zvec::ailego::Logger::LEVEL_INFO);
    check(argc >= 2,
          "usage: benchmark search INDEX MODEL_PATH_OR_hnsw ROUND "
          "SECONDS PREFIX");
    if (std::getenv("BENCH_TOPK")) TOPK = std::stoi(std::getenv("BENCH_TOPK"));
    if (std::getenv("BENCH_M")) M = std::stoi(std::getenv("BENCH_M"));
    if (std::getenv("BENCH_EFC")) EFC = std::stoi(std::getenv("BENCH_EFC"));
    if (std::getenv("BENCH_DEGREE"))
      DEGREE = std::stoi(std::getenv("BENCH_DEGREE"));
    if (std::getenv("BENCH_BUILD_THREADS"))
      BUILD_THREADS = std::stoi(std::getenv("BENCH_BUILD_THREADS"));
    check(TOPK == 10 || TOPK == 50 || TOPK == 100, "supported topk 10/50/100");
    check(DEGREE == 32 || DEGREE == 64, "supported degree 32/64");
    std::cout << "CONFIG topk=" << TOPK << " m=" << M
              << " efConstruction=" << EFC << " degree=" << DEGREE
              << " build_threads=" << BUILD_THREADS << std::endl;
    std::string mode = argv[1];
    if (mode == "search" && argc == 7)
      search(argv[2], std::string(argv[3]), std::stoi(argv[4]),
             std::stoi(argv[5]), argv[6]);
    else
      throw std::runtime_error("invalid command line");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return 1;
  }
}
