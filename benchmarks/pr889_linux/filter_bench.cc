// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
// Local PR #889 benchmark. Generation is deterministic but is not the Mac
// dataset.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#ifdef PR889_INTERNAL_STATS
#include "db/sqlengine/planner/scalar_doc_filter.h"
#endif

using namespace zvec;
using Clock = std::chrono::steady_clock;
using Args = std::map<std::string, std::string>;
constexpr uint64_t kSeed = 868;
constexpr int kDim = 62;
constexpr int kThreshold = 536870911;
constexpr size_t kPool = 1024;

double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
void check(const Status &s) {
  if (!s.ok()) throw std::runtime_error(s.message());
}
template <class T>
T take(Result<T> r) {
  if (!r) throw std::runtime_error(r.error().message());
  return std::move(r).value();
}
std::string get(const Args &a, const std::string &k,
                const std::string &d = "") {
  auto it = a.find(k);
  return it == a.end() ? d : it->second;
}
uint64_t number(const Args &a, const std::string &k, uint64_t d) {
  return std::stoull(get(a, k, std::to_string(d)));
}
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
int32_t data_time(uint64_t id) {
  return static_cast<int32_t>(mix(id ^ (kSeed << 32)) & 0x7fffffffULL);
}
std::string mail(uint64_t id) {
  return "user" + std::to_string(id) + "@example.test";
}
std::vector<float> vector_for(uint64_t id, bool query) {
  uint64_t state =
      mix(id ^ kSeed ^ (query ? 0xfedcba9876543210ULL : 0x123456789abcdef0ULL));
  std::vector<float> v(kDim);
  double norm = 0;
  for (auto &x : v) {
    state = mix(state);
    x = static_cast<float>(static_cast<double>(state >> 40) / 8388608.0 - 1.0);
    norm += static_cast<double>(x) * x;
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(norm));
  for (auto &x : v) x *= inv;
  return v;
}
std::string quoted(const std::string &s) {
  std::string r = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') r += '\\';
    if (c == '\n') {
      r += "\\n";
      continue;
    }
    if (c == '\r') {
      r += "\\r";
      continue;
    }
    if (c == '\t') {
      r += "\\t";
      continue;
    }
    r += c;
  }
  return r + '"';
}
void emit(const Args &args, const std::string &json) {
  auto path = get(args, "output");
  if (!path.empty()) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open " + path);
    out << json << '\n';
  }
  std::cout << json << std::endl;
}
void initialize(const Args &args) {
  GlobalConfig::ConfigData config;
  config.query_thread_count = number(args, "workers", 8);
  config.optimize_thread_count = number(args, "build-workers", 16);
  config.query_thread_binding = false;
  config.optimize_thread_binding = false;
  config.invert_to_forward_scan_ratio =
      get(args, "strategy", "forward") == "inverted" ? 0.9f : 0.5f;
  config.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      get(args, "debug", "0") == "1" ? GlobalConfig::LogLevel::kDebug
                                     : GlobalConfig::LogLevel::kWarn);
  check(GlobalConfig::Instance().initialize(config));
}
std::string storage_description(const Collection::Ptr &coll) {
  auto schema = take(coll->schema());
  auto field = schema.get_field("vector");
  if (!field || field->index_type() != IndexType::HNSW)
    throw std::runtime_error("Expected HNSW vector schema");
  auto options = take(coll->options());
  if (!options.read_only_ || !options.enable_mmap_)
    throw std::runtime_error("Expected read-only mmap open");
  // The debug API currently inspects the first sub-index, which can be the
  // Flat refiner of a combined INT8 HNSW index. Preserve this diagnostic
  // instead of mistaking it for a missing graph; DEBUG open logs give storage.
  auto mode = coll->debug_get_hnsw_storage_mode("vector");
  return mode ? mode.value()
              : "debug_api_unavailable: " + mode.error().message();
}
void build(const Args &args) {
  const auto path = get(args, "path");
  if (path.empty()) throw std::runtime_error("--path is required");
  if (std::filesystem::exists(path))
    throw std::runtime_error("Refusing to overwrite existing collection " +
                             path);
  const uint64_t rows = number(args, "rows", 10000000);
  CollectionSchema schema("issue868_linux");
  schema.set_max_doc_count_per_segment(10000000);
  schema.add_field(
      std::make_shared<FieldSchema>("pid", DataType::INT64, false));
  schema.add_field(
      std::make_shared<FieldSchema>("mail", DataType::STRING, false));
  schema.add_field(
      std::make_shared<FieldSchema>("dataTime", DataType::INT32, false,
                                    std::make_shared<InvertIndexParams>()));
  // Delay the HNSW build until the data is inserted so optimize can build it in
  // parallel.
  schema.add_field(std::make_shared<FieldSchema>(
      "vector", DataType::VECTOR_FP32, kDim, false));
  auto start = Clock::now();
  auto coll = take(
      Collection::CreateAndOpen(path, schema, CollectionOptions{false, true}));
  uint64_t matches = 0;
  const size_t batch_size = number(args, "batch", 1024);
  for (uint64_t begin = 0; begin < rows; begin += batch_size) {
    std::vector<Doc> docs;
    docs.reserve(std::min<uint64_t>(batch_size, rows - begin));
    for (uint64_t id = begin; id < std::min<uint64_t>(begin + batch_size, rows);
         ++id) {
      Doc doc;
      doc.set_pk(std::to_string(id));
      doc.set<int64_t>("pid", id);
      doc.set<std::string>("mail", mail(id));
      doc.set<int32_t>("dataTime", data_time(id));
      doc.set<std::vector<float>>("vector", vector_for(id, false));
      matches += data_time(id) > kThreshold;
      docs.push_back(std::move(doc));
    }
    for (const auto &status : take(coll->insert(docs))) check(status);
    if (begin % (1024 * 1024) == 0)
      std::cerr << "inserted " << (begin + docs.size()) << "/" << rows
                << " elapsed=" << seconds(start) << "s\n";
  }
  check(coll->flush());
  const auto insert_seconds = seconds(start);
  auto indexing = Clock::now();
  check(coll->create_index(
      "vector",
      std::make_shared<HnswIndexParams>(MetricType::COSINE, 16, 200,
                                        QuantizeType::INT8),
      CreateIndexOptions{static_cast<int>(number(args, "build-workers", 16))}));
  const auto create_seconds = seconds(indexing);
  auto optimizing = Clock::now();
  check(coll->optimize(
      OptimizeOptions{static_cast<int>(number(args, "build-workers", 16))}));
  const auto optimize_seconds = seconds(optimizing);
  const auto stats = take(coll->stats()).to_string();
  check(coll->close());
  coll = take(Collection::Open(path, CollectionOptions{true, true}));
  const auto mode = storage_description(coll);
  check(coll->close());
  std::ostringstream out;
  out << std::setprecision(12) << "{\"mode\":\"build\",\"rows\":" << rows
      << ",\"dimension\":62,\"seed\":868,\"matching_rows\":" << matches
      << ",\"match_fraction\":" << double(matches) / rows
      << ",\"insert_seconds\":" << insert_seconds
      << ",\"create_index_seconds\":" << create_seconds
      << ",\"optimize_seconds\":" << optimize_seconds
      << ",\"total_seconds\":" << seconds(start)
      << ",\"storage_mode\":" << quoted(mode) << ",\"stats\":" << quoted(stats)
      << "}";
  emit(args, out.str());
}
int threshold_for(const Args &args, size_t q) {
  if (get(args, "vary", "0") == "1")
    return kThreshold - 134217727 +
           static_cast<int>(mix(q ^ kSeed) % 268435455);
  return std::stoi(get(args, "threshold", std::to_string(kThreshold)));
}
SearchQuery make_query(const Args &args, size_t q) {
  SearchQuery query;
  const std::string strategy = get(args, "strategy", "forward");
  query.topk_ = number(args, "candidate-topk", 21);
  if (strategy != "post" && strategy != "none") query.topk_ = 21;
  query.target_.field_name_ = "vector";
  query.target_.query_params_ =
      std::make_shared<HnswQueryParams>(number(args, "ef", 300));
  auto vector = vector_for(q, true);
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(vector.data()),
                  vector.size() * sizeof(float)));
  if (strategy == "forward" || strategy == "inverted")
    query.filter_ = "dataTime > " + std::to_string(threshold_for(args, q));
  // output_fields_ is deliberately unset: all scalar fields, as in the PR.
  return query;
}
void validate_and_post(const Args &args, size_t q, DocPtrList &docs,
                       uint64_t &checksum) {
  const auto strategy = get(args, "strategy", "forward");
  const bool filtered = strategy != "none";
  const int threshold = threshold_for(args, q);
  if (strategy == "post") {
    docs.erase(std::remove_if(docs.begin(), docs.end(),
                              [&](const Doc::Ptr &d) {
                                return d->get_ref<int32_t>("dataTime") <=
                                       threshold;
                              }),
               docs.end());
    if (docs.size() > 21) docs.resize(21);
  }
  if (docs.size() != 21)
    throw std::runtime_error("Short/wrong result count " +
                             std::to_string(docs.size()));
  for (const auto &doc : docs) {
    const uint64_t id = std::stoull(doc->pk_ref());
    const int32_t value = doc->get_ref<int32_t>("dataTime");
    if (value != data_time(id) ||
        doc->get_ref<int64_t>("pid") != static_cast<int64_t>(id) ||
        doc->get_ref<std::string>("mail") != mail(id))
      throw std::runtime_error("Incorrect scalar fields");
    if (filtered && value <= threshold)
      throw std::runtime_error("Filter violation");
    if (!std::isfinite(doc->score()))
      throw std::runtime_error("Nonfinite score");
    checksum += id;
  }
}
#ifdef PR889_INTERNAL_STATS
using Stats = zvec::sqlengine::scalar_jit::ScalarPredicateCacheStats;
Stats cache_stats() {
  return zvec::sqlengine::ScalarDocFilter::CacheStats();
}
std::string stats_json(const Stats &s) {
  std::ostringstream out;
  out << "{\"lookups\":" << s.lookups << ",\"hits\":" << s.hits
      << ",\"misses\":" << s.misses << ",\"compilations\":" << s.compilations
      << ",\"failed_compilations\":" << s.failed_compilations
      << ",\"evictions\":" << s.evictions << ",\"waits\":" << s.waits
      << ",\"entries\":" << s.entries << "}";
  return out.str();
}
#else
struct Stats {};
Stats cache_stats() {
  return {};
}
std::string stats_json(const Stats &) {
  return "null";
}
#endif
void run(const Args &args) {
  auto coll =
      take(Collection::Open(get(args, "path"), CollectionOptions{true, true}));
  const auto storage = storage_description(coll);
  const size_t clients = number(args, "clients", 1),
               count = number(args, "count", 4096),
               warmup = number(args, "warmup", 1024);
  const double duration_seconds = std::stod(get(args, "seconds", "0"));
  if (!clients || (!count && duration_seconds <= 0) || duration_seconds < 0 ||
      !std::isfinite(duration_seconds))
    throw std::runtime_error("Invalid clients/count/seconds");
  std::vector<SearchQuery> queries;
  for (size_t q = 0; q < kPool; ++q) queries.push_back(make_query(args, q));
  std::vector<std::vector<double>> times(clients), request_times(clients);
  Clock::time_point deadline;
  std::vector<uint64_t> checksums(clients, 0);
  std::vector<std::string> errors(clients);
  std::vector<std::thread> threads;
  std::mutex mu;
  std::condition_variable cv;
  size_t ready = 0, finished = 0;
  bool go = false;
  auto cold_begin = Clock::now();
  {
    auto docs = take(coll->query(queries[0]));
    validate_and_post(args, 0, docs, checksums[0]);
  }
  const double first_query_ms = seconds(cold_begin) * 1000;
  const auto after_cold = cache_stats();
  for (size_t client = 0; client < clients; ++client)
    threads.emplace_back([&, client] {
      try {
        for (size_t j = 0; j < warmup; ++j) {
          size_t q = j % kPool;
          auto docs = take(coll->query(queries[q]));
          validate_and_post(args, q, docs, checksums[client]);
        }
      } catch (const std::exception &e) {
        errors[client] = e.what();
      }
      {
        std::unique_lock<std::mutex> lock(mu);
        ++ready;
        cv.notify_all();
        cv.wait(lock, [&] { return go; });
      }
      if (errors[client].empty()) {
        try {
          times[client].reserve((count + clients - 1) / clients);
          request_times[client].reserve((count + clients - 1) / clients);
          for (size_t j = client;
               duration_seconds > 0 ? Clock::now() < deadline : j < count;
               j += clients) {
            size_t q = j % kPool;
            auto request_begin = Clock::now();
            double api_ms;
            {
              auto begin = Clock::now();
              auto result = coll->query(queries[q]);
              api_ms = seconds(begin) * 1000;
              auto docs = take(std::move(result));
              validate_and_post(args, q, docs, checksums[client]);
            }  // Dispose results before ending the request-wall sample.
            const double request_ms = seconds(request_begin) * 1000;
            times[client].push_back(api_ms);
            request_times[client].push_back(request_ms);
          }
        } catch (const std::exception &e) {
          errors[client] = e.what();
        }
      }
      {
        std::lock_guard<std::mutex> lock(mu);
        ++finished;
        cv.notify_all();
      }
    });
  {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return ready == clients; });
  }
  auto before = cache_stats();
  auto begin = Clock::now();
  deadline = begin + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(duration_seconds));
  {
    std::lock_guard<std::mutex> lock(mu);
    go = true;
    cv.notify_all();
  }
  {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return finished == clients; });
  }
  const double elapsed = seconds(begin);
  auto after = cache_stats();
  for (auto &t : threads) t.join();
  std::vector<double> all, requests;
  for (auto &v : times) all.insert(all.end(), v.begin(), v.end());
  for (auto &v : request_times)
    requests.insert(requests.end(), v.begin(), v.end());
  for (const auto &error : errors)
    if (!error.empty()) throw std::runtime_error(error);
  if (duration_seconds == 0 && all.size() != count)
    throw std::runtime_error("Incorrect timed query count");
  const size_t actual_count = all.size();
  if (!actual_count) throw std::runtime_error("No timed queries completed");
  std::sort(all.begin(), all.end());
  const auto percentile = [&](double p) {
    return all[std::min(all.size() - 1,
                        static_cast<size_t>(p * (all.size() - 1)))];
  };
  std::ostringstream out;
  out << std::setprecision(12) << "{\"mode\":\"run\",\"strategy\":"
      << quoted(get(args, "strategy", "forward")) << ",\"jit_env\":"
      << quoted(std::getenv("ZVEC_SCALAR_JIT") ? std::getenv("ZVEC_SCALAR_JIT")
                                               : "unset")
      << ",\"clients\":" << clients
      << ",\"workers\":" << number(args, "workers", 8)
      << ",\"count\":" << actual_count << ",\"requested_count\":" << count
      << ",\"requested_seconds\":" << duration_seconds
      << ",\"warmup_per_client\":" << warmup
      << ",\"ef\":" << number(args, "ef", 300)
      << ",\"candidate_topk\":" << number(args, "candidate-topk", 21)
      << ",\"elapsed_seconds\":" << elapsed
      << ",\"qps\":" << actual_count / elapsed << ",\"mean_api_ms\":"
      << std::accumulate(all.begin(), all.end(), 0.) / actual_count
      << ",\"mean_request_ms\":"
      << std::accumulate(requests.begin(), requests.end(), 0.) / actual_count
      << ",\"p50_api_ms\":" << percentile(.5)
      << ",\"p95_api_ms\":" << percentile(.95)
      << ",\"p99_api_ms\":" << percentile(.99)
      << ",\"first_query_wall_ms\":" << first_query_ms
      << ",\"storage_mode\":" << quoted(storage)
      << ",\"errors\":0,\"short_results\":0,\"filter_violations\":0,"
         "\"checksum\":"
      << std::accumulate(checksums.begin(), checksums.end(), uint64_t{0})
      << ",\"cache_after_first_query\":" << stats_json(after_cold)
      << ",\"cache_before_timing\":" << stats_json(before)
      << ",\"cache_after_timing\":" << stats_json(after) << ",\"raw_api_ms\":[";
  for (size_t i = 0; i < all.size(); ++i) {
    if (i) out << ',';
    out << all[i];
  }
  out << "],\"raw_request_ms\":[";
  std::sort(requests.begin(), requests.end());
  for (size_t i = 0; i < requests.size(); ++i) {
    if (i) out << ',';
    out << requests[i];
  }
  out << "]}";
  emit(args, out.str());
}
void dump(const Args &args) {
  auto coll =
      take(Collection::Open(get(args, "path"), CollectionOptions{true, true}));
  const size_t count = number(args, "count", 1024);
  std::ostringstream out;
  out << std::setprecision(9) << "{\"mode\":\"dump\",\"strategy\":"
      << quoted(get(args, "strategy", "forward"))
      << ",\"vary\":" << (get(args, "vary", "0") == "1" ? "true" : "false")
      << ",\"queries\":[";
  uint64_t checksum = 0;
  for (size_t q = 0; q < count; ++q) {
    auto docs = take(coll->query(make_query(args, q % kPool)));
    validate_and_post(args, q % kPool, docs, checksum);
    if (q) out << ',';
    out << "{\"query\":" << q
        << ",\"threshold\":" << threshold_for(args, q % kPool)
        << ",\"results\":[";
    for (size_t j = 0; j < docs.size(); ++j) {
      if (j) out << ',';
      const auto &d = docs[j];
      out << "{\"id\":" << quoted(d->pk_ref()) << ",\"score\":" << d->score()
          << ",\"pid\":" << d->get_ref<int64_t>("pid")
          << ",\"dataTime\":" << d->get_ref<int32_t>("dataTime")
          << ",\"mail\":" << quoted(d->get_ref<std::string>("mail")) << "}";
    }
    out << "]}";
  }
  out << "],\"cache\":" << stats_json(cache_stats())
      << ",\"checksum\":" << checksum << "}";
  emit(args, out.str());
}
int main(int argc, char **argv) {
  try {
    Args args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0)
        throw std::runtime_error("Arguments must be --key value pairs");
      args[argv[i] + 2] = argv[i + 1];
    }
    initialize(args);
    const auto mode = get(args, "mode", "run");
    if (mode == "build")
      build(args);
    else if (mode == "run")
      run(args);
    else if (mode == "dump")
      dump(args);
    else
      throw std::runtime_error("Unknown mode");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "BENCHMARK FAILED: " << e.what() << std::endl;
    return 1;
  }
}
