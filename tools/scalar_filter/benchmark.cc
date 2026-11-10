// Public-API scalar-filter workload driver; no diagnostic/profiling symbols.
// Build/bench use separate processes because GlobalConfig is initialized once.
#include <sys/resource.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/schema.h>
#ifdef SCALAR_FILTER_INTERNAL_STATS
#include "db/sqlengine/planner/scalar_doc_filter.h"
#endif

using Clock = std::chrono::steady_clock;
using namespace zvec;
constexpr int kDim = 62;
constexpr int32_t kThreshold = 536870911;
constexpr int kTopK = 21;

struct Args {
  std::string command;
  std::map<std::string, std::string> values;
  std::string get(const std::string &name, std::string fallback = "") const {
    const auto it = values.find(name);
    return it == values.end() ? fallback : it->second;
  }
  uint64_t number(const std::string &name, uint64_t fallback) const {
    return std::stoull(get(name, std::to_string(fallback)));
  }
};

Args parse(int argc, char **argv) {
  if (argc < 2) throw std::runtime_error("Expected build or bench subcommand");
  Args a;
  a.command = argv[1];
  const std::set<std::string> known_options = {
      "path",           "rows",          "seed",
      "batch",          "build-threads", "optimize-threads",
      "deferred-index", "distribution",  "segment-rows",
      "mode",           "threads",       "pool",
      "warmup",         "queries",       "seconds",
      "dump",           "ratio",         "query-workers",
      "memory-gib",     "sequence",      "output-fields",
      "candidate-topk", "ef-search",     "dump-count",
      "dump-only",      "threshold",     "thresholds"};
  for (int i = 2; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0 || i + 1 == argc)
      throw std::runtime_error("Options require --name value pairs: " + key);
    key = key.substr(2);
    if (!known_options.count(key))
      throw std::runtime_error("Unknown option: --" + key);
    a.values[key] = argv[++i];
  }
  if (a.get("path").empty()) throw std::runtime_error("--path is required");
  return a;
}

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void check(const Status &s) {
  if (!s.ok()) throw std::runtime_error(s.message());
}

uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Separate salted streams make timestamps independent of vectors and row IDs.
int32_t timestamp(uint64_t row, uint64_t seed, uint64_t rows,
                  const std::string &distribution) {
  if (distribution == "ordered")
    return static_cast<int32_t>((row * uint64_t(INT32_MAX)) / rows);
  if (distribution == "hourly" || distribution == "buckets96")
    return static_cast<int32_t>((mix(row ^ seed ^ 0x31c9a7ULL) % 96) *
                                uint64_t(INT32_MAX) / 95);
  return static_cast<int32_t>(mix(row ^ seed ^ 0x31c9a7ULL) & INT32_MAX);
}

std::vector<float> vector_for(uint64_t row, uint64_t seed) {
  std::vector<float> result(kDim);
  uint64_t state = mix(row ^ seed ^ 0xe4dc083dULL);
  float norm = 0;
  for (auto &v : result) {
    state = mix(state);
    v = float(state >> 40) / 8388608.0f - 1.0f;
    norm += v * v;
  }
  norm = 1.0f / std::sqrt(norm);
  for (auto &v : result) v *= norm;
  return result;
}

std::string json_string(const std::string &s) {
  std::string result = "\"";
  for (const char c : s) {
    if (c == '\\' || c == '"') result += '\\';
    if (c == '\n')
      result += "\\n";
    else
      result += c;
  }
  return result + '"';
}

void build(const Args &a) {
  const auto path = a.get("path");
  if (std::filesystem::exists(path))
    throw std::runtime_error("Refusing to replace existing path: " + path);
  const uint64_t rows = a.number("rows", 1000000);
  const uint64_t seed = a.number("seed", 868);
  const uint64_t batch_size = a.number("batch", 1000);
  const int workers = static_cast<int>(
      a.number("optimize-threads", a.number("build-threads", 8)));
  const bool deferred = a.number("deferred-index", 0) != 0;
  const auto distribution = a.get("distribution", "random");
  if (!rows || !batch_size || workers < 1)
    throw std::runtime_error("rows, batch and build-threads must be positive");
  if (batch_size > 1024)
    throw std::runtime_error("batch exceeds Zvec limit of 1024");
  if (rows > uint64_t(INT32_MAX))
    throw std::runtime_error("rows exceeds synthetic INT32 pid limit");
  if (distribution != "random" && distribution != "ordered" &&
      distribution != "hourly" && distribution != "buckets96")
    throw std::runtime_error(
        "distribution must be random, ordered or buckets96");
  auto hnsw = std::make_shared<HnswIndexParams>(MetricType::COSINE, 16, 200,
                                                QuantizeType::INT8);
  IndexParams::Ptr initial_index = hnsw;
  if (deferred)
    initial_index = std::make_shared<FlatIndexParams>(MetricType::COSINE);
  CollectionSchema schema("scalar_filter_workload");
  if (a.number("segment-rows", 0))
    schema.set_max_doc_count_per_segment(a.number("segment-rows", 0));
  schema.add_field(std::make_shared<FieldSchema>("vec", DataType::VECTOR_FP32,
                                                 kDim, false, initial_index));
  schema.add_field(std::make_shared<FieldSchema>(
      "pid", DataType::INT32, true, std::make_shared<InvertIndexParams>()));
  schema.add_field(
      std::make_shared<FieldSchema>("mail", DataType::STRING, false));
  auto created =
      Collection::CreateAndOpen(path, schema, CollectionOptions{false, true});
  if (!created) throw std::runtime_error(created.error().message());
  auto col = created.value();
  check(col->add_column(
      std::make_shared<FieldSchema>("dataTime", DataType::INT32, true,
                                    std::make_shared<InvertIndexParams>()),
      ""));
  const auto started = Clock::now();
  uint64_t matches = 0, next_progress = 100000;
  for (uint64_t base = 0; base < rows; base += batch_size) {
    std::vector<Doc> docs;
    const auto end = std::min(rows, base + batch_size);
    docs.reserve(end - base);
    for (uint64_t i = base; i < end; ++i) {
      Doc doc;
      doc.set_pk(std::to_string(i));
      doc.set<int32_t>("pid", static_cast<int32_t>(i));
      const auto time = timestamp(i, seed, rows, distribution);
      matches += time > kThreshold;
      doc.set<int32_t>("dataTime", time);
      doc.set<std::string>("mail",
                           "user" + std::to_string(i) + "@example.test");
      doc.set<std::vector<float>>("vec", vector_for(i, seed));
      docs.push_back(std::move(doc));
    }
    auto inserted = col->insert(docs);
    if (!inserted) throw std::runtime_error(inserted.error().message());
    if (inserted.value().size() != docs.size())
      throw std::runtime_error("Unexpected insert status count");
    for (const auto &status : inserted.value()) check(status);
    if (end >= next_progress || end == rows) {
      std::cerr << "inserted=" << end << " seconds=" << seconds(started)
                << std::endl;
      next_progress = end + 100000;
    }
  }
  const auto load_seconds = seconds(started);
  std::cerr << "optimizing scalar/index segments" << std::endl;
  check(col->optimize(OptimizeOptions{workers}));
  if (deferred) {
    std::cerr << "building HNSW after Flat bulk load" << std::endl;
    check(col->create_index("vec", hnsw, CreateIndexOptions{workers}));
    check(col->optimize(OptimizeOptions{workers}));
  }
  auto stats = col->stats();
  if (!stats) throw std::runtime_error(stats.error().message());
  if (stats.value().doc_count != rows)
    throw std::runtime_error("Unexpected final document count");
  for (const auto &entry : stats.value().index_completeness)
    if (entry.second != 1.0f)
      throw std::runtime_error("Index incomplete after optimize: " +
                               entry.first);
  std::cerr << "final_stats=" << stats.value().to_string() << std::endl;
  auto final_schema = col->schema();
  if (!final_schema) throw std::runtime_error(final_schema.error().message());
  std::cerr << "final_schema=" << final_schema.value().to_string() << std::endl;
  check(col->close());
  std::cout << std::setprecision(10)
            << "{\"event\":\"build\",\"path\":" << json_string(path)
            << ",\"rows\":" << rows << ",\"seed\":" << seed
            << ",\"distribution\":" << json_string(distribution)
            << ",\"matches\":" << matches
            << ",\"match_fraction\":" << double(matches) / rows
            << ",\"deferred_index\":" << (deferred ? "true" : "false")
            << ",\"optimize_threads\":" << workers
            << ",\"memory_gib\":" << std::stod(a.get("memory-gib", "12"))
            << ",\"load_seconds\":" << load_seconds
            << ",\"total_seconds\":" << seconds(started) << "}" << std::endl;
}

SearchQuery make_query(const std::string &mode, uint64_t query_id,
                       uint64_t seed, uint64_t candidate_topk,
                       uint64_t ef_search, int32_t threshold) {
  SearchQuery q;
  q.topk_ = static_cast<int>(candidate_topk);
  q.output_fields_ = std::vector<std::string>{"mail", "dataTime"};
  if (mode == "filtered" || mode == "scalar")
    q.filter_ = "dataTime > " + std::to_string(threshold);
  if (mode != "scalar") {
    const auto vec = vector_for(query_id, seed ^ 0x89c294e7ULL);
    q.target_.field_name_ = "vec";
    q.target_.set_vector(std::string(reinterpret_cast<const char *>(vec.data()),
                                     vec.size() * sizeof(float)));
    q.target_.query_params_ =
        std::make_shared<HnswQueryParams>(static_cast<int>(ef_search));
  }
  return q;
}

struct alignas(64) Counts {
  uint64_t queries = 0, errors = 0, short_results = 0, violations = 0,
           returned = 0;
  uint64_t query_api_ns = 0;
  std::vector<double> latencies;
  std::vector<double> api_latencies;
};

double process_cpu_seconds() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage))
    throw std::runtime_error("getrusage failed");
  return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
         (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000000.0;
}

void account(const Result<DocPtrList> &result, const std::string &mode,
             Counts &c, int32_t threshold) {
  ++c.queries;
  if (!result) {
    if (c.errors++ == 0)
      std::cerr << "query error=" << result.error().message() << std::endl;
    return;
  }
  const bool postfilter = mode.rfind("postfilter", 0) == 0;
  uint64_t accepted = 0;
  for (const auto &doc : result.value()) {
    auto time = doc->get<int32_t>("dataTime");
    if (!time) {
      ++c.violations;
      continue;
    }
    if (postfilter && *time <= threshold) continue;
    if ((mode == "filtered" || mode == "scalar") && *time <= threshold)
      ++c.violations;
    ++accepted;
    if (postfilter && accepted == kTopK) break;
  }
  c.returned += accepted;
  if (accepted < kTopK) ++c.short_results;
}

double percentile(const std::vector<double> &samples, double p) {
  if (samples.empty()) return 0;
  return samples[std::min(
      samples.size() - 1,
      static_cast<size_t>(std::ceil(p * samples.size()) - 1))];
}

void bench(const Args &a) {
  const auto path = a.get("path");
  const auto mode = a.get("mode", "filtered");
  if (mode != "filtered" && mode != "unfiltered" && mode != "scalar" &&
      mode != "postfilter" && mode != "postfilter64" && mode != "postfilter128")
    throw std::runtime_error("Unknown query mode");
  const auto thread_count = a.number("threads", 1);
  const auto pool_size = a.number("pool", 256);
  const auto warmup = a.number("warmup", 32);
  const auto query_limit = a.number("queries", 0);
  const auto seed = a.number("seed", 868);
  const auto sequence = a.get("sequence", "mixed");
  const bool postfilter = mode.rfind("postfilter", 0) == 0;
  const auto candidate_topk =
      a.number("candidate-topk", mode == "postfilter128" ? 128
                                 : postfilter            ? 64
                                                         : kTopK);
  const auto ef_search = a.number("ef-search", 300);
  if (candidate_topk < kTopK || candidate_topk > 2048 ||
      (!postfilter && candidate_topk != kTopK) || ef_search < candidate_topk ||
      ef_search > 2048)
    throw std::runtime_error(
        "candidate-topk must be 21 outside postfilter, or 21..2048 in "
        "postfilter; ef-search must be candidate-topk..2048");
  if (a.number("dump-only", 0) && a.get("dump").empty())
    throw std::runtime_error("dump-only requires --dump");
  const auto output_fields = a.get("output-fields", "all-scalars");
  if (output_fields != "mail-time" && output_fields != "all-scalars")
    throw std::runtime_error("output-fields must be mail-time or all-scalars");
  if (sequence != "mixed" && sequence != "sequential")
    throw std::runtime_error("sequence must be mixed or sequential");
  const double duration = std::stod(a.get("seconds", "15"));
  if (!thread_count || !pool_size || duration <= 0)
    throw std::runtime_error("threads, pool and seconds must be positive");
  const auto open_started = Clock::now();
  auto opened = Collection::Open(path, CollectionOptions{true, true});
  if (!opened) throw std::runtime_error(opened.error().message());
  auto col = opened.value();
  const auto open_seconds = seconds(open_started);
  const auto stats = col->stats();
  if (!stats) throw std::runtime_error(stats.error().message());
  std::cerr << "opened " << stats.value().to_string() << std::endl;
  const auto threshold_policy = a.get("thresholds", "fixed");
  if (threshold_policy != "fixed" && threshold_policy != "varying")
    throw std::runtime_error("thresholds must be fixed or varying");
  const auto requested_threshold =
      std::stoll(a.get("threshold", std::to_string(kThreshold)));
  if (requested_threshold < INT32_MIN || requested_threshold > INT32_MAX)
    throw std::runtime_error("threshold outside INT32");
  std::vector<int32_t> thresholds;
  std::vector<SearchQuery> queries;
  for (uint64_t i = 0; i < pool_size; ++i) {
    // Keep varying predicates between about 62.5% and 87.5% selectivity on
    // uniform data, so ratio=0.5 continues selecting the forward-filter path.
    thresholds.push_back(
        threshold_policy == "fixed"
            ? static_cast<int32_t>(requested_threshold)
            : static_cast<int32_t>(uint64_t(INT32_MAX) / 8 +
                                   mix(i ^ seed ^ 0x642b7aULL) %
                                       (uint64_t(INT32_MAX) / 4)));
    queries.push_back(make_query(mode, i, seed, candidate_topk, ef_search,
                                 thresholds.back()));
    // The issue's C++ snippet leaves output_fields_ unspecified.
    if (output_fields == "all-scalars") queries.back().output_fields_.reset();
  }
  if (!a.get("dump").empty()) {
    std::ofstream out(a.get("dump"));
    if (!out) throw std::runtime_error("Cannot open --dump file");
    for (uint64_t i = 0;
         i < std::min<uint64_t>(a.number("dump-count", 64), pool_size); ++i) {
      auto result = col->query(queries[i]);
      if (!result) throw std::runtime_error(result.error().message());
      Counts checked;
      account(result, mode, checked, thresholds[i]);
      if (checked.errors || checked.violations)
        throw std::runtime_error("Dump result validation failed");
      DocPtrList selected;
      for (const auto &doc : result.value()) {
        if (postfilter &&
            doc->get<int32_t>("dataTime").value_or(0) <= thresholds[i])
          continue;
        selected.push_back(doc);
        if (selected.size() == kTopK) break;
      }
      out << std::setprecision(10) << "{\"query_id\":" << i
          << ",\"threshold\":" << thresholds[i]
          << ",\"raw_returned\":" << result.value().size()
          << ",\"violations\":" << checked.violations << ",\"ids\":[";
      for (size_t j = 0; j < selected.size(); ++j) {
        if (j) out << ',';
        out << json_string(selected[j]->pk());
      }
      out << "],\"scores\":[";
      for (size_t j = 0; j < selected.size(); ++j) {
        if (j) out << ',';
        out << selected[j]->score();
      }
      out << "],\"times\":[";
      for (size_t j = 0; j < selected.size(); ++j) {
        if (j) out << ',';
        out << selected[j]->get<int32_t>("dataTime").value();
      }
      out << "],\"pids\":[";
      for (size_t j = 0; j < selected.size(); ++j) {
        if (j) out << ',';
        const auto value = selected[j]->get<int32_t>("pid");
        if (value)
          out << *value;
        else
          out << "null";
      }
      out << "],\"mails\":[";
      for (size_t j = 0; j < selected.size(); ++j) {
        if (j) out << ',';
        const auto value = selected[j]->get<std::string>("mail");
        if (value)
          out << json_string(*value);
        else
          out << "null";
      }
      out << "]}\n";
    }
    out.flush();
    if (!out) throw std::runtime_error("Failed writing --dump file");
  }
  if (a.number("dump-only", 0)) {
    check(col->close());
    return;
  }
  std::mutex mutex;
  std::condition_variable cv;
  size_t ready = 0;
  bool go = false;
  Clock::time_point started, deadline;
  double cpu_started = 0;
  std::atomic<uint64_t> ticket{0};
  std::vector<Counts> counts(thread_count);
  std::vector<std::thread> threads;
  std::atomic<bool> warmup_failed{false};
#ifdef SCALAR_FILTER_INTERNAL_STATS
  uint64_t jit_bindings_before = 0;
  gandiva::ScalarPredicateCacheStats jit_cache_before;
#endif
  for (uint64_t t = 0; t < thread_count; ++t) {
    threads.emplace_back([&, t] {
      Counts warm;
      for (uint64_t i = 0; i < warmup; ++i) {
        const auto id = (t + i) % pool_size;
        account(col->query(queries[id]), mode, warm, thresholds[id]);
      }
      if (warm.errors || warm.violations ||
          ((mode == "filtered" || mode == "unfiltered") && warm.short_results))
        warmup_failed = true;
      auto &c = counts[t];
      c.latencies.reserve(100000);
      c.api_latencies.reserve(100000);
      {
        std::unique_lock<std::mutex> lock(mutex);
        ++ready;
        cv.notify_all();
        cv.wait(lock, [&] { return go; });
      }
      uint64_t i = 0;
      while (!warmup_failed) {
        uint64_t ordinal;
        if (query_limit) {
          ordinal = ticket.fetch_add(1, std::memory_order_relaxed);
          if (ordinal >= query_limit) break;
        } else {
          if (Clock::now() >= deadline) break;
          ordinal = i++;
        }
        const auto before = Clock::now();
        // Fixed-count runs share one ticket stream, making their query multiset
        // independent of thread scheduling. Duration runs retain independent
        // worker streams, avoiding synchronized visits to identical neighbors.
        const auto query_id =
            sequence == "sequential"
                ? (query_limit ? ordinal : ordinal * thread_count + t) %
                      pool_size
                : mix(seed ^ (query_limit ? ordinal : (t << 32) ^ ordinal)) %
                      pool_size;
        {
          const auto &request = queries[query_id];
          const auto api_started = Clock::now();
          auto result = col->query(request);
          const auto api_finished = Clock::now();
          const auto api_ns = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  api_finished - api_started)
                  .count());
          c.query_api_ns += api_ns;
          c.api_latencies.push_back(double(api_ns) / 1000000.0);
          account(result, mode, c, thresholds[query_id]);
        }
        c.latencies.push_back(seconds(before) * 1000.0);
      }
    });
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return ready == thread_count; });
    std::cerr << "benchmark_start mode=" << mode << " threads=" << thread_count
              << std::endl;
#ifdef SCALAR_FILTER_INTERNAL_STATS
    jit_bindings_before = sqlengine::ScalarDocFilter::SuccessfulBindings();
    jit_cache_before = sqlengine::ScalarDocFilter::CacheStats();
#endif
    cpu_started = process_cpu_seconds();
    started = Clock::now();
    deadline = started + std::chrono::duration_cast<Clock::duration>(
                             std::chrono::duration<double>(duration));
    go = true;
  }
  cv.notify_all();
  for (auto &thread : threads) thread.join();
  const double elapsed = seconds(started);
  const double cpu_seconds = process_cpu_seconds() - cpu_started;
#ifdef SCALAR_FILTER_INTERNAL_STATS
  const auto jit_bindings_after =
      sqlengine::ScalarDocFilter::SuccessfulBindings();
  const auto jit_cache_after = sqlengine::ScalarDocFilter::CacheStats();
#endif
  Counts total;
  for (auto &c : counts) {
    total.queries += c.queries;
    total.errors += c.errors;
    total.short_results += c.short_results;
    total.violations += c.violations;
    total.returned += c.returned;
    total.query_api_ns += c.query_api_ns;
    total.latencies.insert(total.latencies.end(), c.latencies.begin(),
                           c.latencies.end());
    total.api_latencies.insert(total.api_latencies.end(),
                               c.api_latencies.begin(), c.api_latencies.end());
  }
  const double mean_ms = total.queries
                             ? std::accumulate(total.latencies.begin(),
                                               total.latencies.end(), 0.0) /
                                   total.queries
                             : 0.0;
  const double query_api_mean_ms =
      total.queries ? double(total.query_api_ns) / total.queries / 1000000.0
                    : 0.0;
  std::sort(total.latencies.begin(), total.latencies.end());
  std::sort(total.api_latencies.begin(), total.api_latencies.end());
  std::cout << std::setprecision(10)
            << "{\"event\":\"bench\",\"path\":" << json_string(path)
            << ",\"rows\":" << stats.value().doc_count
            << ",\"mode\":" << json_string(mode)
            << ",\"threads\":" << thread_count
            << ",\"query_workers\":" << a.number("query-workers", 8)
            << ",\"memory_gib\":" << std::stod(a.get("memory-gib", "12"))
            << ",\"ratio\":" << std::stod(a.get("ratio", "0.9"))
            << ",\"candidate_topk\":" << candidate_topk
            << ",\"ef_search\":" << ef_search << ",\"seed\":" << seed
            << ",\"pool\":" << pool_size
            << ",\"sequence\":" << json_string(sequence)
            << ",\"output_fields\":" << json_string(output_fields)
            << ",\"threshold_policy\":" << json_string(threshold_policy)
            << ",\"fixed_threshold\":" << requested_threshold
            << ",\"query_limit\":" << query_limit
            << ",\"warmup_per_thread\":" << warmup
            << ",\"open_seconds\":" << open_seconds
            << ",\"seconds\":" << elapsed << ",\"queries\":" << total.queries
            << ",\"qps\":" << total.queries / elapsed
            << ",\"process_cpu_seconds\":" << cpu_seconds
            << ",\"average_cpu_cores\":" << cpu_seconds / elapsed
            << ",\"mean_ms\":" << mean_ms
            << ",\"query_api_mean_ms\":" << query_api_mean_ms
            << ",\"query_api_p50_ms\":" << percentile(total.api_latencies, .50)
            << ",\"query_api_p95_ms\":" << percentile(total.api_latencies, .95)
            << ",\"query_api_p99_ms\":" << percentile(total.api_latencies, .99)
            << ",\"p50_ms\":" << percentile(total.latencies, .50)
            << ",\"p95_ms\":" << percentile(total.latencies, .95)
            << ",\"p99_ms\":" << percentile(total.latencies, .99)
            << ",\"errors\":" << total.errors
            << ",\"short_results\":" << total.short_results
            << ",\"filter_violations\":" << total.violations
            << ",\"returned\":" << total.returned
#ifdef SCALAR_FILTER_INTERNAL_STATS
            << ",\"jit_successful_bindings\":"
            << jit_bindings_after - jit_bindings_before
            << ",\"jit_total_successful_bindings\":" << jit_bindings_after
            << ",\"jit_cache_lookups\":"
            << jit_cache_after.lookups - jit_cache_before.lookups
            << ",\"jit_cache_hits\":"
            << jit_cache_after.hits - jit_cache_before.hits
            << ",\"jit_cache_compilations\":"
            << jit_cache_after.compilations - jit_cache_before.compilations
            << ",\"jit_cache_total_compilations\":"
            << jit_cache_after.compilations
            << ",\"jit_cache_entries\":" << jit_cache_after.entries
#endif
            << "}" << std::endl;
  check(col->close());
  if (warmup_failed || total.errors || total.violations)
    throw std::runtime_error("Query validation failed");
}

int main(int argc, char **argv) {
  try {
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
      std::cout
          << "Scalar filter workload; options use --name value pairs.\n"
          << "Build (refuses an existing path):\n"
          << "  scalar_filter_benchmark build --path /tmp/scalar-filter-100k "
             "--rows 100000 --optimize-threads 8 --memory-gib 12\n"
          << "Benchmark (one fresh process per ratio/mode):\n"
          << "  scalar_filter_benchmark bench --path /tmp/scalar-filter-100k "
             "--mode filtered --ratio 0.9 --threads 8 --seconds 15\n"
          << "Modes: filtered, unfiltered, scalar, postfilter, postfilter64, "
             "postfilter128.\n"
          << "Build options: --rows 1000000 --batch 1000 (max1024) --seed 868\n"
          << "  --distribution random|ordered|buckets96 --segment-rows 0 "
             "(schema default)\n"
          << "  --deferred-index 0 (direct HNSW schema; inserts already use "
             "Flat internally)\n"
          << "Bench options: --threads 1 --seconds 15 --queries 0 (duration "
             "mode)\n"
          << "  --pool 256 --warmup 32 (per thread) --seed 868 --dump "
             "/tmp/results.jsonl\n"
          << "  --candidate-topk 64 (postfilter default; final21) --ef-search "
             "300\n"
          << "  --dump-count 64 --dump-only 0 (1 skips timing; dump count "
             "capped at pool)\n"
          << "  --sequence mixed|sequential (default mixed); fixed-count runs "
             "use global query tickets\n"
          << "  --output-fields mail-time|all-scalars (default all-scalars; "
             "issue snippet uses all-scalars)\n"
          << "  --thresholds fixed|varying --threshold 536870911 (fixed mode)\n"
          << "Global options: --ratio 0.9 --query-workers 8 --optimize-threads "
             "8 --memory-gib 12\n"
          << "62d COSINE INT8 HNSW M16 efConstruction200; query ef300 top21.\n"
          << "All query modes return all scalar fields by default; postfilter "
             "modes retain first21 matches.\n"
          << "Timing includes field validation/postfilter, excludes open and "
             "warmup.\n"
          << "query_api_*_ms times only col->query; mean_ms includes "
             "validation and result disposal.\n"
          << "Postfilter result counts/ID overlap do not establish exact "
             "nearest-neighbor recall.\n";
      return 0;
    }
    const auto a = parse(argc, argv);
    GlobalConfig::ConfigData config;
    config.invert_to_forward_scan_ratio = std::stof(a.get("ratio", "0.9"));
    config.query_thread_count =
        static_cast<uint32_t>(a.number("query-workers", 8));
    config.optimize_thread_count = static_cast<uint32_t>(
        a.number("optimize-threads", a.number("build-threads", 8)));
    config.memory_limit_bytes = static_cast<uint64_t>(
        std::stod(a.get("memory-gib", "12")) * (1ULL << 30));
    check(GlobalConfig::Instance().initialize(config));
    if (a.command == "build")
      build(a);
    else if (a.command == "bench")
      bench(a);
    else
      throw std::runtime_error("Expected build or bench subcommand");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
