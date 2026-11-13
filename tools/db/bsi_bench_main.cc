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

// Reproduces the high-cardinality workload described in issue #868.
// Layouts must share the same seed and, for ANN, the same optimized graph.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <gflags/gflags.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include "db/index/column/inverted_column/inverted_indexer.h"

DEFINE_string(mode, "scalar_build",
              "scalar_build/query, hybrid_build/query, switch");
DEFINE_string(path, "",
              "Dedicated index directory (never overwritten by build)");
DEFINE_string(layout, "TERM_RANGE", "TERM_RANGE or TERM_BSI");
DEFINE_uint64(rows, 10000000, "Number of documents");
DEFINE_uint64(seed, 868, "Portable deterministic dataset seed");
DEFINE_int32(bound, 536870911, "Exclusive INT32 lower bound");
DEFINE_int32(threads, 1, "Concurrent query clients");
DEFINE_int32(build_threads, 8, "Index optimization threads");
DEFINE_int32(rounds, 3, "Independent measured rounds");
DEFINE_int32(warmup, 64, "Unmeasured warm-up queries");
DEFINE_double(seconds, 3.0, "Measured seconds per round");
DEFINE_string(reference, "", "ANN result file: create or compare exactly");
DEFINE_bool(unfiltered, false, "ANN control run without scalar filter");
DEFINE_uint64(memory_gib, 8, "Application memory limit");

namespace {
using namespace zvec;
using Clock = std::chrono::steady_clock;

void Check(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
void Check(const Status &status) {
  Check(status.ok(), status.message());
}
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
uint64_t Mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
int32_t Timestamp(uint64_t id) {
  return static_cast<int32_t>(Mix(id + FLAGS_seed) & 0x7fffffffULL);
}
std::vector<float> Vector(uint64_t id) {
  std::vector<float> result(62);
  for (uint64_t dim = 0; dim < result.size(); ++dim) {
    // Exactly representable operations give identical inputs across platforms.
    const auto bits = Mix(id * 64 + dim + (FLAGS_seed ^ 0x98c81ULL));
    result[dim] = static_cast<float>(bits >> 40) * (1.0f / 8388608.0f) - 1.0f;
  }
  return result;
}
InvertIndexParams::OPtr Params() {
  Check(FLAGS_layout == "TERM_RANGE" || FLAGS_layout == "TERM_BSI",
        "Bad layout");
  return std::make_shared<InvertIndexParams>(
      FLAGS_layout == "TERM_BSI" ? InvertIndexLayout::TERM_BSI
                                 : InvertIndexLayout::TERM_RANGE);
}
uint64_t DiskBytes(const std::string &path) {
  uint64_t bytes = 0;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file()) bytes += entry.file_size();
  }
  return bytes;
}
void Event(const std::string &name, double seconds, uint64_t bytes = 0) {
  std::cout << "{\"event\":" << std::quoted(name)
            << ",\"mode\":" << std::quoted(FLAGS_mode)
            << ",\"layout\":" << std::quoted(FLAGS_layout)
            << ",\"rows\":" << FLAGS_rows << ",\"seconds\":" << seconds
            << ",\"bytes\":" << bytes << "}" << std::endl;
}
void Benchmark(const std::function<uint64_t(size_t)> &query) {
  for (int i = 0; i < FLAGS_warmup; ++i) query(i);
  for (int round = 0; round < FLAGS_rounds; ++round) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::vector<double>> samples(FLAGS_threads);
    std::vector<uint64_t> sums(FLAGS_threads);
    std::vector<std::exception_ptr> errors(FLAGS_threads);
    std::vector<std::thread> clients;
    Clock::time_point start;
    for (int t = 0; t < FLAGS_threads; ++t) {
      clients.emplace_back([&, t]() {
        ++ready;
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        try {
          size_t i = t;
          while (Seconds(start) < FLAGS_seconds) {
            auto before = Clock::now();
            sums[t] += query(i);
            samples[t].push_back(Seconds(before) * 1000);
            i += FLAGS_threads;
          }
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    }
    while (ready != FLAGS_threads) std::this_thread::yield();
    start = Clock::now();
    go.store(true, std::memory_order_release);
    for (auto &client : clients) client.join();
    const double wall = Seconds(start);
    for (const auto &error : errors)
      if (error) std::rethrow_exception(error);
    std::vector<double> all;
    for (const auto &sample : samples)
      all.insert(all.end(), sample.begin(), sample.end());
    Check(!all.empty(), "No timed samples");
    std::sort(all.begin(), all.end());
    auto percentile = [&](double p) {
      return all[static_cast<size_t>((all.size() - 1) * p)];
    };
    const double mean =
        std::accumulate(all.begin(), all.end(), 0.0) / all.size();
    std::cout << "{\"event\":\"query\",\"mode\":" << std::quoted(FLAGS_mode)
              << ",\"layout\":" << std::quoted(FLAGS_layout)
              << ",\"rows\":" << FLAGS_rows << ",\"threads\":" << FLAGS_threads
              << ",\"round\":" << round << ",\"bound\":" << FLAGS_bound
              << ",\"unfiltered\":" << FLAGS_unfiltered
              << ",\"queries\":" << all.size() << ",\"wall_s\":" << wall
              << ",\"qps\":" << all.size() / wall << ",\"mean_ms\":" << mean
              << ",\"p50_ms\":" << percentile(.5)
              << ",\"p95_ms\":" << percentile(.95)
              << ",\"p99_ms\":" << percentile(.99) << ",\"checksum\":"
              << std::accumulate(sums.begin(), sums.end(), uint64_t{0}) << "}"
              << std::endl;
  }
}
void Scalar() {
  const bool build = FLAGS_mode == "scalar_build";
  Check(!build || !std::filesystem::exists(FLAGS_path),
        "Build directory already exists");
  FieldSchema field("dataTime", DataType::INT32, true, Params());
  auto owner = InvertedIndexer::CreateAndOpen("bsi_bench", FLAGS_path, build,
                                              {field}, !build);
  Check(owner != nullptr, "Cannot open scalar index");
  auto index = (*owner)["dataTime"];
  auto start = Clock::now();
  if (build) {
    for (uint32_t id = 0; id < FLAGS_rows; ++id) {
      const int32_t value = Timestamp(id);
      Check(index->insert(
          id,
          std::string(reinterpret_cast<const char *>(&value), sizeof(value))));
      if ((id + 1) % 100000 == 0) {
        std::cout << "progress scalar " << FLAGS_layout << " " << id + 1 << " "
                  << Seconds(start) << "s" << std::endl;
      }
    }
    Event("insert", Seconds(start));
    start = Clock::now();
    Check(owner->seal());
    Event("seal", Seconds(start), DiskBytes(FLAGS_path));
  }
  const std::string term(reinterpret_cast<const char *>(&FLAGS_bound),
                         sizeof(FLAGS_bound));
  start = Clock::now();
  // Isolate filter construction here; pure scalar recall continues to use term.
  auto result = index->search_filter(term, CompareOp::GT);
  Check(result != nullptr, "Scalar search failed");
  Event("first_query", Seconds(start));
  uint64_t expected = 0;
  for (uint32_t id = 0; id < FLAGS_rows; ++id) {
    const bool match = Timestamp(id) > FLAGS_bound;
    Check(result->contains(id) == match,
          "Incorrect scalar membership at " + std::to_string(id));
    expected += match;
  }
  Check(result->count() == expected, "Incorrect scalar cardinality");
  std::cout << "{\"event\":\"verified\",\"rows\":" << FLAGS_rows
            << ",\"matches\":" << expected << "}" << std::endl;
  if (!build)
    Benchmark([&](size_t) {
      auto found = index->search_filter(term, CompareOp::GT);
      Check(found && found->count() == expected, "Scalar query failed");
      return found->count();
    });
}
Collection::Ptr Open(bool read_only) {
  auto opened =
      Collection::Open(FLAGS_path, CollectionOptions{read_only, true});
  Check(opened.has_value(), opened ? "" : opened.error().message());
  return opened.value();
}
void BuildHybrid() {
  Check(!std::filesystem::exists(FLAGS_path), "Build directory already exists");
  CollectionSchema schema("issue868");
  Check(schema.add_field(std::make_shared<FieldSchema>(
      "vec", DataType::VECTOR_FP32, 62, false,
      std::make_shared<HnswIndexParams>(MetricType::COSINE, 16, 200,
                                        QuantizeType::INT8))));
  Check(schema.add_field(std::make_shared<FieldSchema>(
      "pid", DataType::INT32, true, std::make_shared<InvertIndexParams>())));
  auto created =
      Collection::CreateAndOpen(FLAGS_path, schema, CollectionOptions{});
  Check(created.has_value(), created ? "" : created.error().message());
  auto collection = created.value();
  Check(collection->add_column(std::make_shared<FieldSchema>(
                                   "dataTime", DataType::INT32, true, Params()),
                               ""));
  auto start = Clock::now();
  for (uint64_t begin = 0; begin < FLAGS_rows; begin += 512) {
    std::vector<Doc> docs;
    for (uint64_t id = begin; id < std::min(begin + 512, FLAGS_rows); ++id) {
      Doc doc;
      doc.set_pk(std::to_string(id));
      Check(doc.set<int32_t>("pid", id % 1024), "Cannot set pid");
      Check(doc.set<int32_t>("dataTime", Timestamp(id)),
            "Cannot set timestamp");
      Check(doc.set<std::vector<float>>("vec", Vector(id)),
            "Cannot set vector");
      docs.emplace_back(std::move(doc));
    }
    auto written = collection->insert(docs);
    Check(written.has_value(), written ? "" : written.error().message());
    for (const auto &status : written.value()) Check(status);
    if (begin / 100000 != (begin + docs.size()) / 100000) {
      std::cout << "progress hybrid " << begin + docs.size() << " "
                << Seconds(start) << "s" << std::endl;
    }
  }
  Event("insert", Seconds(start));
  start = Clock::now();
  Check(collection->optimize(OptimizeOptions{FLAGS_build_threads}));
  Event("optimize", Seconds(start), DiskBytes(FLAGS_path));
  Check(collection->close());
}
void QueryHybrid() {
  auto collection = Open(true);
  auto schema = collection->schema();
  Check(schema.has_value(), "Cannot read schema");
  const auto actual = std::dynamic_pointer_cast<InvertIndexParams>(
      schema->get_field("dataTime")->index_params());
  Check(actual && actual->layout() == Params()->layout(),
        "Unexpected persisted layout");
  std::vector<SearchQuery> queries(256);
  for (size_t i = 0; i < queries.size(); ++i) {
    auto values = Vector(FLAGS_rows + 10000 + i);
    auto &q = queries[i];
    q.target_.field_name_ = "vec";
    q.target_.set_vector(
        std::string(reinterpret_cast<const char *>(values.data()),
                    values.size() * sizeof(float)));
    q.target_.query_params_ = std::make_shared<HnswQueryParams>(300);
    q.topk_ = 21;
    q.output_fields_ = {"pid", "dataTime"};
    if (!FLAGS_unfiltered)
      q.filter_ = "dataTime > " + std::to_string(FLAGS_bound);
  }
  std::ifstream reference(FLAGS_reference);
  std::ofstream output;
  if (!FLAGS_reference.empty() && !reference) output.open(FLAGS_reference);
  for (const auto &query : queries) {
    auto result = collection->query(query);
    Check(result.has_value(), result ? "" : result.error().message());
    Check(result->size() == 21, "Unexpected ANN result count");
    std::string signature;
    for (const auto &doc : *result) {
      const auto id = std::stoull(doc->pk());
      Check(
          id < FLAGS_rows && (FLAGS_unfiltered || Timestamp(id) > FLAGS_bound),
          "Incorrect ANN filter");
      signature += doc->pk() + ",";
    }
    if (reference) {
      std::string expected;
      Check(bool(std::getline(reference, expected)) && expected == signature,
            "ANN layouts disagree");
    } else if (output)
      output << signature << '\n';
  }
  Event("ann_verified_256_queries", 0);
  Benchmark([&](size_t i) {
    auto found = collection->query(queries[i % queries.size()]);
    Check(found.has_value() && found->size() == 21, "ANN query failed");
    return static_cast<uint64_t>(found->size());
  });
  Check(collection->close());
}
}  // namespace

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    Check(!FLAGS_path.empty() && FLAGS_rows > 0 &&
              FLAGS_rows < (uint64_t{1} << 32),
          "Invalid path/rows");
    Check(FLAGS_threads > 0 && FLAGS_rounds > 0 && FLAGS_seconds > 0,
          "Invalid query settings");
    GlobalConfig::ConfigData config;
    config.memory_limit_bytes = FLAGS_memory_gib << 30;
    config.query_thread_count = 8;
    config.optimize_thread_count = FLAGS_build_threads;
    config.invert_to_forward_scan_ratio = .9f;
    config.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
        GlobalConfig::LogLevel::kError);
    Check(GlobalConfig::Instance().initialize(config));
    if (FLAGS_mode == "scalar_build" || FLAGS_mode == "scalar_query")
      Scalar();
    else if (FLAGS_mode == "hybrid_build")
      BuildHybrid();
    else if (FLAGS_mode == "hybrid_query")
      QueryHybrid();
    else if (FLAGS_mode == "switch") {
      auto collection = Open(false);
      const auto start = Clock::now();
      Check(collection->create_index("dataTime", Params(),
                                     CreateIndexOptions{FLAGS_build_threads}));
      Check(collection->close());
      Event("rebuild_scalar_index", Seconds(start), DiskBytes(FLAGS_path));
    } else
      throw std::runtime_error("Unknown mode");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAILED: " << error.what() << std::endl;
    return 1;
  }
}
