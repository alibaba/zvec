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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <gflags/gflags.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>

DEFINE_string(path, "", "Collection path; never removes existing data");
DEFINE_bool(build, false, "Create deterministic data, then exit");
DEFINE_int32(docs, 16384, "Document count when building");
DEFINE_int32(segments, 8, "Segment count when building; must divide docs");
DEFINE_int32(dim, 128, "Vector dimension when building");
DEFINE_int32(threads, 1, "Concurrent query callers");
DEFINE_int32(query_threads, 4, "Zvec query pool size");
DEFINE_double(seconds, 3, "Measurement duration after warmup");
DEFINE_string(mode, "vector", "vector, filtered, or scan");

namespace {

using Clock = std::chrono::steady_clock;

bool Check(const zvec::Status &status) {
  if (!status.ok()) {
    std::fprintf(stderr, "%s\n", status.message().c_str());
    return false;
  }
  return true;
}

std::vector<float> RandomVector(std::mt19937 &rng, int dimension) {
  std::uniform_real_distribution<float> value(-1.0f, 1.0f);
  std::vector<float> vector(dimension);
  for (auto &v : vector) {
    v = value(rng);
  }
  return vector;
}

bool Build() {
  zvec::CollectionSchema schema("query_threading_bench");
  if (!Check(schema.add_field(std::make_shared<zvec::FieldSchema>(
          "bucket", zvec::DataType::INT32))) ||
      !Check(schema.add_field(std::make_shared<zvec::FieldSchema>(
          "vector", zvec::DataType::VECTOR_FP32, FLAGS_dim, false,
          std::make_shared<zvec::FlatIndexParams>(zvec::MetricType::L2))))) {
    return false;
  }
  schema.set_max_doc_count_per_segment(FLAGS_docs / FLAGS_segments);
  auto opened = zvec::Collection::CreateAndOpen(FLAGS_path, schema, {});
  if (!opened) {
    return Check(opened.error());
  }
  auto collection = opened.value();
  std::mt19937 rng(42);
  for (int begin = 0; begin < FLAGS_docs; begin += 128) {
    std::vector<zvec::Doc> docs;
    for (int i = begin; i < std::min(begin + 128, FLAGS_docs); ++i) {
      zvec::Doc doc;
      doc.set_pk("doc_" + std::to_string(i));
      if (!doc.set<int32_t>("bucket", i % 100) ||
          !doc.set<std::vector<float>>("vector",
                                       RandomVector(rng, FLAGS_dim))) {
        return false;
      }
      docs.push_back(std::move(doc));
    }
    auto inserted = collection->insert(docs);
    if (!inserted) {
      return Check(inserted.error());
    }
    for (const auto &status : inserted.value()) {
      if (!Check(status)) {
        return false;
      }
    }
  }
  return Check(collection->flush()) && Check(collection->close());
}

struct Measurements {
  std::vector<double> latency_us;
  size_t errors{0};
};

bool Measure() {
  auto opened = zvec::Collection::Open(FLAGS_path, {true, true});
  if (!opened) {
    return Check(opened.error());
  }
  auto collection = opened.value();
  auto schema = collection->schema();
  if (!schema) {
    return Check(schema.error());
  }
  const auto *field = schema->get_field("vector");
  if (field == nullptr) {
    return false;
  }
  std::mt19937 rng(123);
  std::vector<zvec::SearchQuery> queries(32);
  for (auto &query : queries) {
    query.topk_ = 10;
    query.output_fields_ = {"bucket"};
    if (FLAGS_mode != "scan") {
      auto vector = RandomVector(rng, field->dimension());
      query.target_.field_name_ = "vector";
      query.target_.set_vector(
          std::string(reinterpret_cast<const char *>(vector.data()),
                      vector.size() * sizeof(float)));
    }
    if (FLAGS_mode != "vector") {
      // No scalar index: this exercises the forward-filter subplan.
      query.filter_ = "bucket >= 50";
    }
  }

  // Validate and warm all query shapes before starting the timed region.
  // The digest makes before/after result differences visible for vector
  // queries.
  uint64_t digest = 14695981039346656037ULL;
  for (const auto &query : queries) {
    auto result = collection->query(query);
    if (!result) {
      return Check(result.error());
    }
    if (result->size() != 10) {
      std::fprintf(stderr, "Expected 10 results, got %zu\n", result->size());
      return false;
    }
    for (const auto &doc : result.value()) {
      for (unsigned char c : doc->pk()) {
        digest = (digest ^ c) * 1099511628211ULL;
      }
    }
  }

  std::vector<Measurements> measurements(FLAGS_threads);
  std::vector<std::thread> threads;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  Clock::time_point deadline;
  for (int t = 0; t < FLAGS_threads; ++t) {
    threads.emplace_back([&, t] {
      auto &measurement = measurements[t];
      measurement.latency_us.reserve(65536);
      ready.fetch_add(1);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      size_t sequence = t;
      while (Clock::now() < deadline) {
        const auto begin = Clock::now();
        auto result = collection->query(queries[sequence++ % queries.size()]);
        measurement.latency_us.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - begin)
                .count());
        if (!result || result->size() != 10) {
          ++measurement.errors;
        }
      }
    });
  }
  while (ready.load() != FLAGS_threads) {
    std::this_thread::yield();
  }
  const auto begin = Clock::now();
  deadline = begin + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(FLAGS_seconds));
  const auto cpu_begin = std::clock();
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }
  const double cpu_seconds =
      static_cast<double>(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
  const double elapsed =
      std::chrono::duration<double>(Clock::now() - begin).count();
  std::vector<double> latencies;
  size_t errors = 0;
  for (const auto &measurement : measurements) {
    latencies.insert(latencies.end(), measurement.latency_us.begin(),
                     measurement.latency_us.end());
    errors += measurement.errors;
  }
  if (latencies.empty()) {
    return false;
  }
  std::sort(latencies.begin(), latencies.end());
  std::printf(
      "{\"mode\":\"%s\",\"threads\":%d,\"queries\":%zu,\"errors\":%zu,"
      "\"qps\":%.2f,\"p50_us\":%.2f,\"p95_us\":%.2f,"
      "\"cpu_us_per_query\":%.2f,\"digest\":\"%zu\"}\n",
      FLAGS_mode.c_str(), FLAGS_threads, latencies.size(), errors,
      latencies.size() / elapsed, latencies[latencies.size() / 2],
      latencies[latencies.size() * 95 / 100],
      cpu_seconds * 1e6 / latencies.size(), static_cast<size_t>(digest));
  return errors == 0 && Check(collection->close());
}

}  // namespace

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_path.empty() || FLAGS_docs < 100 || FLAGS_segments < 1 ||
      FLAGS_docs % FLAGS_segments != 0 || FLAGS_dim < 1 || FLAGS_threads < 1 ||
      FLAGS_query_threads < 1 || FLAGS_seconds <= 0 ||
      (FLAGS_mode != "vector" && FLAGS_mode != "filtered" &&
       FLAGS_mode != "scan")) {
    std::fprintf(stderr, "Invalid benchmark arguments\n");
    return 1;
  }
  zvec::GlobalConfig::ConfigData config;
  config.query_thread_count = FLAGS_query_threads;
  config.query_thread_binding = false;
  config.memory_limit_bytes = 4ULL * 1024 * 1024 * 1024;
  if (!Check(zvec::GlobalConfig::Instance().initialize(config))) {
    return 1;
  }
  return (FLAGS_build ? Build() : Measure()) ? 0 : 1;
}
