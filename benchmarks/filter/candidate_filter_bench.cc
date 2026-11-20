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
#include <cstdio>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include "db/common/constants.h"

namespace {
using namespace zvec;
using Clock = std::chrono::steady_clock;

void check(const Status &status) {
  if (!status.ok()) throw std::runtime_error(status.message());
}

void prepare(const std::string &path, int rows) {
  CollectionSchema schema("candidate_filter_bench");
  schema.set_max_doc_count_per_segment(std::max(1000, rows + 1));
  schema.add_field(
      std::make_shared<FieldSchema>("value", DataType::INT32, false,
                                    std::make_shared<InvertIndexParams>(true)));
  schema.add_field(
      std::make_shared<FieldSchema>("label", DataType::STRING, false,
                                    std::make_shared<InvertIndexParams>(true)));
  schema.add_field(std::make_shared<FieldSchema>("bitmap", DataType::INT32));
  schema.add_field(std::make_shared<FieldSchema>(
      "vec", DataType::VECTOR_FP32, 4, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  // 只创建新目录，不删除或覆盖已有数据。不同版本打开同一个只读数据集。
  auto created =
      Collection::CreateAndOpen(path, schema, CollectionOptions{false, true});
  if (!created) throw std::runtime_error(created.error().message());
  auto collection = created.value();
  std::vector<Doc> batch;
  std::vector<std::string> deleted;
  for (int i = 0; i < rows; ++i) {
    Doc doc;
    doc.set_pk("pk_" + std::to_string(i));
    doc.set<int32_t>("value", i);
    doc.set<int32_t>("bitmap", i);
    doc.set<std::string>("label", "k" + std::to_string(i % 10));
    doc.set<std::vector<float>>("vec", {static_cast<float>(i), 1, 2, 3});
    if (i % 11 == 0) deleted.push_back(doc.pk());
    batch.push_back(std::move(doc));
    if (batch.size() == 512 || i + 1 == rows) {
      auto result = collection->insert(batch);
      if (!result) throw std::runtime_error(result.error().message());
      for (const auto &status : *result) check(status);
      batch.clear();
    }
  }
  // 先固化再删除，以保留约 9% tombstone 的候选过滤开销。
  check(collection->optimize());
  auto result = collection->delete_(deleted);
  if (!result) throw std::runtime_error(result.error().message());
  for (const auto &status : *result) check(status);
  check(collection->close());
}

struct Case {
  const char *name;
  std::string filter;
  bool label_filter;
};

// 只在计时前探测实际路径；兼容没有 evaluator 日志的 PR905 基线。
class BackendProbe final : public ailego::Logger {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  void log(int, const char *, int, const char *format, va_list args) override {
    char text[4096];
    vsnprintf(text, sizeof(text), format, args);
    const std::string message(text);
    if (message.find("Forward filter view:") != std::string::npos) ++views;
    if (message.find("Forward filter locked fetch:") != std::string::npos)
      ++locked;
    if (message.find("backend[stile]") != std::string::npos) ++stile;
  }
  std::atomic<int> views{0}, locked{0}, stile{0};
};

class ProbeScope {
 public:
  explicit ProbeScope(const std::shared_ptr<BackendProbe> &probe)
      : previous_(ailego::LoggerBroker::Register(probe)) {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_DEBUG);
  }
  ~ProbeScope() {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
    ailego::LoggerBroker::Register(previous_);
  }

 private:
  ailego::Logger::Pointer previous_;
};

std::set<std::string> expected_keys(int rows, bool label_filter) {
  std::vector<int> selected;
  for (int i = 0; i < rows; ++i)
    if (i % 11 != 0 && (!label_filter || (i % 10 >= 2 && i % 10 < 8)))
      selected.push_back(i);
  const double center = rows / 2 + 0.25;
  std::sort(selected.begin(), selected.end(), [&](int a, int b) {
    return (a - center) * (a - center) < (b - center) * (b - center);
  });
  std::set<std::string> keys;
  for (size_t i = 0; i < std::min(size_t{10}, selected.size()); ++i)
    keys.insert("pk_" + std::to_string(selected[i]));
  return keys;
}

void run(const std::string &path, int rows, int iterations, int threads,
         const std::string &selected_case) {
  auto opened = Collection::Open(path, CollectionOptions{true, true});
  if (!opened) throw std::runtime_error(opened.error().message());
  auto collection = opened.value();
  for (const auto &item :
       std::vector<Case>{{"numeric", "value >= 0", false},
                         {"string", "label >= 'k2' AND label < 'k8'", true},
                         {"compound",
                          "value >= 0 AND value < " + std::to_string(rows) +
                              " AND label >= 'k2' AND label < 'k8'",
                          true},
                         {"bitmap", "bitmap >= 0", false}}) {
    if (selected_case != "all" && selected_case != item.name) continue;
    SearchQuery query;
    query.topk_ = 10;
    query.filter_ = item.filter;
    query.output_fields_ = std::vector<std::string>{"value"};
    query.target_.field_name_ = "vec";
    const std::vector<float> vector{rows / 2 + 0.25f, 1, 2, 3};
    query.target_.set_vector(
        std::string(reinterpret_cast<const char *>(vector.data()),
                    vector.size() * sizeof(float)));
    const auto expected = expected_keys(rows, item.label_filter);
    auto execute = [&] {
      const auto start = Clock::now();
      auto result = collection->query(query);
      const double micros =
          std::chrono::duration<double, std::micro>(Clock::now() - start)
              .count();
      if (!result) throw std::runtime_error(result.error().message());
      std::set<std::string> actual;
      for (const auto &doc : *result) actual.insert(doc->pk());
      if (actual != expected || result->size() != expected.size())
        throw std::runtime_error("结果与 FLAT 距离及过滤条件的精确期望不符");
      return micros;
    };
    auto probe = std::make_shared<BackendProbe>();
    {
      ProbeScope scope(probe);
      execute();
    }
    const bool bitmap = std::string(item.name) == "bitmap";
    if (probe->locked != 0 || (bitmap ? probe->views != 0 : probe->views == 0))
      throw std::runtime_error(
          "执行计划未选择预期的 sealed IPC 候选/bitmap 路径");
    for (int i = 0; i < 10; ++i) execute();
    std::atomic<int> ready{0};
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<std::vector<double>>> workers;
    for (int t = 0; t < threads; ++t) {
      workers.push_back(std::async(std::launch::async, [&] {
        ++ready;
        gate.wait();
        std::vector<double> times;
        times.reserve(iterations);
        for (int i = 0; i < iterations; ++i) times.push_back(execute());
        return times;
      }));
    }
    while (ready.load() != threads) std::this_thread::yield();
    const auto begin = Clock::now();
    start.set_value();
    std::vector<double> times;
    for (auto &worker : workers) {
      auto local = worker.get();
      times.insert(times.end(), local.begin(), local.end());
    }
    const double seconds =
        std::chrono::duration<double>(Clock::now() - begin).count();
    std::sort(times.begin(), times.end());
    const auto percentile = [&](size_t percent) {
      return times[(times.size() * percent + 99) / 100 - 1];
    };
    std::cout << "{\"case\":\"" << item.name << "\",\"rows\":" << rows
              << ",\"view_prepares\":" << probe->views.load()
              << ",\"stile_prepares\":" << probe->stile.load()
              << ",\"threads\":" << threads << ",\"samples\":" << times.size()
              << ",\"p50_us\":" << percentile(50)
              << ",\"p95_us\":" << percentile(95)
              << ",\"p99_us\":" << percentile(99)
              << ",\"qps\":" << times.size() / seconds << "}" << std::endl;
  }
  check(collection->close());
}
}  // namespace

int main(int argc, char **argv) try {
  if (argc < 4 || argc > 7 ||
      (std::string(argv[1]) != "prepare" && std::string(argv[1]) != "run")) {
    std::cerr << "用法：candidate_filter_bench prepare|run PATH ROWS "
                 "[ITERATIONS=100] [THREADS=1] [CASE=all]\n"
              << "CASE: all、numeric、string、compound、bitmap。\n"
              << "对比时使用同一数据目录、参数和构建配置，基线为 PR905 "
                 "的实际提交。\n";
    return 2;
  }
  const int rows = std::stoi(argv[3]);
  const int iterations = argc > 4 ? std::stoi(argv[4]) : 100;
  const int threads = argc > 5 ? std::stoi(argv[5]) : 1;
  const std::string selected_case = argc > 6 ? argv[6] : "all";
  if (selected_case != "all" && selected_case != "numeric" &&
      selected_case != "string" && selected_case != "compound" &&
      selected_case != "bitmap")
    throw std::runtime_error("未知 CASE");
  if (rows < 32 || rows > 1000000 || iterations < 1 || threads < 1 ||
      threads > 64)
    throw std::runtime_error("参数超出范围");
  GlobalConfig::ConfigData config;
  config.memory_limit_bytes = MIN_MEMORY_LIMIT_BYTES;
  config.query_thread_count = 4;
  config.invert_to_forward_scan_ratio = 0;
  check(GlobalConfig::Instance().initialize(config));
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
  if (std::string(argv[1]) == "prepare")
    prepare(argv[2], rows);
  else
    run(argv[2], rows, iterations, threads, selected_case);
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
