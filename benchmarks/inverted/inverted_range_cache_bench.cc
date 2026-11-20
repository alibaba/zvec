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
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <mach/mach.h>
#include <malloc/malloc.h>
#endif
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/rocksdb_context.h"
#include "db/index/column/inverted_column/inverted_column_indexer.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"

namespace {

using Clock = std::chrono::steady_clock;
using zvec::CompareOp;
constexpr uint32_t kDocCount = 10000000;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_status(const zvec::Status &status, const std::string &operation) {
  require(status.ok(), operation + ": " + status.message());
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double thread_cpu_ms() {
  timespec ts{};
  require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0,
          "clock_gettime failed");
  return static_cast<double>(ts.tv_sec) * 1000.0 + ts.tv_nsec / 1e6;
}

std::string json_string(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      out << '\\' << static_cast<char>(c);
    } else if (c < 32) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<unsigned>(c) << std::dec;
    } else {
      out << static_cast<char>(c);
    }
  }
  out << '"';
  return out.str();
}

std::string value_bytes(int32_t value) {
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}

int32_t decode_int32_key(const std::string &key) {
  require(key.size() == sizeof(uint32_t), "unexpected checkpoint key size");
  uint32_t value = 0;
  for (unsigned char byte : key) {
    value = (value << 8) | byte;
  }
  value ^= 0x80000000U;
  int32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct MemorySnapshot {
  uint64_t rss_bytes{0};
  uint64_t physical_footprint_bytes{0};
  uint64_t peak_rss_bytes{0};
  uint64_t pool_external_bytes{0};

  static MemorySnapshot capture() {
    MemorySnapshot value;
#ifdef __APPLE__
    mach_task_basic_info_data_t basic{};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic),
                  &basic_count) == KERN_SUCCESS) {
      value.rss_bytes = basic.resident_size;
    }
    task_vm_info_data_t vm{};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm),
                  &vm_count) == KERN_SUCCESS) {
      value.physical_footprint_bytes = vm.phys_footprint;
    }
#else
    std::ifstream statm("/proc/self/statm");
    uint64_t pages = 0, resident = 0;
    if (statm >> pages >> resident) {
      value.rss_bytes = resident * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    }
#endif
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
      value.peak_rss_bytes = usage.ru_maxrss;
#else
      value.peak_rss_bytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
    }
    value.pool_external_bytes =
        zvec::ailego::MemoryLimitPool::get_instance().external_used();
    return value;
  }

  void write(std::ostream &out) const {
    out << "{\"rss_bytes\":" << rss_bytes
        << ",\"physical_footprint_bytes\":" << physical_footprint_bytes
        << ",\"peak_rss_bytes\":" << peak_rss_bytes
        << ",\"pool_external_bytes\":" << pool_external_bytes << '}';
  }
};

struct Samples {
  std::vector<double> wall_ms;
  std::vector<double> cpu_ms;

  static void write_values(std::ostream &out,
                           const std::vector<double> &values) {
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    auto quantile = [&](double q) {
      return sorted[static_cast<size_t>(std::ceil(q * sorted.size())) - 1];
    };
    out << "{\"mean_ms\":"
        << std::accumulate(values.begin(), values.end(), 0.0) / values.size()
        << ",\"p50_ms\":" << quantile(0.5) << ",\"p95_ms\":" << quantile(0.95)
        << ",\"samples_ms\":[";
    for (size_t i = 0; i < values.size(); ++i) {
      out << (i ? "," : "") << values[i];
    }
    out << "]}";
  }

  void write(std::ostream &out) const {
    out << "{\"wall\":";
    write_values(out, wall_ms);
    out << ",\"thread_cpu\":";
    write_values(out, cpu_ms);
    out << '}';
  }
};

struct Query {
  std::string name;
  CompareOp op;
  int32_t value;
  int target_match_pct{-1};
  int threshold_pct{-1};
};

const char *op_name(CompareOp op) {
  switch (op) {
    case CompareOp::LT:
      return "LT";
    case CompareOp::LE:
      return "LE";
    case CompareOp::GT:
      return "GT";
    case CompareOp::GE:
      return "GE";
    default:
      throw std::runtime_error("unsupported benchmark comparison");
  }
}

bool matches(int32_t value, const Query &query) {
  switch (query.op) {
    case CompareOp::LT:
      return value < query.value;
    case CompareOp::LE:
      return value <= query.value;
    case CompareOp::GT:
      return value > query.value;
    case CompareOp::GE:
      return value >= query.value;
    default:
      throw std::runtime_error("unsupported benchmark comparison");
  }
}

const std::vector<Query> kQueries = {
    {"gt_75pct", CompareOp::GT, 536870911},
    {"le_25pct", CompareOp::LE, 536870911},
    {"gt_50pct", CompareOp::GT, 1073741823},
    {"gt_25pct", CompareOp::GT, 1610612735},
    {"gt_90pct", CompareOp::GT, 214748364},
    {"le_1pct", CompareOp::LE, 21474836},
    {"le_5pct", CompareOp::LE, 107374182},
    {"le_10pct", CompareOp::LE, 214748364},
    {"le_50pct", CompareOp::LE, 1073741823},
    {"le_75pct", CompareOp::LE, 1610612735},
    {"le_90pct", CompareOp::LE, 1932735282},
    {"le_99pct", CompareOp::LE, 2126008810},
    {"le_3p2pct", CompareOp::LE, 68719476},
    {"le_6p4pct", CompareOp::LE, 137438953},
    {"le_28pct", CompareOp::LE, 601295421},
};

std::vector<Query> queries_for_suite(const std::string &suite) {
  if (suite == "default") {
    return kQueries;
  }
  std::vector<Query> queries;
  for (int match_pct :
       {1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99}) {
    for (const auto op : {CompareOp::LT, CompareOp::GT}) {
      const int threshold_pct =
          op == CompareOp::LT ? match_pct : 100 - match_pct;
      const auto threshold =
          static_cast<int32_t>(int64_t{2147483647} * threshold_pct / 100);
      queries.push_back(
          {std::string(op == CompareOp::LT ? "lt_match_" : "gt_match_") +
               std::to_string(match_pct) + "pct",
           op, threshold, match_pct, threshold_pct});
    }
  }
  return queries;
}

uint64_t verify(const zvec::InvertedSearchResult::Ptr &result,
                const std::vector<int32_t> &values, const Query &query) {
  require(result != nullptr, "null query result: " + query.name);
  // Compare all IDs with an independent value predicate. Iterating the
  // returned set avoids ten million separate Roaring membership lookups.
  auto iter = result->create_iterator();
  uint64_t count = 0;
  for (uint32_t id = 0; id < values.size(); ++id) {
    if (matches(values[id], query)) {
      if (!iter->valid() || iter->doc_id() != id) {
        throw std::runtime_error("membership mismatch in " + query.name +
                                 " at id " + std::to_string(id));
      }
      ++count;
      iter->next();
    }
  }
  require(!iter->valid(), "unexpected trailing ID in " + query.name);
  require(result->count() == count, "cardinality mismatch in " + query.name);
  return count;
}

struct Options {
  std::string distribution{"random"};
  std::string mode{"adaptive"};
  std::string suite{"default"};
  std::string data_root{"build-bitmap-release/benchmark-data"};
  int samples{100};
  int warmup{20};
  bool open_only{false};
  bool validate_boundaries{false};
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--open-only") {
      options.open_only = true;
    } else if (arg == "--validate-boundaries") {
      options.validate_boundaries = true;
    } else {
      require(i + 1 < argc, "missing argument for " + arg);
      const std::string value = argv[++i];
      if (arg == "--distribution")
        options.distribution = value;
      else if (arg == "--mode")
        options.mode = value;
      else if (arg == "--suite")
        options.suite = value;
      else if (arg == "--data-root")
        options.data_root = value;
      else if (arg == "--samples")
        options.samples = std::stoi(value);
      else if (arg == "--warmup")
        options.warmup = std::stoi(value);
      else
        throw std::runtime_error("unknown option " + arg);
    }
  }
  require(options.distribution == "random" || options.distribution == "ordered",
          "distribution must be random or ordered");
  require(options.mode == "off" || options.mode == "lower" ||
              options.mode == "adaptive",
          "mode must be off, lower or adaptive");
  require(options.suite == "default" || options.suite == "ratios",
          "suite must be default or ratios");
  require(options.samples > 0 && options.warmup >= 0,
          "invalid iteration count");
  return options;
}

// Diagnostic access to CRoaring 2.0.4 container layouts. No production query
// uses this code. In particular, avoid roaring_bitmap_statistics here because
// that implementation enumerates IDs for min/max/sum statistics.
struct ArrayLayout {
  int32_t cardinality;
  int32_t capacity;
  uint16_t *array;
};
struct BitsetLayout {
  int32_t cardinality;
  uint64_t *words;
};
struct RunLayout {
  int32_t n_runs;
  int32_t capacity;
  void *runs;
};

struct BitmapStats {
  uint64_t arrays{0}, bitsets{0}, runs{0};
  uint64_t array_ids{0}, run_count{0}, payload_bytes{0}, capacity_bytes{0};
  uint64_t malloc_bytes{0}, cardinality{0};

  static size_t allocation_size(const void *ptr) {
#ifdef __APPLE__
    return ptr ? malloc_size(ptr) : 0;
#else
    return 0;
#endif
  }

  void add(const roaring_bitmap_t *bitmap) {
    const auto &ra = bitmap->high_low_container;
    require((ra.flags & 3) == 0, "expected owned, non-COW cache bitmap");
    cardinality += roaring_bitmap_get_cardinality(bitmap);
    malloc_bytes += allocation_size(bitmap) + allocation_size(ra.containers);
    for (int i = 0; i < ra.size; ++i) {
      const void *container = ra.containers[i];
      malloc_bytes += allocation_size(container);
      switch (ra.typecodes[i]) {
        case 1: {
          const auto *value = reinterpret_cast<const BitsetLayout *>(container);
          ++bitsets;
          payload_bytes += 8192;
          capacity_bytes += 8192;
          malloc_bytes += allocation_size(value->words);
          break;
        }
        case 2: {
          const auto *value = reinterpret_cast<const ArrayLayout *>(container);
          ++arrays;
          array_ids += value->cardinality;
          payload_bytes += 2ULL * value->cardinality;
          capacity_bytes += 2ULL * value->capacity;
          malloc_bytes += allocation_size(value->array);
          break;
        }
        case 3: {
          const auto *value = reinterpret_cast<const RunLayout *>(container);
          ++runs;
          run_count += value->n_runs;
          payload_bytes += 4ULL * value->n_runs;
          capacity_bytes += 4ULL * value->capacity;
          malloc_bytes += allocation_size(value->runs);
          break;
        }
        default:
          throw std::runtime_error("unexpected cache container type");
      }
    }
  }

  void write(std::ostream &out) const {
    out << "{\"array\":" << arrays << ",\"bitset\":" << bitsets
        << ",\"run\":" << runs << ",\"array_ids\":" << array_ids
        << ",\"run_count\":" << run_count
        << ",\"cardinality_sum\":" << cardinality
        << ",\"payload_bytes\":" << payload_bytes
        << ",\"capacity_bytes\":" << capacity_bytes
        << ",\"malloc_bytes\":" << malloc_bytes
#ifdef __APPLE__
        << ",\"malloc_measurement_available\":true"
#else
        << ",\"malloc_measurement_available\":false"
#endif
        << '}';
  }
};

std::string cache_json(const zvec::InvertedColumnIndexer &indexer) {
  const auto *cache = indexer.range_cache();
  if (!cache) return "null";
  BitmapStats total;
  for (const auto &prefix : cache->prefixes) total.add(prefix.bitmap.get());
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "{\"build_ms\":" << cache->stats.build_ms
      << ",\"accounted_bytes\":" << cache->stats.accounted_bytes
      << ",\"build_reservation_bytes\":" << cache->stats.build_reservation_bytes
      << ",\"source_bytes\":" << cache->stats.source_bytes
      << ",\"total_cardinality\":" << cache->total_cardinality
      << ",\"range_count\":" << cache->buckets.size()
      << ",\"prefix_count\":" << cache->prefixes.size() << ",\"bitmap_stats\":";
  total.write(out);
  out << ",\"notes\":\"accounted_bytes is conservative pool accounting, not "
         "RSS. bitmap malloc_bytes counts Roaring allocations only, excluding "
         "cache vectors/keys and allocator retained pages.\""
      << ",\"prefixes\":[";
  for (size_t i = 0; i < cache->prefixes.size(); ++i) {
    const auto &prefix = cache->prefixes[i];
    require(prefix.end_bucket > 0 && prefix.end_bucket <= cache->buckets.size(),
            "invalid cached prefix boundary");
    const auto &bucket = cache->buckets[prefix.end_bucket - 1];
    BitmapStats stats;
    stats.add(prefix.bitmap.get());
    out << (i ? "," : "") << "{\"index\":" << i
        << ",\"end_bucket\":" << prefix.end_bucket
        << ",\"end_value\":" << decode_int32_key(bucket.end)
        << ",\"cardinality\":" << prefix.cardinality
        << ",\"copy_bytes\":" << prefix.copy_bytes << ",\"stats\":";
    stats.write(out);
    out << '}';
  }
  out << "]}";
  return out.str();
}

std::vector<int32_t> cache_boundaries(
    const zvec::InvertedColumnIndexer &indexer) {
  std::vector<int32_t> values;
  if (const auto *cache = indexer.range_cache()) {
    for (const auto &prefix : cache->prefixes) {
      values.push_back(
          decode_int32_key(cache->buckets[prefix.end_bucket - 1].end));
    }
  }
  return values;
}

const char *path_name(zvec::RangeQueryPath path) {
  switch (path) {
    case zvec::RangeQueryPath::Legacy:
      return "legacy";
    case zvec::RangeQueryPath::ScanLeft:
      return "scan_left";
    case zvec::RangeQueryPath::ScanRight:
      return "scan_right";
    case zvec::RangeQueryPath::PrefixAdd:
      return "prefix_add";
    case zvec::RangeQueryPath::PrefixSubtract:
      return "prefix_subtract";
  }
  throw std::runtime_error("unexpected range plan path");
}

std::string plan_json(const zvec::InvertedColumnIndexer &indexer,
                      const Query &query) {
  const auto plan = indexer.explain_range(value_bytes(query.value), query.op);
  const bool metadata_available = indexer.range_cache() != nullptr;
  std::ostringstream out;
  out << "{\"path\":" << json_string(path_name(plan.path))
      << ",\"metadata_available\":" << (metadata_available ? "true" : "false")
      << ",\"residual_estimation_note\":\"When available, bytes/cardinality "
         "include exact whole residual buckets plus the entire boundary "
         "bucket, not the actual matching boundary terms. Without cache "
         "metadata these values are unknown.\""
      << ",\"prefix_index\":";
  if (plan.prefix_index == SIZE_MAX)
    out << "null";
  else
    out << plan.prefix_index;
  out << ",\"range_begin\":";
  if (metadata_available)
    out << plan.range_begin;
  else
    out << "null";
  out << ",\"range_end\":";
  if (metadata_available)
    out << plan.range_end;
  else
    out << "null";
  out << ",\"residual_whole_buckets\":";
  if (metadata_available)
    out << plan.range_end - plan.range_begin;
  else
    out << "null";
  out << ",\"boundary_bucket\":";
  if (plan.boundary_bucket == SIZE_MAX)
    out << "null";
  else
    out << plan.boundary_bucket;
  out << ",\"left_inclusive\":" << (plan.left_inclusive ? "true" : "false")
      << ",\"flip\":" << (plan.flip ? "true" : "false")
      << ",\"residual_bytes\":";
  if (metadata_available)
    out << plan.residual_bytes;
  else
    out << "null";
  out << ",\"residual_cardinality\":";
  if (metadata_available)
    out << plan.residual_cardinality;
  else
    out << "null";
  out << ",\"estimated_work\":";
  if (metadata_available)
    out << plan.estimated_work;
  else
    out << "null";
  out << '}';
  return out.str();
}

}  // namespace

int main(int argc, char **argv) try {
  const Options options = parse_options(argc, argv);
  // Keep stdout exclusively JSON. Errors from this program go to stderr.
  zvec::ailego::LoggerBroker::Unregister();
  require(zvec::ailego::MemoryLimitPool::get_instance().init(1ULL << 30) == 0,
          "failed to initialize memory accounting");
  const auto before_open = MemorySnapshot::capture();
  const std::string path =
      options.data_root + "/" + options.distribution + "-10m-prefix0";
  zvec::RocksdbContext context;
  const auto open_begin = Clock::now();
  require_status(
      context.open(path, true,
                   std::make_shared<zvec::InvertedRocksdbValueMerger>()),
      "open DB");
  const auto rocksdb_open_end = Clock::now();
  zvec::FieldSchema field(
      "dataTime", zvec::DataType::INT32, true,
      std::make_shared<zvec::InvertIndexParams>(true, false));
  zvec::RangePrefixCacheOptions cache_options;
  if (options.mode == "off") cache_options.max_bytes = 0;
  cache_options.allow_subtraction = options.mode == "adaptive";
  auto indexer = zvec::InvertedColumnIndexer::CreateAndOpen(
      "range_cache_bench", field, context, true, cache_options);
  const auto open_end = Clock::now();
  require(indexer != nullptr, "failed to open column index");
  const auto after_open = MemorySnapshot::capture();
  const std::string cache_report = cache_json(*indexer);
  require(options.mode == "off" || indexer->range_cache() != nullptr,
          "requested cache was not admitted");

  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "{\"distribution\":" << json_string(options.distribution)
      << ",\"mode\":" << json_string(options.mode)
      << ",\"suite\":" << json_string(options.suite)
      << ",\"data_path\":" << json_string(path)
      << ",\"doc_count\":" << kDocCount << ",\"samples\":" << options.samples
      << ",\"warmup\":" << options.warmup
      << ",\"open_only\":" << (options.open_only ? "true" : "false")
      << ",\"open_ms\":" << elapsed_ms(open_begin, open_end)
      << ",\"rocksdb_open_ms\":" << elapsed_ms(open_begin, rocksdb_open_end)
      << ",\"column_open_ms\":" << elapsed_ms(rocksdb_open_end, open_end)
      << ",\"memory_before_open\":";
  before_open.write(out);
  out << ",\"memory_after_open\":";
  after_open.write(out);
  out << ",\"cache\":" << cache_report
      << ",\"notes\":\"Real single-column search; includes range/term reads, "
         "decode, merge, boundary handling and flip. Excludes result "
         "destruction, validation, ANN and SQL execution. Dataset generated "
         "with mt19937 seed 868, low 31 bits; ordered uses sorted values.\""
      << ",\"queries\":[";

  size_t verified_queries = 0, verified_boundary_queries = 0;
  if (!options.open_only) {
    std::vector<int32_t> values(kDocCount);
    std::mt19937 rng(868);
    for (auto &value : values)
      value = static_cast<int32_t>(rng() & 0x7fffffffU);
    if (options.distribution == "ordered")
      std::sort(values.begin(), values.end());
    const auto queries = queries_for_suite(options.suite);
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      const auto &query = queries[qi];
      const auto encoded = value_bytes(query.value);
      const std::string plan = plan_json(*indexer, query);
      for (int n = 0; n < options.warmup; ++n) {
        auto result = indexer->search(encoded, query.op);
        require(result != nullptr, "warmup failed: " + query.name);
      }
      Samples samples;
      samples.wall_ms.reserve(options.samples);
      samples.cpu_ms.reserve(options.samples);
      zvec::InvertedSearchResult::Ptr last;
      for (int n = 0; n < options.samples; ++n) {
        const double cpu_begin = thread_cpu_ms();
        const auto begin = Clock::now();
        auto result = indexer->search(encoded, query.op);
        const auto end = Clock::now();
        const double cpu_end = thread_cpu_ms();
        require(result != nullptr, "search failed: " + query.name);
        samples.wall_ms.push_back(elapsed_ms(begin, end));
        samples.cpu_ms.push_back(cpu_end - cpu_begin);
        // Free each result outside both timers; retain only the final sample
        // for validation, so earlier samples do not retain a second bitmap.
        if (n + 1 == options.samples) last = std::move(result);
      }
      const auto count = verify(last, values, query);
      ++verified_queries;
      out << (qi ? "," : "") << "{\"name\":" << json_string(query.name)
          << ",\"op\":" << json_string(op_name(query.op))
          << ",\"value\":" << query.value << ",\"matched\":" << count
          << ",\"target_match_pct\":" << query.target_match_pct
          << ",\"threshold_pct\":" << query.threshold_pct
          << ",\"plan\":" << plan << ",\"timing\":";
      samples.write(out);
      out << '}';
    }
    if (options.validate_boundaries) {
      std::set<int32_t> thresholds;
      for (int32_t boundary : cache_boundaries(*indexer)) {
        thresholds.insert(boundary);
        if (boundary > std::numeric_limits<int32_t>::min())
          thresholds.insert(boundary - 1);
        if (boundary < std::numeric_limits<int32_t>::max())
          thresholds.insert(boundary + 1);
      }
      for (int32_t threshold : thresholds) {
        for (CompareOp op :
             {CompareOp::LT, CompareOp::LE, CompareOp::GT, CompareOp::GE}) {
          Query query{
              "checkpoint_" + std::to_string(threshold) + "_" + op_name(op), op,
              threshold};
          verify(indexer->search(value_bytes(threshold), op), values, query);
          ++verified_boundary_queries;
        }
      }
    }
  }
  out << "],\"verified_queries\":" << verified_queries
      << ",\"verified_boundary_queries\":" << verified_boundary_queries
      << ",\"memory_after_work\":";
  MemorySnapshot::capture().write(out);
  indexer.reset();
  require_status(context.close(), "close DB");
  out << ",\"memory_after_close\":";
  MemorySnapshot::capture().write(out);
  out << "}\n";
  std::cout << out.str();
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
