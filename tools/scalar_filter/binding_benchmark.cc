// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

// Isolate query preparation from scalar execution. Build against the same
// ZVEC_ENABLE_SCALAR_JIT library being measured by benchmark.cc. This program
// does not run HNSW: its candidates are uniformly distributed random row IDs.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include "db/sqlengine/planner/scalar_doc_filter.h"

#ifndef ZVEC_ENABLE_SCALAR_JIT
#error "binding_benchmark requires ZVEC_ENABLE_SCALAR_JIT"
#endif

namespace {

namespace cp = arrow::compute;
using Clock = std::chrono::steady_clock;
using zvec::sqlengine::ScalarDocFilter;

constexpr int64_t kRows = 10'000'000;
constexpr int kChunks = 2645;
constexpr size_t kCandidates = 1777;
constexpr size_t kWarmupQueries = 16;

template <typename T>
T Unwrap(arrow::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return std::move(result).ValueOrDie();
}

void Check(const arrow::Status &status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}

uint64_t Mix(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

int32_t Value(uint64_t row) {
  // Signed 31-bit values give approximately 25% passage at threshold 2^29-1.
  return static_cast<int32_t>(Mix(row) & 0x7fffffffU) - (int32_t{1} << 30);
}

bool Valid(uint64_t row) {
  return Mix(row ^ 0xa5a5a5a5U) % 20 != 0;
}

double Nanoseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::nano>(end - start).count();
}

struct Inputs {
  std::shared_ptr<arrow::Schema> schema =
      arrow::schema({arrow::field("dataTime", arrow::int32(), true)});
  std::shared_ptr<arrow::Buffer> data;
  std::shared_ptr<arrow::Buffer> validity;
  std::vector<uint64_t> candidates;
  std::vector<int32_t> thresholds;
  std::vector<cp::Expression> expressions;
};

Inputs MakeInputs(size_t iterations) {
  Inputs input;
  input.data = Unwrap(arrow::AllocateBuffer(kRows * sizeof(int32_t)));
  input.validity = Unwrap(arrow::AllocateBitmap(kRows));
  std::memset(input.validity->mutable_data(), 0xff, input.validity->size());
  input.candidates.reserve(iterations * kCandidates);
  input.thresholds.reserve(iterations);
  input.expressions.reserve(iterations);
  auto *values = reinterpret_cast<int32_t *>(input.data->mutable_data());
  auto *validity = input.validity->mutable_data();
  for (size_t query = 0; query < iterations; ++query) {
    const auto jitter = static_cast<int32_t>(Mix(query + 123) & 0x07ffffffU) -
                        (int32_t{1} << 26);
    const int32_t threshold = 536870911 + jitter;
    input.thresholds.push_back(threshold);
    // The production Make API takes an already bound Arrow expression. Binding
    // it and generating random candidates are deliberately outside the timer.
    input.expressions.push_back(
        Unwrap(cp::greater(cp::field_ref("dataTime"), cp::literal(threshold))
                   .Bind(*input.schema)));
    for (size_t candidate = 0; candidate < kCandidates; ++candidate) {
      const auto row =
          Mix(query * kCandidates + candidate + 0x12345678ULL) % kRows;
      input.candidates.push_back(row);
      // Initialize only rows that will be read. Repeated rows get identical
      // values; all initialization and page faults here precede measurement.
      values[row] = Value(row);
      if (!Valid(row)) validity[row >> 3] &= ~(uint8_t{1} << (row & 7));
    }
  }
  return input;
}

std::vector<int64_t> ChunkSizes(bool irregular) {
  std::vector<int64_t> sizes(kChunks);
  if (!irregular) {
    const int64_t stride = (kRows + kChunks - 1) / kChunks;
    std::fill(sizes.begin(), sizes.end(), stride);
    sizes.back() = kRows - stride * (kChunks - 1);
    return sizes;
  }
  std::vector<int64_t> weights(kChunks);
  int64_t total = 0;
  for (int i = 0; i < kChunks; ++i) {
    weights[i] = 1 + Mix(i + 6789) % 16;
    total += weights[i];
  }
  int64_t previous_end = 0;
  int64_t cumulative = 0;
  for (int i = 0; i < kChunks; ++i) {
    cumulative += weights[i];
    const int64_t end = kRows * cumulative / total;
    sizes[i] = end - previous_end;
    previous_end = end;
  }
  return sizes;
}

std::shared_ptr<arrow::Table> MakeSnapshot(const Inputs &input, bool irregular,
                                           bool has_validity) {
  arrow::ArrayVector chunks;
  chunks.reserve(kChunks);
  int64_t begin = 0;
  for (const auto size : ChunkSizes(irregular)) {
    // Each chunk has its own buffer view and a possible nonzero Array offset.
    // Retain shared allocations while exercising slice/bitmap offset handling.
    const int64_t offset = begin & 7;
    const int64_t aligned_begin = begin - offset;
    auto data = arrow::SliceBuffer(input.data, aligned_begin * sizeof(int32_t),
                                   (size + offset) * sizeof(int32_t));
    auto validity = has_validity
                        ? arrow::SliceBuffer(input.validity, aligned_begin / 8,
                                             (size + offset + 7) / 8)
                        : nullptr;
    chunks.push_back(arrow::MakeArray(
        arrow::ArrayData::Make(arrow::int32(), size, {validity, data},
                               has_validity ? -1 : 0, offset)));
    begin += size;
  }
  if (begin != kRows) throw std::runtime_error("Incomplete chunk layout");
  auto column =
      std::make_shared<arrow::ChunkedArray>(std::move(chunks), arrow::int32());
  auto snapshot = arrow::Table::Make(input.schema, {column});
  Check(snapshot->Validate());
  return snapshot;
}

uint64_t Verify(const Inputs &input, size_t query, bool has_validity,
                const std::vector<uint8_t> &results) {
  uint64_t passed = 0;
  for (size_t candidate = 0; candidate < kCandidates; ++candidate) {
    const auto row = input.candidates[query * kCandidates + candidate];
    const bool expected =
        (!has_validity || Valid(row)) && Value(row) > input.thresholds[query];
    if (results[candidate] != static_cast<uint8_t>(expected)) {
      throw std::runtime_error("Wrong result at query " +
                               std::to_string(query) + ", row " +
                               std::to_string(row));
    }
    passed += results[candidate];
  }
  return passed;
}

void Evaluate(const ScalarDocFilter &filter, const uint64_t *ids,
              std::vector<uint8_t> *results) {
  for (size_t i = 0; i < kCandidates; ++i) {
    const auto result = filter.Evaluate(ids[i]);
    (*results)[i] = result.has_value() ? static_cast<uint8_t>(*result) : 2;
  }
}

void Run(const Inputs &input, size_t iterations, bool irregular,
         bool has_validity) {
  const auto snapshot = MakeSnapshot(input, irregular, has_validity);
  std::vector<uint8_t> results(kCandidates);
  for (size_t warmup = 0; warmup < kWarmupQueries; ++warmup) {
    const auto query = warmup % iterations;
    auto filter =
        Unwrap(ScalarDocFilter::Make(input.expressions[query], snapshot));
    if (filter->num_batches() != kChunks)
      throw std::runtime_error("Unexpected aligned batch count");
    if (filter->Evaluate(kRows).has_value() ||
        filter->Evaluate(std::numeric_limits<uint64_t>::max()).has_value())
      throw std::runtime_error("Out-of-range row unexpectedly accepted");
    Evaluate(*filter, input.candidates.data() + query * kCandidates, &results);
    Verify(input, query, has_validity, results);
  }

  double validate_ns = 0;
  for (size_t query = 0; query < iterations; ++query) {
    const auto start = Clock::now();
    const auto status = snapshot->Validate();
    const auto end = Clock::now();
    validate_ns += Nanoseconds(start, end);
    Check(status);
  }

  const auto cache_before = ScalarDocFilter::CacheStats();
  double make_ns = 0;
  double evaluate_ns = 0;
  double destroy_ns = 0;
  uint64_t passed = 0;
  for (size_t query = 0; query < iterations; ++query) {
    const auto make_start = Clock::now();
    auto made = ScalarDocFilter::Make(input.expressions[query], snapshot);
    const auto make_end = Clock::now();
    auto filter = Unwrap(std::move(made));
    make_ns += Nanoseconds(make_start, make_end);

    const auto *ids = input.candidates.data() + query * kCandidates;
    const auto evaluate_start = Clock::now();
    Evaluate(*filter, ids, &results);
    const auto evaluate_end = Clock::now();
    evaluate_ns += Nanoseconds(evaluate_start, evaluate_end);
    // Consume and check every result outside the measured execution interval.
    passed += Verify(input, query, has_validity, results);

    const auto destroy_start = Clock::now();
    filter.reset();
    const auto destroy_end = Clock::now();
    destroy_ns += Nanoseconds(destroy_start, destroy_end);
  }
  const auto cache_after = ScalarDocFilter::CacheStats();
  const auto compilations =
      cache_after.compilations - cache_before.compilations;
  const auto hits = cache_after.hits - cache_before.hits;
  if (compilations != 0 || hits != iterations)
    throw std::runtime_error(
        "Timed Make calls did not all reuse the warm cache");

  const double ns_per_query = static_cast<double>(iterations);
  std::cout << std::fixed << std::setprecision(3)
            << "{\"event\":\"binding_benchmark\",\"rows\":" << kRows
            << ",\"chunks\":" << kChunks << ",\"layout\":\""
            << (irregular ? "irregular" : "uniform")
            << "\",\"validity_bitmap\":" << (has_validity ? "true" : "false")
            << ",\"iterations\":" << iterations
            << ",\"candidates_per_query\":" << kCandidates
            << ",\"candidate_distribution\":\"uniform_random\""
            << ",\"threshold_mode\":\"varying_runtime_parameter\""
            << ",\"arrow_expression_bind_timed\":false"
            << ",\"validate_us\":" << validate_ns / ns_per_query / 1000.0
            << ",\"make_us\":" << make_ns / ns_per_query / 1000.0
            << ",\"destroy_us\":" << destroy_ns / ns_per_query / 1000.0
            << ",\"make_plus_destroy_us\":"
            << (make_ns + destroy_ns) / ns_per_query / 1000.0
            << ",\"evaluate_1777_us\":" << evaluate_ns / ns_per_query / 1000.0
            << ",\"evaluate_ns_per_id\":"
            << evaluate_ns / ns_per_query / kCandidates
            << ",\"timed_compilations\":" << compilations
            << ",\"timed_cache_hits\":" << hits
            << ",\"verified\":true,\"passed_candidates\":" << passed << "}\n";
}

size_t Iterations(int argc, char **argv) {
  std::string value = "1000";
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "Usage: binding_benchmark [iterations | --iterations N]\n";
    return 0;
  }
  if (argc == 2)
    value = argv[1];
  else if (argc == 3 && std::string(argv[1]) == "--iterations")
    value = argv[2];
  else if (argc != 1)
    throw std::runtime_error(
        "Usage: binding_benchmark [iterations | --iterations N]");
  size_t consumed = 0;
  const auto count = std::stoull(value, &consumed);
  if (consumed != value.size() || count == 0 || count > 100000)
    throw std::runtime_error("iterations must be in [1, 100000]");
  return static_cast<size_t>(count);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const auto iterations = Iterations(argc, argv);
    if (!iterations) return 0;
    Check(cp::Initialize());
    const auto input = MakeInputs(iterations);
    for (bool irregular : {false, true})
      for (bool has_validity : {false, true})
        Run(input, iterations, irregular, has_validity);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "binding_benchmark: " << error.what() << '\n';
    return 1;
  }
}
