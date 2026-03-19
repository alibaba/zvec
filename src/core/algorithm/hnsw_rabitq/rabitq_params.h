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
#pragma once

#include <string>

namespace zvec {
namespace core {

// Local metric type enum that mirrors rabitqlib::MetricType,
// without exposing rabitqlib headers to consumers of this file.
enum class RabitqMetricType {
  kL2 = 0,
  kIP = 1,
};

// RaBitQ Converter parameters
static const std::string PARAM_RABITQ_NUM_CLUSTERS(
    "proxima.rabitq.num_clusters");
static const std::string PARAM_RABITQ_TOTAL_BITS("proxima.rabitq.total_bits");
static const std::string PARAM_RABITQ_METRIC_NAME("proxima.rabitq.metric_name");
static const std::string PARAM_RABITQ_ROTATOR_TYPE(
    "proxima.rabitq.rotator.type");
static const std::string PARAM_RABITQ_SAMPLE_COUNT(
    "proxima.rabitq.sample_count");

// Default values
constexpr size_t kDefaultNumClusters = 16;
// 4-bit, 5-bit, and 7-bit quantization typically achieve 90%, 95%, and 99%
// recall, respectively—without accessing raw vectors for reranking
constexpr size_t kDefaultRabitqTotalBits = 7;

constexpr int kMinRabitqDimSize = 64;
constexpr int kMaxRabitqDimSize = 4095;


}  // namespace core
}  // namespace zvec
