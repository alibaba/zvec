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

// IVF RabitQ specific params
static const std::string PARAM_IVF_RABITQ_NLIST("proxima.ivf_rabitq.nlist");
static const std::string PARAM_IVF_RABITQ_NPROBE("proxima.ivf_rabitq.nprobe");
static const std::string PARAM_IVF_RABITQ_SCAN_RATIO(
    "proxima.ivf_rabitq.scan_ratio");
static const std::string PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD(
    "proxima.ivf_rabitq.brute_force_threshold");
static const std::string PARAM_IVF_RABITQ_BUILDER_THREAD_COUNT(
    "proxima.ivf_rabitq.builder.thread_count");

// Segment IDs
static const std::string IVF_RABITQ_HEADER_SEG_ID{"ivf_rabitq.header"};
static const std::string IVF_RABITQ_BATCH_DATA_SEG_ID{"ivf_rabitq.batch_data"};
static const std::string IVF_RABITQ_EX_DATA_SEG_ID{"ivf_rabitq.ex_data"};
static const std::string IVF_RABITQ_CLUSTER_META_SEG_ID{
    "ivf_rabitq.cluster_meta"};
static const std::string IVF_RABITQ_KEYS_SEG_ID{"ivf_rabitq.keys"};
static const std::string IVF_RABITQ_MAPPING_SEG_ID{"ivf_rabitq.mapping"};
static const std::string IVF_RABITQ_CENTROID_SEG_ID{"ivf_rabitq.centroid"};

// Defaults
constexpr uint32_t kDefaultIvfRabitqNlist = 1024;
constexpr uint32_t kDefaultIvfRabitqNprobe = 10;
constexpr float kDefaultIvfRabitqScanRatio = 0.1f;
constexpr uint32_t kDefaultIvfRabitqBruteForceThreshold = 1000;

}  // namespace core
}  // namespace zvec
