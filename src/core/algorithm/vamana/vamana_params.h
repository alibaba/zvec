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

// Builder parameters
static const std::string PARAM_VAMANA_BUILDER_THREAD_COUNT(
    "proxima.vamana.builder.thread_count");
static const std::string PARAM_VAMANA_BUILDER_MEMORY_QUOTA(
    "proxima.vamana.builder.memory_quota");
static const std::string PARAM_VAMANA_BUILDER_MAX_DEGREE(
    "proxima.vamana.builder.max_degree");
static const std::string PARAM_VAMANA_BUILDER_SEARCH_LIST_SIZE(
    "proxima.vamana.builder.search_list_size");
static const std::string PARAM_VAMANA_BUILDER_ALPHA(
    "proxima.vamana.builder.alpha");
static const std::string PARAM_VAMANA_BUILDER_MAX_OCCLUSION_SIZE(
    "proxima.vamana.builder.max_occlusion_size");

// Searcher parameters
static const std::string PARAM_VAMANA_SEARCHER_SEARCH_LIST_SIZE(
    "proxima.vamana.searcher.search_list_size");
static const std::string PARAM_VAMANA_SEARCHER_BRUTE_FORCE_THRESHOLD(
    "proxima.vamana.searcher.brute_force_threshold");

// Streamer parameters
static const std::string PARAM_VAMANA_STREAMER_MAX_DEGREE(
    "proxima.vamana.streamer.max_degree");
static const std::string PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE(
    "proxima.vamana.streamer.search_list_size");
static const std::string PARAM_VAMANA_STREAMER_ALPHA(
    "proxima.vamana.streamer.alpha");
static const std::string PARAM_VAMANA_STREAMER_MAX_OCCLUSION_SIZE(
    "proxima.vamana.streamer.max_occlusion_size");
static const std::string PARAM_VAMANA_STREAMER_EF("proxima.vamana.streamer.ef");
static const std::string PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD(
    "proxima.vamana.streamer.brute_force_threshold");
static const std::string PARAM_VAMANA_STREAMER_MAX_SCAN_RATIO(
    "proxima.vamana.streamer.max_scan_ratio");
static const std::string PARAM_VAMANA_STREAMER_DOCS_HARD_LIMIT(
    "proxima.vamana.streamer.docs_hard_limit");
static const std::string PARAM_VAMANA_STREAMER_DOCS_SOFT_LIMIT(
    "proxima.vamana.streamer.docs_soft_limit");
static const std::string PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE(
    "proxima.vamana.streamer.max_index_size");
static const std::string PARAM_VAMANA_STREAMER_CHUNK_SIZE(
    "proxima.vamana.streamer.chunk_size");
static const std::string PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE(
    "proxima.vamana.streamer.get_vector_enable");
static const std::string PARAM_VAMANA_STREAMER_FORCE_PADDING_RESULT_ENABLE(
    "proxima.vamana.streamer.force_padding_result_enable");
static const std::string PARAM_VAMANA_STREAMER_USE_ID_MAP(
    "proxima.vamana.streamer.use_id_map");
static const std::string PARAM_VAMANA_STREAMER_MAX_SCAN_LIMIT(
    "proxima.vamana.streamer.max_scan_limit");
static const std::string PARAM_VAMANA_STREAMER_MIN_SCAN_LIMIT(
    "proxima.vamana.streamer.min_scan_limit");
static const std::string PARAM_VAMANA_STREAMER_SATURATE_GRAPH(
    "proxima.vamana.streamer.saturate_graph");
static const std::string PARAM_VAMANA_STREAMER_USE_CONTIGUOUS_MEMORY(
    "proxima.vamana.streamer.use_contiguous_memory");

}  // namespace core
}  // namespace zvec
