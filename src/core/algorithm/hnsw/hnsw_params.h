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

static const std::string PARAM_HNSW_BUILDER_THREAD_COUNT(
    "proxima.hnsw.builder.thread_count");
static const std::string PARAM_HNSW_BUILDER_MEMORY_QUOTA(
    "proxima.hnsw.builder.memory_quota");
static const std::string PARAM_HNSW_BUILDER_EFCONSTRUCTION(
    "proxima.hnsw.builder.efconstruction");
static const std::string PARAM_HNSW_BUILDER_SCALING_FACTOR(
    "proxima.hnsw.builder.scaling_factor");
static const std::string PARAM_HNSW_BUILDER_CHECK_INTERVAL_SECS(
    "proxima.hnsw.builder.check_interval_secs");
static const std::string PARAM_HNSW_BUILDER_NEIGHBOR_PRUNE_MULTIPLIER(
    "proxima.hnsw.builder.neighbor_prune_multiplier");
static const std::string PARAM_HNSW_BUILDER_MIN_NEIGHBOR_COUNT(
    "proxima.hnsw.builder.min_neighbor_count");
static const std::string PARAM_HNSW_BUILDER_MAX_NEIGHBOR_COUNT(
    "proxima.hnsw.builder.max_neighbor_count");
static const std::string PARAM_HNSW_BUILDER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER(
    "proxima.hnsw.builder.l0_max_neighbor_count_multiplier");

static const std::string PARAM_HNSW_SEARCHER_EF("proxima.hnsw.searcher.ef");
static const std::string PARAM_HNSW_SEARCHER_BRUTE_FORCE_THRESHOLD(
    "proxima.hnsw.searcher.brute_force_threshold");
static const std::string PARAM_HNSW_SEARCHER_NEIGHBORS_IN_MEMORY_ENABLE(
    "proxima.hnsw.searcher.neighbors_in_memory_enable");
static const std::string PARAM_HNSW_SEARCHER_MAX_SCAN_RATIO(
    "proxima.hnsw.searcher.max_scan_ratio");
static const std::string PARAM_HNSW_SEARCHER_CHECK_CRC_ENABLE(
    "proxima.hnsw.searcher.check_crc_enable");
static const std::string PARAM_HNSW_SEARCHER_VISIT_BLOOMFILTER_ENABLE(
    "proxima.hnsw.searcher.visit_bloomfilter_enable");
static const std::string PARAM_HNSW_SEARCHER_VISIT_BLOOMFILTER_NEGATIVE_PROB(
    "proxima.hnsw.searcher.visit_bloomfilter_negative_prob");
static const std::string PARAM_HNSW_SEARCHER_FORCE_PADDING_RESULT_ENABLE(
    "proxima.hnsw.searcher.force_padding_result_enable");

static const std::string PARAM_HNSW_STREAMER_MAX_SCAN_RATIO(
    "proxima.hnsw.streamer.max_scan_ratio");
static const std::string PARAM_HNSW_STREAMER_MIN_SCAN_LIMIT(
    "proxima.hnsw.streamer.min_scan_limit");
static const std::string PARAM_HNSW_STREAMER_MAX_SCAN_LIMIT(
    "proxima.hnsw.streamer.max_scan_limit");
static const std::string PARAM_HNSW_STREAMER_EF("proxima.hnsw.streamer.ef");
static const std::string PARAM_HNSW_STREAMER_EFCONSTRUCTION(
    "proxima.hnsw.streamer.efconstruction");
static const std::string PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT(
    "proxima.hnsw.streamer.max_neighbor_count");
static const std::string PARAM_HNSW_STREAMER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER(
    "proxima.hnsw.streamer.l0_max_neighbor_count_multiplier");
static const std::string PARAM_HNSW_STREAMER_SCALING_FACTOR(
    "proxima.hnsw.streamer.scaling_factor");
static const std::string PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD(
    "proxima.hnsw.streamer.brute_force_threshold");
static const std::string PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT(
    "proxima.hnsw.streamer.docs_hard_limit");
static const std::string PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT(
    "proxima.hnsw.streamer.docs_soft_limit");
static const std::string PARAM_HNSW_STREAMER_MAX_INDEX_SIZE(
    "proxima.hnsw.streamer.max_index_size");
static const std::string PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_ENABLE(
    "proxima.hnsw.streamer.visit_bloomfilter_enable");
static const std::string PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB(
    "proxima.hnsw.streamer.visit_bloomfilter_negative_prob");
static const std::string PARAM_HNSW_STREAMER_CHECK_CRC_ENABLE(
    "proxima.hnsw.streamer.check_crc_enable");
static const std::string PARAM_HNSW_STREAMER_NEIGHBOR_PRUNE_MULTIPLIER(
    "proxima.hnsw.streamer.neighbor_prune_multiplier");
static const std::string PARAM_HNSW_STREAMER_CHUNK_SIZE(
    "proxima.hnsw.streamer.chunk_size");
static const std::string PARAM_HNSW_STREAMER_FILTER_SAME_KEY(
    "proxima.hnsw.streamer.filter_same_key");
static const std::string PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE(
    "proxima.hnsw.streamer.get_vector_enable");
static const std::string PARAM_HNSW_STREAMER_MIN_NEIGHBOR_COUNT(
    "proxima.hnsw.streamer.min_neighbor_count");
static const std::string PARAM_HNSW_STREAMER_FORCE_PADDING_RESULT_ENABLE(
    "proxima.hnsw.streamer.force_padding_result_enable");
static const std::string PARAM_HNSW_STREAMER_ESTIMATE_DOC_COUNT(
    "proxima.hnsw.streamer.estimate_doc_count");
static const std::string PARAM_HNSW_STREAMER_USE_ID_MAP(
    "proxima.hnsw.streamer.use_id_map");

static const std::string PARAM_HNSW_REDUCER_WORKING_PATH(
    "proxima.hnsw.reducer.working_path");
static const std::string PARAM_HNSW_REDUCER_NUM_OF_ADD_THREADS(
    "proxima.hnsw.reducer.num_of_add_threads");
static const std::string PARAM_HNSW_REDUCER_INDEX_NAME(
    "proxima.hnsw.reducer.index_name");
static const std::string PARAM_HNSW_REDUCER_EFCONSTRUCTION(
    "proxima.hnsw.reducer.efconstruction");

}  // namespace core
}  // namespace zvec
