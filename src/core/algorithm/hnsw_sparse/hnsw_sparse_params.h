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

static const std::string PARAM_HNSW_SPARSE_BUILDER_THREAD_COUNT(
    "proxima.hnsw.sparse_builder.thread_count");
static const std::string PARAM_HNSW_SPARSE_BUILDER_MEMORY_QUOTA(
    "proxima.hnsw.sparse_builder.memory_quota");
static const std::string PARAM_HNSW_SPARSE_BUILDER_EFCONSTRUCTION(
    "proxima.hnsw.sparse_builder.efconstruction");
static const std::string PARAM_HNSW_SPARSE_BUILDER_SCALING_FACTOR(
    "proxima.hnsw.sparse_builder.scaling_factor");
static const std::string PARAM_HNSW_SPARSE_BUILDER_CHECK_INTERVAL_SECS(
    "proxima.hnsw.sparse_builder.check_interval_secs");
static const std::string PARAM_HNSW_SPARSE_BUILDER_NEIGHBOR_PRUNE_MULTIPLIER(
    "proxima.hnsw.sparse_builder.neighbor_prune_multiplier");
static const std::string PARAM_HNSW_SPARSE_BUILDER_MIN_NEIGHBOR_COUNT(
    "proxima.hnsw.sparse_builder.min_neighbor_count");
static const std::string PARAM_HNSW_SPARSE_BUILDER_MAX_NEIGHBOR_COUNT(
    "proxima.hnsw.sparse_builder.max_neighbor_count");
static const std::string
    PARAM_HNSW_SPARSE_BUILDER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER(
        "proxima.hnsw.sparse_builder.l0_max_neighbor_count_multiplier");

static const std::string PARAM_HNSW_SPARSE_SEARCHER_EF(
    "proxima.hnsw.sparse_searcher.ef");
static const std::string PARAM_HNSW_SPARSE_SEARCHER_BRUTE_FORCE_THRESHOLD(
    "proxima.hnsw.sparse_searcher.brute_force_threshold");
static const std::string PARAM_HNSW_SPARSE_SEARCHER_NEIGHBORS_IN_MEMORY_ENABLE(
    "proxima.hnsw.sparse_searcher.neighbors_in_memory_enable");
static const std::string PARAM_HNSW_SPARSE_SEARCHER_MAX_SCAN_RATIO(
    "proxima.hnsw.sparse_searcher.max_scan_ratio");
static const std::string PARAM_HNSW_SPARSE_SEARCHER_CHECK_CRC_ENABLE(
    "proxima.hnsw.sparse_searcher.check_crc_enable");
static const std::string PARAM_HNSW_SPARSE_SEARCHER_VISIT_BLOOMFILTER_ENABLE(
    "proxima.hnsw.sparse_searcher.visit_bloomfilter_enable");
static const std::string
    PARAM_HNSW_SPARSE_SEARCHER_VISIT_BLOOMFILTER_NEGATIVE_PROB(
        "proxima.hnsw.sparse_searcher.visit_bloomfilter_negative_prob");
static const std::string PARAM_HNSW_SPARSE_SEARCHER_FORCE_PADDING_RESULT_ENABLE(
    "proxima.hnsw.sparse_searcher.force_padding_result_enable");
static const std::string PARAM_HNSW_SPARSE_SEARCHER_QUERY_FILTERING_RATIO(
    "proxima.hnsw.sparse_searcher.query_filtering_ratio");

static const std::string PARAM_HNSW_SPARSE_STREAMER_MAX_SCAN_RATIO(
    "proxima.hnsw.sparse_streamer.max_scan_ratio");
static const std::string PARAM_HNSW_SPARSE_STREAMER_MIN_SCAN_LIMIT(
    "proxima.hnsw.sparse_streamer.min_scan_limit");
static const std::string PARAM_HNSW_SPARSE_STREAMER_MAX_SCAN_LIMIT(
    "proxima.hnsw.sparse_streamer.max_scan_limit");
static const std::string PARAM_HNSW_SPARSE_STREAMER_EF(
    "proxima.hnsw.sparse_streamer.ef");
static const std::string PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION(
    "proxima.hnsw.sparse_streamer.efconstruction");
static const std::string PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT(
    "proxima.hnsw.sparse_streamer.max_neighbor_count");
static const std::string
    PARAM_HNSW_SPARSE_STREAMER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER(
        "proxima.hnsw.sparse_streamer.l0_max_neighbor_count_multiplier");
static const std::string PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR(
    "proxima.hnsw.sparse_streamer.scaling_factor");
static const std::string PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD(
    "proxima.hnsw.sparse_streamer.brute_force_threshold");
static const std::string PARAM_HNSW_SPARSE_STREAMER_DOCS_HARD_LIMIT(
    "proxima.hnsw.sparse_streamer.docs_hard_limit");
static const std::string PARAM_HNSW_SPARSE_STREAMER_DOCS_SOFT_LIMIT(
    "proxima.hnsw.sparse_streamer.docs_soft_limit");
static const std::string PARAM_HNSW_SPARSE_STREAMER_MAX_INDEX_SIZE(
    "proxima.hnsw.sparse_streamer.max_index_size");
static const std::string PARAM_HNSW_SPARSE_STREAMER_VISIT_BLOOMFILTER_ENABLE(
    "proxima.hnsw.sparse_streamer.visit_bloomfilter_enable");
static const std::string
    PARAM_HNSW_SPARSE_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB(
        "proxima.hnsw.sparse_streamer.visit_bloomfilter_negative_prob");
static const std::string PARAM_HNSW_SPARSE_STREAMER_CHECK_CRC_ENABLE(
    "proxima.hnsw.sparse_streamer.check_crc_enable");
static const std::string PARAM_HNSW_SPARSE_STREAMER_NEIGHBOR_PRUNE_MULTIPLIER(
    "proxima.hnsw.sparse_streamer.neighbor_prune_multiplier");
static const std::string PARAM_HNSW_SPARSE_STREAMER_CHUNK_SIZE(
    "proxima.hnsw.sparse_streamer.chunk_size");
static const std::string PARAM_HNSW_SPARSE_STREAMER_FILTER_SAME_KEY(
    "proxima.hnsw.sparse_streamer.filter_same_key");
static const std::string PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE(
    "proxima.hnsw.sparse_streamer.get_vector_enable");
static const std::string PARAM_HNSW_SPARSE_STREAMER_MIN_NEIGHBOR_COUNT(
    "proxima.hnsw.sparse_streamer.min_neighbor_count");
static const std::string PARAM_HNSW_SPARSE_STREAMER_FORCE_PADDING_RESULT_ENABLE(
    "proxima.hnsw.sparse_streamer.force_padding_result_enable");
static const std::string PARAM_HNSW_SPARSE_STREAMER_QUERY_FILTERING_RATIO(
    "proxima.hnsw.sparse_streamer.query_filtering_ratio");

static const std::string PARAM_HNSW_SPARSE_REDUCER_WORKING_PATH(
    "proxima.hnsw.sparse_reducer.working_path");
static const std::string PARAM_HNSW_SPARSE_REDUCER_NUM_OF_ADD_THREADS(
    "proxima.hnsw.sparse_reducer.num_of_add_threads");
static const std::string PARAM_HNSW_SPARSE_REDUCER_INDEX_NAME(
    "proxima.hnsw.sparse_reducer.index_name");
static const std::string PARAM_HNSW_SPARSE_REDUCER_EFCONSTRUCTION(
    "proxima.hnsw.sparse_reducer.efconstruction");

}  // namespace core
}  // namespace zvec
