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

inline const std::string PARAM_HNSW_RABITQ_GENERAL_DIMENSION(
    "proxima.hnsw_rabitq.general.dimension");

inline const std::string PARAM_HNSW_RABITQ_BUILDER_THREAD_COUNT(
    "proxima.hnsw_rabitq.builder.thread_count");
inline const std::string PARAM_HNSW_RABITQ_BUILDER_MEMORY_QUOTA(
    "proxima.hnsw_rabitq.builder.memory_quota");
inline const std::string PARAM_HNSW_RABITQ_BUILDER_EFCONSTRUCTION(
    "proxima.hnsw_rabitq.builder.efconstruction");
inline const std::string PARAM_HNSW_RABITQ_BUILDER_SCALING_FACTOR(
    "proxima.hnsw_rabitq.builder.scaling_factor");
inline const std::string PARAM_HNSW_RABITQ_BUILDER_CHECK_INTERVAL_SECS(
    "proxima.hnsw_rabitq.builder.check_interval_secs");
inline const std::string PARAM_HNSW_RABITQ_BUILDER_NEIGHBOR_PRUNE_MULTIPLIER(
    "proxima.hnsw_rabitq.builder.neighbor_prune_multiplier");
inline const std::string PARAM_HNSW_RABITQ_BUILDER_MIN_NEIGHBOR_COUNT(
    "proxima.hnsw_rabitq.builder.min_neighbor_count");
inline const std::string PARAM_HNSW_RABITQ_BUILDER_MAX_NEIGHBOR_COUNT(
    "proxima.hnsw_rabitq.builder.max_neighbor_count");
inline const std::string
    PARAM_HNSW_RABITQ_BUILDER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER(
        "proxima.hnsw_rabitq.builder.l0_max_neighbor_count_multiplier");

inline const std::string PARAM_HNSW_RABITQ_SEARCHER_EF(
    "proxima.hnsw_rabitq.searcher.ef");
inline const std::string PARAM_HNSW_RABITQ_SEARCHER_BRUTE_FORCE_THRESHOLD(
    "proxima.hnsw_rabitq.searcher.brute_force_threshold");
inline const std::string PARAM_HNSW_RABITQ_SEARCHER_NEIGHBORS_IN_MEMORY_ENABLE(
    "proxima.hnsw_rabitq.searcher.neighbors_in_memory_enable");
inline const std::string PARAM_HNSW_RABITQ_SEARCHER_MAX_SCAN_RATIO(
    "proxima.hnsw_rabitq.searcher.max_scan_ratio");
inline const std::string PARAM_HNSW_RABITQ_SEARCHER_CHECK_CRC_ENABLE(
    "proxima.hnsw_rabitq.searcher.check_crc_enable");
inline const std::string PARAM_HNSW_RABITQ_SEARCHER_VISIT_BLOOMFILTER_ENABLE(
    "proxima.hnsw_rabitq.searcher.visit_bloomfilter_enable");
inline const std::string
    PARAM_HNSW_RABITQ_SEARCHER_VISIT_BLOOMFILTER_NEGATIVE_PROB(
        "proxima.hnsw_rabitq.searcher.visit_bloomfilter_negative_prob");
inline const std::string PARAM_HNSW_RABITQ_SEARCHER_FORCE_PADDING_RESULT_ENABLE(
    "proxima.hnsw_rabitq.searcher.force_padding_result_enable");

inline const std::string PARAM_HNSW_RABITQ_STREAMER_MAX_SCAN_RATIO(
    "proxima.hnsw_rabitq.streamer.max_scan_ratio");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_MIN_SCAN_LIMIT(
    "proxima.hnsw_rabitq.streamer.min_scan_limit");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_MAX_SCAN_LIMIT(
    "proxima.hnsw_rabitq.streamer.max_scan_limit");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_EF(
    "proxima.hnsw_rabitq.streamer.ef");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_EFCONSTRUCTION(
    "proxima.hnsw_rabitq.streamer.efconstruction");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_MAX_NEIGHBOR_COUNT(
    "proxima.hnsw_rabitq.streamer.max_neighbor_count");
inline const std::string
    PARAM_HNSW_RABITQ_STREAMER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER(
        "proxima.hnsw_rabitq.streamer.l0_max_neighbor_count_multiplier");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_SCALING_FACTOR(
    "proxima.hnsw_rabitq.streamer.scaling_factor");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_BRUTE_FORCE_THRESHOLD(
    "proxima.hnsw_rabitq.streamer.brute_force_threshold");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_DOCS_HARD_LIMIT(
    "proxima.hnsw_rabitq.streamer.docs_hard_limit");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_DOCS_SOFT_LIMIT(
    "proxima.hnsw_rabitq.streamer.docs_soft_limit");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_MAX_INDEX_SIZE(
    "proxima.hnsw_rabitq.streamer.max_index_size");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_VISIT_BLOOMFILTER_ENABLE(
    "proxima.hnsw_rabitq.streamer.visit_bloomfilter_enable");
inline const std::string
    PARAM_HNSW_RABITQ_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB(
        "proxima.hnsw_rabitq.streamer.visit_bloomfilter_negative_prob");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_CHECK_CRC_ENABLE(
    "proxima.hnsw_rabitq.streamer.check_crc_enable");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_NEIGHBOR_PRUNE_MULTIPLIER(
    "proxima.hnsw_rabitq.streamer.neighbor_prune_multiplier");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_CHUNK_SIZE(
    "proxima.hnsw_rabitq.streamer.chunk_size");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_FILTER_SAME_KEY(
    "proxima.hnsw_rabitq.streamer.filter_same_key");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_GET_VECTOR_ENABLE(
    "proxima.hnsw_rabitq.streamer.get_vector_enable");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_MIN_NEIGHBOR_COUNT(
    "proxima.hnsw_rabitq.streamer.min_neighbor_count");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_FORCE_PADDING_RESULT_ENABLE(
    "proxima.hnsw_rabitq.streamer.force_padding_result_enable");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_ESTIMATE_DOC_COUNT(
    "proxima.hnsw_rabitq.streamer.estimate_doc_count");
inline const std::string PARAM_HNSW_RABITQ_STREAMER_USE_ID_MAP(
    "proxima.hnsw_rabitq.streamer.use_id_map");

inline const std::string PARAM_HNSW_RABITQ_REDUCER_WORKING_PATH(
    "proxima.hnsw_rabitq.reducer.working_path");
inline const std::string PARAM_HNSW_RABITQ_REDUCER_NUM_OF_ADD_THREADS(
    "proxima.hnsw_rabitq.reducer.num_of_add_threads");
inline const std::string PARAM_HNSW_RABITQ_REDUCER_INDEX_NAME(
    "proxima.hnsw_rabitq.reducer.index_name");
inline const std::string PARAM_HNSW_RABITQ_REDUCER_EFCONSTRUCTION(
    "proxima.hnsw_rabitq.reducer.efconstruction");

}  // namespace core
}  // namespace zvec
