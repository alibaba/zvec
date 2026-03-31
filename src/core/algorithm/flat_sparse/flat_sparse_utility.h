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

static constexpr uint32_t PARAM_FLAT_SPARSE_MAX_DIM_SIZE = 16384;
static const std::string PARAM_FLAT_SPARSE_META_SEG_ID =
    "bruteforce_sparse_meta";

// streamer
static const std::string PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID =
    "bruteforce_sparse_streamer_meta";
static const std::string PARAM_FLAT_SPARSE_OFFSET_SEG_ID_PREFIX =
    "bruteforce_sparse_streamer_offset_";
static const std::string PARAM_FLAT_SPARSE_DATA_SEG_ID_PREFIX =
    "bruteforce_sparse_streamer_data_";

// searcher
static const std::string PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID =
    "bruteforce_sparse_searcher_offset_segment";
static const std::string PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID =
    "bruteforce_sparse_searcher_data_segment";
static const std::string PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID =
    "bruteforce_sparse_searcher_keys_segment";
static const std::string PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID =
    "bruteforce_sparse_searcher_mapping_segment";

// streamer
static const std::string PARAM_FLAT_SPARSE_STREAMER_OFFSET_CHUNK_SIZE(
    "proxima.bruteforce.sparse_streamer.offset_chunk_size");

static const std::string PARAM_FLAT_SPARSE_STREAMER_DATA_CHUNK_SIZE(
    "proxima.bruteforce.sparse_streamer.data_chunk_size");

static const std::string PARAM_FLAT_SPARSE_STREAMER_MAX_DOC_CNT(
    "proxima.bruteforce.sparse_streamer.max_doc_cnt");

static const std::string PARAM_FLAT_SPARSE_STREAMER_MAX_DATA_CHUNK_CNT(
    "proxima.bruteforce.sparse_streamer.max_data_chunk_cnt");

}  // namespace core
}  // namespace zvec