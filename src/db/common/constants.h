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

#include <cstdint>
#include <regex>
#include <string>

namespace zvec {

// global config
const float DEFAULT_MEMORY_LIMIT_RATIO = 0.8f;

const uint32_t MIN_MEMORY_LIMIT_BYTES = 100 * 1024 * 1024;

const uint64_t INVALID_DOC_ID = -1UL;

const std::string LOCAL_ROW_ID = "_zvec_row_id_";

const std::string GLOBAL_DOC_ID = "_zvec_g_doc_id_";

const std::string USER_ID = "_zvec_uid_";

const int kSparseMaxDimSize = 4096;

const int64_t kMaxRecordBatchNumRows = 4096;

constexpr uint32_t MAX_ARRAY_FIELD_LEN = 32;

const float COMPACT_DELETE_RATIO_THRESHOLD = 0.3f;

const std::regex COLLECTION_NAME_REGEX("^[a-zA-Z0-9_-]{3,64}$");

const std::regex FIELD_NAME_REGEX("^[a-zA-Z0-9_-]{1,32}$");

const std::regex DOC_PK_REGEX("^[a-zA-Z0-9_!@#$%+=.-]{1,64}$");

const std::regex COLLECTION_PATH_REGEX(
    R"(^/?(?:[a-zA-Z0-9_.\-]+/)*[a-zA-Z0-9_.\-]+$)");

constexpr uint32_t kMaxDenseDimSize = 20000;

constexpr uint32_t kMaxScalarFieldSize = 1024;

constexpr uint32_t kMaxVectorFieldSize = 5;

constexpr uint32_t kMaxQueryTopk = 1024;

constexpr uint32_t kMaxOutputFieldSize = 1024;

constexpr uint32_t kMaxWriteBatchSize = 1024;

// Inverted index
const std::string INVERT_SUFFIX_TERMS{"$TERMS"};

const std::string INVERT_SUFFIX_REVERSED_TERMS{"$SMRET"};

const std::string INVERT_SUFFIX_ARRAY_LEN{"$ARRAY_LEN"};

const std::string INVERT_SUFFIX_RANGES{"$RANGES"};

const std::string INVERT_CDF{"$CDF"};

const std::string INVERT_KEY_MAX_ID{"$ZVEC$MAX_ID"};

const std::string INVERT_KEY_NULL{"$ZVEC$NULL"};

const std::string INVERT_KEY_SEALED{"$ZVEC$SEALED"};

const uint32_t INVERT_ID_LIST_SIZE_THRESHOLD = 3;


}  // namespace zvec
