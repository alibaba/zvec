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


inline const std::string PARAM_MIXED_STREAMER_REDUCER_ENABLE_PK_REWRITE(
    "proxima.mixed.reducer.enable_pk_rewrite");
inline const std::string PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS(
    "proxima.mixed.reducer.num_of_add_threads");

inline const std::string PARAM_MIXED_REDUCER_WORKING_PATH(
    "proxima.mixed.reducer.working_path");
inline const std::string PARAM_MIXED_REDUCER_NUM_OF_ADD_THREADS(
    "proxima.mixed.reducer.num_of_add_threads");
inline const std::string PARAM_MIXED_REDUCER_STREAMER_CLASS(
    "proxima.mixed.reducer.streamer_class");
inline const std::string PARAM_MIXED_REDUCER_HYBRID_VECTOR_ENABLE(
    "proxima.mixed.reducer.hybrid_vector_enable");
inline const std::string PARAM_MIXED_REDUCER_INDEX_NAME(
    "proxima.mixed.reducer.index_name");
inline const std::string PARAM_MIXED_REDUCER_QUANTIZER_CLASS(
    "proxima.mixed.reducer.quantizer_class");


}  // namespace core
}  // namespace zvec
