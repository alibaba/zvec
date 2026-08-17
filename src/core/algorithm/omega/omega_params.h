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

namespace zvec::core {

// OMEGA searcher parameters (used at query time)
static const std::string PARAM_OMEGA_SEARCHER_TARGET_RECALL(
    "proxima.omega.searcher.target_recall");

// Training query ID for parallel training searches
static const std::string PARAM_OMEGA_SEARCHER_TRAINING_QUERY_ID(
    "proxima.omega.searcher.training_query_id");

// OMEGA streamer parameters (used at index time)
static const std::string PARAM_OMEGA_STREAMER_TARGET_RECALL(
    "proxima.omega.streamer.target_recall");

}  // namespace zvec::core
