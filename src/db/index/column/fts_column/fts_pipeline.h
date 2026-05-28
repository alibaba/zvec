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

#include <memory>
#include <zvec/db/index_params.h>
#include <zvec/db/status.h>

namespace zvec {

namespace fts {
class TokenizerPipeline;
}  // namespace fts

namespace detail {

// Internal entry to lazily acquire (and cache, per FtsIndexParams instance)
// the tokenizer pipeline.  Thread-safe; same params instance returns the
// same shared_ptr on subsequent calls; the manager-side reference is
// released when the params instance is destroyed.
Result<std::shared_ptr<fts::TokenizerPipeline>> AcquireFtsPipeline(
    FtsIndexParams &params);

}  // namespace detail
}  // namespace zvec
