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

// Internal header — NOT in public includes (src/include/).
// Defines the opaque HNSW-RaBitQ runtime state that HnswRabitqIndexParams only
// forward declares, so that the public header does not have to pull in the
// core/framework engine interfaces.  Used by segment_helper.cc (producer) and
// engine_helper.hpp (consumer).
#pragma once

#include <memory>
#include <utility>
#include <zvec/db/index_params.h>
#include "zvec/core/framework/index_provider.h"
#include "zvec/core/framework/index_reformer.h"

namespace zvec {
namespace detail {

struct HnswRabitqRuntime {
  core::IndexProvider::Pointer raw_vector_provider;
  core::IndexReformer::Pointer rabitq_reformer;
};

// Friend accessor for HnswRabitqIndexParams::runtime_.  Always allocates a
// fresh runtime on set(), which keeps the pre-pimpl semantics where assigning
// the engine objects on a clone never affected the params it was cloned from.
struct HnswRabitqRuntimeAccess {
  static void set(HnswRabitqIndexParams &params,
                  core::IndexProvider::Pointer raw_vector_provider,
                  core::IndexReformer::Pointer rabitq_reformer) {
    params.runtime_ = std::make_shared<HnswRabitqRuntime>();
    params.runtime_->raw_vector_provider = std::move(raw_vector_provider);
    params.runtime_->rabitq_reformer = std::move(rabitq_reformer);
  }

  static core::IndexProvider::Pointer raw_vector_provider(
      const HnswRabitqIndexParams &params) {
    return params.runtime_ ? params.runtime_->raw_vector_provider : nullptr;
  }

  static core::IndexReformer::Pointer rabitq_reformer(
      const HnswRabitqIndexParams &params) {
    return params.runtime_ ? params.runtime_->rabitq_reformer : nullptr;
  }
};

}  // namespace detail
}  // namespace zvec
