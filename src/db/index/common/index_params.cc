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

#include <mutex>
#include <new>
#include <sstream>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/index_params.h>
#include "db/index/column/fts_column/fts_pipeline.h"
#include "db/index/column/fts_column/fts_types.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_pipeline_manager.h"
#include "type_helper.h"

namespace zvec {

std::string InvertIndexParams::to_string() const {
  std::ostringstream oss;
  oss << "InvertIndexParams{"
      << "enable_range_optimization:"
      << (enable_range_optimization_ ? "true" : "false")
      << ", enable_extended_wildcard:"
      << (enable_extended_wildcard_ ? "true" : "false") << "}";
  return oss.str();
}

std::string VectorIndexParams::vector_index_params_to_string(
    const std::string &class_name, MetricType metric_type,
    QuantizeType quantize_type) const {
  std::ostringstream oss;
  oss << class_name << "{"
      << "metric:" << MetricTypeCodeBook::AsString(metric_type)
      << ",quantize:" << QuantizeTypeCodeBook::AsString(quantize_type);
  return oss.str();
}

// ============================================================
// FtsIndexParams — helpers
// ============================================================

static fts::FtsIndexParams to_internal(const FtsIndexParams &params) {
  fts::FtsIndexParams p;
  p.tokenizer_name = params.tokenizer_name();
  p.filters = params.filters();
  p.extra_params = params.extra_params();
  return p;
}

// ============================================================
// FtsIndexParams — opaque pipeline state (Pimpl)
// ============================================================

namespace detail {
struct FtsState {
  std::once_flag once;
  std::shared_ptr<fts::TokenizerPipeline> pipeline;
  bool created{false};
};

struct FtsPipelineHelper {
  static std::unique_ptr<FtsState> &state(FtsIndexParams &p) {
    return p.state_;
  }
};
}  // namespace detail

// ============================================================
// FtsIndexParams — ctor / dtor / move
// ============================================================

FtsIndexParams::FtsIndexParams(std::string tokenizer_name,
                               std::vector<std::string> filters,
                               std::string extra_params)
    : IndexParams(IndexType::FTS),
      tokenizer_name_(std::move(tokenizer_name)),
      filters_(std::move(filters)),
      extra_params_(std::move(extra_params)),
      state_(std::make_unique<detail::FtsState>()) {}

FtsIndexParams::FtsIndexParams(FtsIndexParams &&other) noexcept
    : IndexParams(IndexType::FTS),
      tokenizer_name_(std::move(other.tokenizer_name_)),
      filters_(std::move(other.filters_)),
      extra_params_(std::move(other.extra_params_)),
      state_(std::move(other.state_)) {}

FtsIndexParams::~FtsIndexParams() {
  if (state_ && state_->created) {
    auto internal = to_internal(*this);
    fts::TokenizerPipelineManager::Instance().release(internal);
  }
}

// ============================================================
// FtsIndexParams — pipeline acquisition (internal)
// ============================================================

namespace detail {

Result<std::shared_ptr<fts::TokenizerPipeline>> AcquireFtsPipeline(
    FtsIndexParams &params) {
  auto &state_uptr = FtsPipelineHelper::state(params);
  if (!state_uptr) {
    // Lazily reconstruct after a move-from; not thread-safe vs. a concurrent
    // move on the same instance, but moves on a live instance already need
    // external synchronisation.
    state_uptr = std::make_unique<FtsState>();
  }
  auto &st = *state_uptr;
  std::call_once(st.once, [&]() {
    auto internal = to_internal(params);
    st.pipeline = fts::TokenizerPipelineManager::Instance().acquire(internal);
    if (st.pipeline) {
      st.created = true;
    }
  });
  if (!st.pipeline) {
    return tl::make_unexpected(
        Status::InternalError("Failed to create tokenizer pipeline"));
  }
  return st.pipeline;
}

}  // namespace detail

}  // namespace zvec