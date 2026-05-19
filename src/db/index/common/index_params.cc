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
// FtsIndexParams — destructor
// ============================================================

FtsIndexParams::~FtsIndexParams() {
  if (pipeline_created_) {
    auto internal = to_internal(*this);
    fts::TokenizerPipelineManager::Instance().release(internal);
  }
}

// ============================================================
// FtsIndexParams — move semantics
// ============================================================

FtsIndexParams::FtsIndexParams(FtsIndexParams &&other) noexcept
    : IndexParams(IndexType::FTS),
      tokenizer_name_(std::move(other.tokenizer_name_)),
      filters_(std::move(other.filters_)),
      extra_params_(std::move(other.extra_params_)),
      pipeline_(std::move(other.pipeline_)),
      pipeline_created_(other.pipeline_created_) {
  other.pipeline_created_ = false;
  other.pipeline_.reset();
  // std::once_flag is not movable; default-initialise ours (already done by
  // the member initialiser) and leave other's in a valid but used state.
  // If the source had already called create_pipeline(), we inherit the
  // cached result.  If not, our fresh once_flag will allow a future call.
  if (pipeline_created_) {
    // Mark our once_flag as "already called" by running a no-op through it.
    std::call_once(pipeline_once_, [] {});
  }
}


// ============================================================
// FtsIndexParams — create_pipeline
// ============================================================

Result<FtsIndexParams::PipelinePtr> FtsIndexParams::create_pipeline() {
  std::call_once(pipeline_once_, [this]() {
    auto internal = to_internal(*this);
    pipeline_ = fts::TokenizerPipelineManager::Instance().acquire(internal);
    if (pipeline_) {
      pipeline_created_ = true;
    }
  });
  if (!pipeline_) {
    return tl::make_unexpected(
        Status::InternalError("Failed to create tokenizer pipeline"));
  }
  return pipeline_;
}

}  // namespace zvec