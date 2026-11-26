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
#include <sstream>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/index_params.h>
#include "ailego/internal/cpu_features.h"
#if !RABITQ_SUPPORTED
#include "core/algorithm/hnsw_rabitq/rabitq_params.h"
#endif
#include "db/index/column/fts_column/fts_pipeline.h"
#include "db/index/column/fts_column/fts_types.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_factory.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_pipeline_manager.h"
#include "type_helper.h"

namespace zvec {

namespace {

Status validate_rabitq_params(MetricType metric, int total_bits) {
  if (metric != MetricType::L2 && metric != MetricType::IP &&
      metric != MetricType::COSINE) {
    return Status::InvalidArgument("RabitQ supports L2/IP/COSINE metric; got ",
                                   MetricTypeCodeBook::AsString(metric));
  }
  if (total_bits < 0 || total_bits > 9) {
    return Status::InvalidArgument(
        "RabitQ total_bits must be in [0, 9] (0 selects the default); got ",
        total_bits);
  }
#if !RABITQ_SUPPORTED
  return Status::NotSupported(core::kRabitqUnsupportedPlatform);
#endif
  const auto &flags = ailego::internal::CpuFeatures::static_flags_;
  const bool supports_avx2 = flags.AVX2 && flags.FMA;
  const bool supports_avx512 =
      flags.AVX512F && flags.AVX512BW && flags.AVX512DQ;
  if (!supports_avx2 && !supports_avx512 && !flags.NEON) {
    return Status::NotSupported(
        "RabitQ requires AVX2/FMA, AVX512F/BW/DQ, or NEON to be supported");
  }
  return Status::OK();
}

}  // namespace

Status QuantizerParam::validate() const {
  if (num_chunk_ <= 0) {
    return Status::InvalidArgument("num_chunk must be greater than 0; got ",
                                   num_chunk_);
  }
  if (num_bits_ != 4 && num_bits_ != 8) {
    return Status::InvalidArgument("num_bits must be 4 or 8; got ", num_bits_);
  }
  if (fast_scan_ && num_bits_ != 4) {
    return Status::InvalidArgument("fast_scan requires num_bits=4; got ",
                                   num_bits_);
  }
  if (opq_iter_ == 0) {
    return Status::InvalidArgument("opq_iter must be greater than 0; got 0");
  }
  if (opq_pq_iter_ == 0) {
    return Status::InvalidArgument("opq_pq_iter must be greater than 0; got 0");
  }
  return Status::OK();
}

Status VectorIndexParams::validate() const {
  if (auto s = quantizer_param_.validate(); !s.ok()) {
    return s;
  }
  const auto storage_type = flat_data_type();
  if (storage_type != DataType::UNDEFINED &&
      storage_type != DataType::VECTOR_FP32 &&
      storage_type != DataType::VECTOR_FP16 &&
      storage_type != DataType::VECTOR_UINT8) {
    return Status::InvalidArgument(
        "flat_data_type must be VECTOR_FP32, VECTOR_FP16, or VECTOR_UINT8, "
        "but got ",
        DataTypeCodeBook::AsString(storage_type));
  }
  if (storage_type == DataType::VECTOR_UINT8 &&
      metric_type_ != MetricType::L2) {
    return Status::InvalidArgument(
        "VECTOR_UINT8 Flat reference storage only supports L2 metric; got ",
        MetricTypeCodeBook::AsString(metric_type_));
  }
  if (quantize_type_ == QuantizeType::PQ) {
    if (type_ != IndexType::IVF && type_ != IndexType::HNSW) {
      return Status::InvalidArgument(
          "PQ quantization requires IVF or HNSW index_type; got ",
          IndexTypeCodeBook::AsString(type_));
    }
    if (metric_type_ != MetricType::L2 && metric_type_ != MetricType::IP &&
        metric_type_ != MetricType::COSINE) {
      return Status::InvalidArgument(
          "PQ quantization requires L2/IP/COSINE metric; got ",
          MetricTypeCodeBook::AsString(metric_type_));
    }
    if (quantizer_param_.fast_scan() && type_ != IndexType::IVF) {
      return Status::InvalidArgument("fast_scan requires IVF index_type; got ",
                                     IndexTypeCodeBook::AsString(type_));
    }
  }
  const bool is_uniform = quantize_type_ == QuantizeType::UNIFORM_UINT7 ||
                          quantize_type_ == QuantizeType::UNIFORM_UINT8 ||
                          quantize_type_ == QuantizeType::UNIFORM_UINT4;
  if (is_uniform) {
    // These DB index paths do not support Uniform training or restoration.
    if (type_ == IndexType::FLAT || type_ == IndexType::IVF ||
        type_ == IndexType::DISKANN) {
      return Status::InvalidArgument(
          QuantizeTypeCodeBook::AsString(quantize_type_),
          " quantization is not supported with ",
          IndexTypeCodeBook::AsString(type_), " index");
    }
    if (metric_type_ != MetricType::L2) {
      return Status::InvalidArgument(
          QuantizeTypeCodeBook::AsString(quantize_type_),
          " quantization only supports L2 metric; got ",
          MetricTypeCodeBook::AsString(metric_type_));
    }
  }
  return Status::OK();
}

Status HnswRabitqIndexParams::validate() const {
  if (auto s = VectorIndexParams::validate(); !s.ok()) {
    return s;
  }
  return validate_rabitq_params(metric_type_, total_bits_);
}

Status IvfRabitqIndexParams::validate() const {
  if (auto s = VectorIndexParams::validate(); !s.ok()) {
    return s;
  }
  if (nlist_ <= 0) {
    return Status::InvalidArgument(
        "IVF_RABITQ nlist must be greater than 0; got ", nlist_);
  }
  if (sample_count_ < 0) {
    return Status::InvalidArgument(
        "IVF_RABITQ sample_count must be greater than or equal to 0; got ",
        sample_count_);
  }
  return validate_rabitq_params(metric_type_, total_bits_);
}

Status IVFIndexParams::validate() const {
  if (auto s = VectorIndexParams::validate(); !s.ok()) {
    return s;
  }
  if (quantize_type_ == QuantizeType::RABITQ) {
    return Status::InvalidArgument(
        "IVF index does not support RABITQ quantization; use the dedicated "
        "IVF_RABITQ index instead");
  }
  return Status::OK();
}

Status DiskAnnIndexParams::validate() const {
  if (auto s = VectorIndexParams::validate(); !s.ok()) {
    return s;
  }
#if !DISKANN_SUPPORTED
  return Status::NotSupported("DiskANN is not supported on this platform.");
#endif
  return Status::OK();
}

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
  if (quantize_type == QuantizeType::PQ) {
    oss << ",num_chunk:" << quantizer_param_.num_chunk()
        << ",num_bits:" << quantizer_param_.num_bits()
        << ",fast_scan:" << quantizer_param_.fast_scan()
        << ",opq_iter:" << quantizer_param_.opq_iter()
        << ",opq_pq_iter:" << quantizer_param_.opq_pq_iter();
  }
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

Status FtsIndexParams::validate() const {
  auto pipeline = fts::TokenizerFactory::create(to_internal(*this));
  if (!pipeline.has_value()) {
    return Status::InvalidArgument("invalid FTS index params: ",
                                   pipeline.error().message());
  }
  return Status::OK();
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
