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
#include "ivf_entity.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <core/quantizer/quantizer_params.h>
#include <turbo/quantizer/common/pq_quantizer/packed_code_quantizer.h>
#include "metric/metric_params.h"
#include "utility/batch_topk.h"
#include "ivf_utility.h"
namespace zvec {
namespace core {

namespace {

// Reuse bounded output scratch after each distance batch is added to the heap.
constexpr size_t kDistanceBatchVectors = 512;

//! Cursor over a scatter read. IVF blocks and vectors are logical byte
//! ranges, so neither needs to be aligned with an underlying storage page.
class ScatterCursor {
 public:
  explicit ScatterCursor(const IndexStorage::Segment::ScatterBlock &block)
      : spans_(block.spans) {}

  const uint8_t *data() {
    normalize();
    return span_index_ < spans_.size() ? spans_[span_index_].data + span_offset_
                                       : nullptr;
  }

  size_t contiguous_size() {
    normalize();
    return span_index_ < spans_.size() ? spans_[span_index_].size - span_offset_
                                       : 0;
  }

  const uint8_t *take_contiguous(size_t size) {
    normalize();
    if (span_index_ == spans_.size() ||
        size > spans_[span_index_].size - span_offset_) {
      return nullptr;
    }
    const auto *data = spans_[span_index_].data + span_offset_;
    span_offset_ += size;
    return data;
  }

  bool skip(size_t size) {
    while (size != 0) {
      normalize();
      if (span_index_ == spans_.size()) {
        return false;
      }
      const size_t available = spans_[span_index_].size - span_offset_;
      const size_t consumed = std::min(size, available);
      span_offset_ += consumed;
      size -= consumed;
    }
    return true;
  }

  bool copy(void *output, size_t size) {
    auto *dst = static_cast<uint8_t *>(output);
    while (size != 0) {
      normalize();
      if (span_index_ == spans_.size()) {
        return false;
      }
      const auto &span = spans_[span_index_];
      const size_t available = span.size - span_offset_;
      const size_t copied = std::min(size, available);
      std::memcpy(dst, span.data + span_offset_, copied);
      dst += copied;
      span_offset_ += copied;
      size -= copied;
    }
    return true;
  }

 private:
  void normalize() {
    while (span_index_ < spans_.size() &&
           span_offset_ == spans_[span_index_].size) {
      ++span_index_;
      span_offset_ = 0;
    }
  }

  const std::vector<IndexStorage::Segment::ReadSpan> &spans_;
  size_t span_index_{0};
  size_t span_offset_{0};
};

//! Calculate a row-major IVF block directly from its resident page spans.
//! Only a vector split by a page boundary is copied into scratch.
template <typename Distance>
bool calculate_row_major_block(ScatterCursor *cursor, const Distance &distance,
                               size_t vector_count, size_t element_size,
                               size_t serialized_size, uint8_t *scratch,
                               float *distances, bool calculate) {
  if (element_size == 0 ||
      vector_count > std::numeric_limits<size_t>::max() / element_size) {
    return false;
  }
  const size_t vector_bytes = vector_count * element_size;
  if (serialized_size < vector_bytes) {
    return false;
  }
  if (!calculate) {
    return cursor->skip(serialized_size);
  }

  if (const auto *data = cursor->take_contiguous(serialized_size)) {
    distance(data, vector_count, distances);
    return true;
  }

  size_t processed = 0;
  while (processed < vector_count) {
    const size_t contiguous = cursor->contiguous_size();
    const size_t direct_count =
        std::min(vector_count - processed, contiguous / element_size);
    if (direct_count != 0) {
      distance(cursor->data(), direct_count, distances + processed);
      if (!cursor->skip(direct_count * element_size)) {
        return false;
      }
      processed += direct_count;
      continue;
    }

    if (!scratch || !cursor->copy(scratch, element_size)) {
      return false;
    }
    distance(scratch, 1, distances + processed);
    ++processed;
  }
  return cursor->skip(serialized_size - vector_bytes);
}

}  // namespace

//! Initialize
int IVFEntity::IVFReformerWrapper::init(const IndexMeta &imeta) {
  auto &name = imeta.reformer_name();

  if (name.empty()) {
    type_ = kReformerTpNone;
    return 0;
  }

  auto reformer = IndexFactory::CreateReformer(name);
  if (!reformer) {
    LOG_ERROR("Failed to create reformer %s", name.c_str());
    return IndexError_NoExist;
  }
  int ret = reformer->init(imeta.reformer_params());
  ivf_check_with_msg(ret, "Failed to init reformer %s", name.c_str());

  reformer_ = std::move(reformer);

  if (name == kInt8ReformerName) {
    if (imeta.metric_name() == kIPMetricName) {
      type_ = kReformerTpInnerProductInt8;
      return 0;
    }
    auto &key = INT8_QUANTIZER_REFORMER_SCALE;
    if (!imeta.reformer_params().has(key)) {
      LOG_ERROR("Missing param %s in reformer %s", key.c_str(), name.c_str());
      return IndexError_InvalidArgument;
    };
    float scale = imeta.reformer_params().get_as_float(key);
    reciprocal_ = scale == 0.0 ? 1.0 : (1.0 / scale);
    type_ = kReformerTpInt8;
  } else if (name == kInt4ReformerName) {
    if (imeta.metric_name() == kIPMetricName) {
      type_ = kReformerTpInnerProductInt4;
      return 0;
    }
    auto &key = INT4_QUANTIZER_REFORMER_SCALE;
    if (!imeta.reformer_params().has(key)) {
      LOG_ERROR("Missing param %s in reformer %s", key.c_str(), name.c_str());
      return IndexError_InvalidArgument;
    };
    float scale = imeta.reformer_params().get_as_float(key);
    reciprocal_ = scale == 0.0 ? 1.0 : (1.0 / scale);
    type_ = kReformerTpInt4;
  } else {
    type_ = kReformerTpDefault;
  }

  LOG_DEBUG("Init QcReformer with %s, type=%u", name.c_str(), type_);

  return 0;
}

//! Load reformer state (e.g. rotation matrix) from storage
int IVFEntity::IVFReformerWrapper::load(const IndexStorage::Pointer &storage) {
  if (!reformer_) {
    return 0;
  }
  int ret = reformer_->load(storage);
  ivf_check_with_msg(ret, "Failed to load reformer state");
  return 0;
}

//! Update the params, Called by gpu searcher only
int IVFEntity::IVFReformerWrapper::update(const IndexMeta &meta) {
  auto &name = meta.reformer_name();
  if (name == kInt4ReformerName && meta.metric_name() == kL2MetricName) {
    auto &key = INT4_QUANTIZER_REFORMER_SCALE;
    if (!meta.reformer_params().has(key)) {
      LOG_ERROR("Missing param %s in reformer %s", key.c_str(), name.c_str());
      return IndexError_InvalidArgument;
    };
    float scale = meta.reformer_params().get_as_float(key);
    reciprocal_ = scale == 0.0 ? 1.0 : (1.0 / scale / kNormalizeScaleFactor);
    type_ = kReformerTpInt8;

    ailego::Params params;
    float int8_scale = scale * kNormalizeScaleFactor;
    params.set(INT8_QUANTIZER_REFORMER_SCALE, int8_scale);
    float bias =
        meta.reformer_params().get_as_float(INT4_QUANTIZER_REFORMER_BIAS);
    params.set(INT8_QUANTIZER_REFORMER_BIAS, bias);
    params.set(
        INT4_QUANTIZER_REFORMER_METRIC,
        meta.reformer_params().get_as_string(INT4_QUANTIZER_REFORMER_METRIC));

    auto reformer = IndexFactory::CreateReformer(kInt8ReformerName);
    if (!reformer) {
      LOG_ERROR("Failed to create reformer %s", name.c_str());
      return IndexError_NoExist;
    }
    int ret = reformer->init(params);
    ivf_check_with_msg(ret, "Failed to init reformer %s", name.c_str());

    reformer_ = reformer;

    LOG_DEBUG("Init QcReformer with %s, type=%u", name.c_str(), type_);
  }

  return 0;
}

//! Transform a query
int IVFEntity::IVFReformerWrapper::transform(const void *query,
                                             const IndexQueryMeta &qmeta,
                                             const void **out,
                                             IndexQueryMeta *ometa) {
  int ret = 0;

  switch (type_) {
    case kReformerTpNone:
      *out = query;
      *ometa = qmeta;
      break;

    case kReformerTpInnerProductInt8:
      if (qmeta.data_type() != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }
      scales_.resize(1);
      buffer_.resize(IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8,
                                              qmeta.dimension()));
      this->transform(0, static_cast<const float *>(query), qmeta.dimension(),
                      reinterpret_cast<int8_t *>(&buffer_[0]));
      *ometa = qmeta;
      ometa->set_meta(IndexMeta::DataType::DT_INT8, qmeta.dimension());
      *out = buffer_.data();
      break;

    case kReformerTpInnerProductInt4:
      if (qmeta.data_type() != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }
      scales_.resize(1);
      buffer_.resize(IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT4,
                                              qmeta.dimension()));
      this->transform(0, static_cast<const float *>(query), qmeta.dimension(),
                      reinterpret_cast<uint8_t *>(&buffer_[0]));
      *ometa = qmeta;
      ometa->set_meta(IndexMeta::DataType::DT_INT4, qmeta.dimension());
      *out = buffer_.data();
      break;

    case kReformerTpInt8:
    case kReformerTpInt4:
      /* FALLTHRU */
    case kReformerTpDefault:
      ret = reformer_->transform(query, qmeta, &buffer_, ometa);
      *out = buffer_.data();
      break;

    default:
      ret = IndexError_Unsupported;
      break;
  }

  return ret;
}

//! Transform querys
int IVFEntity::IVFReformerWrapper::transform(const void *query,
                                             const IndexQueryMeta &qmeta,
                                             uint32_t count, const void **out,
                                             IndexQueryMeta *ometa) {
  int ret = 0;

  switch (type_) {
    case kReformerTpNone:
      *out = query;
      *ometa = qmeta;
      break;

    case kReformerTpInnerProductInt8:
      if (qmeta.data_type() != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }
      scales_.resize(count);
      buffer_.resize(count *
                     IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8,
                                              qmeta.dimension()));
      {
        const float *ivec = reinterpret_cast<const float *>(query);
        int8_t *ovec = reinterpret_cast<int8_t *>(&buffer_[0]);
        for (size_t i = 0; i < count; ++i) {
          this->transform(i, &ivec[i * qmeta.dimension()], qmeta.dimension(),
                          &ovec[i * qmeta.dimension()]);
        }
      }
      *ometa = qmeta;
      ometa->set_meta(IndexMeta::DataType::DT_INT8, qmeta.dimension());
      *out = buffer_.data();
      break;

    case kReformerTpInnerProductInt4:
      if (qmeta.data_type() != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }
      scales_.resize(count);
      buffer_.resize(count *
                     IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT4,
                                              qmeta.dimension()));
      {
        const float *ivec = reinterpret_cast<const float *>(query);
        uint8_t *ovec = reinterpret_cast<uint8_t *>(&buffer_[0]);
        for (size_t i = 0; i < count; ++i) {
          this->transform(i, &ivec[i * qmeta.dimension()], qmeta.dimension(),
                          &ovec[i * qmeta.dimension() / 2]);
        }
      }
      *ometa = qmeta;
      ometa->set_meta(IndexMeta::DataType::DT_INT4, qmeta.dimension());
      *out = buffer_.data();
      break;

    case kReformerTpInt8:
    case kReformerTpInt4:
      /* FALLTHRU */
    case kReformerTpDefault:
      ret = reformer_->transform(query, qmeta, count, &buffer_, ometa);
      *out = buffer_.data();
      break;

    default:
      ret = IndexError_Unsupported;
      break;
  }

  return ret;
}

//! Transform querys
int IVFEntity::IVFReformerWrapper::transform_gpu(const void *query,
                                                 const IndexQueryMeta &qmeta,
                                                 uint32_t count,
                                                 const void **out,
                                                 IndexQueryMeta *ometa) {
  int ret = 0;

  switch (type_) {
    case kReformerTpNone:
    case kReformerTpDefault:
      *out = query;
      *ometa = qmeta;
      break;

    case kReformerTpInnerProductInt4:
    case kReformerTpInnerProductInt8:
      if (qmeta.data_type() != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }
      scales_.resize(count);
      buffer_.resize(count *
                     IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8,
                                              qmeta.dimension()));
      {
        const float *ivec = reinterpret_cast<const float *>(query);
        int8_t *ovec = reinterpret_cast<int8_t *>(&buffer_[0]);
        for (size_t i = 0; i < count; ++i) {
          this->transform(i, &ivec[i * qmeta.dimension()], qmeta.dimension(),
                          &ovec[i * qmeta.dimension()]);
        }
      }
      *ometa = qmeta;
      ometa->set_meta(IndexMeta::DataType::DT_INT8, qmeta.dimension());
      *out = buffer_.data();
      break;

    case kReformerTpInt8:
    case kReformerTpInt4:
      ret = reformer_->transform(query, qmeta, count, &buffer_, ometa);
      *out = buffer_.data();
      break;

    default:
      ret = IndexError_Unsupported;
      break;
  }

  return ret;
}


//! Convert a record
int IVFEntity::IVFReformerWrapper::convert(const void *record,
                                           const IndexQueryMeta &rmeta,
                                           const void **out,
                                           IndexQueryMeta *ometa) {
  if (type_ == kReformerTpNone) {
    *out = record;
    *ometa = rmeta;
    return 0;
  }

  int ret = reformer_->convert(record, rmeta, &buffer_, ometa);
  *out = buffer_.data();
  return ret;
}

//! Convert records
int IVFEntity::IVFReformerWrapper::convert(const void *records,
                                           const IndexQueryMeta &rmeta,
                                           uint32_t count, const void **out,
                                           IndexQueryMeta *ometa) {
  if (type_ == kReformerTpNone) {
    *out = records;
    *ometa = rmeta;
    return 0;
  }
  int ret = reformer_->convert(records, rmeta, count, &buffer_, ometa);
  *out = buffer_.data();
  return ret;
}

//! Normalize score
void IVFEntity::IVFReformerWrapper::normalize(size_t qidx,
                                              IndexDocumentHeap *heap) const {
  switch (type_) {
    case kReformerTpNone:
      return;

    case kReformerTpInnerProductInt8:
    case kReformerTpInnerProductInt4:
      ailego_assert_with(qidx < scales_.size(), "invalid index");
      {
        auto reciprocal = 1.0f / scales_[qidx];
        for (auto &it : heap->mutable_container()) {
          *it.mutable_score() *= reciprocal;
        }
      }
      break;

    case kReformerTpInt8:
    case kReformerTpInt4:
      for (auto &it : heap->mutable_container()) {
        *it.mutable_score() *= reciprocal_;
      }
      break;

    default:
      // Not support
      break;
  }
}

//! Normalize score
void IVFEntity::IVFReformerWrapper::normalize(size_t qidx, const void *query,
                                              const IndexQueryMeta &qmeta,
                                              IndexDocumentHeap *heap) const {
  switch (type_) {
    case kReformerTpNone:
      return;

    case kReformerTpInnerProductInt8:
    case kReformerTpInnerProductInt4:
      ailego_assert_with(qidx < scales_.size(), "invalid index");
      {
        auto reciprocal = 1.0f / scales_[qidx];
        for (auto &it : heap->mutable_container()) {
          *it.mutable_score() *= reciprocal;
        }
      }
      break;

    case kReformerTpInt8:
    case kReformerTpInt4:
      for (auto &it : heap->mutable_container()) {
        *it.mutable_score() *= reciprocal_;
      }
      break;

    case kReformerTpDefault:
      reformer_->normalize(query, qmeta, heap->mutable_container());
      break;

    default:
      // Not support
      LOG_ERROR("Not a supported type in QC reformer, type: %u", type_);
      break;
  }
}

void IVFEntity::IVFReformerWrapper::transform(size_t qidx, const float *in,
                                              size_t dim, int8_t *out) {
  ailego_assert_with(qidx < scales_.size(), "invalid index");

  float abs_max = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    auto abs = std::abs(in[i]);
    if (abs > abs_max) {
      abs_max = abs;
    }
  }

  if (abs_max > 0.0f) {
    float scale = 127 / abs_max;
    for (size_t i = 0; i < dim; ++i) {
      out[i] = static_cast<int8_t>(std::round(in[i] * scale));
    }
    scales_[qidx] = scale;
  } else {
    std::fill(out, out + dim, static_cast<int8_t>(1));
    scales_[qidx] = std::numeric_limits<float>::max();
  }
}

void IVFEntity::IVFReformerWrapper::transform(size_t qidx, const float *in,
                                              size_t dim, uint8_t *out) {
  ailego_assert_with(qidx < scales_.size(), "invalid index");
  ailego_assert_with(dim % 2 == 0, "invalid dim");

  float abs_max = 0.0f;
  float max = -std::numeric_limits<float>::max();
  for (size_t i = 0; i < dim; ++i) {
    float abs = std::abs(in[i]);
    abs_max = std::max(abs_max, abs);
    max = std::max(max, in[i]);
  }
  if (abs_max > 0.0f) {
    float scale = ((7 * abs_max > 8 * max) ? 8 : 7) / abs_max;
    for (size_t i = 0; i < dim; i += 2) {
      auto v1 = static_cast<int8_t>(std::round(in[i] * scale));
      auto v2 = static_cast<int8_t>(std::round(in[i + 1] * scale));
      out[i / 2] =
          (static_cast<uint8_t>(v1) & 0xF) | (static_cast<uint8_t>(v2) << 4);
    }
    scales_[qidx] = scale;
  } else {
    std::fill(out, out + dim / 2, static_cast<uint8_t>(9));
    scales_[qidx] = std::numeric_limits<float>::max();
  }
}

int IVFEntity::load_header(const IndexStorage::Pointer &container) {
  //! Load the Header Segment
  auto header = container->get(IVF_INVERTED_HEADER_SEG_ID);
  if (!header) {
    LOG_ERROR("Failed to get segment %s", IVF_INVERTED_HEADER_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  if (header->data_size() < sizeof(header_)) {
    LOG_ERROR("Invalid format for segment %s",
              IVF_INVERTED_HEADER_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock data;
  if (header->read(0, data, header->data_size()) != header->data_size()) {
    LOG_ERROR("Failed to read data, segment %s",
              IVF_INVERTED_HEADER_SEG_ID.c_str());
    return IndexError_ReadData;
  }
  std::memcpy(&header_, data.data(), sizeof(header_));
  if (header_.header_size < sizeof(header_) + header_.index_meta_size ||
      header_.header_size > header->data_size()) {
    LOG_ERROR("Invalid header size %u", header_.header_size);
    return IndexError_InvalidFormat;
  }

  //! Load the index meta
  if (!meta_.deserialize(
          reinterpret_cast<const uint8_t *>(data.data()) + sizeof(header_),
          header_.index_meta_size)) {
    LOG_ERROR("Failed to deserialize index meta");
    return IndexError_InvalidFormat;
  }

  if (header_.block_vector_count == 0 || meta_.element_size() == 0 ||
      header_.block_size < static_cast<uint64_t>(header_.block_vector_count) *
                               meta_.element_size()) {
    LOG_ERROR("Invalid IVF posting block layout");
    return IndexError_InvalidFormat;
  }

  const auto &attributes = meta_.attributes();
  if (attributes.has(IVF_TURBO_FORMAT_VERSION_ATTRIBUTE)) {
    uint32_t version = 0;
    if (!attributes.get(IVF_TURBO_FORMAT_VERSION_ATTRIBUTE, &version) ||
        version != kIVFTurboFormatVersion) {
      LOG_ERROR("Unsupported Turbo IVF posting format");
      return IndexError_InvalidFormat;
    }
    // A marked Turbo index must have its state segment, even for FP32.
    return load_quantizer(container);
  }
  // Earlier Turbo indexes have no version attribute but always contain this
  // segment. Legacy indexes may have a quantizer descriptor without Turbo
  // codes.
  if (container->has(IVF_TURBO_QUANTIZER_SEG_ID)) {
    return load_quantizer(container);
  }
  quantizer_.reset();
  query_distance_ = turbo::DistanceImpl{};

  int ret = reformer_.init(meta_);
  ivf_check_error_code(ret);

  //! Create the distance calculator
  auto metric = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric) {
    LOG_ERROR("Failed to create metric %s", meta_.metric_name().c_str());
    return IndexError_NoExist;
  }
  ret = metric->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failed to initialize metric %s", meta_.metric_name().c_str());
    return ret;
  }
  calculator_ = std::make_shared<IVFDistanceCalculator>(
      meta_, metric->query_metric() ? metric->query_metric() : metric,
      header_.block_vector_count);
  if (!calculator_) {
    return IndexError_NoMemory;
  }

  return 0;
}

int IVFEntity::load_quantizer(const IndexStorage::Pointer &container) {
  if (meta_.quantizer_name().empty() ||
      meta_.major_order() != IndexMeta::MajorOrder::MO_ROW ||
      !meta_.reformer_name().empty() || header_.block_vector_count > 64) {
    LOG_ERROR("Turbo IVF requires row-major codes without a legacy reformer");
    return IndexError_InvalidFormat;
  }

  IndexMeta raw_meta;
  int ret = IndexHelper::DeserializeFromStorage(container.get(), &raw_meta);
  ivf_check_error_code(ret);
  raw_meta.set_quantizer(meta_.quantizer_name(), meta_.quantizer_revision(),
                         meta_.quantizer_params());
  auto quantizer = IndexFactory::CreateQuantizer(meta_.quantizer_name());
  if (!quantizer) {
    LOG_ERROR("Failed to create Turbo quantizer %s",
              meta_.quantizer_name().c_str());
    return IndexError_NoExist;
  }
  if (dynamic_cast<turbo::PackedCodeQuantizer *>(quantizer.get())) {
    LOG_ERROR("Packed Turbo codes are incompatible with row-major IVF");
    return IndexError_Unsupported;
  }
  ret = quantizer->init(raw_meta, meta_.quantizer_params());
  ivf_check_with_msg(ret, "Failed to initialize Turbo IVF quantizer");
  const bool matched_input =
      (raw_meta.data_type() == IndexMeta::DT_FP32 &&
       quantizer->input_data_type() == turbo::DataType::kFp32) ||
      (raw_meta.data_type() == IndexMeta::DT_FP16 &&
       quantizer->input_data_type() == turbo::DataType::kFp16);
  if (!matched_input || raw_meta.extra_meta_size() != 0) {
    LOG_ERROR("Turbo quantizer does not match IVF input metadata");
    return IndexError_InvalidFormat;
  }

  auto segment = container->get(IVF_TURBO_QUANTIZER_SEG_ID);
  if (!segment || (quantizer->require_train() && segment->data_size() == 0)) {
    LOG_ERROR("Missing Turbo IVF quantizer state");
    return IndexError_InvalidFormat;
  }
  if (segment->data_size() != 0) {
    IndexStorage::MemoryBlock data;
    if (segment->read(0, data, segment->data_size()) != segment->data_size()) {
      return IndexError_ReadData;
    }
    ret = quantizer->deserialize(data.data(), segment->data_size());
    ivf_check_with_msg(ret, "Failed to restore Turbo IVF quantizer");
  }

  const auto &output = quantizer->meta();
  if (output.data_type() != meta_.data_type() ||
      output.dimension() != meta_.dimension() ||
      output.element_size() != meta_.element_size() ||
      output.extra_meta_size() != meta_.extra_meta_size() ||
      output.metric_name() != meta_.metric_name() ||
      quantizer->quantized_datapoint_vector_length() != meta_.element_size() ||
      quantizer->dim() != static_cast<int>(raw_meta.dimension())) {
    LOG_ERROR("Turbo quantizer does not match the persisted IVF code layout");
    return IndexError_InvalidFormat;
  }

  quantizer_ = std::move(quantizer);
  query_distance_ = turbo::DistanceImpl{};
  calculator_ = std::make_shared<IVFDistanceCalculator>(
      meta_, nullptr, header_.block_vector_count);
  return 0;
}

int IVFEntity::bind_query(const void *query, const IndexQueryMeta &qmeta) {
  if (!quantizer_) {
    return 0;
  }
  query_distance_ = turbo::DistanceImpl{};
  std::string encoded;
  IndexQueryMeta encoded_meta;
  int ret = quantizer_->quantize(query, qmeta, &encoded, &encoded_meta);
  ivf_check_with_msg(ret, "Failed to encode Turbo IVF query");
  if (encoded.size() != quantizer_->quantized_query_vector_length()) {
    LOG_ERROR("Unexpected Turbo IVF query code size");
    return IndexError_InvalidArgument;
  }
  query_distance_ = quantizer_->distance(encoded.data(), encoded_meta);
  if (!query_distance_.valid()) {
    LOG_ERROR("Turbo quantizer does not support IVF posting distances");
    return IndexError_Unsupported;
  }
  return 0;
}

int IVFEntity::load(const IndexStorage::Pointer &container) {
  all_keys_valid_ = false;
  int ret = this->load_header(container);
  ivf_check_error_code(ret);

  //! Load the remaining segments
  container_ = container;

  //! Load reformer state (e.g. rotation matrix) from the main container,
  //! which holds the rotator segment dumped at build time.
  ret = reformer_.load(container);
  ivf_check_error_code(ret);

  size_t expect_size = header_.inverted_body_size;
  inverted_ = load_segment(IVF_INVERTED_BODY_SEG_ID, expect_size);
  if (!inverted_) {
    LOG_ERROR("Failed to load segment, inverted_size=%zu block_count=%u",
              static_cast<size_t>(header_.inverted_body_size),
              header_.block_count);
    return IndexError_InvalidFormat;
  }

  expect_size = header_.inverted_list_count * sizeof(InvertedListMeta);
  inverted_meta_ = load_segment(IVF_INVERTED_META_SEG_ID, expect_size);
  if (!inverted_meta_) {
    LOG_ERROR("Failed to load segment, inverted_lists=%u",
              header_.inverted_list_count);
    return IndexError_InvalidFormat;
  }

  expect_size = header_.total_vector_count * sizeof(uint64_t);
  keys_ = load_segment(IVF_KEYS_SEG_ID, expect_size);
  if (!keys_) {
    return IndexError_InvalidFormat;
  }

  expect_size = header_.total_vector_count * sizeof(InvertedVecLocation);
  offsets_ = load_segment(IVF_OFFSETS_SEG_ID, expect_size);
  if (!offsets_) {
    return IndexError_InvalidFormat;
  }

  expect_size = header_.total_vector_count * sizeof(uint32_t);
  mapping_ = load_segment(IVF_MAPPING_SEG_ID, expect_size);
  if (!mapping_) {
    return IndexError_InvalidFormat;
  }

  // The mapping orders keys ascending, so any invalid key (UINT64_MAX)
  // sorts last. Empty indexes contain no invalid keys either.
  bool all_keys_valid = true;
  if (vector_count() != 0) {
    uint32_t last_id;
    if (get_key_order_mapping(vector_count() - 1, &last_id, 1) != 1) {
      return IndexError_ReadData;
    }
    if (last_id >= vector_count()) {
      LOG_ERROR("Invalid IVF key-order mapping id=%u", last_id);
      return IndexError_InvalidFormat;
    }
    IndexStorage::MemoryBlock key_block;
    const auto *key = get_keys(last_id, 1, key_block);
    if (!key) {
      return IndexError_ReadData;
    }
    all_keys_valid = *key != kInvalidKey;
  }

  norm_value_sqrt_ =
      meta_.metric_name() == "Euclidean" || meta_.metric_name() == "Manhattan";
  if (container_->get(IVF_INT8_QUANTIZED_PARAMS_SEG_ID) ||
      container->get(IVF_INT4_QUANTIZED_PARAMS_SEG_ID)) {
    expect_size =
        header_.inverted_list_count * sizeof(InvertedIntegerQuantizerParams);
    auto &seg_id = meta_.reformer_name() == kInt8ReformerName
                       ? IVF_INT8_QUANTIZED_PARAMS_SEG_ID
                       : IVF_INT4_QUANTIZED_PARAMS_SEG_ID;
    integer_quantizer_params_ = load_segment(seg_id, expect_size);
    if (!integer_quantizer_params_) {
      return IndexError_InvalidFormat;
    }
    norm_value_ = 0.0f;
  } else if (meta_.reformer_name() == kInt8ReformerName ||
             meta_.reformer_name() == kInt4ReformerName) {
    auto &scale_key = meta_.reformer_name() == kInt8ReformerName
                          ? INT8_QUANTIZER_REFORMER_SCALE
                          : INT4_QUANTIZER_REFORMER_SCALE;
    auto scale = meta_.reformer_params().get_as_float(scale_key);
    norm_value_ = this->convert_to_normalize_value(scale);
  } else {
    norm_value_ = 1.0f;
  }

  if (container_->get(IVF_FEATURES_SEG_ID)) {
    features_ = load_segment(IVF_FEATURES_SEG_ID, 0);
    if (!features_) {
      return IndexError_InvalidFormat;
    }
    if ((vector_count() == 0 && features_->data_size() != 0) ||
        (vector_count() != 0 && features_->data_size() % vector_count() != 0)) {
      LOG_ERROR("Invalid featureSegment size=%zu, totalVecs=%zu",
                features_->data_size(), vector_count());
      return IndexError_InvalidFormat;
    }
  }

  all_keys_valid_ = all_keys_valid;
  LOG_DEBUG(
      "Load inverted index done, docs=%u invertedListCnt=%u "
      "elementSize=%u metric=%s reformer=%s",
      header_.total_vector_count, header_.inverted_list_count,
      meta_.element_size(), meta_.metric_name().c_str(),
      meta_.reformer_name().c_str());
  return 0;
}

int IVFEntity::search(size_t inverted_list_id, const void *query,
                      const IndexFilter &filter, uint32_t *scan_count,
                      IndexDocumentHeap *heap,
                      IndexContext::Stats *context_stats) const {
  if (quantizer_ && !query_distance_.valid()) {
    return IndexError_InvalidArgument;
  }
  ailego_assert_with(inverted_list_id < header_.inverted_list_count,
                     "invalid id");
  IndexStorage::MemoryBlock meta_block;
  auto list_meta = this->inverted_list_meta(inverted_list_id, meta_block);
  ivf_assert(list_meta, IndexError_ReadData);

  IndexStorage::MemoryBlock data;
  const size_t block_vecs = header_.block_vector_count;
  std::vector<float> distances(block_vecs);
  const size_t batch_size = kBatchBlocks;
  const size_t block_size = header_.block_size;
  const size_t element_size = meta_.element_size();
  const auto distance = [this, query](const void *features, size_t count,
                                      float *out) {
    query_features_distance(query, features, count, out);
  };
  const bool row_major = meta_.major_order() == IndexMeta::MO_ROW;
  if (row_major) {
    scatter_vector_.resize(element_size);
  }
  const auto norm_val = this->inverted_list_normalize_value(inverted_list_id);
  for (size_t i = 0; i < list_meta->block_count; i += batch_size) {
    const size_t blocks = std::min(batch_size, list_meta->block_count - i);
    //! Read keys
    size_t items = std::min(blocks * block_vecs,
                            list_meta->vector_count - (i * block_vecs));
    IndexStorage::MemoryBlock keys_block;
    auto keys =
        get_keys(list_meta->id_offset + i * block_vecs, items, keys_block);
    if (!keys) {
      return IndexError_ReadData;
    }

    //! Read vectors after keys. A resident scatter read pins no pages while a
    //! second segment is faulted into a small buffer pool.
    const size_t off = list_meta->offset + i * block_size;
    const size_t size =
        std::min(blocks * block_size,
                 static_cast<size_t>(header_.inverted_body_size - off));
    IndexStorage::Segment::ScatterBlock scatter_data;
    if (row_major) {
      if (inverted_->read_scatter(off, scatter_data, size) != size) {
        LOG_ERROR("Failed to scatter-read block, off=%zu, size=%zu", off, size);
        return IndexError_ReadData;
      }
    } else if (inverted_->read(off, data, size) != size) {
      LOG_ERROR("Failed to read block, off=%zu, size=%zu", off, size);
      return IndexError_ReadData;
    }
    ScatterCursor scatter_cursor(scatter_data);

    //! Compute distances for each block
    for (size_t b = 0; b < blocks; ++b) {
      const size_t vecs_count =
          std::min(block_vecs, list_meta->vector_count - (i + b) * block_vecs);
      auto block_keys = keys + b * block_vecs;
      size_t keeps = 0;
      ailego_assert_with(block_vecs <= sizeof(keeps) * 8, "bits overflow");
      for (size_t k = 0; k < vecs_count; ++k) {
        if (!filter(block_keys[k])) {
          keeps |= (1ULL << k);
        } else {
          ++(*context_stats->mutable_filtered_count());
        }
      }
      const size_t block_offset = b * block_size;
      if (block_offset > size) {
        LOG_ERROR("Invalid IVF block range, off=%zu, size=%zu", block_offset,
                  size);
        return IndexError_ReadData;
      }
      const size_t serialized_size = std::min(block_size, size - block_offset);
      if (keeps == 0) {
        if (row_major && !calculate_row_major_block(
                             &scatter_cursor, distance, vecs_count,
                             element_size, serialized_size,
                             scatter_vector_.data(), distances.data(), false)) {
          LOG_ERROR("Invalid scatter-read IVF block, off=%zu, size=%zu",
                    off + block_offset, serialized_size);
          return IndexError_ReadData;
        }
        continue;
      }

      if (row_major) {
        if (!calculate_row_major_block(&scatter_cursor, distance, vecs_count,
                                       element_size, serialized_size,
                                       scatter_vector_.data(), distances.data(),
                                       true)) {
          LOG_ERROR("Invalid scatter-read IVF block, off=%zu, size=%zu",
                    off + block_offset, serialized_size);
          return IndexError_ReadData;
        }
      } else {
        const void *block_data =
            static_cast<const char *>(data.data()) + block_offset;
        query_features_distance(query, block_data, vecs_count,
                                distances.data());
      }

      *(context_stats->mutable_dist_calced_count()) += vecs_count;

      uint32_t id_off = list_meta->id_offset + (i + b) * block_vecs;
      for (size_t k = 0; k < vecs_count; ++k) {
        if (keeps & (1ULL << k)) {
          if (block_keys[k] != kInvalidKey) {
            heap->emplace(block_keys[k], distances[k] * norm_val, id_off + k);
          }
        }
      }
    }
  }

  *scan_count = list_meta->vector_count;
  return 0;
}

//! search in inverted list without filter
int IVFEntity::search(size_t inverted_list_id, const void *query,
                      uint32_t *scan_count, IndexDocumentHeap *heap,
                      IndexContext::Stats *context_stats) const {
  if (quantizer_ && !query_distance_.valid()) {
    return IndexError_InvalidArgument;
  }
  ailego_assert_with(inverted_list_id < header_.inverted_list_count,
                     "invalid id");
  IndexStorage::MemoryBlock meta_block;
  auto list_meta = inverted_list_meta(inverted_list_id, meta_block);
  ivf_assert(list_meta, IndexError_ReadData);

  const bool row_major = meta_.major_order() == IndexMeta::MO_ROW;
  const bool packed_rows =
      row_major &&
      header_.block_size == header_.block_vector_count * meta_.element_size();
  const auto *base = inverted_->base_data();
  const auto norm_val = this->inverted_list_normalize_value(inverted_list_id);
  int ret;
  if (packed_rows && base) {
    const size_t block_vecs = header_.block_vector_count;
    if (list_meta->id_offset > header_.total_vector_count ||
        list_meta->vector_count >
            header_.total_vector_count - list_meta->id_offset ||
        block_vecs == 0 ||
        list_meta->block_count !=
            (list_meta->vector_count + block_vecs - 1) / block_vecs) {
      LOG_ERROR("Invalid contiguous IVF list range or block count");
      return IndexError_ReadData;
    }
    // An unlimited result heap needs no selection or additional O(n) buffer.
    if (all_keys_valid_ &&
        heap->limit() != std::numeric_limits<size_t>::max()) {
      ret = search_contiguous_rows_all_valid(list_meta, base, query, norm_val,
                                             heap, context_stats);
    } else {
      ret = search_contiguous_rows(list_meta, base, query, norm_val, heap,
                                   context_stats);
    }
  } else if (packed_rows) {
    ret =
        search_scattered_rows(list_meta, query, norm_val, heap, context_stats);
  } else if (row_major) {
    ret = search_padded_rows(list_meta, query, norm_val, heap, context_stats);
  } else {
    ret = search_column_major(list_meta, query, norm_val, heap, context_stats);
  }
  if (ret != 0) {
    return ret;
  }
  *scan_count = list_meta->vector_count;
  return 0;
}

int IVFEntity::search_contiguous_rows_all_valid(
    const InvertedListMeta *list_meta, const uint8_t *base, const void *query,
    float norm_val, IndexDocumentHeap *heap,
    IndexContext::Stats *context_stats) const {
  // Select once for the entire list. Each instantiated scan calls its concrete
  // BatchTopK kernel directly throughout the block and distance-batch loops.
  using Impl = BatchTopK::Implementation;
  switch (BatchTopK::implementation()) {
    case Impl::kAVX512:
      return search_contiguous_rows_all_valid_impl<
          BatchTopKImpl<Impl::kAVX512>>(list_meta, base, query, norm_val, heap,
                                        context_stats);
    case Impl::kAVX2:
      return search_contiguous_rows_all_valid_impl<BatchTopKImpl<Impl::kAVX2>>(
          list_meta, base, query, norm_val, heap, context_stats);
    case Impl::kSSE2:
      return search_contiguous_rows_all_valid_impl<BatchTopKImpl<Impl::kSSE2>>(
          list_meta, base, query, norm_val, heap, context_stats);
    case Impl::kNEON:
      return search_contiguous_rows_all_valid_impl<BatchTopKImpl<Impl::kNEON>>(
          list_meta, base, query, norm_val, heap, context_stats);
    default:
      return search_contiguous_rows_all_valid_impl<
          BatchTopKImpl<Impl::kScalar>>(list_meta, base, query, norm_val, heap,
                                        context_stats);
  }
}

template <typename TopK>
int IVFEntity::search_contiguous_rows_all_valid_impl(
    const InvertedListMeta *list_meta, const uint8_t *base, const void *query,
    float norm_val, IndexDocumentHeap *heap,
    IndexContext::Stats *context_stats) const {
  const size_t block_vecs = header_.block_vector_count;
  const size_t batch_blocks =
      std::max(size_t{1}, kDistanceBatchVectors / block_vecs);
  const size_t element_size = meta_.element_size();
  std::array<float, kDistanceBatchVectors> distances;
  const size_t body_size = std::min(
      static_cast<size_t>(header_.inverted_body_size), inverted_->capacity());
  if (element_size == 0 || list_meta->offset > body_size ||
      list_meta->vector_count >
          (body_size - list_meta->offset) / element_size) {
    LOG_ERROR("Invalid contiguous IVF list range, off=%zu",
              static_cast<size_t>(list_meta->offset));
    return IndexError_ReadData;
  }
  float threshold = heap->threshold();
  bool strict_threshold = false;
  if (heap->full() && heap->front().score() <= threshold) {
    threshold = heap->front().score();
    strict_threshold = true;
  }
  TopK topk;
  topk.reset(std::min(heap->limit(), size_t{list_meta->vector_count}),
             threshold, strict_threshold);
  const auto *features = base + list_meta->offset;
  for (size_t i = 0; i < list_meta->block_count; i += batch_blocks) {
    const size_t blocks = std::min(batch_blocks, list_meta->block_count - i);
    const size_t items =
        std::min(blocks * block_vecs, list_meta->vector_count - i * block_vecs);
    const uint32_t id_offset = list_meta->id_offset + i * block_vecs;
    const auto *batch_features = features + i * header_.block_size;
    for (size_t offset = 0; offset < items; offset += kDistanceBatchVectors) {
      const size_t count = std::min(kDistanceBatchVectors, items - offset);
      query_features_distance(query, batch_features + offset * element_size,
                              count, distances.data());
      topk.push_block(distances.data(), count, id_offset + offset, norm_val);
      *(context_stats->mutable_dist_calced_count()) += count;
    }
  }
  topk.finish();
  for (size_t i = 0; i < topk.size(); ++i) {
    const auto &candidate = topk.data()[i];
    // Resolve only the final query results' keys through retrieve_keys().
    heap->emplace(0, candidate.score, candidate.id);
  }
  return 0;
}

int IVFEntity::search_contiguous_rows(
    const InvertedListMeta *list_meta, const uint8_t *base, const void *query,
    float norm_val, IndexDocumentHeap *heap,
    IndexContext::Stats *context_stats) const {
  const size_t block_vecs = header_.block_vector_count;
  const size_t batch_blocks =
      std::max(size_t{1}, kDistanceBatchVectors / block_vecs);
  const size_t element_size = meta_.element_size();
  std::array<float, kDistanceBatchVectors> distances;
  // Stable contiguous storage needs no per-batch read or scatter traversal.
  const size_t body_size = std::min(
      static_cast<size_t>(header_.inverted_body_size), inverted_->capacity());
  if (element_size == 0 || list_meta->offset > body_size ||
      list_meta->vector_count >
          (body_size - list_meta->offset) / element_size) {
    LOG_ERROR("Invalid contiguous IVF list range, off=%zu",
              static_cast<size_t>(list_meta->offset));
    return IndexError_ReadData;
  }
  const auto *features = base + list_meta->offset;
  for (size_t i = 0; i < list_meta->block_count; i += batch_blocks) {
    const size_t blocks = std::min(batch_blocks, list_meta->block_count - i);
    const size_t items =
        std::min(blocks * block_vecs, list_meta->vector_count - i * block_vecs);
    const uint32_t id_offset = list_meta->id_offset + i * block_vecs;
    IndexStorage::MemoryBlock keys_block;
    const auto *keys = get_keys(id_offset, items, keys_block);
    if (!keys) {
      return IndexError_ReadData;
    }
    const auto *batch_features = features + i * header_.block_size;
    // Keep distance batches bounded even for older, larger physical blocks.
    for (size_t offset = 0; offset < items; offset += kDistanceBatchVectors) {
      const size_t count = std::min(kDistanceBatchVectors, items - offset);
      query_features_distance(query, batch_features + offset * element_size,
                              count, distances.data());
      for (size_t k = 0; k < count; ++k) {
        if (keys[offset + k] != kInvalidKey) {
          const uint32_t id = id_offset + offset + k;
          heap->emplace(keys[offset + k], distances[k] * norm_val, id);
        }
      }
      *(context_stats->mutable_dist_calced_count()) += count;
    }
  }
  return 0;
}

int IVFEntity::search_scattered_rows(const InvertedListMeta *list_meta,
                                     const void *query, float norm_val,
                                     IndexDocumentHeap *heap,
                                     IndexContext::Stats *context_stats) const {
  const size_t block_vecs = header_.block_vector_count;
  const size_t batch_blocks =
      std::max(size_t{1}, kDistanceBatchVectors / block_vecs);
  const size_t block_size = header_.block_size;
  const size_t element_size = meta_.element_size();
  const auto distance = [this, query](const void *features, size_t count,
                                      float *out) {
    query_features_distance(query, features, count, out);
  };
  std::array<float, kDistanceBatchVectors> distances;
  // Page-backed rows may span several independently pinned buffers.
  scatter_vector_.resize(element_size);
  for (size_t i = 0; i < list_meta->block_count; i += batch_blocks) {
    const size_t blocks = std::min(batch_blocks, list_meta->block_count - i);
    const size_t items =
        std::min(blocks * block_vecs, list_meta->vector_count - i * block_vecs);
    const uint32_t id_offset = list_meta->id_offset + i * block_vecs;
    IndexStorage::MemoryBlock keys_block;
    const auto *keys = get_keys(id_offset, items, keys_block);
    if (!keys) {
      return IndexError_ReadData;
    }
    const size_t off = list_meta->offset + i * block_size;
    const size_t size =
        std::min(blocks * block_size,
                 static_cast<size_t>(header_.inverted_body_size - off));
    IndexStorage::Segment::ScatterBlock scatter_data;
    if (inverted_->read_scatter(off, scatter_data, size) != size) {
      LOG_ERROR("Failed to scatter-read block, off=%zu, size=%zu", off, size);
      return IndexError_ReadData;
    }
    ScatterCursor cursor(scatter_data);
    for (size_t offset = 0; offset < items; offset += kDistanceBatchVectors) {
      const size_t count = std::min(kDistanceBatchVectors, items - offset);
      if (!calculate_row_major_block(
              &cursor, distance, count, element_size, count * element_size,
              scatter_vector_.data(), distances.data(), true)) {
        LOG_ERROR("Invalid scatter-read IVF batch, off=%zu", off);
        return IndexError_ReadData;
      }
      for (size_t k = 0; k < count; ++k) {
        if (keys[offset + k] != kInvalidKey) {
          heap->emplace(keys[offset + k], distances[k] * norm_val,
                        id_offset + offset + k);
        }
      }
      *(context_stats->mutable_dist_calced_count()) += count;
    }
  }
  return 0;
}

int IVFEntity::search_padded_rows(const InvertedListMeta *list_meta,
                                  const void *query, float norm_val,
                                  IndexDocumentHeap *heap,
                                  IndexContext::Stats *context_stats) const {
  const size_t block_vecs = header_.block_vector_count;
  const size_t batch_blocks =
      std::max(size_t{1}, kDistanceBatchVectors / block_vecs);
  const size_t block_size = header_.block_size;
  const size_t element_size = meta_.element_size();
  const auto distance = [this, query](const void *features, size_t count,
                                      float *out) {
    query_features_distance(query, features, count, out);
  };
  std::array<float, kDistanceBatchVectors> distances;
  // Padded rows must retain physical block boundaries and skip padding.
  scatter_vector_.resize(element_size);
  for (size_t i = 0; i < list_meta->block_count; i += batch_blocks) {
    const size_t blocks = std::min(batch_blocks, list_meta->block_count - i);
    const size_t items =
        std::min(blocks * block_vecs, list_meta->vector_count - i * block_vecs);
    const uint32_t id_offset = list_meta->id_offset + i * block_vecs;
    IndexStorage::MemoryBlock keys_block;
    const auto *keys = get_keys(id_offset, items, keys_block);
    if (!keys) {
      return IndexError_ReadData;
    }
    const size_t off = list_meta->offset + i * block_size;
    const size_t size =
        std::min(blocks * block_size,
                 static_cast<size_t>(header_.inverted_body_size - off));
    IndexStorage::Segment::ScatterBlock scatter_data;
    if (inverted_->read_scatter(off, scatter_data, size) != size) {
      LOG_ERROR("Failed to scatter-read block, off=%zu, size=%zu", off, size);
      return IndexError_ReadData;
    }
    ScatterCursor cursor(scatter_data);
    for (size_t b = 0; b < blocks; ++b) {
      const size_t row_offset = b * block_vecs;
      const size_t count = std::min(block_vecs, items - row_offset);
      const size_t byte_offset = b * block_size;
      if (byte_offset > size || element_size == 0) {
        return IndexError_ReadData;
      }
      const size_t block_bytes = std::min(block_size, size - byte_offset);
      if (count > block_bytes / element_size) {
        return IndexError_ReadData;
      }
      // Older row-major indexes can have physical blocks larger than 512.
      for (size_t offset = 0; offset < count; offset += kDistanceBatchVectors) {
        const size_t batch = std::min(kDistanceBatchVectors, count - offset);
        if (!calculate_row_major_block(
                &cursor, distance, batch, element_size, batch * element_size,
                scatter_vector_.data(), distances.data(), true)) {
          LOG_ERROR("Invalid padded IVF block, off=%zu", off + byte_offset);
          return IndexError_ReadData;
        }
        for (size_t k = 0; k < batch; ++k) {
          const size_t row = row_offset + offset + k;
          if (keys[row] != kInvalidKey) {
            heap->emplace(keys[row], distances[k] * norm_val, id_offset + row);
          }
        }
        *(context_stats->mutable_dist_calced_count()) += batch;
      }
      if (!cursor.skip(block_bytes - count * element_size)) {
        return IndexError_ReadData;
      }
    }
  }
  return 0;
}

int IVFEntity::search_column_major(const InvertedListMeta *list_meta,
                                   const void *query, float norm_val,
                                   IndexDocumentHeap *heap,
                                   IndexContext::Stats *context_stats) const {
  const size_t block_vecs = header_.block_vector_count;
  const size_t batch_blocks =
      std::max(size_t{1}, kDistanceBatchVectors / block_vecs);
  const size_t block_size = header_.block_size;
  const size_t element_size = meta_.element_size();
  std::array<float, kDistanceBatchVectors> distances;
  // Column-major kernels consume one physical matrix block at a time.
  if (block_vecs > kDistanceBatchVectors) {
    LOG_ERROR("Column-major IVF block exceeds distance batch capacity");
    return IndexError_Unsupported;
  }
  IndexStorage::MemoryBlock data;
  for (size_t i = 0; i < list_meta->block_count; i += batch_blocks) {
    const size_t blocks = std::min(batch_blocks, list_meta->block_count - i);
    const size_t items =
        std::min(blocks * block_vecs, list_meta->vector_count - i * block_vecs);
    const uint32_t id_offset = list_meta->id_offset + i * block_vecs;
    IndexStorage::MemoryBlock keys_block;
    const auto *keys = get_keys(id_offset, items, keys_block);
    if (!keys) {
      return IndexError_ReadData;
    }
    const size_t off = list_meta->offset + i * block_size;
    const size_t size =
        std::min(blocks * block_size,
                 static_cast<size_t>(header_.inverted_body_size - off));
    if (inverted_->read(off, data, size) != size) {
      LOG_ERROR("Failed to read block, off=%zu, size=%zu", off, size);
      return IndexError_ReadData;
    }
    const size_t last_block_offset = (blocks - 1) * block_size;
    const size_t last_count = items - (blocks - 1) * block_vecs;
    if (last_block_offset > size || element_size == 0 ||
        last_count > (size - last_block_offset) / element_size) {
      LOG_ERROR("Invalid column-major IVF batch, off=%zu, size=%zu", off, size);
      return IndexError_ReadData;
    }
    const auto *features = static_cast<const uint8_t *>(data.data());
    for (size_t b = 0; b < blocks; ++b) {
      const size_t offset = b * block_vecs;
      const size_t count = std::min(block_vecs, items - offset);
      query_features_distance(query, features + b * block_size, count,
                              distances.data());
      for (size_t k = 0; k < count; ++k) {
        if (keys[offset + k] != kInvalidKey) {
          heap->emplace(keys[offset + k], distances[k] * norm_val,
                        id_offset + offset + k);
        }
      }
      *(context_stats->mutable_dist_calced_count()) += count;
    }
  }
  return 0;
}

//! search all inverted list with filter
int IVFEntity::search(const void *query, const IndexFilter &filter,
                      IndexDocumentHeap *heap,
                      IndexContext::Stats *context_stats) const {
  for (size_t i = 0; i < header_.inverted_list_count; ++i) {
    uint32_t scan_count;
    int ret = this->search(i, query, filter, &scan_count, heap, context_stats);
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

//! search all inverted list without filter
int IVFEntity::search(const void *query, IndexDocumentHeap *heap,
                      IndexContext::Stats *context_stats) const {
  for (size_t i = 0; i < header_.inverted_list_count; ++i) {
    uint32_t scan_count;
    int ret = this->search(i, query, &scan_count, heap, context_stats);
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

int IVFEntity::search_by_p_keys(const void *query,
                                const std::vector<uint64_t> &p_keys,
                                const IndexFilter &filter,
                                IndexDocumentHeap *heap,
                                IndexContext::Stats *context_stats) const {
  if (quantizer_ && !query_distance_.valid()) {
    return IndexError_InvalidArgument;
  }
  for (uint64_t key : p_keys) {
    if (key == kInvalidKey) {
      continue;
    }
    if (filter.is_valid() && filter(key)) {
      ++(*context_stats->mutable_filtered_count());
      continue;
    }
    uint32_t id;
    int ret = key_to_id(key, &id);
    ivf_check_error_code(ret);
    if (id == std::numeric_limits<uint32_t>::max()) {
      continue;
    }
    float norm_value;
    ret = key_normalize_value(id, &norm_value);
    ivf_check_error_code(ret);
    IndexStorage::MemoryBlock vector;
    ret = read_posting_vector(id, vector);
    ivf_check_error_code(ret);
    float distance;
    if (quantizer_) {
      calculator_->query_features_distance(query_distance_, vector.data(), 1,
                                           &distance);
    } else {
      // Column-major postings have been gathered into one row by the reader.
      calculator_->query_features_distance(query, vector.data(), false, 1,
                                           &distance);
    }
    ++(*context_stats->mutable_dist_calced_count());
    heap->emplace(key, distance * norm_value, id);
  }
  return 0;
}

int IVFEntity::key_normalize_value(uint32_t id, float *value) const {
  if (quantizer_ || !integer_quantizer_params_) {
    *value = quantizer_ ? 1.0f : norm_value_;
    return 0;
  }
  // Legacy per-centroid integer quantization uses a different posting scale
  // for each list. Local IDs are contiguous within each inverted list.
  size_t begin = 0;
  size_t end = inverted_list_count();
  while (begin < end) {
    const size_t mid = begin + (end - begin) / 2;
    size_t list_id = mid;
    IndexStorage::MemoryBlock meta_block;
    const auto *list = inverted_list_meta(list_id, meta_block);
    // Empty lists have no ID range. In particular, trailing lists in older
    // indexes can retain id_offset=0, so only compare nonempty lists.
    while (list && list->vector_count == 0 && list_id > begin) {
      list = inverted_list_meta(--list_id, meta_block);
    }
    if (!list) {
      return IndexError_ReadData;
    }
    if (list->vector_count == 0) {
      begin = mid + 1;
      continue;
    }
    if (id < list->id_offset) {
      end = list_id;
    } else if (id - list->id_offset >= list->vector_count) {
      begin = mid + 1;
    } else {
      InvertedIntegerQuantizerParams params;
      if (integer_quantizer_params_->fetch(list_id * sizeof(params), &params,
                                           sizeof(params)) != sizeof(params)) {
        return IndexError_ReadData;
      }
      *value = convert_to_normalize_value(params.scale);
      return 0;
    }
  }
  return IndexError_InvalidFormat;
}

int IVFEntity::get_vector(size_t id, IndexStorage::MemoryBlock &block) const {
  if (features_) {
    size_t element_size = features_->data_size() / vector_count();
    size_t off = id * element_size;
    if (features_->read(off, block, element_size) != element_size) {
      LOG_ERROR("Failed to read segment, off=%zu size=%zu", off, element_size);
      return IndexError_Runtime;
    }
    return 0;
  }


  return read_posting_vector(id, block);
}

int IVFEntity::read_posting_vector(size_t id,
                                   IndexStorage::MemoryBlock &block) const {
  if (id >= vector_count()) {
    return IndexError_InvalidArgument;
  }
  IndexStorage::MemoryBlock data_block;
  size_t size = sizeof(InvertedVecLocation);
  if (offsets_->read(id * size, data_block, size) != size) {
    LOG_ERROR("Failed to read offsets segment, id=%zu", id);
    return IndexError_ReadData;
  }
  const auto loc = *static_cast<const InvertedVecLocation *>(data_block.data());
  data_block.reset();
  if (loc.column_major) {
    vector_.resize(meta_.element_size());
    auto unit_size = IndexMeta::AlignSizeof(meta_.data_type());
    size_t cols = meta_.element_size() / unit_size;
    size_t step = block_vector_count() * unit_size;
    size_t rd_size = step * (cols - 1) + unit_size;
    IndexStorage::MemoryBlock vector_block;
    if (inverted_->read(loc.offset, vector_block, rd_size) != rd_size) {
      LOG_ERROR("Failed to read data, off=%zu size=%zu",
                static_cast<size_t>(loc.offset), rd_size);
      return IndexError_ReadData;
    }
    for (size_t c = 0; c < cols; ++c) {
      vector_.replace(
          c * unit_size, unit_size,
          reinterpret_cast<const char *>(vector_block.data()) + c * step,
          unit_size);
    }
    block.reset(vector_.data());
    return 0;
  } else {
    if (inverted_->read(loc.offset, block, meta_.element_size()) !=
        meta_.element_size()) {
      LOG_ERROR("Failed to read data, off=%zu size=%u",
                static_cast<size_t>(loc.offset), meta_.element_size());
      return IndexError_ReadData;
    }
    return 0;
  }
}

uint32_t IVFEntity::key_to_id(uint64_t key) const {
  uint32_t id = std::numeric_limits<uint32_t>::max();
  if (key_to_id(key, &id) != 0) {
    return std::numeric_limits<uint32_t>::max();
  }
  return id;
}

int IVFEntity::key_to_id(uint64_t key, uint32_t *id) const {
  *id = std::numeric_limits<uint32_t>::max();
  //! Do binary search
  uint32_t start = 0UL;
  uint32_t end = vector_count();
  IndexStorage::MemoryBlock data;
  uint32_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    if (ailego_unlikely(mapping_->read(idx * sizeof(uint32_t), data,
                                       sizeof(uint32_t)) != sizeof(uint32_t))) {
      LOG_ERROR("Failed to read mapping segment, idx=%u", idx);
      return IndexError_ReadData;
    }
    IndexStorage::MemoryBlock key_block;
    uint32_t local_id = *reinterpret_cast<const uint32_t *>(data.data());
    if (local_id >= vector_count()) {
      return IndexError_InvalidFormat;
    }
    if (ailego_unlikely(keys_->read(local_id * sizeof(uint64_t), key_block,
                                    sizeof(uint64_t)) != sizeof(uint64_t))) {
      LOG_ERROR("Read key from segment failed");
      return IndexError_ReadData;
    }
    const auto mkey = *static_cast<const uint64_t *>(key_block.data());
    if (mkey < key) {
      start = idx + 1;
    } else if (mkey > key) {
      end = idx;
    } else {
      *id = local_id;
      return 0;
    }
  }
  return 0;
}

int IVFEntity::get_vector_by_key(uint64_t key,
                                 IndexStorage::MemoryBlock &block) const {
  uint32_t id = this->key_to_id(key);
  if (id != std::numeric_limits<uint32_t>::max()) {
    return get_vector(id, block);
  } else {
    return IndexError_Runtime;
  }
}

IVFEntity::Pointer IVFEntity::clone() const {
  auto entity = std::make_shared<IVFEntity>();
  return clone(entity);
}

IVFEntity::Pointer IVFEntity::clone(const IVFEntity::Pointer &entity) const {
  if (!entity) {
    LOG_ERROR("Failed to alloc IVFEntity");
    return nullptr;
  }

  auto inverted = inverted_->clone();
  ivf_assert_with_msg(inverted, nullptr, "Failed to clone inverted segment");

  auto inverted_meta = inverted_meta_->clone();
  ivf_assert_with_msg(inverted_meta, nullptr,
                      "Failed to clone inverted meta segment");

  auto keys = keys_->clone();
  ivf_assert_with_msg(keys, nullptr, "Failed to clone keys segment");

  auto offsets = offsets_->clone();
  ivf_assert_with_msg(offsets, nullptr, "Failed to clone offsets segment");

  auto mapping = mapping_->clone();
  ivf_assert_with_msg(mapping, nullptr, "Failed to clone mapping segment");

  IndexStorage::Segment::Pointer integer_quantizer_params;
  if (integer_quantizer_params_) {
    integer_quantizer_params = integer_quantizer_params_->clone();
    if (!integer_quantizer_params) {
      LOG_ERROR("Failed to clone integer quantizer params segment");
      return nullptr;
    }
  }
  IndexStorage::Segment::Pointer features;
  if (features_) {
    features = features_->clone();
    if (!features) {
      LOG_ERROR("Failed to clone features segment");
      return nullptr;
    }
  }

  entity->meta_ = this->meta_;
  entity->reformer_ = this->reformer_;
  entity->calculator_ = this->calculator_;
  entity->quantizer_ = this->quantizer_;
  entity->query_distance_ = turbo::DistanceImpl{};
  entity->header_ = this->header_;
  entity->container_ = this->container_;

  entity->inverted_ = inverted;
  entity->inverted_meta_ = inverted_meta;
  entity->keys_ = keys;
  entity->offsets_ = offsets;
  entity->mapping_ = mapping;
  entity->integer_quantizer_params_ = integer_quantizer_params;
  entity->features_ = features;
  entity->norm_value_ = this->norm_value_;
  entity->norm_value_sqrt_ = this->norm_value_sqrt_;
  entity->all_keys_valid_ = this->all_keys_valid_;

  return entity;
}

IndexStorage::Segment::Pointer IVFEntity::load_segment(
    const std::string &seg_id, size_t expect_size) const {
  auto segment = container_->get(seg_id);
  if (!segment) {
    LOG_ERROR("Failed to get segment %s", seg_id.c_str());
    return nullptr;
  }
  if (expect_size && segment->data_size() != expect_size) {
    LOG_ERROR("Invalid segment %s size=%zu, total_vecs=%u", seg_id.c_str(),
              segment->data_size(), header_.total_vector_count);
    return nullptr;
  }
  return segment;
}

}  // namespace core
}  // namespace zvec
