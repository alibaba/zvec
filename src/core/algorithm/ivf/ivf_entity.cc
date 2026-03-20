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
#include <iostream>
#include "ivf_utility.h"
namespace zvec {
namespace core {

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
        for (auto &it : *heap) {
          *it.mutable_score() *= reciprocal;
        }
      }
      break;

    case kReformerTpInt8:
    case kReformerTpInt4:
      for (auto &it : *heap) {
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
        for (auto &it : *heap) {
          *it.mutable_score() *= reciprocal;
        }
      }
      break;

    case kReformerTpInt8:
    case kReformerTpInt4:
      for (auto &it : *heap) {
        *it.mutable_score() *= reciprocal_;
      }
      break;

    case kReformerTpDefault:
      reformer_->normalize(query, qmeta, *heap);
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
  const void *data = nullptr;
  if (header->read(0, &data, header->data_size()) != header->data_size()) {
    LOG_ERROR("Failed to read data, segment %s",
              IVF_INVERTED_HEADER_SEG_ID.c_str());
    return IndexError_ReadData;
  }
  std::memcpy(&header_, data, sizeof(header_));
  if (header_.header_size < sizeof(header_) + header_.index_meta_size ||
      header_.header_size > header->data_size()) {
    LOG_ERROR("Invalid header size %u", header_.header_size);
    return IndexError_InvalidFormat;
  }

  //! Load the index meta
  if (!meta_.deserialize(
          reinterpret_cast<const uint8_t *>(data) + sizeof(header_),
          header_.index_meta_size)) {
    LOG_ERROR("Failed to deserialize index meta");
    return IndexError_InvalidFormat;
  }

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

int IVFEntity::load(const IndexStorage::Pointer &container) {
  int ret = this->load_header(container);
  ivf_check_error_code(ret);

  //! Load the remaining segments
  container_ = container;
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
    if (features_->data_size() % vector_count() != 0) {
      LOG_ERROR("Invalid featureSegment size=%zu, totalVecs=%zu",
                features_->data_size(), vector_count());
      return IndexError_InvalidFormat;
    }
  }

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
  ailego_assert_with(inverted_list_id < header_.inverted_list_count,
                     "invalid id");
  auto list_meta = this->inverted_list_meta(inverted_list_id);
  ivf_assert(list_meta, IndexError_ReadData);

  const void *data = nullptr;
  const size_t block_vecs = header_.block_vector_count;
  std::vector<float> distances(block_vecs);
  const size_t batch_size = kBatchBlocks;
  const size_t block_size = header_.block_size;
  const auto norm_val = this->inverted_list_normalize_value(inverted_list_id);
  for (size_t i = 0; i < list_meta->block_count; i += batch_size) {
    //! Read vecs
    const size_t off = list_meta->offset + i * block_size;
    const size_t blocks = std::min(batch_size, list_meta->block_count - i);
    const size_t size =
        std::min(blocks * block_size,
                 static_cast<size_t>(header_.inverted_body_size - off));
    if (inverted_->read(off, &data, size) != size) {
      LOG_ERROR("Failed to read block, off=%zu, size=%zu", off, size);
      return IndexError_ReadData;
    }

    //! Read keys
    size_t items = std::min(blocks * block_vecs,
                            list_meta->vector_count - (i * block_vecs));
    auto keys = get_keys(list_meta->id_offset + i * block_vecs, items);
    if (!keys) {
      return IndexError_ReadData;
    }

    //! Compute distances for each block
    for (size_t b = 0; b < blocks; ++b) {
      const size_t vecs_count =
          std::min(block_vecs, list_meta->vector_count - (i + b) * block_vecs);
      auto block_keys = keys + b * block_vecs;
      size_t keeps = 0;
      ailego_assert_with(block_vecs < sizeof(keeps) * 8, "bits overflow");
      for (size_t k = 0; k < vecs_count; ++k) {
        if (!filter(block_keys[k])) {
          keeps |= (1 << k);
        } else {
          ++(*context_stats->mutable_filtered_count());
        }
      }
      if (keeps == 0) {
        continue;
      }

      const void *block_data = static_cast<const char *>(data) + b * block_size;
      calculator_->query_features_distance(query, block_data, vecs_count,
                                           distances.data());

      *(context_stats->mutable_dist_calced_count()) += vecs_count;

      uint32_t id_off = list_meta->id_offset + (i + b) * block_vecs;
      for (size_t k = 0; k < vecs_count; ++k) {
        if (keeps & (1 << k)) {
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
  ailego_assert_with(inverted_list_id < header_.inverted_list_count,
                     "invalid id");
  auto list_meta = inverted_list_meta(inverted_list_id);
  ivf_assert(list_meta, IndexError_ReadData);

  const void *data = nullptr;
  const size_t block_vecs = header_.block_vector_count;
  std::vector<float> distances(block_vecs);
  const size_t batch_size = kBatchBlocks;
  const size_t block_size = header_.block_size;
  const auto norm_val = this->inverted_list_normalize_value(inverted_list_id);
  for (size_t i = 0; i < list_meta->block_count; i += batch_size) {
    //! Read vecs
    const size_t off = list_meta->offset + i * block_size;
    const size_t blocks = std::min(batch_size, list_meta->block_count - i);
    const size_t size =
        std::min(blocks * block_size,
                 static_cast<size_t>(header_.inverted_body_size - off));
    if (inverted_->read(off, &data, size) != size) {
      LOG_ERROR("Failed to read block, off=%zu, size=%zu", off, size);
      return IndexError_ReadData;
    }

    //! Read keys
    size_t items = std::min(blocks * block_vecs,
                            list_meta->vector_count - (i * block_vecs));
    auto keys = get_keys(list_meta->id_offset + i * block_vecs, items);
    if (!keys) {
      return IndexError_ReadData;
    }

    //! Compute distances for each block
    for (size_t b = 0; b < blocks; ++b) {
      const size_t vecs_count =
          std::min(block_vecs, list_meta->vector_count - (i + b) * block_vecs);
      auto block_keys = keys + b * block_vecs;
      const void *block_data = static_cast<const char *>(data) + b * block_size;
      calculator_->query_features_distance(query, block_data, vecs_count,
                                           distances.data());
      for (size_t k = 0; k < vecs_count; ++k) {
        if (block_keys[k] != kInvalidKey) {
          uint32_t id = list_meta->id_offset + (i + b) * block_vecs + k;
          heap->emplace(block_keys[k], distances[k] * norm_val, id);
        }
      }
      *(context_stats->mutable_dist_calced_count()) += vecs_count;
    }
  }

  *scan_count = list_meta->vector_count;
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

const void *IVFEntity::get_vector(size_t id) const {
  if (features_) {
    const void *data = nullptr;
    size_t element_size = features_->data_size() / vector_count();
    size_t off = id * element_size;
    if (features_->read(off, &data, element_size) != element_size) {
      LOG_ERROR("Failed to read segment, off=%zu size=%zu", off, element_size);
      return nullptr;
    }
    return data;
  }

  const void *data = nullptr;
  size_t size = sizeof(InvertedVecLocation);
  if (offsets_->read(id * size, &data, size) != size) {
    LOG_ERROR("Failed to read offsets segment, id=%zu", id);
    return nullptr;
  }
  auto &loc = *reinterpret_cast<const InvertedVecLocation *>(data);
  if (loc.column_major) {
    vector_.resize(meta_.element_size());
    auto unit_size = IndexMeta::AlignSizeof(meta_.data_type());
    size_t cols = meta_.element_size() / unit_size;
    size_t step = block_vector_count() * unit_size;
    size_t rd_size = step * (cols - 1) + unit_size;
    if (inverted_->read(loc.offset, &data, rd_size) != rd_size) {
      LOG_ERROR("Failed to read data, off=%zu size=%zu",
                static_cast<size_t>(loc.offset), rd_size);
      return nullptr;
    }
    for (size_t c = 0; c < cols; ++c) {
      vector_.replace(c * unit_size, unit_size,
                      reinterpret_cast<const char *>(data) + c * step,
                      unit_size);
    }
    return vector_.data();
  } else {
    if (inverted_->read(loc.offset, &data, meta_.element_size()) !=
        meta_.element_size()) {
      LOG_ERROR("Failed to read data, off=%zu size=%u",
                static_cast<size_t>(loc.offset), meta_.element_size());
      return nullptr;
    }
    return data;
  }
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


  IndexStorage::MemoryBlock data_block;
  size_t size = sizeof(InvertedVecLocation);
  if (offsets_->read(id * size, data_block, size) != size) {
    LOG_ERROR("Failed to read offsets segment, id=%zu", id);
    return IndexError_Runtime;
  }
  const void *data = data_block.data();
  auto &loc = *reinterpret_cast<const InvertedVecLocation *>(data);
  if (loc.column_major) {
    vector_.resize(meta_.element_size());
    auto unit_size = IndexMeta::AlignSizeof(meta_.data_type());
    size_t cols = meta_.element_size() / unit_size;
    size_t step = block_vector_count() * unit_size;
    size_t rd_size = step * (cols - 1) + unit_size;
    if (inverted_->read(loc.offset, &data, rd_size) != rd_size) {
      LOG_ERROR("Failed to read data, off=%zu size=%zu",
                static_cast<size_t>(loc.offset), rd_size);
      return IndexError_Runtime;
    }
    for (size_t c = 0; c < cols; ++c) {
      vector_.replace(c * unit_size, unit_size,
                      reinterpret_cast<const char *>(data) + c * step,
                      unit_size);
    }
    block.reset(vector_.data());
    return 0;
  } else {
    if (inverted_->read(loc.offset, block, meta_.element_size()) !=
        meta_.element_size()) {
      LOG_ERROR("Failed to read data, off=%zu size=%u",
                static_cast<size_t>(loc.offset), meta_.element_size());
      return IndexError_Runtime;
    }
    return 0;
  }
}

uint32_t IVFEntity::key_to_id(uint64_t key) const {
  //! Do binary search
  uint32_t start = 0UL;
  uint32_t end = vector_count();
  const void *data = nullptr;
  uint32_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    if (ailego_unlikely(mapping_->read(idx * sizeof(uint32_t), &data,
                                       sizeof(uint32_t)) != sizeof(uint32_t))) {
      LOG_ERROR("Failed to read mapping segment, idx=%u", idx);
      return std::numeric_limits<uint32_t>::max();
    }
    const uint64_t *mkey;
    uint32_t local_id = *reinterpret_cast<const uint32_t *>(data);
    if (ailego_unlikely(keys_->read(local_id * sizeof(uint64_t),
                                    (const void **)(&mkey),
                                    sizeof(uint64_t)) != sizeof(uint64_t))) {
      LOG_ERROR("Read key from segment failed");
      return std::numeric_limits<uint32_t>::max();
    }
    if (*mkey < key) {
      start = idx + 1;
    } else if (*mkey > key) {
      end = idx;
    } else {
      return local_id;
    }
  }
  return std::numeric_limits<uint32_t>::max();
}

const void *IVFEntity::get_vector_by_key(uint64_t key) const {
  uint32_t id = this->key_to_id(key);
  if (id != std::numeric_limits<uint32_t>::max()) {
    return get_vector(id);
  } else {
    return nullptr;
  }
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

IVFEntity::Pointer IVFEntity::clone(void) const {
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
