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

#include <ailego/algorithm/integer_quantizer.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math/normalizer.h>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include "record_quantizer.h"

namespace zvec {
namespace core {

/*! Integer Quantizer Reformer
 */
template <class Quantizer>
class IntegerQuantizerReformer : public IndexReformer {
 public:
  using IndexReformer::transform;

  //! Constructor
  IntegerQuantizerReformer(IndexMeta::DataType dst_type)
      : data_type_(dst_type) {}

//! Get param name
#define P_NAME(NAME)                                                 \
  data_type_ == IndexMeta::DataType::DT_INT8 ? INT8_QUANTIZER_##NAME \
                                             : INT4_QUANTIZER_##NAME

  //! Initialize Reformer
  int init(const ailego::Params &params) override {
    float bias;
    float scale;
    if (!params.get(P_NAME(REFORMER_BIAS), &bias) ||
        !params.get(P_NAME(REFORMER_SCALE), &scale)) {
      LOG_ERROR("Init IntegerReformer failed, required params bias and scale");
      return IndexError_InvalidArgument;
    }

    quantizer_.set_bias(bias);
    quantizer_.set_scale(scale);

    auto metric = params.get_as_string(P_NAME(REFORMER_METRIC));
    auto reciprocal = scale == 0.0 ? 1.0f : (1.0f / scale);
    if (metric == "SquaredEuclidean") {
      scale_reciprocal_ = reciprocal * reciprocal;
    } else if (metric == "Euclidean") {
      scale_reciprocal_ = reciprocal;
    } else if (metric == "Manhattan") {
      scale_reciprocal_ = reciprocal;
    } else if (metric == "InnerProduct" || metric == "MipsSquaredEuclidean") {
      inner_product_ = true;
      scale_reciprocal_ = reciprocal;  // missing query part
    } else {
      LOG_WARN("Unsupported normalize the score for %s", metric.c_str());
      scale_reciprocal_ = 1.0f;
    }
    LOG_DEBUG("Init integer reformer, bias %f, scale %f", bias, scale);
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    inner_product_ = false;
    return 0;
  }

  //! Load index from container
  int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  int unload(void) override {
    return 0;
  }

  //! Transform query
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = qmeta.data_type();

    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = qmeta;
    ometa->set_meta(data_type_, qmeta.dimension());
    out->resize(
        IndexMeta::ElementSizeof(ometa->data_type(), ometa->dimension()));
    const float *vec = reinterpret_cast<const float *>(query);
    auto ovec = reinterpret_cast<typename Quantizer::ValueType *>(&(*out)[0]);

    if (!inner_product_) {
      quantizer_.encode(vec, qmeta.dimension(), ovec);
    } else {
      this->transform(vec, qmeta.dimension(), ovec);
    }
    return 0;
  }

  //! Transform queries
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = qmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = qmeta;
    ometa->set_meta(data_type_, qmeta.dimension());
    out->resize(count * IndexMeta::ElementSizeof(ometa->data_type(),
                                                 ometa->dimension()));
    const float *vec = reinterpret_cast<const float *>(query);

    if (!inner_product_) {
      quantizer_.encode(
          vec, qmeta.dimension() * count,
          reinterpret_cast<typename Quantizer::ValueType *>(&(*out)[0]));
    } else if (ometa->data_type() == IndexMeta::DataType::DT_INT8) {
      int8_t *ovec = reinterpret_cast<int8_t *>(&(*out)[0]);
      for (size_t i = 0; i < count; ++i) {
        this->transform(&vec[i * qmeta.dimension()], qmeta.dimension(),
                        &ovec[i * qmeta.dimension()]);
      }
    } else {
      uint8_t *ovec = reinterpret_cast<uint8_t *>(&(*out)[0]);
      for (size_t i = 0; i < count; ++i) {
        this->transform(&vec[i * qmeta.dimension()], qmeta.dimension(),
                        &ovec[i * qmeta.dimension() / 2]);
      }
    }

    return 0;
  }

  //! Convert a record
  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = rmeta.data_type();

    if (ft != IndexMeta::DataType::DT_FP32 ||
        rmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = rmeta;
    ometa->set_meta(data_type_, rmeta.dimension());
    out->resize(ometa->element_size());
    const float *vec = reinterpret_cast<const float *>(record);
    auto ovec = reinterpret_cast<typename Quantizer::ValueType *>(&(*out)[0]);

    quantizer_.encode(vec, rmeta.dimension(), ovec);

    return 0;
  }

  //! Convert records
  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = rmeta.data_type();

    if (ft != IndexMeta::DataType::DT_FP32 ||
        rmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = rmeta;
    ometa->set_meta(data_type_, rmeta.dimension());
    out->resize(count * ometa->element_size());
    const float *vec = reinterpret_cast<const float *>(records);
    quantizer_.encode(
        vec, rmeta.dimension() * count,
        reinterpret_cast<typename Quantizer::ValueType *>(&(*out)[0]));

    return 0;
  }

  //! Normalize results
  int normalize(const void *query, const IndexQueryMeta &qmeta,
                IndexDocumentList &result) const override {
    IndexMeta::DataType ft = qmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    auto scale = scale_reciprocal_;
    if (inner_product_) {
      float abs_max = 0.0f;
      const float *vec = static_cast<const float *>(query);
      if (data_type_ == IndexMeta::DataType::DT_INT8) {
        for (size_t i = 0; i < qmeta.dimension(); ++i) {
          float abs = std::abs(vec[i]);
          abs_max = std::max(abs, abs_max);
        }
        scale *= abs_max / 127;
      } else {
        float max = -std::numeric_limits<float>::max();
        for (size_t i = 0; i < qmeta.dimension(); ++i) {
          float abs = std::abs(vec[i]);
          abs_max = std::max(abs_max, abs);
          max = std::max(max, vec[i]);
        }
        scale *= abs_max / ((7 * abs_max > 8 * max) ? 8 : 7);
      }
    }
    for (auto &it : result) {
      *it.mutable_score() *= scale;
    }

    return 0;
  }

 private:
  //! Quantize the query to int8 in InnerProduct
  void transform(const float *in, size_t dim, int8_t *out) const {
    float abs_max = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      float abs = std::abs(in[i]);
      abs_max = std::max(abs, abs_max);
    }
    float scale = 127 / abs_max;
    for (size_t i = 0; i < dim; ++i) {
      out[i] = static_cast<int8_t>(std::round(in[i] * scale));
    }
  }

  //! Quantize the query to int4 in InnerProduct
  void transform(const float *in, size_t dim, uint8_t *out) const {
    float abs_max = 0.0f;
    float max = -std::numeric_limits<float>::max();
    for (size_t i = 0; i < dim; ++i) {
      float abs = std::abs(in[i]);
      abs_max = std::max(abs_max, abs);
      max = std::max(max, in[i]);
    }
    float scale = ((7 * abs_max > 8 * max) ? 8 : 7) / abs_max;
    for (size_t i = 0; i < dim; i += 2) {
      auto lo = std::round(in[i] * scale);
      auto hi = std::round(in[i + 1] * scale);
      out[i / 2] = (static_cast_from_float_to_uint8(hi) << 4) |
                   (static_cast_from_float_to_uint8(lo) & 0xF);
    }
  }

 private:
  //! Members
  Quantizer quantizer_;
  float scale_reciprocal_{1.0};
  bool inner_product_{false};
  IndexMeta::DataType data_type_{};
};


/*! Reformer of Integer Streaming Quantizer
 */
class IntegerStreamingReformer : public IndexReformer {
 public:
  //! Constructor
  IntegerStreamingReformer(IndexMeta::DataType dst_type)
      : data_type_(dst_type),
        extra_dimension_(data_type_ == IndexMeta::DataType::DT_INT8 ? 16 : 32) {
  }

  //! Initialize Reformer
  int init(const ailego::Params &params) override {
    params.get(INTEGER_STREAMING_REFORMER_ENABLE_NORMALIZE, &enable_normalize_);
    params.get(INTEGER_STREAMING_REFORMER_IS_EUCLIDEAN, &is_euclidean_);
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  int unload(void) override {
    return 0;
  }

  //! Transform query
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = qmeta.data_type();

    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = qmeta;
    ometa->set_meta(data_type_, qmeta.dimension() + extra_dimension_);
    out->resize(ometa->element_size());
    const float *vec = reinterpret_cast<const float *>(query);
    std::unique_ptr<float[]> normalized;
    if (enable_normalize_) {
      normalized.reset(new float[qmeta.dimension()]);
      vec = normalize(query, qmeta, normalized.get());
    }

    RecordQuantizer::quantize_record(vec, qmeta.dimension(), data_type_,
                                     is_euclidean_, &(*out)[0]);

    return 0;
  }

  //! Transform queries
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = qmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = qmeta;
    ometa->set_meta(data_type_, qmeta.dimension() + extra_dimension_);
    out->resize(count * ometa->element_size());
    std::unique_ptr<float[]> normalized;
    if (enable_normalize_) {
      normalized.reset(new float[qmeta.dimension()]);
    }
    for (size_t i = 0; i < count; ++i) {
      const float *vec =
          reinterpret_cast<const float *>(query) + i * qmeta.dimension();
      if (enable_normalize_) {
        vec = normalize(vec, qmeta, normalized.get());
      }

      RecordQuantizer::quantize_record(vec, qmeta.dimension(), data_type_,
                                       is_euclidean_,
                                       &(*out)[i * ometa->element_size()]);
    }

    return 0;
  }

  //! Convert a record
  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = rmeta.data_type();

    if (ft != IndexMeta::DataType::DT_FP32 ||
        rmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = rmeta;
    ometa->set_meta(data_type_, rmeta.dimension() + extra_dimension_);
    out->resize(ometa->element_size());
    const float *vec = reinterpret_cast<const float *>(record);
    std::unique_ptr<float[]> normalized;
    if (enable_normalize_) {
      normalized.reset(new float[rmeta.dimension()]);
      vec = normalize(record, rmeta, normalized.get());
    }

    RecordQuantizer::quantize_record(vec, rmeta.dimension(), data_type_,
                                     is_euclidean_, &(*out)[0]);

    return 0;
  }

  //! Convert records
  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = rmeta.data_type();

    if (ft != IndexMeta::DataType::DT_FP32 ||
        rmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = rmeta;
    ometa->set_meta(data_type_, rmeta.dimension() + extra_dimension_);
    out->resize(count * ometa->element_size());
    std::unique_ptr<float[]> normalized;
    if (enable_normalize_) {
      normalized.reset(new float[rmeta.dimension()]);
    }
    for (size_t i = 0; i < count; ++i) {
      const float *vec =
          reinterpret_cast<const float *>(records) + i * rmeta.dimension();
      if (enable_normalize_) {
        vec = normalize(vec, rmeta, normalized.get());
      }

      RecordQuantizer::quantize_record(vec, rmeta.dimension(), data_type_,
                                       is_euclidean_,
                                       &(*out)[i * ometa->element_size()]);
    }

    return 0;
  }

  //! Normalize results
  int normalize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                IndexDocumentList & /*result*/) const override {
    return 0;
  }

 private:
  //! Normalize a query to `normalized`
  float *normalize(const void *query, const IndexQueryMeta &qmeta,
                   float *normalized) const {
    memcpy(normalized, query, qmeta.element_size());
    float norm = 0.0;
    ailego::Normalizer<float>::L2(normalized, qmeta.dimension(), &norm);
    return normalized;
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    if (enable_normalize_) {
      LOG_ERROR("Unsupported revert for normalized value");

      return IndexError_Unsupported;
    }

    out->resize((qmeta.dimension() - extra_dimension_) * sizeof(float));
    float *out_buf = reinterpret_cast<float *>(out->data());

    RecordQuantizer::unquantize_record(in, qmeta.dimension() - extra_dimension_,
                                       data_type_, out_buf);

    return 0;
  }

  //! Members
  IndexMeta::DataType data_type_{};
  uint32_t extra_dimension_{0};
  bool enable_normalize_{false};
  bool is_euclidean_{false};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(
    Int8QuantizerReformer,
    IntegerQuantizerReformer<ailego::EntropyInt8Quantizer>,
    IndexMeta::DataType::DT_INT8);
INDEX_FACTORY_REGISTER_REFORMER_ALIAS(
    Int4QuantizerReformer,
    IntegerQuantizerReformer<ailego::EntropyInt4Quantizer>,
    IndexMeta::DataType::DT_INT4);
INDEX_FACTORY_REGISTER_REFORMER_ALIAS(Int8StreamingReformer,
                                      IntegerStreamingReformer,
                                      IndexMeta::DataType::DT_INT8);
INDEX_FACTORY_REGISTER_REFORMER_ALIAS(Int4StreamingReformer,
                                      IntegerStreamingReformer,
                                      IndexMeta::DataType::DT_INT4);

}  // namespace core
}  // namespace zvec
