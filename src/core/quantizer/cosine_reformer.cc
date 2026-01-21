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
#include <memory>
#include <ailego/algorithm/integer_quantizer.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math/normalizer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include "record_quantizer.h"

namespace zvec {
namespace core {

/*! Reformer of Cosine
 */
class CosineReformer : public IndexReformer {
 public:
  static constexpr size_t NORM_SIZE = sizeof(float);

  //! Constructor
  CosineReformer(IndexMeta::DataType original_type,
                 IndexMeta::DataType dst_type)
      : original_type_(original_type), dst_type_(dst_type) {}

  //! Constructor
  CosineReformer(IndexMeta::DataType dst_type)
      : original_type_(IndexMeta::DataType::DT_FP32), dst_type_(dst_type) {}

  //! Constructor
  CosineReformer()
      : original_type_(IndexMeta::DataType::DT_UNDEFINED),
        dst_type_(IndexMeta::DataType::DT_UNDEFINED) {}

  //! Initialize Reformer
  int init(const ailego::Params & /*params*/) override {
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
    IndexMeta::DataType type = qmeta.data_type();

    if (type == IndexMeta::DataType::DT_FP32) {
      if (dst_type_ != IndexMeta::DataType::DT_FP32 &&
          dst_type_ != IndexMeta::DataType::DT_FP16 &&
          dst_type_ != IndexMeta::DataType::DT_INT4 &&
          dst_type_ != IndexMeta::DataType::DT_INT8) {
        return IndexError_Unsupported;
      }

      if (qmeta.unit_size() != sizeof(float)) {
        return IndexError_Unsupported;
      }

      *ometa = qmeta;
      ometa->set_meta(dst_type_, qmeta.dimension() + ExtraDimension(dst_type_));
      out->resize(ometa->element_size());

      float norm = 0.0f;
      size_t origin_dimension = qmeta.dimension();
      std::string normalized_buffer(reinterpret_cast<const char *>(query),
                                    qmeta.element_size());

      float *buf = reinterpret_cast<float *>(&normalized_buffer[0]);

      ailego::Normalizer<float>::L2(buf, origin_dimension, &norm);

      ::memcpy(reinterpret_cast<uint8_t *>(&(*out)[0]) + ometa->element_size() -
                   NORM_SIZE,
               &norm, NORM_SIZE);

      if (dst_type_ == IndexMeta::DataType::DT_FP32) {
        ::memcpy(reinterpret_cast<uint8_t *>(&(*out)[0]), buf,
                 ometa->element_size() - NORM_SIZE);
      } else if (dst_type_ == IndexMeta::DataType::DT_FP16) {
        RecordQuantizer::quantize_record(buf, origin_dimension, dst_type_,
                                         false, &(*out)[0]);
      } else if (dst_type_ == IndexMeta::DataType::DT_INT4 ||
                 dst_type_ == IndexMeta::DataType::DT_INT8) {
        RecordQuantizer::quantize_record(buf, qmeta.dimension(), dst_type_,
                                         false, &(*out)[0]);
      }
    } else if (type == IndexMeta::DataType::DT_FP16) {
      if (dst_type_ != IndexMeta::DataType::DT_FP16) {
        return IndexError_Unsupported;
      }

      if (qmeta.unit_size() != sizeof(ailego::Float16)) {
        return IndexError_Unsupported;
      }

      *ometa = qmeta;
      ometa->set_meta(
          IndexMeta::DataType::DT_FP16,
          qmeta.dimension() + ExtraDimension(IndexMeta::DataType::DT_FP16));
      out->resize(ometa->element_size());

      ::memcpy(reinterpret_cast<uint8_t *>(&(*out)[0]), query,
               ometa->element_size() - NORM_SIZE);

      float norm = 0.0f;
      auto data = reinterpret_cast<ailego::Float16 *>(&(*out)[0]);
      ailego::Normalizer<ailego::Float16>::L2(
          data,
          ometa->dimension() - ExtraDimension(IndexMeta::DataType::DT_FP16),
          &norm);

      ::memcpy(reinterpret_cast<uint8_t *>(&(*out)[0]) + ometa->element_size() -
                   NORM_SIZE,
               &norm, NORM_SIZE);
    } else {
      return IndexError_Unsupported;
    }

    return 0;
  }

  //! Transform queries
  int transform(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                uint32_t /*count*/, std::string * /*out*/,
                IndexQueryMeta * /*ometa*/) const override {
    return IndexError_Unsupported;
  }

  //! Convert records
  int convert(const void * /*records*/, const IndexQueryMeta & /*rmeta*/,
              uint32_t /*count*/, std::string * /*out*/,
              IndexQueryMeta * /*ometa*/) const override {
    return IndexError_Unsupported;
  }

  //! Normalize results
  int normalize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                IndexDocumentList & /*result*/) const override {
    return 0;
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    IndexMeta::DataType type = qmeta.data_type();

    if (type != IndexMeta::DataType::DT_FP32 &&
        type != IndexMeta::DataType::DT_INT8 &&
        type != IndexMeta::DataType::DT_INT4 &&
        type != IndexMeta::DataType::DT_FP16) {
      return IndexError_Unsupported;
    }

    size_t dimension = qmeta.dimension() - ExtraDimension(dst_type_);
    out->resize(dimension * IndexMeta::UnitSizeof(original_type_));

    float norm;
    ::memcpy(&norm,
             reinterpret_cast<const uint8_t *>(in) + qmeta.element_size() -
                 NORM_SIZE,
             NORM_SIZE);

    if (type == IndexMeta::DataType::DT_FP32) {
      if (dst_type_ != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }

      float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
      const float *in_buf = reinterpret_cast<const float *>(in);

      this->denormalize(in_buf, out_buf, qmeta, norm);
    } else if (type == IndexMeta::DataType::DT_FP16) {
      if (dst_type_ != IndexMeta::DataType::DT_FP16) {
        return IndexError_Unsupported;
      }

      if (original_type_ != IndexMeta::DataType::DT_FP16 &&
          original_type_ != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }

      if (original_type_ == IndexMeta::DataType::DT_FP32) {
        float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
        RecordQuantizer::unquantize_record(in, dimension, dst_type_, out_buf);

        this->denormalize(out_buf, out_buf, qmeta, norm);
      } else {
        ailego::Float16 *out_buf =
            reinterpret_cast<ailego::Float16 *>(&(*out)[0]);
        const ailego::Float16 *in_buf =
            reinterpret_cast<const ailego::Float16 *>(in);
        this->denormalize(in_buf, out_buf, qmeta, norm);
      }
    } else if (type == IndexMeta::DataType::DT_INT8 ||
               type == IndexMeta::DataType::DT_INT4) {
      if (dst_type_ != IndexMeta::DataType::DT_INT8 &&
          dst_type_ != IndexMeta::DataType::DT_INT4) {
        return IndexError_Unsupported;
      }

      float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
      RecordQuantizer::unquantize_record(in, dimension, dst_type_, out_buf);

      this->denormalize(out_buf, out_buf, qmeta, norm);
    }

    return 0;
  }

 private:
  template <typename T>
  void denormalize(const T *in, T *out, const IndexQueryMeta &qmeta,
                   float norm) const {
    size_t origin_dim = qmeta.dimension() - ExtraDimension(dst_type_);

    for (size_t d = 0; d < origin_dim; ++d) {
      out[d] = in[d] * norm;
    }
  }

  static size_t ExtraDimension(IndexMeta::DataType type) {
    // The extra quantized params storage size to save for each vector
    if (type == IndexMeta::DataType::DT_INT4)
      return 40;  // 5 * sizeof(float) / sizeof(FT_INT4)
    else if (type == IndexMeta::DataType::DT_INT8)
      return 20;  // 5 * sizeof(float) / sizeof(FT_INT8)
    else if (type == IndexMeta::DataType::DT_FP16)
      return 2;  // sizeof(float) / sizeof(FT_FP16)
    else if (type == IndexMeta::DataType::DT_FP32) {
      return 1;  // sizeof(float) / sizeof(FT_FP32)
    } else {
      return 0;
    }
  }

  //! Members
  IndexMeta::DataType original_type_{IndexMeta::DataType::DT_UNDEFINED};
  IndexMeta::DataType dst_type_{IndexMeta::DataType::DT_UNDEFINED};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(CosineNormalizeReformer, CosineReformer,
                                      IndexMeta::DataType::DT_FP32);

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(CosineFp32Reformer, CosineReformer,
                                      IndexMeta::DataType::DT_FP32);

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(CosineFp16Reformer, CosineReformer,
                                      IndexMeta::DataType::DT_FP16);

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(CosineInt8Reformer, CosineReformer,
                                      IndexMeta::DataType::DT_INT8);

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(CosineInt4Reformer, CosineReformer,
                                      IndexMeta::DataType::DT_INT4);

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(CosineHalfFloatReformer, CosineReformer,
                                      IndexMeta::DataType::DT_FP16,
                                      IndexMeta::DataType::DT_FP16);

}  // namespace core
}  // namespace zvec
