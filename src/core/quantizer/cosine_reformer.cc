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
#include <vector>
#include <ailego/algorithm/integer_quantizer.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math/normalizer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include "record_quantizer.h"
#include "record_rotator.h"

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
  int init(const ailego::Params &params) override {
    params.get(COSINE_REFORMER_ENABLE_ROTATE, &enable_rotate_);
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  //! Auto-detects rotation by checking for rotator segment in storage.
  int load(IndexStorage::Pointer storage) override {
    if (enable_rotate_ || storage->get(RECORD_ROTATOR_SEG_ID)) {
      rotator_ = std::make_shared<RecordRotator>();
      int ret = rotator_->open(storage);
      if (ret != 0) {
        if (enable_rotate_) {
          LOG_ERROR("CosineReformer: load rotator failed, ret=%d", ret);
          rotator_.reset();
          return ret;
        }
        rotator_.reset();
      } else {
        enable_rotate_ = true;
        LOG_DEBUG(
            "CosineReformer: rotator auto-loaded, origin_dim=%zu, "
            "padded_dim=%zu",
            rotator_->dimension(), rotator_->padded_dim());
      }
    }
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

      size_t origin_dimension = qmeta.dimension();
      const float *vec = reinterpret_cast<const float *>(query);

      // Apply rotation if enabled
      std::unique_ptr<float[]> rotate_buffer;
      if (enable_rotate_ && rotator_) {
        rotate_buffer.reset(new float[rotator_->padded_dim()]);
        rotator_->rotate(vec, rotate_buffer.get());
        vec = rotate_buffer.get();
        origin_dimension = rotator_->padded_dim();
      }

      // Normalize (L2)
      float norm = 0.0f;
      std::string normalized_buffer(reinterpret_cast<const char *>(query),
                                    qmeta.element_size());
      float *buf = reinterpret_cast<float *>(&normalized_buffer[0]);
      if (enable_rotate_ && rotator_) {
        // Already rotated, normalize the rotated vector
        ailego::Normalizer<float>::L2(const_cast<float *>(vec),
                                      origin_dimension, &norm);
      } else {
        ailego::Normalizer<float>::L2(buf, origin_dimension, &norm);
        vec = buf;
      }

      ::memcpy(reinterpret_cast<uint8_t *>(&(*out)[0]) + ometa->element_size() -
                   NORM_SIZE,
               &norm, NORM_SIZE);

      if (dst_type_ == IndexMeta::DataType::DT_FP32) {
        ::memcpy(reinterpret_cast<uint8_t *>(&(*out)[0]), vec,
                 ometa->element_size() - NORM_SIZE);
      } else if (dst_type_ == IndexMeta::DataType::DT_FP16) {
        RecordQuantizer::quantize_record(const_cast<float *>(vec),
                                         qmeta.dimension(), dst_type_, false,
                                         &(*out)[0]);
      } else if (dst_type_ == IndexMeta::DataType::DT_INT4 ||
                 dst_type_ == IndexMeta::DataType::DT_INT8) {
        RecordQuantizer::quantize_record(vec, qmeta.dimension(), dst_type_,
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

    // For FP32 input type, rotation may have been applied during transform.
    // For FP16 input type, rotation was NOT applied — skip inverse rotation.
    const bool need_inv_rotate =
        (type == IndexMeta::DataType::DT_FP32 && enable_rotate_ && rotator_);

    if (type == IndexMeta::DataType::DT_FP32) {
      if (dst_type_ != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }

      float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
      const float *in_buf = reinterpret_cast<const float *>(in);

      this->denormalize(in_buf, out_buf, qmeta, norm);
      if (need_inv_rotate) {
        std::vector<float> tmp(dimension);
        rotator_->unrotate(out_buf, tmp.data());
        std::memcpy(out_buf, tmp.data(), dimension * sizeof(float));
      }
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
        // FP16 type path: no rotation was applied, skip inverse
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
      if (need_inv_rotate) {
        std::vector<float> tmp(dimension);
        rotator_->unrotate(out_buf, tmp.data());
        std::memcpy(out_buf, tmp.data(), dimension * sizeof(float));
      }
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
      return 24;  // (5 * sizeof(float) + sizeof(int)) / sizeof(FT_INT8)
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
  bool enable_rotate_{false};
  std::shared_ptr<RecordRotator> rotator_{};
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
