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
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "rotator/rotator.h"
#include "record_quantizer.h"

namespace zvec {
namespace core {

namespace {

void Fp32ToFp16Fallback(const float *input, size_t dimension, void *output) {
  ailego::FloatHelper::ToFP16(input, dimension,
                              static_cast<uint16_t *>(output));
}

turbo::ConvertFunc ResolveFp16ConvertFunc() {
  auto convert = turbo::get_convert_func(turbo::DataType::kFp16);
  return convert ? convert : Fp32ToFp16Fallback;
}

}  // namespace

/*! Reformer of Cosine
 */
class CosineReformer : public IndexReformer {
 public:
  static constexpr size_t NORM_SIZE = sizeof(float);

  //! Constructor
  CosineReformer(IndexMeta::DataType original_type,
                 IndexMeta::DataType dst_type, bool raw_fp16_storage = false)
      : original_type_(original_type),
        dst_type_(dst_type),
        raw_fp16_storage_(raw_fp16_storage) {}

  //! Constructor
  CosineReformer(IndexMeta::DataType dst_type)
      : original_type_(IndexMeta::DataType::DT_FP32), dst_type_(dst_type) {}

  //! Constructor
  CosineReformer()
      : original_type_(IndexMeta::DataType::DT_UNDEFINED),
        dst_type_(IndexMeta::DataType::DT_UNDEFINED) {}

  //! Initialize Reformer
  int init(const ailego::Params & /*params*/) override {
    if (raw_fp16_storage_ && (original_type_ != IndexMeta::DataType::DT_FP32 ||
                              dst_type_ != IndexMeta::DataType::DT_FP16)) {
      LOG_ERROR("Raw FP16 cosine storage requires FP32 input and FP16 output");
      return IndexError_Unsupported;
    }
    if (raw_fp16_storage_) {
      fp16_convert_func_ = ResolveFp16ConvertFunc();
    }
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  //! Auto-detects rotation by checking for rotator segment in storage.
  int load(IndexStorage::Pointer storage) override {
    if (enable_rotate_ || storage->get(ROTATOR_SEG_ID)) {
      int ret = Rotator::open(&rotator_, storage);
      if (ret != 0) {
        if (enable_rotate_) {
          LOG_ERROR("CosineReformer: load rotator failed, ret=%d", ret);
          return ret;
        }
      } else {
        enable_rotate_ = true;
        LOG_DEBUG("CosineReformer: rotator auto-loaded, dim=%zu",
                  rotator_->dimension());
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
      float norm = 0.0f;

      if (raw_fp16_storage_) {
        auto *buf = reinterpret_cast<ailego::Float16 *>(out->data());
        fp16_convert_func_(vec, origin_dimension, buf);
        ailego::Normalizer<ailego::Float16>::L2(buf, origin_dimension, &norm);
        ::memcpy(out->data() + ometa->element_size() - NORM_SIZE, &norm,
                 NORM_SIZE);
        return 0;
      }

      // Fast path: no rotation — matches main branch behavior exactly
      std::string normalized_buffer(reinterpret_cast<const char *>(query),
                                    qmeta.element_size());
      float *buf = reinterpret_cast<float *>(&normalized_buffer[0]);

      if (enable_rotate_ && rotator_) {
        rotator_->rotate(vec, buf);
      }
      ailego::Normalizer<float>::L2(buf, origin_dimension, &norm);
      vec = buf;

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

    // Rotation only applies to INT8/INT4 targets (guarded at converter init).
    // For FP32/FP16 stored types, rotator_ is always null.
    const bool need_inv_rotate = (enable_rotate_ && rotator_);

    if (type == IndexMeta::DataType::DT_FP32) {
      if (dst_type_ != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }

      float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
      const float *in_buf = reinterpret_cast<const float *>(in);

      this->denormalize(in_buf, out_buf, qmeta, norm);
      if (need_inv_rotate) {
        rotator_->unrotate(out_buf, out_buf);
      }
    } else if (type == IndexMeta::DataType::DT_FP16) {
      if (dst_type_ != IndexMeta::DataType::DT_FP16) {
        return IndexError_Unsupported;
      }

      if (original_type_ != IndexMeta::DataType::DT_FP16 &&
          original_type_ != IndexMeta::DataType::DT_FP32) {
        return IndexError_Unsupported;
      }

      if (raw_fp16_storage_) {
        float *out_buf = reinterpret_cast<float *>(out->data());
        const ailego::Float16 *in_buf =
            reinterpret_cast<const ailego::Float16 *>(in);
        for (size_t d = 0; d < dimension; ++d) {
          ailego::Float16 restored;
          restored = static_cast<float>(in_buf[d]) * norm;
          out_buf[d] = static_cast<float>(restored);
        }
      } else if (original_type_ == IndexMeta::DataType::DT_FP32) {
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
        rotator_->unrotate(out_buf, out_buf);
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
  bool raw_fp16_storage_{false};
  turbo::ConvertFunc fp16_convert_func_{nullptr};
  std::shared_ptr<Rotator> rotator_{};
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

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(CosineRawFp16Reformer, CosineReformer,
                                      IndexMeta::DataType::DT_FP32,
                                      IndexMeta::DataType::DT_FP16, true);

}  // namespace core
}  // namespace zvec
