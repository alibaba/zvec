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
#include <iterator>
#include <ailego/algorithm/integer_quantizer.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math/normalizer.h>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include "record_quantizer.h"
#include "../metric/metric_params.h"

namespace zvec {
namespace core {

/*! Cosine Converter Holder
 */
class CosineConverterHolder : public IndexHolder {
 public:
  static constexpr size_t NORM_SIZE = sizeof(float);

  class Iterator : public IndexHolder::Iterator {
   public:
    //! Constructor
    Iterator(const CosineConverterHolder *owner,
             IndexHolder::Iterator::Pointer &&iter,
             IndexMeta::DataType original_type, IndexMeta::DataType type)
        : owner_(owner),
          front_iter_(std::move(iter)),
          original_type_(original_type),
          type_(type) {
      dimension_ = owner_->dimension(),
      original_dimension_ = dimension_ - ExtraDimension(type_);
      size_t element_size = owner->element_size();

      if (original_type_ == IndexMeta::DataType::DT_FP16) {
        normalize_buffer_.resize(dimension_ * sizeof(ailego::Float16));
      } else {  // original_type_ == IndexMeta::DataType::DT_FP32
        normalize_buffer_.resize(dimension_ * sizeof(float));

        if (type_ == IndexMeta::DataType::DT_FP16 ||
            type_ == IndexMeta::DataType::DT_INT4 ||
            type_ == IndexMeta::DataType::DT_INT8) {
          buffer_.resize(element_size, 0);
        }
      }

      this->convert_record();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return type_ == original_type_ ? normalize_buffer_.data()
                                     : buffer_.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return front_iter_->is_valid();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return front_iter_->key();
    }

    //! Next iterator
    void next(void) override {
      front_iter_->next();
      this->convert_record();
    }

   private:
    //! Encode the data by quantizer
    void convert_record(void) {
      if (!front_iter_->is_valid()) {
        return;
      }

      size_t element_size = owner_->element_size();
      size_t original_element_size =
          IndexMeta::ElementSizeof(original_type_, original_dimension_);

      if (original_type_ == IndexMeta::DataType::DT_FP16) {
        ::memcpy(reinterpret_cast<char *>(&normalize_buffer_[0]),
                 reinterpret_cast<const char *>(front_iter_->data()),
                 original_element_size);

        ailego::Float16 *buf =
            reinterpret_cast<ailego::Float16 *>(&normalize_buffer_[0]);

        float norm = 0.0f;
        ailego::Normalizer<ailego::Float16>::L2(buf, original_dimension_,
                                                &norm);

        ::memcpy(reinterpret_cast<uint16_t *>(&normalize_buffer_[0]) +
                     original_dimension_,
                 &norm, NORM_SIZE);
      } else {  // original_type_ == IndexMeta::DataType::DT_FP32
        ::memcpy(reinterpret_cast<char *>(&normalize_buffer_[0]),
                 reinterpret_cast<const char *>(front_iter_->data()),
                 original_element_size);

        float *buf = reinterpret_cast<float *>(&normalize_buffer_[0]);

        float norm = 0.0f;
        ailego::Normalizer<float>::L2(buf, original_dimension_, &norm);

        if (type_ == IndexMeta::DataType::DT_FP32) {
          ::memcpy(reinterpret_cast<float *>(&normalize_buffer_[0]) +
                       original_dimension_,
                   &norm, NORM_SIZE);
        } else if (type_ == IndexMeta::DataType::DT_FP16) {
          ailego::FloatHelper::ToFP16(
              buf, original_dimension_,
              reinterpret_cast<uint16_t *>(&buffer_[0]));

          ::memcpy(
              reinterpret_cast<uint16_t *>(&buffer_[0]) + original_dimension_,
              &norm, NORM_SIZE);
        } else if (type_ == IndexMeta::DataType::DT_INT4 ||
                   type_ == IndexMeta::DataType::DT_INT8) {
          RecordQuantizer::quantize_record(
              reinterpret_cast<const float *>(normalize_buffer_.data()),
              original_dimension_, type_, false, &buffer_[0]);

          ::memcpy(reinterpret_cast<uint8_t *>(&buffer_[0]) + element_size -
                       NORM_SIZE,
                   &norm, NORM_SIZE);
        }
      }
    }

    //! Members
    const CosineConverterHolder *owner_{nullptr};
    std::string buffer_{};
    std::string normalize_buffer_{};
    IndexHolder::Iterator::Pointer front_iter_{};
    size_t dimension_{0u};
    size_t original_dimension_{0u};
    IndexMeta::DataType original_type_{IndexMeta::DataType::DT_UNDEFINED};
    IndexMeta::DataType type_{IndexMeta::DataType::DT_UNDEFINED};
  };

  //! Constructor
  CosineConverterHolder(IndexHolder::Pointer front,
                        IndexMeta::DataType original_type,
                        IndexMeta::DataType type)
      : front_(std::move(front)),
        original_type_(original_type),
        type_(type),
        dimension_(front_->dimension()) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_ + ExtraDimension(type_);
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return type_;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return IndexMeta::ElementSizeof(this->data_type(), this->dimension());
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    IndexHolder::Iterator::Pointer iter = front_->create_iterator();

    return iter ? IndexHolder::Iterator::Pointer(
                      new CosineConverterHolder::Iterator(this, std::move(iter),
                                                          this->original_type_,
                                                          this->type_))
                : IndexHolder::Iterator::Pointer();
  }

  static size_t ExtraDimension(IndexMeta::DataType type) {
    // The extra quantized params storage size to save for each vector
    if (type == IndexMeta::DataType::DT_INT4)
      return 40;  // 5 * sizeof(float) / sizeof(FT_INT4)
    else if (type == IndexMeta::DataType::DT_INT8)
      return 20;  // 5 * sizeof(float) / sizeof(FT_INT8)
    else if (type == IndexMeta::DataType::DT_FP16)
      return 2;  // 2* sizeof(float) / sizeof(FT_FP16)
    else if (type == IndexMeta::DataType::DT_FP32) {
      return 1;  // sizeof(float) / sizeof(FT_FP32)
    } else {
      return 0;
    }
  }

 private:
  //! Members
  IndexHolder::Pointer front_{};
  IndexMeta::DataType original_type_{};
  IndexMeta::DataType type_{};
  uint32_t dimension_{0};
};

/*! Converter of Cosine
 */
class CosineConverter : public IndexConverter {
 public:
  static constexpr size_t NORM_SIZE = sizeof(float);

 public:
  //! Constructor
  CosineConverter(IndexMeta::DataType original_type,
                  IndexMeta::DataType dst_type)
      : original_type_(original_type), dst_type_(dst_type) {}

  //! Constructor
  CosineConverter(IndexMeta::DataType dst_type)
      : original_type_(IndexMeta::DataType::DT_FP32), dst_type_(dst_type) {}

  CosineConverter()
      : original_type_(IndexMeta::DataType::DT_UNDEFINED),
        dst_type_(IndexMeta::DataType::DT_UNDEFINED) {}

  //! Destructor
  ~CosineConverter() override {}

  //! Initialize Converter
  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    meta_ = index_meta;

    IndexMeta::DataType type = meta_.data_type();

    if (type != original_type_) {
      LOG_ERROR("Orignal Type Not Matched: (%d, %d)", type, original_type_);
      return IndexError_Mismatch;
    }

    if (meta_.unit_size() != IndexMeta::UnitSizeof(type)) {
      LOG_ERROR("Unsupported type %d with unit size %u", type,
                meta_.unit_size());
      return IndexError_Unsupported;
    }

    ailego::Params reformer_params;

    if (dst_type_ == IndexMeta::DataType::DT_INT8) {
      meta_.set_converter("CosineInt8Converter", 0, params);
      meta_.set_reformer("CosineInt8Reformer", 0, reformer_params);

      ailego::Params metric_params;
      metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME,
                        index_meta.metric_name());
      metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_PARAMS,
                        index_meta.metric_params());
      meta_.set_metric("QuantizedInteger", 0, metric_params);
    } else if (dst_type_ == IndexMeta::DataType::DT_INT4) {
      if (index_meta.dimension() % 2) {
        LOG_ERROR("Unsupported dimension %u for INT4 type",
                  index_meta.dimension());
        return IndexError_Unsupported;
      }

      meta_.set_converter("CosineInt4Converter", 0, params);
      meta_.set_reformer("CosineInt4Reformer", 0, reformer_params);

      ailego::Params metric_params;
      metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME,
                        index_meta.metric_name());
      metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_PARAMS,
                        index_meta.metric_params());
      meta_.set_metric("QuantizedInteger", 0, metric_params);
    } else if (dst_type_ == IndexMeta::DataType::DT_FP16) {
      if (original_type_ == IndexMeta::DataType::DT_FP16) {
        meta_.set_reformer("CosineHalfFloatReformer", 0, reformer_params);
        meta_.set_converter("CosineHalfFloatConverter", 0, params);
      } else {
        meta_.set_reformer("CosineFp16Reformer", 0, reformer_params);
        meta_.set_converter("CosineFp16Converter", 0, params);
      }
    } else {
      dst_type_ = type;

      meta_.set_reformer("CosineFp32Reformer", 0, reformer_params);
      meta_.set_converter("CosineFp32Converter", 0, params);
    }

    meta_.set_meta(dst_type_, meta_.dimension() + ExtraDimension(dst_type_));

    return 0;
  }

  //! Cleanup Converter
  virtual int cleanup(void) override {
    *stats_.mutable_transformed_count() = 0;
    return 0;
  }

  //! Train the data
  int train(IndexHolder::Pointer /*holder*/) override {
    return 0;
  }

  //! Transform the data
  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != original_type_ ||
        holder->dimension() != meta_.dimension() - ExtraDimension(dst_type_)) {
      return IndexError_Mismatch;
    }

    *stats_.mutable_transformed_count() += holder->count();

    holder_ = std::make_shared<CosineConverterHolder>(
        holder, holder->data_type(), dst_type_);
    return 0;
  }

  //! Dump index into storage
  int dump(const IndexDumper::Pointer & /*dumper*/) override {
    return 0;
  }

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve a holder as result
  IndexHolder::Pointer result(void) const override {
    return holder_;
  }

  //! Retrieve Index Meta
  const IndexMeta &meta(void) const override {
    return meta_;
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
  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  IndexMeta::DataType original_type_{IndexMeta::DataType::DT_UNDEFINED};
  IndexMeta::DataType dst_type_{IndexMeta::DataType::DT_UNDEFINED};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(CosineNormalizeConverter,
                                       CosineConverter,
                                       IndexMeta::DataType::DT_FP32);

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(CosineFp32Converter, CosineConverter,
                                       IndexMeta::DataType::DT_FP32);

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(CosineFp16Converter, CosineConverter,
                                       IndexMeta::DataType::DT_FP16);

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(CosineInt8Converter, CosineConverter,
                                       IndexMeta::DataType::DT_INT8);

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(CosineInt4Converter, CosineConverter,
                                       IndexMeta::DataType::DT_INT4);

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(CosineHalfFloatConverter,
                                       CosineConverter,
                                       IndexMeta::DataType::DT_FP16,
                                       IndexMeta::DataType::DT_FP16);

}  // namespace core
}  // namespace zvec