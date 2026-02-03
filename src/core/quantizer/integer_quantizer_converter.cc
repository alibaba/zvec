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

/*! Integer Quantizer Converter Holder
 */
template <class Quantizer>
class IntegerQuantizerConverterHolder : public IndexHolder {
 public:
  /*! Integer Quantizer Converter Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Constructor
    Iterator(const IntegerQuantizerConverterHolder *owner,
             IndexHolder::Iterator::Pointer &&iter)
        : buffer_(owner->element_size(), 0),
          front_iter_(std::move(iter)),
          quantizer_(owner->quantizer_),
          dim_(owner->dimension()) {
      this->encode_record();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return buffer_.data();
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
      this->encode_record();
    }

   private:
    //! Encode the data by quantizer
    inline void encode_record(void) {
      if (front_iter_->is_valid()) {
        const float *vec = reinterpret_cast<const float *>(front_iter_->data());
        quantizer_->encode(
            vec, dim_,
            reinterpret_cast<typename Quantizer::ValueType *>(buffer_.data()));
      }
    }

    //! Members
    std::vector<uint8_t> buffer_{};
    IndexHolder::Iterator::Pointer front_iter_{};
    std::shared_ptr<Quantizer> quantizer_{};
    size_t dim_{0u};
  };

  //! Constructor
  IntegerQuantizerConverterHolder(IndexHolder::Pointer front,
                                  std::shared_ptr<Quantizer> quantizer,
                                  IndexMeta::DataType tp)
      : front_(std::move(front)),
        quantizer_(std::move(quantizer)),
        data_type_(tp) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return front_->dimension();
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return data_type_;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return IndexMeta::ElementSizeof(this->data_type(), front_->dimension());
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    IndexHolder::Iterator::Pointer iter = front_->create_iterator();
    return iter ? IndexHolder::Iterator::Pointer(
                      new IntegerQuantizerConverterHolder::Iterator(
                          this, std::move(iter)))
                : IndexHolder::Iterator::Pointer();
  }

 private:
  //! Members
  IndexHolder::Pointer front_{};
  std::shared_ptr<Quantizer> quantizer_{};
  IndexMeta::DataType data_type_{};
};


/*! Integer Quantizer Converter
 */
template <class Quantizer>
class IntegerQuantizerConverter : public IndexConverter {
 public:
  //! Constructor
  IntegerQuantizerConverter(IndexMeta::DataType dst_type)
      : data_type_(dst_type) {}

  //! Destructor
  virtual ~IntegerQuantizerConverter() {}

//! Get param name
#define P_NAME(NAME)                                                 \
  data_type_ == IndexMeta::DataType::DT_INT8 ? INT8_QUANTIZER_##NAME \
                                             : INT4_QUANTIZER_##NAME

  //! Initialize Converter
  int init(const IndexMeta &mt, const ailego::Params &params) override {
    if (mt.data_type() != IndexMeta::DataType::DT_FP32 ||
        mt.unit_size() != IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      LOG_ERROR("Unsupported type %d with unit size %u", mt.data_type(),
                mt.unit_size());
      return IndexError_Unsupported;
    }
    quantizer_ = std::make_shared<Quantizer>();
    if (!quantizer_) {
      LOG_ERROR("Malloc EntropyIntegerQuantizer failed");
      return IndexError_NoMemory;
    }

    size_t count;
    if (params.get(P_NAME(CONVERTER_HISTOGRAM_BINS_COUNT), &count)) {
      quantizer_->set_histogram_bins(count);
      LOG_DEBUG("Init Converter with bins=%zu", count);
    }
    float scale;
    if (params.get(P_NAME(CONVERTER_SCALE), &scale)) {
      quantizer_->set_scale(scale);
      LOG_DEBUG("Init Converter with scale=%f", scale);
    }
    float bias = 0.0f;
    if (params.get(P_NAME(CONVERTER_BIAS), &bias)) {
      quantizer_->set_bias(bias);
      LOG_DEBUG("Init Converter with bias=%f", bias);
    }

    meta_ = mt;
    meta_.set_meta(data_type_, meta_.dimension());
    meta_.set_converter(data_type_ == IndexMeta::DataType::DT_INT8
                            ? "Int8QuantizerConverter"
                            : "Int4QuantizerConverter",
                        0, params);

    bool disable_bias = false;
    if (meta_.metric_name() == "InnerProduct" ||
        meta_.metric_name() == "MipsSquaredEuclidean") {
      disable_bias = true;
    }
    params.get(P_NAME(CONVERTER_DISABLE_BIAS), &disable_bias);
    quantizer_->set_non_bias(disable_bias);

    return 0;
  }

  //! Cleanup Converter
  int cleanup(void) override {
    return 0;
  }

  //! Train the data
  int train(IndexHolder::Pointer holder) override {
    if (holder->dimension() != meta_.dimension() ||
        holder->data_type() != IndexMeta::DataType::DT_FP32) {
      return IndexError_Mismatch;
    }

    ailego::ElapsedTime timer;
    AILEGO_DEFER([&]() { stats_.set_trained_costtime(timer.milli_seconds()); });

    if (holder->multipass()) {
      {
        //! step1: compute max/min value
        auto iter = holder->create_iterator();
        if (!iter) {
          LOG_ERROR("Failed to create iterator of holder");
          return IndexError_Runtime;
        }
        float max = -std::numeric_limits<float>::max();
        float min = std::numeric_limits<float>::max();
        for (; iter->is_valid(); iter->next()) {
          const float *vec = reinterpret_cast<const float *>(iter->data());
          for (size_t i = 0; i < meta_.dimension(); ++i) {
            max = std::max(max, vec[i]);
            min = std::min(min, vec[i]);
          }
        }
        quantizer_->set_max(max);
        quantizer_->set_min(min);

        //! step2: feed quantizer with training data
        iter = holder->create_iterator();
        if (!iter) {
          LOG_ERROR("Failed to create iterator of holder");
          return IndexError_Runtime;
        }
        for (; iter->is_valid(); iter->next()) {
          (*stats_.mutable_trained_count())++;
          quantizer_->feed(reinterpret_cast<const float *>(iter->data()),
                           meta_.dimension());
        }
      }
    } else {
      //! step1: compute max/min value
      auto iter = holder->create_iterator();
      if (!iter) {
        LOG_ERROR("Failed to create iterator of holder");
        return IndexError_Runtime;
      }
      std::vector<float> features;
      float max = -std::numeric_limits<float>::max();
      float min = std::numeric_limits<float>::max();
      for (; iter->is_valid(); iter->next()) {
        const float *vec = reinterpret_cast<const float *>(iter->data());
        for (size_t i = 0; i < meta_.dimension(); ++i) {
          max = std::max(max, vec[i]);
          min = std::min(min, vec[i]);
          features.emplace_back(vec[i]);
        }
      }
      quantizer_->set_max(max);
      quantizer_->set_min(min);

      //! step2: feed quantizer with training data
      for (size_t i = 0; i < features.size(); i += meta_.dimension()) {
        quantizer_->feed(&features[i], meta_.dimension());
        (*stats_.mutable_trained_count())++;
      }
    }

    //! step3: feed quantizer with training data
    if (!quantizer_->train()) {
      LOG_ERROR("Quantizer train failed");
      return IndexError_Runtime;
    }

    //! Setting of Integer Reformer
    ailego::Params reformer_params;
    float scale = quantizer_->scale();
    float bias = quantizer_->bias();
    float inf = std::numeric_limits<float>::infinity();
    if (scale == inf || bias == inf) {
      reformer_params.set(P_NAME(REFORMER_SCALE), std::to_string(scale));
      reformer_params.set(P_NAME(REFORMER_BIAS), std::to_string(bias));
    } else {
      reformer_params.set(P_NAME(REFORMER_SCALE), scale);
      reformer_params.set(P_NAME(REFORMER_BIAS), bias);
    }
    reformer_params.set(P_NAME(REFORMER_METRIC), meta_.metric_name());
    meta_.set_reformer(data_type_ == IndexMeta::DataType::DT_INT8
                           ? "Int8QuantizerReformer"
                           : "Int4QuantizerReformer",
                       0, reformer_params);

    ailego::Params params = meta_.converter_params();
    if (scale == inf || bias == inf) {
      params.set(P_NAME(CONVERTER_SCALE), std::to_string(scale));
      params.set(P_NAME(CONVERTER_BIAS), std::to_string(bias));
    } else {
      params.set(P_NAME(CONVERTER_SCALE), scale);
      params.set(P_NAME(CONVERTER_BIAS), bias);
    }
    meta_.set_converter(meta_.converter_name(), 0, params);

    LOG_DEBUG(
        "IntegerQuantizerConverter train done, costtime %zums, scale %f, bias "
        "%f",
        (size_t)timer.milli_seconds(), quantizer_->scale(), quantizer_->bias());

    return 0;
  }

  //! Transform the data
  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != meta_.dimension()) {
      return IndexError_Mismatch;
    }

    if (holder->count() > 0) {
      *stats_.mutable_transformed_count() += holder->count();
    }
    holder_ = std::make_shared<IntegerQuantizerConverterHolder<Quantizer>>(
        holder, quantizer_, data_type_);
    return 0;
  }

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &) override {
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

 private:
  //! Members
  IndexMeta meta_{};
  IndexHolder::Pointer holder_{};
  std::shared_ptr<Quantizer> quantizer_{};
  Stats stats_{};
  IndexMeta::DataType data_type_{};
};


/*! Converter of Integer Streaming Quantizer
 */
class IntegerStreamingConverter : public IndexConverter {
 public:
  //! Constructor
  IntegerStreamingConverter(IndexMeta::DataType dst_type)
      : data_type_(dst_type) {}

  //! Destructor
  ~IntegerStreamingConverter() override {}

  //! Initialize Converter
  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    meta_ = index_meta;
    params.get(INTEGER_STREAMING_CONVERTER_ENABLE_NORMALIZE,
               &enable_normalize_);
    ailego::Params reformer_params;
    if (enable_normalize_) {
      reformer_params.set(INTEGER_STREAMING_REFORMER_ENABLE_NORMALIZE, true);
    }

    is_euclidean_ = index_meta.metric_name() == "MipsSquaredEuclidean" ||
                    index_meta.metric_name() == "SquaredEuclidean" ||
                    index_meta.metric_name() == "Euclidean";
    if (is_euclidean_) {
      reformer_params.set(INTEGER_STREAMING_REFORMER_IS_EUCLIDEAN, true);
    }


    if (data_type_ == IndexMeta::DataType::DT_INT8) {
      meta_.set_converter("Int8StreamingConverter", 0, params);
      meta_.set_reformer("Int8StreamingReformer", 0, reformer_params);
    } else {
      if (index_meta.dimension() % 2) {
        LOG_ERROR("Unsupported dimension %u for INT4 type",
                  index_meta.dimension());
        return IndexError_Unsupported;
      }
      meta_.set_converter("Int4StreamingConverter", 0, params);
      meta_.set_reformer("Int4StreamingReformer", 0, reformer_params);
    }
    ailego::Params metric_params;
    metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME,
                      index_meta.metric_name());
    metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_PARAMS,
                      index_meta.metric_params());
    meta_.set_metric("QuantizedInteger", 0, metric_params);
    meta_.set_meta(data_type_, meta_.dimension() + ExtraDimension(data_type_));
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
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != meta_.dimension() - ExtraDimension(data_type_)) {
      return IndexError_Mismatch;
    }

    *stats_.mutable_transformed_count() += holder->count();
    holder_ = std::make_shared<IntegerStreamingConverterHolder>(
        holder, data_type_, enable_normalize_, is_euclidean_);
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

 private:
  //! IndexHolder for IntegerStreamingConverter
  class IntegerStreamingConverterHolder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      //! Constructor
      Iterator(const IntegerStreamingConverterHolder *owner,
               IndexHolder::Iterator::Pointer &&iter)
          : owner_(owner),
            buffer_(owner->element_size(), 0),
            normalize_buffer_(owner->front_->element_size(), 0),
            front_iter_(std::move(iter)) {
        this->encode_record();
      }

      //! Destructor
      virtual ~Iterator(void) {}

      //! Retrieve pointer of data
      const void *data(void) const override {
        return buffer_.data();
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
        this->encode_record();
      }

     private:
      //! Encode the data by quantizer
      void encode_record(void) {
        if (front_iter_->is_valid()) {
          const float *vec =
              reinterpret_cast<const float *>(front_iter_->data());
          if (owner_->enable_normalize_) {
            float norm = 0.0;
            memcpy((void *)normalize_buffer_.data(), vec,
                   owner_->front_->element_size());
            ailego::Normalizer<float>::L2((float *)normalize_buffer_.data(),
                                          owner_->dimension_, &norm);
            vec = (float *)normalize_buffer_.data();
          }

          RecordQuantizer::quantize_record(
              vec, owner_->dimension_, owner_->data_type(),
              owner_->is_euclidean_, buffer_.data());
        }
      }

      //! Members
      const IntegerStreamingConverterHolder *owner_{nullptr};
      std::vector<uint8_t> buffer_{};
      std::string normalize_buffer_{};
      IndexHolder::Iterator::Pointer front_iter_{};
    };

    //! Constructor
    IntegerStreamingConverterHolder(IndexHolder::Pointer front,
                                    IndexMeta::DataType tp,
                                    bool enable_normalize, bool is_euclidean)
        : front_(std::move(front)),
          data_type_(tp),
          dimension_(front_->dimension()),
          enable_normalize_(enable_normalize),
          is_euclidean_(is_euclidean) {}

    //! Retrieve count of elements in holder (-1 indicates unknown)
    size_t count(void) const override {
      return front_->count();
    }

    //! Retrieve dimension
    size_t dimension(void) const override {
      return dimension_ + ExtraDimension(data_type_);
    }

    //! Retrieve type information
    IndexMeta::DataType data_type(void) const override {
      return data_type_;
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
                        new IntegerStreamingConverterHolder::Iterator(
                            this, std::move(iter)))
                  : IndexHolder::Iterator::Pointer();
    }

   private:
    //! Members
    IndexHolder::Pointer front_{};
    IndexMeta::DataType data_type_{};
    uint32_t dimension_{0};
    bool enable_normalize_{false};
    bool is_euclidean_{false};
  };

  static size_t ExtraDimension(IndexMeta::DataType type) {
    // The extra quantized params storage size to save for each vector
    constexpr size_t kExtraSize = 4 * sizeof(float);
    return type == IndexMeta::DataType::DT_INT8 ? kExtraSize : kExtraSize * 2;
  }

  //! Members
  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  IndexMeta::DataType data_type_{};
  bool enable_normalize_{false};
  bool is_euclidean_{false};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(
    Int8QuantizerConverter,
    IntegerQuantizerConverter<ailego::EntropyInt8Quantizer>,
    IndexMeta::DataType::DT_INT8);
INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(
    Int4QuantizerConverter,
    IntegerQuantizerConverter<ailego::EntropyInt4Quantizer>,
    IndexMeta::DataType::DT_INT4);
INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(Int8StreamingConverter,
                                       IntegerStreamingConverter,
                                       IndexMeta::DataType::DT_INT8);
INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(Int4StreamingConverter,
                                       IntegerStreamingConverter,
                                       IndexMeta::DataType::DT_INT4);

}  // namespace core
}  // namespace zvec
