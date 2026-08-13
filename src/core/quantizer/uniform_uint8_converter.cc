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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include "../metric/metric_params.h"

namespace zvec {
namespace core {

class UniformUint8Converter : public IndexConverter {
 public:
  explicit UniformUint8Converter(IndexMeta::DataType /*destination_type*/) {}

  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    meta_ = index_meta;
    original_dimension_ = index_meta.dimension();
    encoded_dimension_ = 0;
    scale_ = 0.0f;
    bias_ = 0.0f;
    holder_.reset();
    stats_.set_trained_count(0);
    stats_.set_transformed_count(0);

    if (original_dimension_ == 0 || original_dimension_ > MAX_DIMENSION) {
      LOG_ERROR("UniformUint8Converter: dimension=%zu must be in [1, %d]",
                original_dimension_, MAX_DIMENSION);
      return IndexError_InvalidArgument;
    }
    encoded_dimension_ = original_dimension_ + kTailBytes;

    meta_.set_converter("UniformUint8Converter", 0, params);
    meta_.set_meta(IndexMeta::DataType::DT_INT8, encoded_dimension_);

    ailego::Params metric_params;
    metric_params.set(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME,
                      index_meta.metric_name());
    meta_.set_metric("UniformUint8", 0, metric_params);

    const bool has_scale = params.get(UNIFORM_UINT8_REFORMER_SCALE, &scale_);
    const bool has_bias = params.get(UNIFORM_UINT8_REFORMER_BIAS, &bias_);
    if (has_scale != has_bias ||
        (has_scale &&
         (!std::isfinite(scale_) || scale_ <= 0.0f || !std::isfinite(bias_)))) {
      LOG_ERROR("UniformUint8Converter: invalid scale/bias params");
      return IndexError_InvalidArgument;
    }

    if (has_scale) {
      ailego::Params reformer_params;
      reformer_params.set(UNIFORM_UINT8_REFORMER_SCALE, scale_);
      reformer_params.set(UNIFORM_UINT8_REFORMER_BIAS, bias_);
      meta_.set_reformer("UniformUint8Reformer", 0, reformer_params);
    }
    return 0;
  }

  int cleanup(void) override {
    stats_.set_trained_count(0);
    stats_.set_transformed_count(0);
    holder_.reset();
    scale_ = 0.0f;
    bias_ = 0.0f;
    return 0;
  }

  int train(IndexHolder::Pointer holder) override {
    if (!holder) {
      LOG_ERROR("UniformUint8Converter: null training holder");
      return IndexError_InvalidArgument;
    }
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != original_dimension_) {
      return IndexError_Mismatch;
    }

    ailego::ElapsedTime timer;
    AILEGO_DEFER([&]() { stats_.set_trained_costtime(timer.milli_seconds()); });
    stats_.set_trained_count(0);

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();
    bool all_integer = true;

    auto iterator = holder->create_iterator();
    if (!iterator) {
      LOG_ERROR("UniformUint8Converter: failed to create iterator");
      return IndexError_Runtime;
    }

    for (; iterator->is_valid(); iterator->next()) {
      const auto *vector = static_cast<const float *>(iterator->data());
      for (size_t i = 0; i < original_dimension_; ++i) {
        const float value = vector[i];
        if (!std::isfinite(value)) {
          LOG_ERROR(
              "UniformUint8Converter: non-finite training value "
              "(record_idx=%zu, dim_idx=%zu, value=%f)",
              static_cast<size_t>(stats_.trained_count()), i, value);
          return IndexError_InvalidArgument;
        }
        global_min = std::min(global_min, value);
        global_max = std::max(global_max, value);
        if (all_integer && std::floor(value) != value) {
          all_integer = false;
        }
      }
      stats_.set_trained_count(stats_.trained_count() + 1);
    }

    if (stats_.trained_count() == 0) {
      LOG_ERROR("UniformUint8Converter: empty training set");
      return IndexError_InvalidArgument;
    }

    const float range = global_max - global_min;
    if (!std::isfinite(range)) {
      LOG_ERROR("UniformUint8Converter: non-finite training range");
      return IndexError_InvalidArgument;
    }
    if (all_integer && range <= 255.0f) {
      scale_ = 1.0f;
      bias_ = -global_min;
    } else {
      constexpr float epsilon = std::numeric_limits<float>::epsilon();
      scale_ = 255.0f / std::max(range, epsilon);
      bias_ = -global_min * scale_;
    }

    ailego::Params reformer_params;
    reformer_params.set(UNIFORM_UINT8_REFORMER_SCALE, scale_);
    reformer_params.set(UNIFORM_UINT8_REFORMER_BIAS, bias_);
    meta_.set_reformer("UniformUint8Reformer", 0, reformer_params);

    ailego::Params converter_params = meta_.converter_params();
    converter_params.set(UNIFORM_UINT8_REFORMER_SCALE, scale_);
    converter_params.set(UNIFORM_UINT8_REFORMER_BIAS, bias_);
    meta_.set_converter(meta_.converter_name(), meta_.converter_revision(),
                        converter_params);

    LOG_INFO(
        "UniformUint8Converter trained: count=%zu min=%f max=%f "
        "scale=%f bias=%f cost=%zums",
        static_cast<size_t>(stats_.trained_count()), global_min, global_max,
        scale_, bias_, static_cast<size_t>(timer.milli_seconds()));
    return 0;
  }

  int transform(IndexHolder::Pointer holder) override {
    if (!holder || holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != original_dimension_ || !std::isfinite(scale_) ||
        scale_ <= 0.0f || !std::isfinite(bias_)) {
      return IndexError_Mismatch;
    }
    stats_.set_transformed_count(stats_.transformed_count() + holder->count());
    holder_ = std::make_shared<UniformUint8Holder>(
        std::move(holder), original_dimension_, encoded_dimension_, scale_,
        bias_);
    return 0;
  }

  int dump(const IndexDumper::Pointer & /*dumper*/) override {
    return 0;
  }

  const Stats &stats(void) const override {
    return stats_;
  }

  IndexHolder::Pointer result(void) const override {
    return holder_;
  }

  const IndexMeta &meta(void) const override {
    return meta_;
  }

 private:
  // Stored squared-sum tail and exact squared L2.
  static_assert(uint64_t{MAX_DIMENSION} * 255 * 255 <=
                (std::numeric_limits<uint32_t>::max)());
  // Signed dot product accumulated by the VNNI query kernel.
  static_assert(uint64_t{MAX_DIMENSION} * 255 * 128 <=
                (std::numeric_limits<int32_t>::max)());
  // Absolute value of sum(q*q) - 256*sum(q), maximized at q=128.
  static_assert(uint64_t{MAX_DIMENSION} * 128 * 128 <=
                (std::numeric_limits<int32_t>::max)());
  static constexpr size_t kTailBytes = sizeof(uint32_t);

  static void EncodeRecord(const float *input, size_t dimension, float scale,
                           float bias, int8_t *output) {
    auto *bytes = reinterpret_cast<uint8_t *>(output);
    for (size_t i = 0; i < dimension; ++i) {
      float value = std::round(input[i] * scale + bias);
      value = std::max(0.0f, std::min(255.0f, value));
      bytes[i] = static_cast<uint8_t>(value);
    }

    int64_t sum_squared = 0;
    for (size_t i = 0; i < dimension; ++i) {
      const int code = static_cast<int>(bytes[i]);
      sum_squared += code * code;
      bytes[i] = static_cast<uint8_t>(code - 128);
    }
    const uint32_t tail = static_cast<uint32_t>(sum_squared);
    std::memcpy(output + dimension, &tail, sizeof(tail));
  }

  class UniformUint8Holder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      Iterator(const UniformUint8Holder *owner,
               IndexHolder::Iterator::Pointer iterator)
          : owner_(owner),
            buffer_(owner->encoded_dimension_, 0),
            iterator_(std::move(iterator)) {
        encode();
      }

      const void *data(void) const override {
        return buffer_.data();
      }

      bool is_valid(void) const override {
        return iterator_ && iterator_->is_valid();
      }

      uint64_t key(void) const override {
        return iterator_->key();
      }

      void next(void) override {
        iterator_->next();
        encode();
      }

     private:
      void encode() {
        if (!is_valid()) {
          return;
        }
        EncodeRecord(static_cast<const float *>(iterator_->data()),
                     owner_->original_dimension_, owner_->scale_, owner_->bias_,
                     buffer_.data());
      }

      const UniformUint8Holder *owner_;
      std::vector<int8_t> buffer_;
      IndexHolder::Iterator::Pointer iterator_;
    };

    UniformUint8Holder(IndexHolder::Pointer holder, size_t original_dimension,
                       size_t encoded_dimension, float scale, float bias)
        : holder_(std::move(holder)),
          original_dimension_(original_dimension),
          encoded_dimension_(encoded_dimension),
          scale_(scale),
          bias_(bias) {}

    size_t count(void) const override {
      return holder_->count();
    }

    size_t dimension(void) const override {
      return encoded_dimension_;
    }

    IndexMeta::DataType data_type(void) const override {
      return IndexMeta::DataType::DT_INT8;
    }

    size_t element_size(void) const override {
      return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8,
                                      encoded_dimension_);
    }

    bool multipass(void) const override {
      return holder_->multipass();
    }

    IndexHolder::Iterator::Pointer create_iterator(void) override {
      auto iterator = holder_->create_iterator();
      return iterator ? std::make_unique<Iterator>(this, std::move(iterator))
                      : nullptr;
    }

   private:
    IndexHolder::Pointer holder_;
    size_t original_dimension_;
    size_t encoded_dimension_;
    float scale_;
    float bias_;
  };

  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  size_t original_dimension_{0};
  size_t encoded_dimension_{0};
  float scale_{0.0f};
  float bias_{0.0f};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(UniformUint8Converter,
                                       UniformUint8Converter,
                                       IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
