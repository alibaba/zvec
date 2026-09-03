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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/turbo/turbo.h>
#include "../metric/metric_params.h"

namespace zvec {
namespace core {
namespace {

constexpr float kAlmostHalf = 0.4999999701976776123046875f;

size_t PaddedDimension(size_t dimension) {
  return (dimension + 127U) / 128U * 128U;
}

uint32_t FloatOrderKey(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000U) != 0 ? ~bits : bits ^ 0x80000000U;
}

float FloatFromOrderKey(uint32_t key) {
  const uint32_t bits = (key & 0x80000000U) != 0 ? key ^ 0x80000000U : ~key;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct RadixSelection {
  size_t rank{0};
  uint32_t prefix{0};
};

bool ChooseBucket(int shift, const std::array<size_t, 256> &histogram,
                  RadixSelection *selection) {
  size_t before = 0;
  for (size_t bucket = 0; bucket < histogram.size(); ++bucket) {
    if (selection->rank < before + histogram[bucket]) {
      selection->rank -= before;
      selection->prefix |= static_cast<uint32_t>(bucket) << shift;
      return true;
    }
    before += histogram[bucket];
  }
  return false;
}

void QuantizeScalar(const float *input, size_t dimension, float minimum,
                    float range, uint8_t *output, size_t encoded_dimension) {
  std::memset(output, 0, encoded_dimension);
  for (size_t d = 0; d < dimension; ++d) {
    float normalized = (input[d] - minimum) / range;
    normalized = std::min(1.0f, std::max(0.0f, normalized));
    const auto code = static_cast<uint8_t>(
        static_cast<int>(normalized * 15.0f + kAlmostHalf));
    if ((d & 1U) == 0) {
      output[d >> 1U] = code;
    } else {
      output[d >> 1U] |= static_cast<uint8_t>(code << 4U);
    }
  }
}

bool IsSupportedSourceType(IndexMeta::DataType data_type) {
  return data_type == IndexMeta::DataType::DT_FP32 ||
         data_type == IndexMeta::DataType::DT_FP16;
}

float SourceValue(const void *record, IndexMeta::DataType data_type,
                  size_t index) {
  switch (data_type) {
    case IndexMeta::DataType::DT_FP32:
      return static_cast<const float *>(record)[index];
    case IndexMeta::DataType::DT_FP16:
      return static_cast<float>(
          static_cast<const ailego::Float16 *>(record)[index]);
    default:
      return std::numeric_limits<float>::quiet_NaN();
  }
}

void DecodeSource(const void *record, IndexMeta::DataType data_type,
                  size_t dimension, std::vector<float> *output) {
  output->resize(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    (*output)[i] = SourceValue(record, data_type, i);
  }
}

}  // namespace

class UniformUint4Converter : public IndexConverter {
 public:
  UniformUint4Converter(IndexMeta::DataType /*dst_type*/) {}
  ~UniformUint4Converter() override = default;

  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    if (index_meta.data_type() != IndexMeta::DataType::DT_FP32) {
      return IndexError_Unsupported;
    }
    const size_t dimension = index_meta.dimension();
    if (dimension == 0 || dimension > MAX_DIMENSION) {
      LOG_ERROR("UniformUint4Converter: dimension=%zu must be in [1, %d]",
                dimension, MAX_DIMENSION);
      return IndexError_InvalidArgument;
    }
    meta_ = index_meta;
    original_dimension_ = dimension;
    encoded_dimension_ = PaddedDimension(original_dimension_) / 2U;
    *stats_.mutable_trained_count() = 0;
    *stats_.mutable_transformed_count() = 0;

    meta_.set_converter("UniformUint4Converter", 0, params);
    meta_.set_meta(IndexMeta::DataType::DT_INT8, encoded_dimension_);

    ailego::Params metric_params;
    metric_params.set(UNIFORM_UINT4_METRIC_ORIGIN_METRIC_NAME,
                      index_meta.metric_name());
    meta_.set_metric("UniformUint4", 0, metric_params);

    const bool has_minimum =
        params.get(UNIFORM_UINT4_REFORMER_MINIMUM, &minimum_);
    const bool has_range = params.get(UNIFORM_UINT4_REFORMER_RANGE, &range_);
    if (has_minimum && has_range && range_ > 0.0f && std::isfinite(minimum_) &&
        std::isfinite(range_)) {
      SetReformerParams();
    }
    return 0;
  }

  int cleanup(void) override {
    *stats_.mutable_trained_count() = 0;
    *stats_.mutable_transformed_count() = 0;
    holder_.reset();
    return 0;
  }

  int train(IndexHolder::Pointer holder) override {
    if (!holder || !IsSupportedSourceType(holder->data_type()) ||
        holder->dimension() != original_dimension_) {
      return IndexError_Mismatch;
    }
    const auto source_type = holder->data_type();

    ailego::ElapsedTime timer;
    AILEGO_DEFER([&]() { stats_.set_trained_costtime(timer.milli_seconds()); });

    size_t record_count = holder->count();
    if (record_count == 0) {
      LOG_ERROR("UniformUint4Converter: empty training set");
      return IndexError_InvalidArgument;
    }
    if (record_count == std::numeric_limits<size_t>::max() ||
        record_count >
            std::numeric_limits<size_t>::max() / original_dimension_) {
      LOG_ERROR("UniformUint4Converter: invalid training count");
      return IndexError_InvalidArgument;
    }

    if (holder->multipass()) {
      const size_t value_count = record_count * original_dimension_;
      // Match reimpl/vamana and KGN exactly: discard
      // floor(float(N) * 0.01) + 1 values at each tail.
      size_t tail =
          static_cast<size_t>(static_cast<float>(value_count) * 0.01f) +
          size_t{1};
      tail = std::max<size_t>(1, std::min(tail, value_count));
      RadixSelection lower{tail - 1, 0};
      RadixSelection upper{value_count - tail, 0};

      for (int shift = 24; shift >= 0; shift -= 8) {
        std::array<size_t, 256> lower_hist{};
        std::array<size_t, 256> upper_hist{};
        const uint32_t prefix_mask =
            shift == 24 ? 0U : ~uint32_t{0} << (shift + 8);
        auto iter = holder->create_iterator();
        if (!iter) {
          LOG_ERROR("UniformUint4Converter: iterator unavailable");
          return IndexError_Runtime;
        }
        size_t actual_records = 0;
        for (; iter->is_valid(); iter->next(), ++actual_records) {
          for (size_t d = 0; d < original_dimension_; ++d) {
            const float value = SourceValue(iter->data(), source_type, d);
            if (!std::isfinite(value)) {
              LOG_ERROR(
                  "UniformUint4Converter: non-finite training "
                  "value (record_idx=%zu, dim_idx=%zu)",
                  actual_records, d);
              return IndexError_InvalidArgument;
            }
            const uint32_t key = FloatOrderKey(value);
            const size_t bucket = (key >> shift) & 0xffU;
            if ((key & prefix_mask) == lower.prefix) ++lower_hist[bucket];
            if ((key & prefix_mask) == upper.prefix) ++upper_hist[bucket];
          }
        }
        if (actual_records != record_count ||
            !ChooseBucket(shift, lower_hist, &lower) ||
            !ChooseBucket(shift, upper_hist, &upper)) {
          LOG_ERROR("UniformUint4Converter: radix selection failed");
          return IndexError_Runtime;
        }
      }
      minimum_ = FloatFromOrderKey(lower.prefix);
      const float maximum = FloatFromOrderKey(upper.prefix);
      range_ = maximum - minimum_;
    } else {
      // IndexConverter wraps one-shot sources in a two-pass holder. Such a
      // holder cannot support exact four-pass order-statistic selection, so
      // retain bounded-memory compatibility by using its exact global range.
      LOG_WARN(
          "UniformUint4Converter: one-pass holder; using global "
          "min/max instead of 1%% clipped calibration");
      float maximum = std::numeric_limits<float>::lowest();
      minimum_ = std::numeric_limits<float>::max();
      auto iter = holder->create_iterator();
      if (!iter) return IndexError_Runtime;
      record_count = 0;
      for (; iter->is_valid(); iter->next(), ++record_count) {
        for (size_t d = 0; d < original_dimension_; ++d) {
          const float value = SourceValue(iter->data(), source_type, d);
          if (!std::isfinite(value)) return IndexError_InvalidArgument;
          minimum_ = std::min(minimum_, value);
          maximum = std::max(maximum, value);
        }
      }
      range_ = maximum - minimum_;
    }

    if (!(range_ > 0.0f)) range_ = 1.0f;
    *stats_.mutable_trained_count() = record_count;
    SetReformerParams();

    ailego::Params converter_params = meta_.converter_params();
    converter_params.set(UNIFORM_UINT4_REFORMER_MINIMUM, minimum_);
    converter_params.set(UNIFORM_UINT4_REFORMER_RANGE, range_);
    converter_params.set(UNIFORM_UINT4_REFORMER_ORIGINAL_DIMENSION,
                         original_dimension_);
    meta_.set_converter(meta_.converter_name(), 0, converter_params);
    LOG_INFO(
        "UniformUint4Converter train done: costtime %zums, "
        "minimum=%f, range=%f, original_dimension=%zu, encoded_dimension=%zu",
        static_cast<size_t>(timer.milli_seconds()), minimum_, range_,
        original_dimension_, encoded_dimension_);
    return 0;
  }

  int transform(IndexHolder::Pointer holder) override {
    if (!holder || !IsSupportedSourceType(holder->data_type()) ||
        holder->dimension() != original_dimension_ || !(range_ > 0.0f)) {
      return IndexError_Mismatch;
    }
    if (holder->count() > 0) {
      *stats_.mutable_transformed_count() += holder->count();
    }
    holder_ = std::make_shared<UniformUint4Holder>(
        std::move(holder), original_dimension_, encoded_dimension_, minimum_,
        range_);
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
  void SetReformerParams() {
    ailego::Params reformer_params;
    reformer_params.set(UNIFORM_UINT4_REFORMER_MINIMUM, minimum_);
    reformer_params.set(UNIFORM_UINT4_REFORMER_RANGE, range_);
    reformer_params.set(UNIFORM_UINT4_REFORMER_ORIGINAL_DIMENSION,
                        original_dimension_);
    meta_.set_reformer("UniformUint4Reformer", 0, reformer_params);
  }

  class UniformUint4Holder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      Iterator(const UniformUint4Holder *owner,
               IndexHolder::Iterator::Pointer &&front)
          : owner_(owner),
            buffer_(owner->encoded_dimension_, 0),
            front_(std::move(front)) {
        Encode();
      }

      const void *data(void) const override {
        return buffer_.data();
      }
      bool is_valid(void) const override {
        return front_->is_valid();
      }
      uint64_t key(void) const override {
        return front_->key();
      }
      void next(void) override {
        front_->next();
        Encode();
      }

     private:
      void Encode() {
        if (!front_->is_valid()) return;
        const float *input = nullptr;
        if (owner_->source_type_ == IndexMeta::DataType::DT_FP32) {
          input = static_cast<const float *>(front_->data());
        } else {
          DecodeSource(front_->data(), owner_->source_type_,
                       owner_->original_dimension_, &decoded_);
          input = decoded_.data();
        }
        if (owner_->quantize_func_) {
          owner_->quantize_func_(input, owner_->original_dimension_,
                                 owner_->minimum_, owner_->range_,
                                 buffer_.data());
        } else {
          QuantizeScalar(input, owner_->original_dimension_, owner_->minimum_,
                         owner_->range_, buffer_.data(),
                         owner_->encoded_dimension_);
        }
      }

      const UniformUint4Holder *owner_{nullptr};
      std::vector<uint8_t> buffer_{};
      std::vector<float> decoded_{};
      IndexHolder::Iterator::Pointer front_{};
    };

    UniformUint4Holder(IndexHolder::Pointer front, size_t original_dimension,
                       size_t encoded_dimension, float minimum, float range)
        : front_(std::move(front)),
          source_type_(front_->data_type()),
          original_dimension_(original_dimension),
          encoded_dimension_(encoded_dimension),
          minimum_(minimum),
          range_(range),
          quantize_func_(
              turbo::get_uniform_uint4_quantize_func(turbo::DataType::kUint4)) {
    }

    size_t count(void) const override {
      return front_->count();
    }
    size_t dimension(void) const override {
      return encoded_dimension_;
    }
    IndexMeta::DataType data_type(void) const override {
      return IndexMeta::DataType::DT_INT8;
    }
    size_t element_size(void) const override {
      return encoded_dimension_;
    }
    bool multipass(void) const override {
      return front_->multipass();
    }
    IndexHolder::Iterator::Pointer create_iterator(void) override {
      auto iter = front_->create_iterator();
      return iter ? IndexHolder::Iterator::Pointer(
                        new Iterator(this, std::move(iter)))
                  : IndexHolder::Iterator::Pointer();
    }

   private:
    IndexHolder::Pointer front_{};
    IndexMeta::DataType source_type_{IndexMeta::DataType::DT_UNDEFINED};
    size_t original_dimension_{0};
    size_t encoded_dimension_{0};
    float minimum_{0.0f};
    float range_{0.0f};
    turbo::UniformUint4QuantizeFunc quantize_func_{nullptr};
  };

  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  size_t original_dimension_{0};
  size_t encoded_dimension_{0};
  float minimum_{0.0f};
  float range_{0.0f};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(UniformUint4Converter,
                                       UniformUint4Converter,
                                       IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
