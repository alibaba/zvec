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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <zvec/core/framework/index_framework.h>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace core {

namespace {

void Fp32ToRawUint8Scalar(const float *input, size_t dimension, void *output) {
  auto *target = static_cast<uint8_t *>(output);
  for (size_t i = 0; i < dimension; ++i) {
    const float value = input[i];
    target[i] = !(value > 0.0F)   ? 0
                : value >= 255.0F ? 255
                                  : static_cast<uint8_t>(value);
  }
}

turbo::ConvertFunc ResolveRawUint8ConvertFunc() {
  auto convert = turbo::get_convert_func(turbo::DataType::kUint8);
  return convert ? convert : Fp32ToRawUint8Scalar;
}

}  // namespace

class RawUint8Holder : public IndexHolder {
 public:
  class Iterator : public IndexHolder::Iterator {
   public:
    Iterator(const RawUint8Holder *owner,
             IndexHolder::Iterator::Pointer &&iterator)
        : owner_(owner),
          buffer_(owner->dimension()),
          front_iterator_(std::move(iterator)) {
      transform_record();
    }

    const void *data(void) const override {
      return buffer_.data();
    }

    bool is_valid(void) const override {
      return front_iterator_->is_valid();
    }

    uint64_t key(void) const override {
      return front_iterator_->key();
    }

    void next(void) override {
      front_iterator_->next();
      transform_record();
    }

   private:
    void transform_record(void) {
      if (front_iterator_->is_valid()) {
        owner_->convert_func_(
            static_cast<const float *>(front_iterator_->data()), buffer_.size(),
            buffer_.data());
      }
    }

    const RawUint8Holder *owner_{nullptr};
    std::vector<uint8_t> buffer_{};
    IndexHolder::Iterator::Pointer front_iterator_{};
  };

  RawUint8Holder(IndexHolder::Pointer holder, turbo::ConvertFunc convert_func)
      : front_(std::move(holder)), convert_func_(convert_func) {}

  size_t count(void) const override {
    return front_->count();
  }

  size_t dimension(void) const override {
    return front_->dimension();
  }

  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UINT8;
  }

  size_t element_size(void) const override {
    return dimension() * sizeof(uint8_t);
  }

  bool multipass(void) const override {
    return front_->multipass();
  }

  IndexHolder::Iterator::Pointer create_iterator(void) override {
    auto iterator = front_->create_iterator();
    return iterator ? std::make_unique<Iterator>(this, std::move(iterator))
                    : nullptr;
  }

 private:
  friend class Iterator;
  IndexHolder::Pointer front_{};
  turbo::ConvertFunc convert_func_{nullptr};
};

class RawUint8Converter : public IndexConverter {
 public:
  int init(const IndexMeta &meta, const ailego::Params &) override {
    if (meta.data_type() != IndexMeta::DataType::DT_FP32 ||
        meta.unit_size() != sizeof(float)) {
      LOG_ERROR("RawUint8Converter only supports FP32 input");
      return IndexError_Unsupported;
    }

    convert_func_ = ResolveRawUint8ConvertFunc();
    meta_ = meta;
    meta_.set_meta(IndexMeta::DataType::DT_UINT8, meta.dimension());
    meta_.set_converter("RawUint8Converter", 0, ailego::Params());
    meta_.set_reformer("RawUint8Reformer", 0, ailego::Params());
    return 0;
  }

  int cleanup(void) override {
    return 0;
  }

  int train(IndexHolder::Pointer) override {
    return 0;
  }

  int transform(IndexHolder::Pointer holder) override {
    if (!holder || holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != meta_.dimension()) {
      return IndexError_Mismatch;
    }
    holder_ =
        std::make_shared<RawUint8Holder>(std::move(holder), convert_func_);
    return 0;
  }

  int dump(const IndexDumper::Pointer &) override {
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
  IndexMeta meta_{};
  IndexHolder::Pointer holder_{};
  Stats stats_{};
  turbo::ConvertFunc convert_func_{nullptr};
};

class RawUint8Reformer : public IndexReformer {
 public:
  int init(const ailego::Params &) override {
    convert_func_ = ResolveRawUint8ConvertFunc();
    return 0;
  }

  int cleanup(void) override {
    return 0;
  }

  int load(IndexStorage::Pointer) override {
    return 0;
  }

  int unload(void) override {
    return 0;
  }

  int transform(const void *query, const IndexQueryMeta &query_meta,
                std::string *output,
                IndexQueryMeta *output_meta) const override {
    return transform(query, query_meta, 1, output, output_meta);
  }

  int transform(const void *query, const IndexQueryMeta &query_meta,
                uint32_t count, std::string *output,
                IndexQueryMeta *output_meta) const override {
    if (!query || !output || !output_meta || count == 0 ||
        query_meta.data_type() != IndexMeta::DataType::DT_FP32 ||
        query_meta.unit_size() != sizeof(float)) {
      return IndexError_Unsupported;
    }

    const size_t value_count =
        static_cast<size_t>(query_meta.dimension()) * count;
    output->resize(value_count * sizeof(uint8_t));
    convert_func_(static_cast<const float *>(query), value_count,
                  output->data());
    *output_meta = query_meta;
    output_meta->set_meta(IndexMeta::DataType::DT_UINT8,
                          query_meta.dimension());
    return 0;
  }

  int normalize(const void *, const IndexQueryMeta &,
                IndexDocumentList &) const override {
    return 0;
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const void *input, const IndexQueryMeta &query_meta,
             std::string *output) const override {
    if (!input || !output ||
        query_meta.data_type() != IndexMeta::DataType::DT_UINT8) {
      return IndexError_Unsupported;
    }

    output->resize(query_meta.dimension() * sizeof(float));
    const auto *source = static_cast<const uint8_t *>(input);
    auto *target = reinterpret_cast<float *>(output->data());
    for (size_t i = 0; i < query_meta.dimension(); ++i) {
      target[i] = static_cast<float>(source[i]);
    }
    return 0;
  }

 private:
  turbo::ConvertFunc convert_func_{nullptr};
};

INDEX_FACTORY_REGISTER_CONVERTER(RawUint8Converter);
INDEX_FACTORY_REGISTER_REFORMER(RawUint8Reformer);

}  // namespace core
}  // namespace zvec
