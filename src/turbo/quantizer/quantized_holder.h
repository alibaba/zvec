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

#pragma once

#include <memory>
#include <string>
#include <utility>
#include "quantizer.h"

namespace zvec {
namespace turbo {

/*! Quantized Index Holder
 *
 */
class QuantizedIndexHolder : public core::IndexHolder {
 public:
  //! Constructor. The layout is taken from the quantizer, which must already
  //! be initialized (and trained, if it requires training).
  QuantizedIndexHolder(core::IndexHolder::Pointer front,
                       Quantizer::Pointer quantizer)
      : front_(std::move(front)),
        quantizer_(std::move(quantizer)),
        type_(quantizer_->meta().data_type()),
        raw_dim_(quantizer_->meta().dimension()),
        code_bytes_(quantizer_->quantized_datapoint_vector_length()) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve dimension (the raw, non-inflated dimension)
  size_t dimension(void) const override {
    return raw_dim_;
  }

  //! Retrieve type information
  core::IndexMeta::DataType data_type(void) const override {
    return type_;
  }

  //! Retrieve element size in bytes (the full encoded record length)
  size_t element_size(void) const override {
    return code_bytes_;
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  core::IndexHolder::Iterator::Pointer create_iterator(void) override {
    core::IndexHolder::Iterator::Pointer iter = front_->create_iterator();
    if (!iter) {
      return core::IndexHolder::Iterator::Pointer();
    }
    return core::IndexHolder::Iterator::Pointer(
        new Iterator(quantizer_.get(), code_bytes_, std::move(iter)));
  }

 private:
  class Iterator : public core::IndexHolder::Iterator {
   public:
    Iterator(const Quantizer *quantizer, size_t code_bytes,
             core::IndexHolder::Iterator::Pointer &&iter)
        : quantizer_(quantizer), front_iter_(std::move(iter)) {
      buffer_.resize(code_bytes, 0);
      this->quantize_record();
    }

    const void *data(void) const override {
      return buffer_.data();
    }

    bool is_valid(void) const override {
      return front_iter_->is_valid();
    }

    uint64_t key(void) const override {
      return front_iter_->key();
    }

    void next(void) override {
      front_iter_->next();
      this->quantize_record();
    }

   private:
    void quantize_record(void) {
      if (!front_iter_->is_valid()) {
        return;
      }
      quantizer_->quantize_data(front_iter_->data(), &buffer_[0]);
    }

    const Quantizer *quantizer_{nullptr};
    core::IndexHolder::Iterator::Pointer front_iter_{};
    std::string buffer_{};
  };

  core::IndexHolder::Pointer front_{};
  Quantizer::Pointer quantizer_{};
  core::IndexMeta::DataType type_{core::IndexMeta::DataType::DT_UNDEFINED};
  size_t raw_dim_{0};
  size_t code_bytes_{0};
};

}  // namespace turbo
}  // namespace zvec
