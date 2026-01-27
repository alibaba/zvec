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
#include <ailego/algorithm/binary_quantizer.h>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

/*! Binary Quantizer Converter Holder
 */
class BinaryConverterHolder : public IndexHolder {
 public:
  /*! Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Constructor
    Iterator(const BinaryConverterHolder *owner,
             IndexHolder::Iterator::Pointer &&iter)
        : buffer_(ailego::BinaryQuantizer::EncodedSizeInBinary32(
                      owner->dimension()),
                  0),
          front_iter_(std::move(iter)),
          quantizer_(owner->quantizer_),
          dim_{owner->dimension()} {
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
        quantizer_->encode(vec, dim_ / 2, buffer_.data());
      }
    }

    //! Members
    std::vector<uint32_t> buffer_{};
    IndexHolder::Iterator::Pointer front_iter_{};
    std::shared_ptr<ailego::BinaryQuantizer> quantizer_{};
    size_t dim_{0u};
  };

  //! Constructor
  BinaryConverterHolder(IndexHolder::Pointer front,
                        std::shared_ptr<ailego::BinaryQuantizer> quantizer)
      : front_(std::move(front)), quantizer_(std::move(quantizer)) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return ailego::BinaryQuantizer::EncodedSizeInBinary32(front_->dimension()) *
           32u;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_BINARY32;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_BINARY32,
                                    this->dimension());
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    IndexHolder::Iterator::Pointer iter = front_->create_iterator();
    return iter
               ? IndexHolder::Iterator::Pointer(
                     new BinaryConverterHolder::Iterator(this, std::move(iter)))
               : IndexHolder::Iterator::Pointer();
  }

 private:
  //! Members
  IndexHolder::Pointer front_{};
  std::shared_ptr<ailego::BinaryQuantizer> quantizer_{};
};

/*! Binary Converter
 */
class BinaryConverter : public IndexConverter {
 public:
  //! Destructor
  virtual ~BinaryConverter(void) {}

  //! Initialize Converter
  int init(const IndexMeta &mt, const ailego::Params &params) override {
    if (ailego_unlikely(mt.data_type() != IndexMeta::DataType::DT_FP32 ||
                        mt.unit_size() != IndexMeta::UnitSizeof(
                                              IndexMeta::DataType::DT_FP32))) {
      LOG_ERROR("Unsupported type %d with unit size %u.", mt.data_type(),
                mt.unit_size());
      return IndexError_Unsupported;
    }

    meta_ = mt;

    ailego::Params reformer_params;
    meta_.set_reformer("BinaryReformer", 0, reformer_params);

    if (meta_.metric_name() != "InnerProduct") {
      LOG_ERROR("Only InnerProduct Supported");
      return IndexError_Unsupported;
    }

    dimension_ = meta_.dimension();

    size_t dim =
        ailego::BinaryQuantizer::EncodedSizeInBinary32(dimension_) * 32u;

    meta_.set_metric("Hamming", 0, ailego::Params());
    meta_.set_converter("BinaryConverter", 0, params);
    meta_.set_meta(IndexMeta::DataType::DT_BINARY32, dim);

    return 0;
  }

  //! Cleanup Converter
  int cleanup(void) override {
    return 0;
  }

  //! Train the data
  int train(IndexHolder::Pointer holder) override {
    if (holder->dimension() != dimension_ ||
        holder->data_type() != IndexMeta::DataType::DT_FP32) {
      return IndexError_Mismatch;
    }

    return 0;
  }

  //! Transform the data
  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != dimension_) {
      return IndexError_Mismatch;
    }

    if (holder->count() > 0) {
      *stats_.mutable_transformed_count() += holder->count();
    }
    holder_ =
        std::make_shared<BinaryConverterHolder>(std::move(holder), quantizer_);
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
  std::shared_ptr<ailego::BinaryQuantizer> quantizer_{};
  Stats stats_{};
  size_t dimension_{0u};
};

INDEX_FACTORY_REGISTER_CONVERTER(BinaryConverter);

}  // namespace core
}  // namespace zvec