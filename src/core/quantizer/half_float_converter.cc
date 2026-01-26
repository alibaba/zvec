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
#include <zvec/ailego/utility/float_helper.h>
#include "framework/index_framework.h"

namespace zvec {
namespace core {

/*! Half Float Holder
 */
class HalfFloatHolder : public IndexHolder {
 public:
  /*! Half Float Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(const HalfFloatHolder *owner,
             IndexHolder::Iterator::Pointer &&iter)
        : buffer_(owner->dimension(), 0), front_iter_(std::move(iter)) {
      this->transform_record();
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
      this->transform_record();
    }

   private:
    inline void transform_record(void) {
      if (front_iter_->is_valid()) {
        ailego::FloatHelper::ToFP16(
            reinterpret_cast<const float *>(front_iter_->data()),
            buffer_.size(), buffer_.data());
      }
    }

    std::vector<uint16_t> buffer_{};
    IndexHolder::Iterator::Pointer front_iter_{};
  };

  //! Constructor
  HalfFloatHolder(IndexHolder::Pointer front) : front_(std::move(front)) {}

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
    return IndexMeta::DataType::DT_FP16;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_FP16,
                                    front_->dimension());
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    IndexHolder::Iterator::Pointer iter = front_->create_iterator();
    return iter ? IndexHolder::Iterator::Pointer(
                      new HalfFloatHolder::Iterator(this, std::move(iter)))
                : IndexHolder::Iterator::Pointer();
  }

 private:
  //! Disable them
  HalfFloatHolder(void) = delete;

  //! Members
  IndexHolder::Pointer front_{};
};

/*! Half Float Converter
 */
class HalfFloatConverter : public IndexConverter {
 public:
  //! Destructor
  virtual ~HalfFloatConverter(void) {}

  //! Initialize Converter
  int init(const IndexMeta &mt, const ailego::Params &) override {
    if (ailego_unlikely(mt.data_type() != IndexMeta::DataType::DT_FP32 ||
                        mt.unit_size() != sizeof(float))) {
      LOG_ERROR("Unsupported type %d with unit size %u.", mt.data_type(),
                mt.unit_size());
      return IndexError_Unsupported;
    }

    meta_ = mt;
    meta_.set_meta(IndexMeta::DataType::DT_FP16, mt.dimension());
    meta_.set_converter("HalfFloatConverter", 0, ailego::Params());
    meta_.set_reformer("HalfFloatReformer", 0, ailego::Params());
    return 0;
  }

  //! Cleanup Converter
  int cleanup(void) override {
    return 0;
  }

  //! Train the data
  int train(IndexHolder::Pointer) override {
    return 0;
  }

  //! Transform the data
  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != meta_.dimension()) {
      return IndexError_Mismatch;
    }
    holder_ = std::make_shared<HalfFloatHolder>(std::move(holder));
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
  IndexMeta meta_{};
  IndexHolder::Pointer holder_{};
  Stats stats_{};
};

/*! Half Float Sparse Holder
 */
class HalfFloatSparseHolder : public IndexSparseHolder {
 public:
  /*! Half Float Holder Iterator
   */
  class Iterator : public IndexSparseHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(const HalfFloatSparseHolder * /*owner*/,
             IndexSparseHolder::Iterator::Pointer &&iter)
        : sparse_buffer_(MAX_DIM_COUNT * sizeof(uint16_t), 0),
          front_iter_(std::move(iter)) {
      this->transform_record();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return front_iter_->is_valid();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return front_iter_->key();
    }

    //! Retrieve sparse count
    uint32_t sparse_count() const override {
      return front_iter_->sparse_count();
    }

    //! Retrieve sparse indices
    const uint32_t *sparse_indices() const override {
      return front_iter_->sparse_indices();
    }

    //! Retrieve sparse data
    const void *sparse_data() const override {
      return sparse_buffer_.data();
    }

    //! Next iterator
    void next(void) override {
      front_iter_->next();
      this->transform_record();
    }

   private:
    inline void transform_record(void) {
      if (front_iter_->is_valid()) {
        ailego::FloatHelper::ToFP16(
            reinterpret_cast<const float *>(front_iter_->sparse_data()),
            front_iter_->sparse_count(), sparse_buffer_.data());
      }
    }

    constexpr static uint32_t MAX_DIM_COUNT = 4096;
    std::vector<uint16_t> sparse_buffer_{};

    IndexSparseHolder::Iterator::Pointer front_iter_{};
  };

  //! Constructor
  HalfFloatSparseHolder(IndexSparseHolder::Pointer front)
      : front_(std::move(front)) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexSparseHolder::Iterator::Pointer create_iterator(void) override {
    IndexSparseHolder::Iterator::Pointer iter = front_->create_iterator();
    return iter
               ? IndexSparseHolder::Iterator::Pointer(
                     new HalfFloatSparseHolder::Iterator(this, std::move(iter)))
               : IndexSparseHolder::Iterator::Pointer();
  }

  size_t total_sparse_count(void) const override {
    return front_->total_sparse_count();
  }

 private:
  //! Disable them
  HalfFloatSparseHolder(void) = delete;

  //! Members
  IndexSparseHolder::Pointer front_{};
};

/*! Half Float Sparse Converter
 */
class HalfFloatSparseConverter : public IndexConverter {
 public:
  //! Destructor
  virtual ~HalfFloatSparseConverter(void) {}

  //! Initialize Converter
  int init(const IndexMeta &mt, const ailego::Params &) override {
    if (ailego_unlikely(mt.data_type() != IndexMeta::DataType::DT_FP32 ||
                        mt.unit_size() != sizeof(float))) {
      LOG_ERROR("Unsupported type %d with unit size %u.", mt.data_type(),
                mt.unit_size());
      return IndexError_Unsupported;
    }

    meta_ = mt;
    meta_.set_data_type(IndexMeta::DataType::DT_FP16);
    meta_.set_converter("HalfFloatSparseConverter", 0, ailego::Params());
    meta_.set_reformer("HalfFloatSparseReformer", 0, ailego::Params());
    return 0;
  }

  //! Cleanup Converter
  int cleanup(void) override {
    return 0;
  }

  //! Train the data
  int train(IndexSparseHolder::Pointer) override {
    return 0;
  }

  //! Transform the data
  int transform(IndexSparseHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32) {
      return IndexError_Mismatch;
    }

    holder_ = std::make_shared<HalfFloatSparseHolder>(std::move(holder));
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
  IndexSparseHolder::Pointer sparse_result(void) const override {
    return holder_;
  }

  //! Retrieve Index Sparse Meta
  const IndexMeta &meta(void) const override {
    return meta_;
  }

 private:
  IndexMeta meta_{};
  IndexSparseHolder::Pointer holder_{};
  Stats stats_{};
};

INDEX_FACTORY_REGISTER_CONVERTER(HalfFloatConverter);
INDEX_FACTORY_REGISTER_CONVERTER(HalfFloatSparseConverter);

}  // namespace core
}  // namespace zvec