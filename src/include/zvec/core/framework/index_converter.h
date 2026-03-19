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

#include <atomic>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_stats.h>
#include "zvec/core/framework/index_reformer.h"

namespace zvec {
namespace core {

/*! Index Converter
 */
class IndexConverter : public IndexModule {
 public:
  //! Index Converter Pointer
  typedef std::shared_ptr<IndexConverter> Pointer;

  /*! Index Converter Stats
   */
  class Stats : public IndexStats {
   public:
    Stats() {}
    Stats(const Stats &stats) {
      *this = stats;
    }
    Stats &operator=(const Stats &stats) {
      this->trained_count_.store(stats.trained_count_.load());
      this->transformed_count_.store(stats.transformed_count_.load());
      this->dumped_size_.store(stats.dumped_size_.load());
      this->discarded_count_.store(stats.discarded_count_.load());
      this->trained_costtime_.store(stats.trained_costtime_.load());
      this->transformed_costtime_.store(stats.transformed_costtime_.load());
      this->dumped_costtime_.store(stats.dumped_costtime_.load());
      return *this;
    }
    //! Set count of documents trained
    void set_trained_count(size_t count) {
      trained_count_ = count;
    }

    //! Set count of documents transformed
    void set_transformed_count(size_t count) {
      transformed_count_ = count;
    }

    //! Set size of documents dumped
    void set_dumped_size(size_t size) {
      dumped_size_ = size;
    }

    //! Set count of documents discarded
    void set_discarded_count(size_t count) {
      discarded_count_ = count;
    }

    //! Set time cost of documents trained
    void set_trained_costtime(uint64_t cost) {
      trained_costtime_ = cost;
    }

    //! Set time cost of documents transformed
    void set_transformed_costtime(uint64_t cost) {
      transformed_costtime_ = cost;
    }

    //! Set time cost of documents dumped
    void set_dumped_costtime(uint64_t cost) {
      dumped_costtime_ = cost;
    }

    //! Retrieve count of documents trained
    size_t trained_count(void) const {
      return trained_count_;
    }

    //! Retrieve count of documents transformed
    size_t transformed_count(void) const {
      return transformed_count_;
    }

    //! Retrieve size of documents dumped
    size_t dumped_size(void) const {
      return dumped_size_;
    }

    //! Retrieve count of documents discarded
    size_t discarded_count(void) const {
      return discarded_count_;
    }

    //! Retrieve time cost of documents trained
    uint64_t trained_costtime(void) const {
      return trained_costtime_;
    }

    //! Retrieve time cost of documents transformed
    uint64_t transformed_costtime(void) const {
      return transformed_costtime_;
    }

    //! Retrieve time cost of documents dumped
    uint64_t dumped_costtime(void) const {
      return dumped_costtime_;
    }

    //! Retrieve count of documents trained (mutable)
    std::atomic<size_t> *mutable_trained_count(void) {
      return &trained_count_;
    }

    //! Retrieve count of documents transformed (mutable)
    std::atomic<size_t> *mutable_transformed_count(void) {
      return &transformed_count_;
    }

    //! Retrieve size of documents dumped (mutable)
    std::atomic<size_t> *mutable_dumped_size(void) {
      return &dumped_size_;
    }

    //! Retrieve count of documents discarded (mutable)
    std::atomic<size_t> *mutable_discarded_count(void) {
      return &discarded_count_;
    }

    //! Retrieve time cost of documents trained (mutable)
    std::atomic<uint64_t> *mutable_trained_costtime(void) {
      return &trained_costtime_;
    }

    //! Retrieve time cost of documents transformed (mutable)
    std::atomic<uint64_t> *mutable_transformed_costtime(void) {
      return &transformed_costtime_;
    }

    //! Retrieve time cost of documents dumped (mutable)
    std::atomic<uint64_t> *mutable_dumped_costtime(void) {
      return &dumped_costtime_;
    }

   private:
    //! Members
    std::atomic<size_t> trained_count_{0u};
    std::atomic<size_t> transformed_count_{0u};
    std::atomic<size_t> dumped_size_{0u};
    std::atomic<size_t> discarded_count_{0u};
    std::atomic<uint64_t> trained_costtime_{0u};
    std::atomic<uint64_t> transformed_costtime_{0u};
    std::atomic<uint64_t> dumped_costtime_{0u};
  };

  //! Destructor
  virtual ~IndexConverter(void) {}

  //! Initialize Converter
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Cleanup Converter
  virtual int cleanup(void) = 0;

  //! Train the data
  virtual int train(IndexHolder::Pointer) {
    return IndexError_NotImplemented;
  }

  //! Train the data
  virtual int train(IndexSparseHolder::Pointer) {
    return IndexError_NotImplemented;
  }

  //! Transform the data
  virtual int transform(IndexHolder::Pointer) {
    return IndexError_NotImplemented;
  };

  //! Transform the data
  virtual int transform(IndexSparseHolder::Pointer) {
    return IndexError_NotImplemented;
  }

  //! Dump index into storage
  virtual int dump(const IndexDumper::Pointer &dumper) = 0;

  //! Retrieve statistics
  virtual const Stats &stats(void) const = 0;

  //! Retrieve a holder as result
  virtual IndexHolder::Pointer result(void) const {
    return nullptr;
  }

  //! Retrieve a holder as result
  virtual IndexSparseHolder::Pointer sparse_result(void) const {
    return nullptr;
  }

  //! Retrieve Index Meta
  virtual const IndexMeta &meta(void) const = 0;

  //! Train and transform the index
  static int TrainAndTransform(const IndexConverter::Pointer &converter,
                               IndexHolder::Pointer holder);

  //! Train, transform and dump the index
  static int TrainTransformAndDump(const IndexConverter::Pointer &converter,
                                   IndexHolder::Pointer holder,
                                   const IndexDumper::Pointer &dumper);

  //! Convert to reformer
  virtual int to_reformer(IndexReformer::Pointer *) {
    return IndexError_NotImplemented;
  }
};

}  // namespace core
}  // namespace zvec
