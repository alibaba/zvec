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

#include <zvec/core/framework/index_bundle.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_stats.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/framework/index_threads.h>

namespace zvec {
namespace core {

/*! Index Trainer
 */
class IndexTrainer : public IndexModule {
 public:
  //! Index Trainer Pointer
  typedef std::shared_ptr<IndexTrainer> Pointer;

  /*! Index Trainer Stats
   */
  class Stats : public IndexStats {
   public:
    //! Set count of documents trained
    void set_trained_count(size_t count) {
      trained_count_ = count;
    }

    //! Set count of documents discarded
    void set_discarded_count(size_t count) {
      discarded_count_ = count;
    }

    //! Set time cost of documents trained
    void set_trained_costtime(uint64_t cost) {
      trained_costtime_ = cost;
    }

    //! Retrieve count of documents trained
    size_t trained_count(void) const {
      return trained_count_;
    }

    //! Retrieve count of documents discarded
    size_t discarded_count(void) const {
      return discarded_count_;
    }

    //! Retrieve time cost of documents trained
    uint64_t trained_costtime(void) const {
      return trained_costtime_;
    }

   private:
    //! Members
    size_t trained_count_{0u};
    size_t discarded_count_{0u};
    uint64_t trained_costtime_{0u};
  };

  //! Destructor
  virtual ~IndexTrainer(void) {}

  //! Initialize Trainer
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Cleanup Trainer
  virtual int cleanup(void) = 0;

  //! Train the data
  virtual int train(IndexHolder::Pointer holder) {
    return this->train(nullptr, holder);
  }

  //! Train the data
  virtual int train(IndexThreads::Pointer threads,
                    IndexHolder::Pointer holder) = 0;

  //! Load index from container
  virtual int load(IndexStorage::Pointer cntr) = 0;

  //! Dump index into storage
  virtual int dump(const IndexDumper::Pointer &dumper) = 0;

  //! Retrieve Index Meta
  virtual const IndexMeta &meta(void) const = 0;

  //! Retrieve statistics
  virtual const Stats &stats(void) const = 0;

  //! Retrieve the output indexes
  virtual IndexBundle::Pointer indexes(void) const = 0;
};

}  // namespace core
}  // namespace zvec
