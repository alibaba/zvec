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
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_document.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_filter.h>
#include <zvec/core/framework/index_groupby.h>
#include <zvec/core/framework/index_metric.h>
#include <zvec/core/framework/index_stats.h>

namespace zvec {
namespace core {

/*! Profiler
 */
struct Profiler {
  Profiler() = default;
  ~Profiler() = default;

  void add(const std::string &name, double time) {
    timings[name] += time;
  }

  std::string display() const {
    std::string info = "================================================\n";

    for (auto itr = timings.begin(); itr != timings.end(); ++itr) {
      info +=
          itr->first + std::string(": ") + std::to_string(itr->second) + " s\n";
    }

    info += "================================================\n";

    return info;
  }

  std::map<std::string, double> timings;
};

/*! Index Context
 */
class IndexContext {
 public:
  //! Index Context Pointer
  typedef std::unique_ptr<IndexContext> Pointer;

  //! Index Context UPointer
  typedef std::unique_ptr<IndexContext> UPointer;

  /*! Index Context Stats
   */
  class Stats : public IndexStats {
   public:
    //! Set count of documents filtered
    void set_filtered_count(size_t count) {
      filtered_count_ = count;
    }

    //! Set count of documents dist calced
    void set_dist_calced_count(size_t count) {
      dist_calced_count_ = count;
    }

    //! Retrieve count of documents filtered
    size_t filtered_count(void) const {
      return filtered_count_;
    }

    //! Retrieve count of documents dist-calced
    size_t dist_calced_count(void) const {
      return dist_calced_count_;
    }

    //! Retrieve count of documents filtered (mutable)
    size_t *mutable_filtered_count(void) {
      return &filtered_count_;
    }

    //! Retrieve count of documents dist-calced (mutable)
    size_t *mutable_dist_calced_count(void) {
      return &dist_calced_count_;
    }

    void clear() {
      this->clear_attributes();

      filtered_count_ = 0u;
      dist_calced_count_ = 0u;
    }

   private:
    //! Members
    size_t filtered_count_{0u};
    size_t dist_calced_count_{0u};
  };

  //! Constructor
  IndexContext() {}

  //! Constructor
  IndexContext(IndexMetric::Pointer index_metric)
      : index_metric_(std::move(index_metric)) {}

  //! Destructor
  virtual ~IndexContext(void) {}

  //! Set topk of search result
  virtual void set_topk(uint32_t topk) = 0;

  virtual uint32_t topk() const {
    return 0;
  }

  virtual void set_group_params(uint32_t /*group_mum*/,
                                uint32_t /*group_topk*/){};

  //! Set brute force threshold
  virtual void set_bruteforce_threshold(uint32_t /*bruteforce_threshold*/) {}

  //! Set mode of debug
  virtual void set_debug_mode(bool /*enable*/) {}

  //! Set fetch vector
  virtual void set_fetch_vector(bool /*enable*/) {}

  //! Retrieve search result
  virtual const IndexDocumentList &result(void) const = 0;

  //! Retrieve search result with index
  virtual const IndexDocumentList &result(size_t /*index*/) const {
    return this->result();
  }

  //! Retrieve mutable result with index
  virtual IndexDocumentList *mutable_result(size_t idx) = 0;

  //! Retrieve search group result with index
  virtual const IndexGroupDocumentList &group_result(void) const {
    // to make it compile
    static const IndexGroupDocumentList empty_list{};
    return empty_list;
  };

  //! Retrieve search group result with index
  virtual const IndexGroupDocumentList &group_result(size_t /*idx*/) const {
    return this->group_result();
  }

  //! Update the parameters of context
  virtual int update(const ailego::Params & /*params*/) {
    return IndexError_NotImplemented;
  }

  //! Retrieve mode of debug
  virtual bool debug_mode(void) const {
    return false;
  }

  //! Retrieve debug information
  virtual std::string debug_string(void) const {
    return std::string();
  }

  //! Retrieve magic number
  virtual uint32_t magic(void) const {
    return 0;
  }

  //! Retrieve search filter
  const IndexFilter &filter(void) const {
    return filter_;
  }

  //! Retrieve fetch vector
  virtual bool fetch_vector(void) const {
    return false;
  }

  //! Reset context
  virtual void reset(void) {}

  //! Set the filter of context
  template <typename T>
  void set_filter(T &&func) {
    filter_.set(std::forward<T>(func));
  }

  //! Reset the filter of context
  void reset_filter(void) {
    filter_.reset();
  }

  //! Retrieve search groupby
  const IndexGroupBy &group_by(void) const {
    return group_by_;
  }

  //! Set the groupby of context
  template <typename T>
  void set_group_by(T &&func) {
    group_by_.set(std::forward<T>(func));
  }

  //! Reset the groupby of context
  void reset_group_by(void) {
    group_by_.reset();
  }

  //! Set threshold for RNN
  void set_threshold(float val) {
    if (index_metric_ && index_metric_->support_normalize()) {
      index_metric_->denormalize(&val);
    }

    threshold_ = val;
  }

  //! Retrieve value of threshold for RNN
  float threshold(void) const {
    return threshold_;
  }

  //! Reset value of threshold for RNN
  void reset_threshold(void) {
    threshold_ = std::numeric_limits<float>::max();
  }

  //! Generate a global magic number
  static uint32_t GenerateMagic(void);

  //! Profiler
  Profiler &profiler() {
    return profiler_;
  }

 private:
  //! Members
  IndexFilter filter_{};
  IndexGroupBy group_by_{};
  float threshold_{std::numeric_limits<float>::max()};


  Profiler profiler_{};

 protected:
  IndexMetric::Pointer index_metric_{nullptr};
};

}  // namespace core
}  // namespace zvec