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

#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_framework.h>
#include "ivf_params.h"
#include "ivf_utility.h"

namespace zvec {
namespace core {

/*! IVF Centroid Index
 */
class IVFCentroidIndex {
 public:
  typedef std::shared_ptr<IVFCentroidIndex> Pointer;

  //! Constructor
  IVFCentroidIndex(void) {}

  //! Destructor
  ~IVFCentroidIndex(void) {
    IndexMemory *instance = IndexMemory::Instance();
    if (instance) {
      if (instance->has(path_)) {
        instance->remove(path_);
      }
    }
  }

  //! Initialize
  int init(const IndexMeta &meta, const ailego::Params &params);

  //! Set Quantizer for the index
  void set_quantizer(const std::string &quantizer_name,
                     ailego::Params &quantizer_params) {
    quantizer_class_ = quantizer_name;
    quantizer_params_ = quantizer_params;
  }

  //! Retrieve data address of the index
  const void *data(void) const {
    return data_;
  }

  //! Retrieve size of the index
  size_t size(void) const {
    return size_;
  }

  //! Create searcher context for centroid index
  IndexSearcher::Context::Pointer create_context() const {
    return searcher_ ? searcher_->create_context() : nullptr;
  }

  //! Similarity search
  int search(const void *query, const IndexQueryMeta &qmeta, size_t count,
             IndexSearcher::Context::Pointer &ctx);

  //! Search the nearest point, must be called in local thread pool
  uint32_t search_nearest_centroid(const void *query, size_t len);

  //! Transform Data and Search the nearest point, called while adding record
  uint32_t transform_and_search_nearest_centroid(
      const void *record, const IndexQueryMeta &rmeta,
      IndexSearcher::Context::Pointer &ctx) const;

  //! Build Centroid Index From Centroid List
  int build(const IndexCluster::CentroidList &centroid_list);

  //! Load Centroid Index From container
  int load(const IndexStorage::Pointer &container, const ailego::Params params);

  //! Retrieve centroid count of the index
  size_t centroids_count(void) const {
    return centroids_count_;
  }

  //! Retrieve meta
  const IndexMeta &meta() const {
    return meta_;
  }

  //! Retrieve reformer of the index
  const IndexReformer::Pointer reformer(void) const {
    return reformer_;
  }

  static constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();

 private:
  /*! Centroids IndexHolder
   */
  class CentroidsIndexHolder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      //! Index Holder Iterator Pointer
      typedef std::unique_ptr<Iterator> Pointer;

      //! Constructor
      Iterator(std::vector<const void *> *features) : features_(features) {}

      //! Destructor
      virtual ~Iterator(void) {}

      //! Retrieve pointer of data
      virtual const void *data(void) const override {
        return (*features_)[id_];
      }

      //! Test if the iterator is valid
      virtual bool is_valid(void) const override {
        return id_ < features_->size();
      }

      //! Retrieve primary key
      virtual uint64_t key(void) const override {
        return id_;
      }

      //! Next iterator
      virtual void next(void) override {
        ++id_;
      }

     private:
      //! Members
      std::vector<const void *> *features_{nullptr};
      uint32_t id_{0};
    };

    //! Constructor
    CentroidsIndexHolder(const IndexMeta &meta,
                         const IndexCluster::CentroidList &centroid_list)
        : dimension_(meta.dimension()),
          element_size_(meta.element_size()),
          data_type_(meta.data_type()) {
      using CentroidList = IndexCluster::CentroidList;

      std::function<void(const CentroidList &)> get_leaf_features =
          [&](const CentroidList &cents) {
            if (cents.empty()) {
              return;
            }
            for (const auto &it : cents) {
              if (it.subitems().empty()) {
                features_.emplace_back(it.feature());
              } else {
                get_leaf_features(it.subitems());
              }
            }
          };

      get_leaf_features(centroid_list);
    }

    //! Retrieve count of elements in holder (-1 indicates unknown)
    virtual size_t count(void) const override {
      return features_.size();
    }

    //! Retrieve dimension
    virtual size_t dimension(void) const override {
      return dimension_;
    }

    //! Retrieve type information
    virtual IndexMeta::DataType data_type(void) const override {
      return data_type_;
    }

    //! Retrieve element size in bytes
    virtual size_t element_size(void) const override {
      return element_size_;
    }

    //! Retrieve if it can multi-pass
    virtual bool multipass(void) const override {
      return true;
    }

    //! Create a new iterator
    virtual IndexHolder::Iterator::Pointer create_iterator(void) override {
      return IndexHolder::Iterator::Pointer(
          new CentroidsIndexHolder::Iterator(&features_));
    }

   private:
    //! Members
    std::vector<const void *> features_{};
    size_t dimension_{0};
    size_t element_size_{0};
    IndexMeta::DataType data_type_{IndexMeta::DataType::DT_UNDEFINED};
  };

  int build_index(const IndexCluster::CentroidList &centroid_list,
                  const IndexDumper::Pointer &dumper);

  //! Prepare trainer for clustering index
  IndexTrainer::Pointer prepare_trainer(
      const IndexCluster::CentroidList &centroid_list);

  //! Quantize the centroid vectors in holder
  IndexHolder::Pointer quantize_holder(const IndexHolder::Pointer &holder);


 private:
  //! Constants
  constexpr static const char *kDefaultBuilder = "FlatBuilder";
  constexpr static const char *kTempralPathPrefix = "IVF";

  //! Members
  IndexMeta meta_{};

  IndexSearcher::Pointer searcher_{};
  IndexReformer::Pointer reformer_{};
  std::string builder_class_{kDefaultBuilder};
  std::string searcher_class_{};
  std::string quantizer_class_{};

  std::string path_{};

  ailego::Params builder_params_{};
  ailego::Params searcher_params_{};
  ailego::Params quantizer_params_{};

  const void *data_{};
  size_t size_{};
  size_t centroids_count_{0};
  bool index_building_{false};
};

}  // namespace core
}  // namespace zvec
