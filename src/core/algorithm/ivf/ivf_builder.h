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

#include <zvec/core/framework/index_builder.h>
#include <zvec/core/framework/index_meta.h>
#include "ivf_centroid_index.h"

namespace zvec {
namespace core {

/*! IVF Builder
 */
class IVFBuilder : public IndexBuilder {
 public:
  //! Constructor
  IVFBuilder();

  //! Destructor
  ~IVFBuilder();

  //! Disable them
  IVFBuilder(const IVFBuilder &) = delete;
  IVFBuilder &operator=(const IVFBuilder &) = delete;

 public:
  //! Initialize the builder
  virtual int init(const IndexMeta &meta,
                   const ailego::Params &params) override;

  //! Cleanup the builder
  virtual int cleanup(void) override;

  //! Train the data
  virtual int train(IndexThreads::Pointer threads,
                    IndexHolder::Pointer holder) override;

  //! Train the data
  virtual int train(const IndexTrainer::Pointer &trainer) override;

  //! Build the index
  virtual int build(IndexThreads::Pointer threads,
                    IndexHolder::Pointer holder) override;

  //! Dump index into file system
  virtual int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  virtual const Stats &stats(void) const override {
    return stats_;
  }

  IVFCentroidIndex::Pointer centroid_index() const {
    return centroid_index_;
  }

 public:
  /*! Random Access Index Holder
   */
  class RandomAccessIndexHolder : public IndexHolder {
   public:
    //! Index Holder Iterator Pointer
    typedef std::shared_ptr<RandomAccessIndexHolder> Pointer;

    /*! Random Access Index Holder Iterator
     */
    class Iterator : public IndexHolder::Iterator {
     public:
      //! Index Holder Iterator Pointer
      typedef std::unique_ptr<Iterator> Pointer;

      //! Constructor
      Iterator(RandomAccessIndexHolder *owner) : holder_(owner) {}

      //! Destructor
      virtual ~Iterator(void) {}

      //! Retrieve pointer of data
      virtual const void *data(void) const override {
        return holder_->element(id_);
      }

      //! Test if the iterator is valid
      virtual bool is_valid(void) const override {
        return id_ < holder_->count();
      }

      //! Retrieve primary key
      virtual uint64_t key(void) const override {
        return holder_->key(id_);
      }

      //! Next iterator
      virtual void next(void) override {
        ++id_;
      }

     private:
      //! Members
      RandomAccessIndexHolder *holder_{nullptr};
      uint32_t id_{0};
    };

    //! Constructor
    RandomAccessIndexHolder(const IndexMeta &meta)
        : features_(std::make_shared<CompactIndexFeatures>(meta)) {}

    //! Retrieve count of elements in holder (-1 indicates unknown)
    virtual size_t count(void) const override {
      return features_->count();
    }

    //! Retrieve dimension
    virtual size_t dimension(void) const override {
      return features_->dimension();
    }

    //! Retrieve type information
    virtual IndexMeta::DataType data_type(void) const override {
      return features_->data_type();
    }

    //! Retrieve element size in bytes
    virtual size_t element_size(void) const override {
      return features_->element_size();
    }

    //! Retrieve if it can multi-pass
    virtual bool multipass(void) const override {
      return true;
    }

    //! Create a new iterator
    virtual IndexHolder::Iterator::Pointer create_iterator(void) override {
      return IndexHolder::Iterator::Pointer(
          new RandomAccessIndexHolder::Iterator(this));
    }

    void reserve(size_t elems) {
      features_->reserve(elems);
      keys_.reserve(elems);
    }

    //! Append an element into holder
    void emplace(uint64_t pkey, const void *vec) {
      features_->emplace(vec);
      keys_.emplace_back(pkey);
    }

    //! Retrieve feature via local id
    const void *element(size_t id) const {
      return features_->element(id);
    }

    //! Retrieve key via local id
    uint64_t key(size_t id) const {
      ailego_assert_with(id < keys_.size(), "Index Overflow");
      return keys_[id];
    }

   private:
    //! Disable them
    RandomAccessIndexHolder(void) = delete;

    //! Members
    CompactIndexFeatures::Pointer features_{};
    std::vector<uint64_t> keys_{};
  };

 private:
  /*! Wrapper of feature
   */
  class Vector {
   public:
    typedef std::shared_ptr<Vector> Pointer;

    Vector(const void *vec, size_t len, uint32_t idx)
        : vec_(reinterpret_cast<const char *>(vec), len), id_{idx} {}

    const void *data() const {
      return vec_.data();
    }

    size_t size() const {
      return vec_.size();
    }

    uint32_t id(void) const {
      return id_;
    }

   private:
    std::string vec_{};
    uint32_t id_{0u};
  };

  using VectorList = std::vector<Vector>;

  //! Check MajorOrder in meta, and update the major order if needed
  int CheckAndUpdateMajorOrder(IndexMeta &meta);

  //! Parse params
  int parse_centroids_num(const ailego::Params &params);
  int parse_clustering_params(const ailego::Params &params);
  int parse_general_params(const ailego::Params &params);

  //! Prepare params for trainer
  int prepare_trainer_params(ailego::Params &params);

  //! Build the index
  int build_label_index(IndexThreads *threads,
                        const IndexHolder::Pointer &holder);

  //! Dump the index to dumper
  int dump_index(const IndexDumper::Pointer &dumper);

  //! Prepare the quantizer for inverted index
  int prepare_quantizer(IndexThreads *threads);

  //! Quantize the centrods list
  int quantize_centroids();

  //! Create converter and init with params
  static IndexConverter::Pointer CreateAndInitConverter(
      const IndexMeta &meta, const std::string &name,
      const ailego::Params &params) {
    auto converter = IndexFactory::CreateConverter(name);
    if (!converter) {
      LOG_ERROR("Failed to create converter %s", name.c_str());
      return IndexConverter::Pointer();
    }
    int ret = converter->init(meta, params);
    if (ret != 0) {
      LOG_ERROR("Failed to initialize converter %s for %s", name.c_str(),
                IndexError::What(ret));
      return IndexConverter::Pointer();
    }
    return converter;
  }

  //! Select the nearest centroid id for the vector
  void label(const std::shared_ptr<VectorList> &vecs) {
    for (size_t i = 0; i < vecs->size(); ++i) {
      auto &vec = (*vecs)[i];

      uint32_t centroid_idx =
          centroid_index_->search_nearest_centroid(vec.data(), vec.size());
      if (centroid_idx == IVFCentroidIndex::kInvalidID) {
        LOG_ERROR("Failed to search nearest centroid in CentroidIndex");
        if (!error_.exchange(true)) {
          err_code_ = IndexError_Runtime;
        }
        return;
      }
      ailego_assert_with(centroid_idx < labels_.size(), "Index Overflow");
      mutex_.lock();
      labels_[centroid_idx].emplace_back(vec.id());
      mutex_.unlock();
    }
  }


 private:
  //! Constants
  static constexpr size_t kThreadPoolQueueSize = 300u;
  static constexpr size_t kBatchSize = 10u;
  static constexpr size_t kDefaultBlockCount = 32u;

  enum BuilderState { INIT = 0, INITED = 1, TRAINED = 2, BUILT = 3 };

  //! Members
  BuilderState state_{INIT};
  Stats stats_{};
  ailego::Params params_{};
  IndexMeta meta_{};

  std::vector<uint32_t> centroid_num_vec_{};
  std::string cluster_class_{};
  std::string converter_class_{};
  std::vector<ailego::Params> cluster_params_{};

  std::vector<std::vector<uint32_t>> labels_{};
  std::mutex mutex_{};
  IVFCentroidIndex::Pointer centroid_index_{};
  IVFCentroidIndex::Pointer searcher_centroid_index_{};
  RandomAccessIndexHolder::Pointer holder_{};
  IndexMeta converted_meta_{};
  IndexConverter::Pointer converter_{};
  IndexMeta quantized_meta_{};
  std::vector<IndexConverter::Pointer> quantizers_{};

  std::atomic_bool error_{false};
  int err_code_{0};

  uint32_t thread_count_{0};
  uint32_t sample_count_{0};
  float sample_ratio_{0.0};
  uint32_t block_vector_count_{kDefaultBlockCount};
  bool cluster_auto_tuning_{false};
  bool store_original_features_{false};
  bool quantize_by_centroid_{false};
};

}  // namespace core
}  // namespace zvec
