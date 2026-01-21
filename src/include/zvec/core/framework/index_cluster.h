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
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_bundle.h>
#include <zvec/core/framework/index_features.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_module.h>
#include <zvec/core/framework/index_threads.h>

namespace zvec {
namespace core {

/*! Index Cluster
 */
struct IndexCluster : public IndexModule {
 public:
  //! Index Cluster Pointer
  typedef std::shared_ptr<IndexCluster> Pointer;

  /*! Index Cluster Centroid
   */
  class Centroid {
   public:
    //! Constructor
    Centroid(void)
        : buffer_(), score_(0.0), follows_(0), similars_(), subitems_() {}

    //! Constructor
    Centroid(const void *feat, size_t bytes)
        : buffer_(std::string(reinterpret_cast<const char *>(feat), bytes)),
          score_(0.0),
          follows_(0),
          similars_(),
          subitems_() {}

    //! Constructor
    Centroid(const Centroid &rhs)
        : buffer_(rhs.buffer_),
          score_(rhs.score_),
          follows_(rhs.follows_),
          similars_(rhs.similars_),
          subitems_(rhs.subitems_) {}

    //! Constructor
    Centroid(Centroid &&rhs)
        : buffer_(std::move(rhs.buffer_)),
          score_(rhs.score_),
          follows_(rhs.follows_),
          similars_(std::move(rhs.similars_)),
          subitems_(std::move(rhs.subitems_)) {}

    //! Assignment
    Centroid &operator=(const Centroid &rhs) {
      buffer_ = rhs.buffer_;
      score_ = rhs.score_;
      follows_ = rhs.follows_;
      similars_ = rhs.similars_;
      subitems_ = rhs.subitems_;
      return *this;
    }

    //! Assignment
    Centroid &operator=(Centroid &&rhs) {
      buffer_ = std::move(rhs.buffer_);
      score_ = rhs.score_;
      follows_ = rhs.follows_;
      similars_ = std::move(rhs.similars_);
      subitems_ = std::move(rhs.subitems_);
      return *this;
    }

    //! Less than
    bool operator<(const Centroid &rhs) const {
      return (this->score_ < rhs.score_);
    }

    //! Test if matchs the meta
    bool is_matched(const IndexMeta &meta) const {
      if (buffer_.size() != meta.element_size()) {
        return false;
      }
      for (const auto &it : subitems_) {
        if (!it.is_matched(meta)) {
          return false;
        }
      }
      return true;
    }

    //! Set feature of centroid
    void set_feature(const void *feat, size_t bytes) {
      buffer_.assign(std::string(reinterpret_cast<const char *>(feat), bytes));
    }

    //! Set feature of centroid
    template <typename T>
    void set_feature(const ailego::NumericalVector<T> &feat) {
      buffer_.assign(feat);
    }

    //! Set feature of centroid
    template <typename T>
    void set_feature(ailego::NumericalVector<T> &&feat) {
      buffer_.assign(std::forward<ailego::NumericalVector<T>>(feat));
    }

    //! Set score of centroid
    void set_score(double val) {
      score_ = val;
    }

    //! Set follows of centroid
    void set_follows(size_t count) {
      follows_ = count;
    }

    //! Set similars of centroid
    void set_similars(const std::vector<const void *> &feats) {
      similars_ = feats;
    }

    //! Set similars of centroid
    void set_similars(std::vector<const void *> &&feats) {
      similars_ = std::move(feats);
    }

    //! Set subitems of centroid
    void set_subitems(const std::vector<Centroid> &cents) {
      subitems_ = cents;
    }

    //! Set subitems of centroid
    void set_subitems(std::vector<Centroid> &&cents) {
      subitems_ = std::move(cents);
    }

    //! Retrieve feature buffer
    std::string *mutable_buffer(void) {
      return &buffer_;
    }

    //! Retrieve feature buffer
    const std::string &buffer(void) const {
      return buffer_;
    }

    //! Retrieve feature vector
    template <typename T>
    ailego::NumericalVector<T> *mutable_vector(void) {
      return static_cast<ailego::NumericalVector<T> *>(&buffer_);
    }

    //! Retrieve feature vector
    template <typename T>
    const ailego::NumericalVector<T> &vector(void) const {
      return static_cast<const ailego::NumericalVector<T> &>(buffer_);
    }

    //! Retrieve feature pointer
    const void *feature(void) const {
      return buffer_.data();
    }

    //! Retrieve size of centroid in bytes
    size_t size(void) const {
      return buffer_.size();
    }

    //! Retrieve score of centroid
    double score(void) const {
      return score_;
    }

    //! Retrieve follows' count of centroid
    size_t follows(void) const {
      return follows_;
    }

    //! Retrieve similars of centroid
    const std::vector<const void *> &similars(void) const {
      return similars_;
    }

    //! Retrieve similars of centroid
    std::vector<const void *> *mutable_similars(void) {
      return &similars_;
    }

    //! Retrieve the sub centroids
    const std::vector<Centroid> &subitems(void) const {
      return subitems_;
    }

    //! Retrieve the sub centroids
    std::vector<Centroid> *mutable_subitems(void) {
      return &subitems_;
    }

    //! Retrieve the count of subitems (includes children's children)
    size_t subcount(void) const {
      size_t total = subitems_.size();
      for (const auto &it : subitems_) {
        total += it.subcount();
      }
      return total;
    }

   private:
    //! Members
    std::string buffer_;
    double score_;
    size_t follows_;
    std::vector<const void *> similars_;
    std::vector<Centroid> subitems_;
  };

  //! Index Cluster Centroid List
  typedef std::vector<Centroid> CentroidList;

  //! Destructor
  virtual ~IndexCluster(void) {}

  //! Deserialize centroids from bundle
  static int Deserialize(const IndexMeta &meta, IndexBundle::Pointer bundle,
                         CentroidList *cents);

  //! Serialize centroids into bundle
  static int Serialize(const IndexMeta &meta, const CentroidList &cents,
                       IndexBundle::Pointer *out);

  //! Initialize Cluster
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Cleanup Cluster
  virtual int cleanup(void) = 0;

  //! Reset Cluster
  virtual int reset(void) = 0;

  //! Update Cluster
  virtual int update(const ailego::Params &params) = 0;

  //! Suggest dividing to K clusters
  virtual void suggest(uint32_t k) = 0;

  //! Mount features
  virtual int mount(IndexFeatures::Pointer feats) = 0;

  //! Cluster
  virtual int cluster(CentroidList &cents) {
    return this->cluster(nullptr, cents);
  }

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads, CentroidList &cents) = 0;

  //! Classify
  virtual int classify(CentroidList &cents) {
    return this->classify(nullptr, cents);
  }

  //! Classify
  virtual int classify(IndexThreads::Pointer threads, CentroidList &cents) = 0;

  //! Label
  virtual int label(const CentroidList &cents, std::vector<uint32_t> *out) {
    return this->label(nullptr, cents, out);
  }

  //! Label
  virtual int label(IndexThreads::Pointer threads, const CentroidList &cents,
                    std::vector<uint32_t> *out) = 0;
};

}  // namespace core
}  // namespace zvec
