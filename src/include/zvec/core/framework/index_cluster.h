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

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_bundle.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_features.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_module.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/framework/index_threads.h>

namespace zvec {
namespace core {

/*! Index Cluster
 */
struct IndexCluster : public IndexModule {
 public:
  // Optional immutable backing for trained centroid rows. Internal consumers
  // use read_feature() with one reusable row; legacy pointer APIs materialize
  // only when explicitly requested.
  class FeatureStorage {
   public:
    virtual ~FeatureStorage() = default;
    virtual int read(size_t offset, void *out, size_t bytes) const = 0;
  };
  //! Index Cluster Pointer
  typedef std::shared_ptr<IndexCluster> Pointer;

  /*! Index Cluster Centroid
   */
  class Centroid {
   public:
    //! Constructor
    Centroid()
        : buffer_(), score_(0.0), follows_(0), similars_(), subitems_() {}

    //! Constructor
    Centroid(const void *feat, size_t bytes) : Centroid() {
      set_feature(feat, bytes);
    }

    Centroid(const Centroid &) = default;
    Centroid(Centroid &&) noexcept = default;
    Centroid &operator=(Centroid rhs) noexcept {
      metadata_memory_.swap(rhs.metadata_memory_);
      buffer_memory_.swap(rhs.buffer_memory_);
      similars_memory_.swap(rhs.similars_memory_);
      buffer_.swap(rhs.buffer_);
      stored_.swap(rhs.stored_);
      std::swap(score_, rhs.score_);
      std::swap(follows_, rhs.follows_);
      similars_.swap(rhs.similars_);
      subitems_.swap(rhs.subitems_);
      return *this;
    }

    //! Less than
    bool operator<(const Centroid &rhs) const {
      return (this->score_ < rhs.score_);
    }

    //! Test if matchs the meta
    bool is_matched(const IndexMeta &meta) const {
      if (size() != meta.element_size()) {
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
      ailego::ReserveStringCapacity(buffer_, buffer_memory_, bytes);
      buffer_.assign(reinterpret_cast<const char *>(feat), bytes);
      stored_.reset();
    }

    void set_feature_storage(std::shared_ptr<const FeatureStorage> storage,
                             size_t offset, size_t bytes) {
      if (!storage) throw std::invalid_argument("Missing centroid storage");
      auto row = std::make_shared<StoredFeature>();
      row->storage = std::move(storage);
      row->offset = offset;
      row->bytes = bytes;
      stored_ = std::move(row);
      std::string().swap(buffer_);
      buffer_memory_.retain_at_most(0);
    }

    // Centroid trees retain their directory reservation for their entire
    // lifetime, including copies of the public CentroidList.
    void adopt_metadata(ailego::MemoryReservation reservation) {
      metadata_memory_ = std::move(reservation);
    }

    void absorb_metadata(Centroid &rhs) noexcept {
      metadata_memory_.absorb(std::move(rhs.metadata_memory_));
    }

    static size_t loaded_metadata_bytes() {
      return sizeof(Centroid) + sizeof(StoredFeature) + 2 * sizeof(void *);
    }

    int read_feature(void *out, size_t bytes) const {
      if (bytes != size() || (!out && bytes)) return IndexError_InvalidArgument;
      if (stored_) return stored_->storage->read(stored_->offset, out, bytes);
      if (bytes) std::memcpy(out, buffer_.data(), bytes);
      return 0;
    }

    bool feature_stored() const {
      return stored_ != nullptr;
    }

    //! Set feature of centroid
    template <typename T>
    void set_feature(const ailego::NumericalVector<T> &feat) {
      set_feature(feat.data(), feat.bytes());
    }

    //! Set feature of centroid
    template <typename T>
    void set_feature(ailego::NumericalVector<T> &&feat) {
      auto &source = static_cast<std::string &>(feat);
      ailego::MemoryReservation memory;
      if (source.capacity() > std::string().capacity())
        memory.add(source.capacity() + 1);
      std::string replacement(std::move(source));
      buffer_.swap(replacement);
      buffer_memory_.swap(memory);
      stored_.reset();
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
      ailego::ReserveVectorCapacity(similars_, similars_memory_, feats.size());
      similars_ = feats;
    }

    //! Set similars of centroid
    void set_similars(std::vector<const void *> &&feats) {
      ailego::MemoryReservation memory;
      memory.add_array(feats.capacity(), sizeof(const void *));
      std::vector<const void *> replacement(std::move(feats));
      similars_.swap(replacement);
      similars_memory_.swap(memory);
    }

    void resize_similars(size_t count) {
      ailego::ReserveVectorCapacity(similars_, similars_memory_, count);
      similars_.resize(count);
    }

    // Preserve existing assignments when publishing newly trained features.
    void swap_assignments(Centroid &other) noexcept {
      similars_memory_.swap(other.similars_memory_);
      similars_.swap(other.similars_);
      subitems_.swap(other.subitems_);
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
    std::string *mutable_buffer() {
      if (stored_) {
        const auto &row = buffer();
        ailego::ReserveStringCapacity(buffer_, buffer_memory_, row.size());
        buffer_ = row;
        stored_.reset();
      }
      return &buffer_;
    }

    //! Retrieve feature buffer
    const std::string &buffer() const {
      if (stored_) {
        std::call_once(stored_->once, [&] {
          auto &pool = ailego::MemoryLimitPool::get_instance();
          const size_t charge = pool.capacity() ? stored_->bytes : 0;
          if (charge && !pool.try_charge_external(charge))
            throw std::bad_alloc();
          try {
            std::string row(stored_->bytes, '\0');
            if (read_feature(row.data(), row.size()) != 0)
              throw std::runtime_error("Failed to read stored centroid");
            stored_->materialized = std::move(row);
            stored_->charged = charge;
          } catch (...) {
            if (charge) pool.release_external(charge);
            throw;
          }
        });
        return stored_->materialized;
      }
      return buffer_;
    }

    //! Retrieve feature vector
    template <typename T>
    ailego::NumericalVector<T> *mutable_vector() {
      return static_cast<ailego::NumericalVector<T> *>(mutable_buffer());
    }

    //! Retrieve feature vector
    template <typename T>
    const ailego::NumericalVector<T> &vector() const {
      return static_cast<const ailego::NumericalVector<T> &>(buffer());
    }

    //! Retrieve feature pointer
    const void *feature() const {
      return buffer().data();
    }

    //! Retrieve size of centroid in bytes
    size_t size() const {
      return stored_ ? stored_->bytes : buffer_.size();
    }

    //! Retrieve score of centroid
    double score() const {
      return score_;
    }

    //! Retrieve follows' count of centroid
    size_t follows() const {
      return follows_;
    }

    //! Retrieve similars of centroid
    const std::vector<const void *> &similars() const {
      return similars_;
    }

    //! Retrieve similars of centroid
    std::vector<const void *> *mutable_similars() {
      return &similars_;
    }

    //! Retrieve the sub centroids
    const std::vector<Centroid> &subitems() const {
      return subitems_;
    }

    //! Retrieve the sub centroids
    std::vector<Centroid> *mutable_subitems() {
      return &subitems_;
    }

    //! Retrieve the count of subitems (includes children's children)
    size_t subcount() const {
      size_t total = subitems_.size();
      for (const auto &it : subitems_) {
        total += it.subcount();
      }
      return total;
    }

   private:
    struct StoredFeature {
      std::shared_ptr<const FeatureStorage> storage;
      size_t offset{0}, bytes{0};
      std::once_flag once;
      std::string materialized;
      size_t charged{0};
      ~StoredFeature() {
        if (charged)
          ailego::MemoryLimitPool::get_instance().release_external(charged);
      }
    };
    //! Members
    ailego::MemoryReservation metadata_memory_;
    ailego::MemoryReservation buffer_memory_;
    ailego::MemoryReservation similars_memory_;
    std::string buffer_;
    std::shared_ptr<StoredFeature> stored_;
    double score_;
    size_t follows_;
    std::vector<const void *> similars_;
    std::vector<Centroid> subitems_;
  };

  //! Index Cluster Centroid List
  typedef std::vector<Centroid> CentroidList;

  //! Destructor
  ~IndexCluster() override = default;

  //! Deserialize centroids from bundle
  static int Deserialize(const IndexMeta &meta, IndexBundle::Pointer bundle,
                         CentroidList *cents);

  //! Load only the centroid segments, with bounded reads and pageable rows.
  static int Load(const IndexMeta &meta, IndexStorage::Pointer storage,
                  CentroidList *cents);

  //! Serialize centroids into bundle
  static int Serialize(const IndexMeta &meta, const CentroidList &cents,
                       IndexBundle::Pointer *out);

  //! Stream the same centroid segments without materializing a whole bundle.
  static int Dump(const IndexMeta &meta, const CentroidList &cents,
                  const IndexDumper::Pointer &dumper);

  //! Initialize Cluster
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Cleanup Cluster
  virtual int cleanup() = 0;

  //! Reset Cluster
  virtual int reset() = 0;

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
