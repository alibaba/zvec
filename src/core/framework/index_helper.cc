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

#include <ailego/utility/memory_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_helper.h>

namespace zvec {
namespace core {

int IndexHelper::SerializeToDumper(const IndexMeta &mt, IndexDumper *dumper,
                                   const std::string &key) {
  std::string buffer;
  mt.serialize(&buffer);

  size_t data_size = buffer.size();
  uint32_t data_crc = ailego::Crc32c::Hash(buffer.data(), buffer.size(), 0);
  buffer.resize((data_size + 31u) & ~31u);

  if (dumper->write(buffer.data(), buffer.size()) != buffer.size()) {
    return IndexError_WriteData;
  }
  if (dumper->append(key, data_size, buffer.size() - data_size, data_crc) !=
      0) {
    return IndexError_WriteData;
  }
  return IndexError_Success;
}

int IndexHelper::SerializeToStorage(const IndexMeta &mt, IndexStorage *storage,
                                    const std::string &key) {
  std::string buffer;
  mt.serialize(&buffer);

  auto segment = storage->get(key);
  if (!segment) {
    const size_t align_size = 4096 * 4;
    size_t meta_size =
        (buffer.size() + align_size - 1) / align_size * align_size;

    if (storage->append(key, meta_size) != 0) {
      return IndexError_WriteData;
    }

    segment = storage->get(key);
    if (!segment) {
      return IndexError_NoExist;
    }
  }

  if (segment->write(0, buffer.data(), buffer.size()) != buffer.size()) {
    return IndexError_WriteData;
  }
  segment->resize(buffer.size());
  segment->update_data_crc(
      ailego::Crc32c::Hash(buffer.data(), buffer.size(), 0));
  return IndexError_Success;
}

int IndexHelper::DeserializeFromStorage(IndexStorage *storage,
                                        const std::string &key,
                                        IndexMeta *out) {
  auto segment = storage->get(key);
  if (!segment) {
    return IndexError_NoExist;
  }

  uint32_t crc = segment->data_crc();
  size_t len = segment->data_size();
  const void *data = nullptr;

  if (segment->read(0, &data, len) != len) {
    return IndexError_ReadData;
  }
  if (crc != 0u && ailego::Crc32c::Hash(data, len, 0u) != crc) {
    return IndexError_InvalidChecksum;
  }
  if (!out->deserialize(data, len)) {
    return IndexError_Deserialize;
  }
  return IndexError_Success;
}

/*! Two Pass Index Holder
 */
class TwoPassIndexHolder : public IndexHolder {
 private:
  /*! First Pass Iterator
   * store elements during iterating for second iterating.
   */
  class FirstPassIterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<FirstPassIterator> Pointer;

    //! Constructor
    FirstPassIterator(TwoPassIndexHolder *owner,
                      IndexHolder::Iterator::Pointer &&iter)
        : holder_(owner), front_iter_(std::move(iter)) {}

    //! Destructor
    virtual ~FirstPassIterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return front_iter_->data();
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
      holder_->features_.emplace_back(
          front_iter_->key(), std::string((const char *)front_iter_->data(),
                                          holder_->front_->element_size()));
      front_iter_->next();
    }

   private:
    TwoPassIndexHolder *holder_{nullptr};
    IndexHolder::Iterator::Pointer front_iter_{};
  };

  class SecondPassIterator : public IndexHolder::Iterator {
   public:
    //! Second Pass Iterator Pointer
    typedef std::unique_ptr<SecondPassIterator> Pointer;

    //! Constructor
    SecondPassIterator(TwoPassIndexHolder *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~SecondPassIterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return features_iter_->second.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      holder_->features_.erase(features_iter_++);
    }

   private:
    TwoPassIndexHolder *holder_{nullptr};
    typename std::list<std::pair<uint64_t, std::string>>::iterator
        features_iter_{};
  };

 public:
  //! Constructor
  TwoPassIndexHolder(IndexHolder::Pointer &&front)
      : front_(std::move(front)),
        data_type_(front_->data_type()),
        dimension_(front_->dimension()),
        element_size_(front_->element_size()),
        count_(front_->count()) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return count_;
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return data_type_;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return element_size_;
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return false;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    ++pass_;
    if (pass_ == 1) {
      IndexHolder::Iterator::Pointer iter = front_->create_iterator();
      return iter ? IndexHolder::Iterator::Pointer(
                        new TwoPassIndexHolder::FirstPassIterator(
                            this, std::move(iter)))
                  : IndexHolder::Iterator::Pointer();
    } else if (pass_ == 2) {
      return IndexHolder::Iterator::Pointer(
          new TwoPassIndexHolder::SecondPassIterator(this));
    }
    return nullptr;
  }

 private:
  //! Disable them
  TwoPassIndexHolder(void) = delete;

  //! Members
  IndexHolder::Pointer front_{};
  std::list<std::pair<uint64_t, std::string>> features_{};
  size_t pass_{0};
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_UNDEFINED};
  size_t dimension_;
  size_t element_size_;
  size_t count_;
};

IndexHolder::Pointer IndexHelper::MakeTwoPassHolder(
    IndexHolder::Pointer holder) {
  if (holder->multipass()) {
    return holder;
  }
  return IndexHolder::Pointer(new TwoPassIndexHolder(std::move(holder)));
}

}  // namespace core
}  // namespace zvec
