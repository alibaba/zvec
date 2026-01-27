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

#include <map>
#include <memory>
#include <string>
#include <zvec/ailego/container/blob.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/io/mmap_file.h>

namespace zvec {
namespace core {

/*! Index Bundle
 */
struct IndexBundle {
  //! Index Bundle Pointer
  typedef std::shared_ptr<IndexBundle> Pointer;

  //! Destructor
  virtual ~IndexBundle(void) {}

  //! Retrieve index buffer via key
  virtual ailego::BlobWrap get(const std::string &key) const = 0;

  //! Test if the key is exist
  virtual bool has(const std::string &key) const = 0;

  //! Retrieve all
  virtual std::map<std::string, ailego::BlobWrap> all(void) const = 0;

  //! Retrieve the count of indexes
  virtual size_t count(void) const = 0;
};

/*! Trivial Index Bundle
 */
class TrivialIndexBundle : public IndexBundle {
 public:
  //! Trivial Index Bundle Pointer
  typedef std::shared_ptr<TrivialIndexBundle> Pointer;

  //! Retrieve index buffer via key
  virtual ailego::BlobWrap get(const std::string &key) const {
    auto iter = map_.find(key);
    if (iter != map_.end()) {
      return iter->second;
    }
    return ailego::BlobWrap();
  }

  //! Test if the key is exist
  virtual bool has(const std::string &key) const {
    return (map_.find(key) != map_.end());
  }

  //! Retrieve all
  virtual std::map<std::string, ailego::BlobWrap> all(void) const {
    return map_;
  }

  //! Retrieve the count of indexes
  virtual size_t count(void) const {
    return map_.size();
  }

  //! Set an index buffer in bundle
  void set(const std::string &key, const ailego::BlobWrap &blob) {
    map_[key] = blob;
  }

  //! Set an index buffer in bundle
  void set(std::string &&key, const ailego::BlobWrap &blob) {
    map_[std::move(key)] = blob;
  }

  //! Set an index buffer in bundle
  void set(const std::string &key, const void *buf, size_t len) {
    map_[key] = ailego::BlobWrap(buf, len);
  }

  //! Set an index buffer in bundle
  void set(std::string &&key, const void *buf, size_t len) {
    map_[std::move(key)] = ailego::BlobWrap(buf, len);
  }

 private:
  std::map<std::string, ailego::BlobWrap> map_;
};

/*! Memory Index Bundle
 */
class MemoryIndexBundle : public IndexBundle {
 public:
  //! Memory Index Bundle Pointer
  typedef std::shared_ptr<MemoryIndexBundle> Pointer;

  //! Retrieve index buffer via key
  virtual ailego::BlobWrap get(const std::string &key) const {
    auto iter = map_.find(key);
    if (iter != map_.end()) {
      return ailego::BlobWrap(iter->second.data(), iter->second.size());
    }
    return ailego::BlobWrap();
  }

  //! Test if the key is exist
  virtual bool has(const std::string &key) const {
    return (map_.find(key) != map_.end());
  }

  //! Retrieve all
  virtual std::map<std::string, ailego::BlobWrap> all(void) const {
    std::map<std::string, ailego::BlobWrap> result;
    for (const auto &it : map_) {
      result.emplace(it.first,
                     ailego::BlobWrap(it.second.data(), it.second.size()));
    }
    return result;
  }

  //! Retrieve the count of indexes
  virtual size_t count(void) const {
    return map_.size();
  }

  //! Set an index buffer in bundle
  void set(const std::string &key, const std::string &buf) {
    map_[key] = buf;
  }

  //! Set an index buffer in bundle
  void set(std::string &&key, const std::string &buf) {
    map_[std::move(key)] = buf;
  }

  //! Set an index buffer in bundle
  void set(const std::string &key, std::string &&buf) {
    map_[key] = std::move(buf);
  }

  //! Set an index buffer in bundle
  void set(std::string &&key, std::string &&buf) {
    map_[std::move(key)] = std::move(buf);
  }

  //! Set an index buffer in bundle
  void set(const std::string &key, const void *buf, size_t len) {
    map_[key].assign(reinterpret_cast<const char *>(buf), len);
  }

  //! Set an index buffer in bundle
  void set(std::string &&key, const void *buf, size_t len) {
    map_[std::move(key)].assign(reinterpret_cast<const char *>(buf), len);
  }

 private:
  std::map<std::string, std::string> map_;
};

/*! MMap File Index Bundle
 */
class MMapFileIndexBundle : public IndexBundle {
 public:
  //! Memory Index Bundle Pointer
  typedef std::shared_ptr<MMapFileIndexBundle> Pointer;

  //! Retrieve index buffer via key
  virtual ailego::BlobWrap get(const std::string &key) const {
    auto iter = map_.find(key);
    if (iter != map_.end()) {
      return ailego::BlobWrap(iter->second.region(), iter->second.size());
    }
    return ailego::BlobWrap();
  }

  //! Test if the key is exist
  virtual bool has(const std::string &key) const {
    return (map_.find(key) != map_.end());
  }

  //! Retrieve all
  virtual std::map<std::string, ailego::BlobWrap> all(void) const {
    std::map<std::string, ailego::BlobWrap> result;
    for (const auto &it : map_) {
      result.emplace(it.first,
                     ailego::BlobWrap(it.second.region(), it.second.size()));
    }
    return result;
  }

  //! Retrieve the count of indexes
  virtual size_t count(void) const {
    return map_.size();
  }

  //! Create a memory mapping file in bundle
  bool create(const std::string &prefix, const std::string &key, size_t len) {
    ailego::MMapFile file;
    if (!file.create(prefix + '/' + key, len)) {
      return false;
    }
    map_[key] = std::move(file);
    return true;
  }

  //! Create a memory mapping file in bundle
  bool create(const std::string &prefix, std::string &&key, size_t len) {
    ailego::MMapFile file;
    if (!file.create(prefix + '/' + key, len)) {
      return false;
    }
    map_[std::move(key)] = std::move(file);
    return true;
  }

  //! Create a memory mapping file in bundle
  bool create(const std::string &path, size_t len) {
    ailego::MMapFile file;
    if (!file.create(path, len)) {
      return false;
    }
    map_[ailego::File::BaseName(path)] = std::move(file);
    return true;
  }

  //! Open a memory mapping file in bundle
  bool open(const std::string &prefix, const std::string &key, bool rdonly) {
    ailego::MMapFile file;
    if (!file.open(prefix + '/' + key, rdonly)) {
      return false;
    }
    map_[key] = std::move(file);
    return true;
  }

  //! Open a memory mapping file in bundle
  bool open(const std::string &prefix, std::string &&key, bool rdonly) {
    ailego::MMapFile file;
    if (!file.open(prefix + '/' + key, rdonly)) {
      return false;
    }
    map_[std::move(key)] = std::move(file);
    return true;
  }

  //! Open a memory mapping file in bundle
  bool open(const std::string &path, bool rdonly) {
    ailego::MMapFile file;
    if (!file.open(path, rdonly)) {
      return false;
    }
    map_[ailego::File::BaseName(path)] = std::move(file);
    return true;
  }

 private:
  std::map<std::string, ailego::MMapFile> map_;
};

}  // namespace core
}  // namespace zvec
