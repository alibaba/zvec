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

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {
namespace {

// This storage lets a test retain a small writer page while allocating new
// segments with a larger reader page, without depending on the CI host OS.
class LayoutTestSegment : public IndexStorage::Segment {
 public:
  explicit LayoutTestSegment(size_t capacity) : bytes(capacity) {}
  size_t data_size() const override {
    return used;
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return bytes.size() - used;
  }
  size_t capacity() const override {
    return bytes.size();
  }
  size_t fetch(size_t offset, void *buf, size_t len) const override {
    if (offset > used) return 0;
    len = std::min(len, used - offset);
    std::memcpy(buf, bytes.data() + offset, len);
    return len;
  }
  size_t read(size_t offset, const void **data, size_t len) override {
    if (offset > used) return 0;
    *data = bytes.data() + offset;
    return std::min(len, used - offset);
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &block,
              size_t len) override {
    const void *data = nullptr;
    size_t size = read(offset, &data, len);
    block.reset(const_cast<void *>(data));
    return size;
  }
  size_t write(size_t offset, const void *data, size_t len) override {
    if (offset > bytes.size() || len > bytes.size() - offset) return 0;
    std::memcpy(bytes.data() + offset, data, len);
    used = std::max(used, offset + len);
    return len;
  }
  size_t resize(size_t size) override {
    used = std::min(size, bytes.size());
    return used;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<LayoutTestSegment>(*this);
  }
  std::vector<char> bytes;
  size_t used{0};
};

class LayoutTestStorage : public IndexStorage {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int open(const std::string &, bool) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  int flush() override {
    ++flushes;
    return 0;
  }
  int append(const std::string &id, size_t size) override {
    ++appends;
    if (has(id)) return IndexError_Duplicate;
    size = (size + physical_page - 1) / physical_page * physical_page;
    segments[id] = std::make_shared<LayoutTestSegment>(size);
    return 0;
  }
  void refresh(uint64_t) override {}
  uint64_t check_point() const override {
    return 0;
  }
  Segment::Pointer get(const std::string &id, int = -1) override {
    auto it = segments.find(id);
    return unreadable || it == segments.end() ? nullptr : it->second;
  }
  bool has(const std::string &id) const override {
    return segments.count(id);
  }
  uint32_t magic() const override {
    return 0;
  }

  void seed_legacy(size_t writer_page) {
    auto meta = std::make_shared<LayoutTestSegment>(writer_page);
    uint64_t words[16]{};
    words[2] = 1;                // metadata chunk count
    words[8] = 2 * 1024 * 1024;  // configured chunk size
    words[9] = writer_page;
    meta->write(0, words, sizeof(words));
    segments["HnswT2S0"] = meta;
  }
  std::map<std::string, std::shared_ptr<LayoutTestSegment>> segments;
  size_t physical_page{16384};
  bool unreadable{false};
  int appends{0};
  int flushes{0};
};

}  // namespace
}  // namespace core
}  // namespace zvec
