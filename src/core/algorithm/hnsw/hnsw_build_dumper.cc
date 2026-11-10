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

#include "hnsw_build_dumper.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/core/framework/index_version.h>

namespace zvec::core {

HnswBuildDumper::~HnswBuildDumper() {
  close();
}

int HnswBuildDumper::init(const ailego::Params &) {
  return 0;
}

int HnswBuildDumper::cleanup() {
  return close();
}

int HnswBuildDumper::create(const std::string &path) {
  // HnswT<type>S<sequence> uses at most 27 bytes plus its terminator, even
  // with a uint64_t sequence. IndexVersion and IndexMeta are shorter.
  constexpr size_t entry_size = sizeof(IndexFormat::SegmentMeta) + 32;
  const size_t max_capacity = std::numeric_limits<uint32_t>::max() -
                              sizeof(IndexFormat::MetaHeader) -
                              sizeof(IndexFormat::MetaFooter) -
                              std::max(ailego::MemoryHelper::PageSize(),
                                       ailego::MemoryHelper::HugePageSize());
  if (segment_count_ == 0) return IndexError_InvalidArgument;
  if (segment_count_ > max_capacity / entry_size) return IndexError_Overflow;
  metadata_remaining_ = segment_count_ * entry_size;
  opened_ = true;
  int ret = mapping_.create(path, metadata_remaining_);
  if (ret != 0) return ret;
  if (!file_.open(path.c_str(), false, false)) return IndexError_OpenFile;
  const char *version = IndexVersion::Details();
  const size_t bytes = std::strlen(version);
  if (write(version, bytes) != bytes) return IndexError_WriteData;
  return append("IndexVersion", bytes, 0,
                ailego::Crc32c::Hash(version, bytes, 0));
}

int HnswBuildDumper::close() {
  int ret = 0;
  if (opened_) {
    mapping_.refresh(0);
    ret = mapping_.flush();
    mapping_.close();
    file_.close();
    opened_ = false;
  }
  return ret;
}

size_t HnswBuildDumper::write(const void *data, size_t bytes) {
  const size_t result =
      file_.write(mapping_.index_size() + pending_, data, bytes);
  pending_ += result;
  return result;
}

int HnswBuildDumper::append(const std::string &id, size_t data_size,
                            size_t padding_size, uint32_t crc) {
  if (pending_ != data_size + padding_size) return IndexError_InvalidFormat;
  const size_t metadata_size = sizeof(IndexFormat::SegmentMeta) + id.size() + 1;
  if (metadata_size > metadata_remaining_) return IndexError_IndexFull;
  int ret = mapping_.append(id, pending_);
  if (ret != 0) return ret;
  metadata_remaining_ -= metadata_size;
  auto segment = mapping_.map(id, false, false);
  if (!segment) return IndexError_MMapFile;
  auto meta = segment->meta();
  meta->padding_size += meta->data_size;
  meta->data_size = data_size;
  meta->padding_size -= data_size;
  // Mutable HNSW chunks use the native format's optional checksum (zero).
  // Subsequent mmap appends update their bytes without recomputing a CRC.
  meta->data_crc = id.rfind("HnswT", 0) == 0 ? 0 : crc;
  mapping_.unmap(id);
  pending_ = 0;
  return 0;
}

uint32_t HnswBuildDumper::magic() const {
  return mapping_.magic();
}

}  // namespace zvec::core
