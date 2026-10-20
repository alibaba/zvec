// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <zvec/core/framework/index_dumper.h>
#include "temporary_buffer_storage.h"

namespace zvec::core {
// Streams one known-size segment directly into storage, without an IndexMemory
// handoff copy. The normal dumper CRC/padding format is retained.
class StorageSegmentDumper : public IndexDumper {
 public:
  int prepare(IndexStorage::Pointer storage, const std::string &id,
              size_t bytes) {
    if (!storage || segment_) return IndexError_InvalidArgument;
    int ret = storage->append(id, bytes);
    if (ret != 0) return ret;
    segment_ = storage->get(id);
    storage_ = std::move(storage);
    offset_ = 0;
    id_ = id;
    return segment_ ? 0 : IndexError_WriteData;
  }
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    segment_.reset();
    storage_.reset();
    offset_ = 0;
    return 0;
  }
  int create(const std::string &) override {
    return IndexError_NotImplemented;
  }
  int close() override {
    return 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  size_t size() const override {
    return offset_;
  }
  size_t write(const void *data, size_t bytes) override {
    if (!segment_) return 0;
    const size_t written = segment_->write(offset_, data, bytes);
    offset_ += written;
    return written;
  }
  int append(const std::string &id, size_t data, size_t padding,
             uint32_t crc) override {
    if (!segment_) return IndexError_NoReady;
    if (id != id_ || data > offset_ || padding != offset_ - data)
      return IndexError_InvalidArgument;
    segment_->resize(data);
    segment_->update_data_crc(crc);
    return 0;
  }

 private:
  std::string id_;
  IndexStorage::Pointer storage_;
  IndexStorage::Segment::Pointer segment_;
  size_t offset_{0};
};

// Aliasing ownership retains the temporary file for every consumer of storage.
inline int CreateTemporaryIndexStorage(IndexStorage::Pointer *out) {
  std::string prefix;
  int ret = TemporaryBufferStorage::DefaultPrefix("zvec-trained", &prefix);
  if (ret != 0) return ret;
  TemporaryBufferStorage::Pointer file;
  ret = TemporaryBufferStorage::CreateEmpty(prefix, &file);
  if (ret == 0) *out = IndexStorage::Pointer(file, file->storage().get());
  return ret;
}
}  // namespace zvec::core
