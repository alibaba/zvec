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

#include "vector_build_source.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::core {
namespace {
class BorrowedBuildReader : public OrdinalAccessHolder::Reader {
 public:
  BorrowedBuildReader(IndexHolder::Pointer holder, Reader::Pointer reader)
      : holder_(std::move(holder)), reader_(std::move(reader)) {}
  int read(size_t ordinal, uint64_t *key, const void **data) override {
    try {
      return reader_->read(ordinal, key, data);
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_ReadData;
    }
  }
  void reset() override {
    reader_->reset();
  }

 private:
  IndexHolder::Pointer holder_;
  Reader::Pointer reader_;
};
}  // namespace

class VectorBuildSource::SpoolReader : public OrdinalAccessHolder::Reader {
 public:
  explicit SpoolReader(std::shared_ptr<VectorBuildSource> source)
      : source_(std::move(source)) {}

  int read(size_t ordinal, uint64_t *key, const void **data) override {
    if (!key || !data) return IndexError_InvalidArgument;
    if (ordinal >= source_->count_) return IndexError_OutOfRange;
    try {
      const size_t chunk = ordinal / source_->rows_per_chunk_;
      const size_t word_offset =
          (ordinal % source_->rows_per_chunk_) * source_->record_words_;
      const uint64_t *record = nullptr;
      if (source_->file_) {
        scratch_.resize(source_->record_words_);
        const size_t bytes = sizeof(uint64_t) + source_->element_bytes_;
        if (source_->segments_[chunk]->fetch(word_offset * sizeof(uint64_t),
                                             scratch_.data(), bytes) != bytes) {
          return IndexError_ReadData;
        }
        record = scratch_.data();
      } else {
        record = source_->memory_chunks_[chunk].data() + word_offset;
      }
      *key = record[0];
      *data = record + 1;
      return 0;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_ReadData;
    }
  }

  void reset() override {
    std::vector<uint64_t>().swap(scratch_);
  }

 private:
  std::shared_ptr<VectorBuildSource> source_;
  std::vector<uint64_t> scratch_;
};

int VectorBuildSource::Create(IndexHolder::Pointer holder, Pointer *out,
                              const std::string &scratch_prefix) {
  if (!out || !holder) return IndexError_InvalidArgument;
  try {
    Pointer source(new VectorBuildSource());
    int ret = source->init(std::move(holder), scratch_prefix);
    if (ret == 0) *out = std::move(source);
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }
}

int VectorBuildSource::init(IndexHolder::Pointer holder,
                            const std::string &scratch_prefix) {
  const size_t expected_count = holder->count();
  const bool known_count = expected_count != std::numeric_limits<size_t>::max();
  if (known_count && expected_count > std::numeric_limits<uint32_t>::max()) {
    return IndexError_Overflow;
  }
  if (known_count) {
    auto *ordinal = dynamic_cast<OrdinalAccessHolder *>(holder.get());
    if (ordinal) {
      Reader::Pointer probe;
      int ret = ordinal->create_ordinal_reader(&probe);
      if (ret == 0) {
        if (!probe) return IndexError_Runtime;
        holder_ = std::move(holder);
        count_ = expected_count;
        return 0;
      }
      if (ret != IndexError_NotImplemented) return ret;
    }
  }

  element_bytes_ = holder->element_size();
  if (element_bytes_ == 0 ||
      element_bytes_ >
          std::numeric_limits<size_t>::max() - 2 * sizeof(uint64_t)) {
    return IndexError_InvalidArgument;
  }
  record_words_ =
      1 + (element_bytes_ + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  rows_per_chunk_ =
      std::max<size_t>(1, (4U << 20) / (record_words_ * sizeof(uint64_t)));
  if (known_count && expected_count != 0) {
    rows_per_chunk_ = std::min(rows_per_chunk_, expected_count);
  }

  auto iter = holder->create_iterator();
  if (!iter) return IndexError_Runtime;
  const bool buffered = ailego::MemoryLimitPool::get_instance().capacity() != 0;
  for (; iter->is_valid(); iter->next()) {
    if (iter->status() != 0) return iter->status();
    if (known_count && count_ >= expected_count)
      return IndexError_InvalidLength;
    if (count_ == std::numeric_limits<uint32_t>::max())
      return IndexError_Overflow;
    const uint64_t key = iter->key();
    if (iter->status() != 0) return iter->status();
    const void *data = iter->data();
    if (iter->status() != 0) return iter->status();
    if (!data) return IndexError_ReadData;
    if (buffered && !file_) {
      std::string prefix = scratch_prefix;
      if (prefix.empty()) {
        std::error_code error;
        auto dir = std::filesystem::temp_directory_path(error);
        if (error) return IndexError_OpenFile;
        prefix = ailego::FileHelper::PathToUtf8(dir / "zvec-build-input");
      }
      int ret = TemporaryBufferStorage::CreateEmpty(prefix, &file_);
      if (ret != 0) return ret;
    }
    int ret = append(key, data);
    if (ret != 0) return ret;
  }
  if (iter->status() != 0) return iter->status();
  if (known_count && count_ != expected_count) return IndexError_InvalidLength;
  return file_ ? file_->flush() : 0;
}

int VectorBuildSource::append(uint64_t key, const void *data) {
  const size_t chunk = count_ / rows_per_chunk_;
  const size_t word_offset = (count_ % rows_per_chunk_) * record_words_;
  const size_t chunk_words = rows_per_chunk_ * record_words_;
  if (file_) {
    if (chunk == segments_.size()) {
      std::string name = std::to_string(chunk);
      int ret = file_->storage()->append(name, chunk_words * sizeof(uint64_t));
      if (ret != 0) return ret;
      auto segment = file_->storage()->get(name);
      if (!segment) return IndexError_Runtime;
      segments_.push_back(std::move(segment));
    }
    auto &segment = segments_[chunk];
    const size_t offset = word_offset * sizeof(uint64_t);
    if (segment->write(offset, &key, sizeof(key)) != sizeof(key) ||
        segment->write(offset + sizeof(key), data, element_bytes_) !=
            element_bytes_) {
      return IndexError_WriteData;
    }
  } else {
    if (chunk == memory_chunks_.size())
      memory_chunks_.emplace_back(chunk_words);
    auto *record = memory_chunks_[chunk].data() + word_offset;
    record[0] = key;
    std::memcpy(record + 1, data, element_bytes_);
  }
  ++count_;
  return 0;
}

int VectorBuildSource::create_ordinal_reader(Reader::Pointer *out) {
  if (!out) return IndexError_InvalidArgument;
  try {
    if (holder_) {
      Reader::Pointer reader;
      auto *source = dynamic_cast<OrdinalAccessHolder *>(holder_.get());
      int ret = source->create_ordinal_reader(&reader);
      if (ret != 0) return ret;
      if (!reader) return IndexError_Runtime;
      *out = std::make_unique<BorrowedBuildReader>(holder_, std::move(reader));
    } else {
      *out = std::make_unique<SpoolReader>(shared_from_this());
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }
}

}  // namespace zvec::core
