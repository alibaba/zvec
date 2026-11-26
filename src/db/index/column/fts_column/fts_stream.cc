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

#include "fts_stream.h"
#include <algorithm>
#include <limits>
#include <zvec/ailego/buffer/vector_page_table.h>

namespace zvec::fts {
namespace {

Status IoError() {
  return {StatusCode::UNAVAILABLE, "FTS temporary section I/O failed"};
}

#ifdef _WIN32
constexpr uint64_t kOffsetLimit = INT64_MAX;
#else
constexpr uint64_t kOffsetLimit = std::numeric_limits<off_t>::max();
#endif

}  // namespace

Result<std::unique_ptr<FtsSpool>> FtsSpool::Create() {
  auto *file = std::tmpfile();
  if (!file) {
    return tl::make_unexpected(IoError());
  }
  return std::unique_ptr<FtsSpool>(new FtsSpool(file));
}

FtsSpool::~FtsSpool() {
  std::fclose(file_);
}

Status FtsSpool::seek(uint64_t offset, bool writing) {
  if (!status_.ok()) {
    return status_;
  }
  if (offset > kOffsetLimit) {
    return {StatusCode::RESOURCE_EXHAUSTED,
            "FTS temporary section exceeds platform offset limit"};
  }
  // Consecutive appends/reads keep stdio buffering intact. Switching between
  // reading and writing still requires positioning, even at the same offset.
  if (position_ == offset && writing_ == writing) {
    return Status::OK();
  }
#ifdef _WIN32
  const int result = _fseeki64(file_, static_cast<int64_t>(offset), SEEK_SET);
#else
  const int result = fseeko(file_, static_cast<off_t>(offset), SEEK_SET);
#endif
  if (result != 0) {
    return status_ = IoError();
  }
  position_ = offset;
  writing_ = writing;
  return Status::OK();
}

Status FtsSpool::append(std::string_view bytes) {
  if (!status_.ok()) {
    return status_;
  }
  if (!format::ValidRange(size_, bytes.size(), kOffsetLimit)) {
    return {StatusCode::RESOURCE_EXHAUSTED,
            "FTS temporary section exceeds platform offset limit"};
  }
  auto status = seek(size_, true);
  if (!status.ok()) {
    return status;
  }
  if (std::fwrite(bytes.data(), 1, bytes.size(), file_) != bytes.size()) {
    return status_ = IoError();
  }
  size_ += bytes.size();
  position_ += bytes.size();
  return Status::OK();
}

Status FtsSpool::overwrite(uint64_t offset, std::string_view bytes) {
  if (!format::ValidRange(offset, bytes.size(), size_)) {
    return Status::InvalidArgument("Invalid FTS temporary section range");
  }
  auto status = seek(offset, true);
  if (!status.ok()) {
    return status;
  }
  if (std::fwrite(bytes.data(), 1, bytes.size(), file_) != bytes.size()) {
    return status_ = IoError();
  }
  position_ += bytes.size();
  return Status::OK();
}

Result<std::string> FtsSpool::read(uint64_t offset, size_t length) {
  if (length > 65536) {
    return tl::make_unexpected(Status(StatusCode::RESOURCE_EXHAUSTED,
                                      "FTS spool read exceeds workspace"));
  }
  if (!format::ValidRange(offset, length, size_)) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS temporary section range"));
  }
  auto status = seek(offset, false);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  std::string bytes(length, '\0');
  if (std::fread(bytes.data(), 1, length, file_) != length) {
    return tl::make_unexpected(status_ = IoError());
  }
  position_ += length;
  return bytes;
}

Status FtsSpool::copy_to(FtsContainerWriter *container,
                         const std::string &section, std::string_view header) {
  if (!status_.ok()) {
    return status_;
  }
  auto status = container->begin_section(section);
  if (status.ok()) {
    status = container->append(header);
  }
  for (uint64_t offset = 0; status.ok() && offset < size_;) {
    const auto length =
        static_cast<size_t>(std::min<uint64_t>(16384, size_ - offset));
    auto bytes = read(offset, length);
    if (!bytes) {
      return bytes.error();
    }
    status = container->append(*bytes);
    offset += length;
  }
  return status.ok() ? container->end_section() : status;
}

Result<std::unique_ptr<FtsSectionReader>> FtsSectionReader::Open(
    FtsContainerReader::Pointer container, const std::string &section) {
  if (!container) {
    return tl::make_unexpected(
        Status::InvalidArgument("Missing FTS container"));
  }
  for (const auto &metadata : container->metadata().sections) {
    if (metadata.name == section) {
      std::unique_ptr<FtsSectionReader> reader(new FtsSectionReader);
      reader->owner_ = std::move(container);
      reader->name_ = section;
      reader->size_ = metadata.length;
      reader->window_granule_ = reader->owner_->metadata().layout_granule;
      if (reader->owner_->buffer_pool_) {
        // Only paged reads need the physical origin. Storage::get creates a
        // segment handle, so avoid a throwaway handle for every mapped cursor.
        const auto physical = reader->owner_->storage_->get(section);
        if (!physical) {
          return tl::make_unexpected(
              Status::InternalError("Missing FTS section storage: ", section));
        }
        reader->window_granule_ = ailego::kVectorPageSize;
        reader->section_page_offset_ =
            physical->data_offset() % ailego::kVectorPageSize;
      }
      return reader;
    }
  }
  return tl::make_unexpected(
      Status::InternalError("Missing FTS section: ", section));
}

Result<std::string_view> FtsSectionReader::refill(uint64_t offset,
                                                  size_t length) {
  if (!format::ValidRange(offset, length, size_)) {
    return tl::make_unexpected(
        Status::InternalError("Invalid FTS section reference"));
  }
  if (length > 65536) {
    return tl::make_unexpected(Status(StatusCode::RESOURCE_EXHAUSTED,
                                      "FTS section read exceeds workspace"));
  }
  if (length == 0) {
    return std::string_view{};
  }
  cache_ = {};
  // Buffer windows follow native backing-file pages, clipped to the
  // section. Mapped views need no page pins or cross-page copies and keep
  // their section-relative layout windows.
  const uint64_t page = window_granule_;
  cache_start_ = offset - offset % page;
  uint64_t end = offset + length;
  uint64_t tail = end % page;
  if (section_page_offset_ != 0) {
    uint64_t within = offset - cache_start_ + section_page_offset_;
    if (within >= page) {
      within -= page;
    }
    cache_start_ = offset - std::min(offset, within);
    tail += section_page_offset_;
    if (tail >= page) {
      tail -= page;
    }
  }
  if (tail != 0) {
    end += std::min<uint64_t>(page - tail, size_ - end);
  }
  auto range = owner_->read(name_, cache_start_, end - cache_start_);
  if (!range) {
    return tl::make_unexpected(range.error());
  }
  cache_ = std::move(*range);
  return cache_.view().substr(static_cast<size_t>(offset - cache_start_),
                              length);
}

}  // namespace zvec::fts
