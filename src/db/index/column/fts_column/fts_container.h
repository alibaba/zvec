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

#include <atomic>
#include <cstdio>
#include <memory>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_storage.h>
#include "fts_format.h"

namespace zvec::fts {

// Low-level section writer for a PREPARED container. The caller supplies an
// already-created private dumper and retains responsibility for fsync, unique
// naming, publication and cleanup. This never commits a database version.
class FtsContainerWriter {
 public:
  static Result<std::unique_ptr<FtsContainerWriter>> Create(
      core::IndexDumper::Pointer dumper, uint32_t layout_granule = 4096);
  ~FtsContainerWriter();
  FtsContainerWriter(const FtsContainerWriter &) = delete;
  FtsContainerWriter &operator=(const FtsContainerWriter &) = delete;

  Status begin_section(const std::string &name);
  Status append(std::string_view bytes);
  Status end_section();
  // sections must be empty; the writer supplies lengths/checksum references.
  Status finish(const format::Metadata &metadata);

 private:
  FtsContainerWriter(core::IndexDumper::Pointer dumper, uint32_t granule,
                     std::FILE *spool);
  Status checksum_chunk();
  Status checksum_page();
  Status write_bytes(std::string_view bytes);

  core::IndexDumper::Pointer dumper_;
  uint32_t granule_;
  std::FILE *spool_;
  uint64_t output_size_{0};
  std::vector<format::SectionMetadata> sections_;
  std::string active_name_;
  uint64_t section_length_{0};
  uint64_t checksum_count_{0};
  uint64_t checksum_bytes_{0};
  uint64_t first_checksum_{0};
  format::Sha256 chunk_hash_;
  size_t chunk_size_{0};
  std::array<uint8_t, 4096> checksum_page_{};
  size_t page_entries_{0};
  bool finished_{false};
  Status status_;
};

class FtsContainerReader;

// A range pins its file owner as well as the native storage's data handle.
// Reuse view() while this handle lives; no storage close is exposed to callers.
class FtsReadRange {
 public:
  FtsReadRange() = default;
  FtsReadRange(FtsReadRange &&other) noexcept;
  FtsReadRange &operator=(FtsReadRange &&other) noexcept;
  FtsReadRange(const FtsReadRange &) = delete;
  FtsReadRange &operator=(const FtsReadRange &) = delete;
  std::string_view view() const;

 private:
  friend class FtsContainerReader;
  void swap(FtsReadRange &other) noexcept;
  std::shared_ptr<void> lease_;
  core::IndexStorage::Segment::Pointer segment_;
  core::IndexStorage::MemoryBlock block_;
  size_t length_{0};
};

class FtsContainerReader
    : public std::enable_shared_from_this<FtsContainerReader> {
 public:
  using Pointer = std::shared_ptr<FtsContainerReader>;
  // enable_mmap must come from the persisted DB configuration at integration.
  // read_budget bounds concurrent range leases, separate from the global pool
  // and the fixed metadata budget. Insufficient budget is an explicit error.
  // file_owner pins generation retirement; it is released after native
  // storage handles, including when only a query's range remains alive.
  static Result<Pointer> Open(const std::string &path, bool enable_mmap,
                              size_t read_budget = 1024 * 1024,
                              std::shared_ptr<void> file_owner = {});
  const format::Metadata &metadata() const {
    return metadata_;
  }
  // Digest of exact persisted metadata, including unknown optional records.
  const format::Digest &metadata_digest() const {
    return metadata_digest_;
  }
  Result<FtsReadRange> read(const std::string &section, uint64_t offset,
                            uint64_t length);
  // Explicit diagnostic only. Normal open/read/publication do not scan data
  // checksums, matching the vector storage's default validation policy.
  Status verify_all();

  struct ReadStats {
    uint64_t range_reads;
    uint64_t verified_pages;
    uint64_t hashed_bytes;
  };
  ReadStats read_stats() const {
    return {range_reads_.load(std::memory_order_relaxed),
            verified_pages_.load(std::memory_order_relaxed),
            hashed_bytes_.load(std::memory_order_relaxed)};
  }

 private:
  struct Lease;
  FtsContainerReader() = default;
  Result<std::shared_ptr<Lease>> reserve(size_t bytes);
  Status verify_chunk(const format::SectionMetadata &meta, uint64_t chunk,
                      std::string_view bytes);

  // Declaration order is intentional: storage must close before retirement.
  std::shared_ptr<void> file_owner_;
  core::IndexStorage::Pointer storage_;
  core::IndexStorage::Segment::Pointer checksums_;
  format::Metadata metadata_;
  size_t budget_{0};
  std::atomic<size_t> reserved_{0};
  format::Digest metadata_digest_{};
  std::atomic<uint64_t> range_reads_{0};
  std::atomic<uint64_t> verified_pages_{0};
  std::atomic<uint64_t> hashed_bytes_{0};
};

}  // namespace zvec::fts
