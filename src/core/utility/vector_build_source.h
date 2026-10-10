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

#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_holder.h>
#include "temporary_buffer_storage.h"

namespace zvec::core {

// Immutable vector build input. Borrow an ordinal-capable holder rather than
// copying it. Otherwise spool each row once, retaining keys by ordinal
// (including duplicates). With a configured Buffer Pool the spool is pageable;
// callers without a pool retain the legacy in-memory fallback. Never fall back
// to the heap after an I/O or budget error. Each spooled reader owns at most
// one row.
class VectorBuildSource
    : public OrdinalAccessHolder,
      public std::enable_shared_from_this<VectorBuildSource> {
 public:
  using Pointer = std::shared_ptr<VectorBuildSource>;
  VectorBuildSource(const VectorBuildSource &) = delete;
  VectorBuildSource &operator=(const VectorBuildSource &) = delete;
  static int Create(IndexHolder::Pointer holder, Pointer *out,
                    const std::string &scratch_prefix = "");

  size_t count() const {
    return count_;
  }
  int create_ordinal_reader(Reader::Pointer *out) override;

 private:
  VectorBuildSource() = default;
  class SpoolReader;
  int init(IndexHolder::Pointer holder, const std::string &scratch_prefix);
  int append(uint64_t key, const void *data);

  IndexHolder::Pointer holder_;
  size_t count_{0};
  size_t element_bytes_{0};
  size_t record_words_{0};
  size_t rows_per_chunk_{0};
  TemporaryBufferStorage::Pointer file_;
  // Destroy segments before the file that owns them.
  std::vector<IndexStorage::Segment::Pointer> segments_;
  std::vector<std::vector<uint64_t>> memory_chunks_;
};

}  // namespace zvec::core
