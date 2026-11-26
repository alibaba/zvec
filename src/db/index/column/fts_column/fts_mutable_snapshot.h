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

#include "fts_field.h"

namespace zvec::fts {

// Captures one read view for all fields after the caller's full-storage
// checkpoint. The caller pins schema/CF identity until conversion finishes;
// this owner pins the context and snapshot. It never drops CFs, converts legacy
// postings in place, publishes a version, or cleans up the database WAL.
class FtsMutableSnapshot {
 public:
  static Result<std::unique_ptr<FtsMutableSnapshot>> Capture(
      std::shared_ptr<RocksdbContext> context, uint64_t rows);
  ~FtsMutableSnapshot();
  FtsMutableSnapshot(const FtsMutableSnapshot &) = delete;
  FtsMutableSnapshot &operator=(const FtsMutableSnapshot &) = delete;

  // Caller supplies field/configuration identity. Statistics and presence are
  // read from the captured view. Output remains an uncommitted candidate.
  Result<format::FieldMetadata> write_field(format::FieldMetadata identity,
                                            BM25Params params,
                                            FtsContainerWriter *container,
                                            uint32_t ordinal) const;

 private:
  FtsMutableSnapshot(std::shared_ptr<RocksdbContext> context, uint64_t rows);
  std::shared_ptr<RocksdbContext> context_;
  rocksdb::ReadOptions options_;
  uint64_t rows_;
};

}  // namespace zvec::fts
