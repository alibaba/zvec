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

#include "db/common/rocksdb_context.h"
#include "fts_reader.h"

namespace zvec::fts {

struct LegacyFtsSource {
  RocksdbContext *context{nullptr};
  std::shared_ptr<RocksdbContext> owner;
  rocksdb::ColumnFamilyHandle *postings{nullptr};
  rocksdb::ColumnFamilyHandle *positions{nullptr};
  rocksdb::ColumnFamilyHandle *frequencies{nullptr};
  rocksdb::ColumnFamilyHandle *max_tf{nullptr};
  rocksdb::ColumnFamilyHandle *lengths{nullptr};
};

// The legacy adapter retains complete RocksDB values. Its budget limits
// retained values, not RocksDB's temporary allocation during Get/MultiGet.
class LegacyFtsReader final
    : public FtsSegmentReader,
      public std::enable_shared_from_this<LegacyFtsReader> {
 public:
  using Ptr = std::shared_ptr<LegacyFtsReader>;
  static Result<Ptr> Create(LegacyFtsSource source, BM25ScorerPtr scorer);
  BM25ScorerPtr scorer() const override;
  Result<FtsReadTerm::Ptr> lookup(std::string_view term) const override;
  Result<std::vector<FtsReadTerm::Ptr>> lookup_many(
      const std::vector<std::string_view> &terms) const override;
  Result<std::vector<FtsReadPosition::Ptr>> positions(
      const std::vector<FtsReadTerm::Ptr> &terms, uint32_t doc) const override;
  void observe_length(uint32_t length);
  // Stops admitting side-CF users and drains existing term handles before
  // callers drop the CFs. New Roaring handles retain the legacy null-CF mode.
  void reset_side_cfs();

 private:
  class Term;
  class Posting;
  class Position;
  struct SideLease;
  LegacyFtsReader(LegacyFtsSource source, BM25ScorerPtr scorer);
  Result<FtsReadTerm::Ptr> decode(
      std::string_view term,
      std::shared_ptr<rocksdb::PinnableSlice> value) const;
  LegacyFtsSource source_;
  BM25ScorerPtr scorer_;
  mutable std::atomic<uint64_t> side_users_{0};
  std::atomic<bool> side_dropped_{false};
  std::atomic<uint32_t> min_length_{UINT32_MAX};
};

}  // namespace zvec::fts
