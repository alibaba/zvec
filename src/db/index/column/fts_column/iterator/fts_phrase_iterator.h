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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "db/common/rocksdb_context.h"
#include "fts_conjunction_iterator.h"
#include "fts_doc_iterator.h"
#include "../bm25_scorer.h"

namespace zvec::fts {

/*! Phrase document iterator (two-phase)
 *
 *  Internally wraps a ConjunctionIterator for phase-1 doc_id intersection.
 *  Phase-2 matches() reads position payloads and checks adjacency.
 */
class PhraseDocIterator : public DocIterator {
 public:
  /*! Construct a phrase iterator.
   *  \param conjunction   ConjunctionIterator over all terms in the phrase
   *  \param terms         Processed (tokenized) term strings in phrase order
   *  \param positions_cf  $POS column family for reading position lists
   */
  PhraseDocIterator(DocIteratorPtr conjunction, std::vector<std::string> terms,
                    RocksdbContext *ctx,
                    rocksdb::ColumnFamilyHandle *positions_cf);

  uint32_t next_doc() override;
  uint32_t advance(uint32_t target) override;

  //! Phase-2: verify position adjacency for the current document.
  //! Reads position lists from $POS CF (deferred IO).
  bool matches() override;

  float score() override;
  uint64_t cost() const override;
  float max_score() const override;

 private:
  // Read position list for a term in a specific document
  std::vector<uint32_t> read_positions(const std::string &term,
                                       uint32_t doc_id) const;

  // Verify that terms appear at consecutive positions in the document
  bool verify_phrase_positions(uint32_t doc_id) const;

  // Decode varint delta-encoded position list
  static std::vector<uint32_t> decode_positions(const std::string &data);

 private:
  DocIteratorPtr conjunction_;
  std::vector<std::string> terms_;
  RocksdbContext *ctx_;
  rocksdb::ColumnFamilyHandle *positions_cf_;
};

}  // namespace zvec::fts
