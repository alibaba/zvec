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

#include "fts_phrase_iterator.h"
#include <algorithm>
#include <cstring>
#include "../fts_utils.h"

namespace zvec::fts {

PhraseDocIterator::PhraseDocIterator(DocIteratorPtr conjunction,
                                     std::vector<std::string> terms,
                                     RocksdbContext *ctx,
                                     rocksdb::ColumnFamilyHandle *positions_cf)
    : conjunction_(std::move(conjunction)),
      terms_(std::move(terms)),
      ctx_(ctx),
      positions_cf_(positions_cf) {
  cached_max_score_ = conjunction_->cached_max_score_;
}

uint32_t PhraseDocIterator::next_doc() {
  cached_doc_id_ = conjunction_->next_doc();
  return cached_doc_id_;
}

uint32_t PhraseDocIterator::advance(uint32_t target) {
  cached_doc_id_ = conjunction_->advance(target);
  return cached_doc_id_;
}

bool PhraseDocIterator::matches() {
  if (cached_doc_id_ == NO_MORE_DOCS) {
    return false;
  }
  // Phase 2: verify position adjacency (deferred IO)
  return verify_phrase_positions(cached_doc_id_);
}

float PhraseDocIterator::score() {
  return conjunction_->score();
}

uint64_t PhraseDocIterator::cost() const {
  return conjunction_->cost();
}

float PhraseDocIterator::max_score() const {
  return conjunction_->max_score();
}

bool PhraseDocIterator::verify_phrase_positions(uint32_t doc_id) const {
  if (terms_.empty()) {
    return false;
  }

  // Read position list of first term as anchor.
  // Empty anchor means the term has no position record for this doc — this is
  // normal for non-matching docs filtered through the conjunction without a
  // position-CF entry, so do NOT log here.
  std::vector<uint32_t> anchor_positions = read_positions(terms_[0], doc_id);
  if (anchor_positions.empty()) {
    return false;
  }

  // For each anchor position, verify if subsequent terms appear at consecutive
  // positions
  for (uint32_t anchor_pos : anchor_positions) {
    bool phrase_matched = true;
    for (size_t term_index = 1; term_index < terms_.size(); ++term_index) {
      const uint32_t expected_pos =
          anchor_pos + static_cast<uint32_t>(term_index);
      std::vector<uint32_t> positions =
          read_positions(terms_[term_index], doc_id);
      bool found =
          std::binary_search(positions.begin(), positions.end(), expected_pos);
      if (!found) {
        phrase_matched = false;
        break;
      }
    }
    if (phrase_matched) {
      return true;
    }
  }

  return false;
}

std::vector<uint32_t> PhraseDocIterator::read_positions(const std::string &term,
                                                        uint32_t doc_id) const {
  const std::string key = fts::make_doc_term_key(term, doc_id);
  std::string value;
  if (!ctx_->db_->Get(ctx_->read_opts_, positions_cf_, key, &value).ok() ||
      value.empty()) {
    return {};
  }
  return decode_positions(value);
}

std::vector<uint32_t> PhraseDocIterator::decode_positions(
    const std::string &data) {
  std::vector<uint32_t> positions;
  size_t index = 0;
  uint32_t current_position = 0;

  while (index < data.size()) {
    // Decode varint
    uint32_t delta = 0;
    uint32_t shift = 0;
    while (index < data.size()) {
      const uint8_t byte = static_cast<uint8_t>(data[index++]);
      delta |= static_cast<uint32_t>(byte & 0x7F) << shift;
      shift += 7;
      if ((byte & 0x80) == 0) {
        break;
      }
    }
    current_position += delta;
    positions.push_back(current_position);
  }

  return positions;
}

}  // namespace zvec::fts
