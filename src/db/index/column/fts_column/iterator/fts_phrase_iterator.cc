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
#include <unordered_map>
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

uint32_t PhraseDocIterator::next_doc(const zvec::IndexFilter *filter) {
  cached_doc_id_ = conjunction_->next_doc(filter);
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
  if (cached_doc_id_ == cached_matches_doc_id_) {
    return cached_matches_result_;
  }
  // Phase 2: verify position adjacency (deferred IO)
  cached_matches_doc_id_ = cached_doc_id_;
  cached_matches_result_ = verify_phrase_positions(cached_doc_id_);
  return cached_matches_result_;
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
  const size_t n = terms_.size();
  if (n == 0) {
    return false;
  }

  // Deduplicate terms within the phrase. Repeated terms (e.g., "to be or not
  // to be") collapse into one $POS lookup; term_to_unique_idx maps each phrase
  // position back to its slot in the unique list.
  std::vector<size_t> term_to_unique_idx(n);
  std::vector<size_t> unique_to_first_term_idx;
  unique_to_first_term_idx.reserve(n);
  std::unordered_map<std::string, size_t> seen;
  seen.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const size_t next_idx = unique_to_first_term_idx.size();
    auto [it, inserted] = seen.try_emplace(terms_[i], next_idx);
    if (inserted) {
      unique_to_first_term_idx.push_back(i);
    }
    term_to_unique_idx[i] = it->second;
  }
  const size_t unique_size = unique_to_first_term_idx.size();

  // Build unique (term, doc_id) keys into a single reusable buffer; reserve
  // up-front so the buffer never reallocates and the Slice pointers below stay
  // valid until the MultiGet returns.
  size_t total_key_bytes = 0;
  for (size_t u = 0; u < unique_size; ++u) {
    total_key_bytes +=
        terms_[unique_to_first_term_idx[u]].size() + 1 + sizeof(uint32_t);
  }
  std::string key_buffer;
  key_buffer.reserve(total_key_bytes);

  std::vector<rocksdb::Slice> key_slices;
  key_slices.reserve(unique_size);
  for (size_t u = 0; u < unique_size; ++u) {
    const std::string &term = terms_[unique_to_first_term_idx[u]];
    const size_t offset = key_buffer.size();
    const size_t bytes = fts::append_doc_term_key(term, doc_id, &key_buffer);
    key_slices.emplace_back(key_buffer.data() + offset, bytes);
  }

  // Batched read across unique (term, doc_id) keys — single MultiGet instead
  // of per-anchor-position Gets.
  std::vector<rocksdb::ColumnFamilyHandle *> cfs(unique_size, positions_cf_);
  std::vector<rocksdb::PinnableSlice> values(unique_size);
  std::vector<rocksdb::Status> statuses(unique_size);
  ctx_->db_->MultiGet(ctx_->read_opts_, unique_size, cfs.data(),
                      key_slices.data(), values.data(), statuses.data());

  // Decode every position list once. A missing entry means this doc cannot
  // be a phrase match — this happens for docs filtered through the conjunction
  // without a position-CF entry, so we do NOT log here.
  std::vector<std::vector<uint32_t>> positions_cache(unique_size);
  for (size_t u = 0; u < unique_size; ++u) {
    if (!statuses[u].ok() || values[u].size() == 0) {
      return false;
    }
    positions_cache[u] = decode_positions(values[u]);
    if (positions_cache[u].empty()) {
      return false;
    }
  }

  // Pick the term with the shortest position list as anchor so the outer
  // loop iterates as few candidates as possible. anchor_term_idx stays in
  // original phrase order — the phrase start equals anchor_pos -
  // anchor_term_idx.
  size_t anchor_term_idx = 0;
  size_t min_size = positions_cache[term_to_unique_idx[0]].size();
  for (size_t i = 1; i < n; ++i) {
    const size_t sz = positions_cache[term_to_unique_idx[i]].size();
    if (sz < min_size) {
      min_size = sz;
      anchor_term_idx = i;
    }
  }

  const auto &anchor_positions =
      positions_cache[term_to_unique_idx[anchor_term_idx]];
  const uint32_t anchor_offset = static_cast<uint32_t>(anchor_term_idx);
  for (uint32_t anchor_pos : anchor_positions) {
    if (anchor_pos < anchor_offset) {
      // phrase start would be negative — impossible
      continue;
    }
    const uint32_t start = anchor_pos - anchor_offset;
    bool phrase_matched = true;
    for (size_t i = 0; i < n; ++i) {
      if (i == anchor_term_idx) {
        continue;
      }
      const uint32_t expected = start + static_cast<uint32_t>(i);
      const auto &positions = positions_cache[term_to_unique_idx[i]];
      if (!std::binary_search(positions.begin(), positions.end(), expected)) {
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

std::vector<uint32_t> PhraseDocIterator::decode_positions(
    const rocksdb::Slice &data) {
  std::vector<uint32_t> positions;
  size_t index = 0;
  uint32_t current_position = 0;
  const char *bytes = data.data();
  const size_t size = data.size();

  while (index < size) {
    // Decode varint
    uint32_t delta = 0;
    uint32_t shift = 0;
    while (index < size) {
      const uint8_t byte = static_cast<uint8_t>(bytes[index++]);
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
