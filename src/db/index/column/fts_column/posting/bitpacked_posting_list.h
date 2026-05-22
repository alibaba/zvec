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
#include <cstring>
#include <string>
#include "bitpacked_simd_dispatch.h"
#include "../bm25_scorer.h"

namespace zvec::fts {

// ============================================================
// BitPacked Posting List encoder
// ============================================================

class BitPackedPostingList {
 public:
  static constexpr uint32_t DOCS_PER_BLOCK = 128;
  static constexpr uint32_t MAGIC = 0x42504B44;  // "BPKD"
  static constexpr uint32_t VERSION = 1;

  /// Skip-list entry stored after the file header.
  struct BlockMeta {
    uint32_t max_doc_id;    ///< Last (largest) doc_id in this block
    uint32_t block_offset;  ///< Byte offset from data start to block header
    float block_max_score;  ///< BM25 score upper bound for this block
  };

  /// File header (16 bytes).
  struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t num_docs;
    uint32_t num_blocks;
  };
  static constexpr size_t HEADER_SIZE = sizeof(Header);

  /// Block header (16 bytes, padded for SIMD alignment).
  struct BlockHeader {
    uint32_t min_doc_id;
    uint8_t bitwidth_id;
    uint8_t bitwidth_tf;
    uint8_t bitwidth_dl;
    uint8_t num_docs;       ///< Number of docs in this block (<=128)
    float block_max_score;  ///< Redundant copy for fast in-block access
    uint32_t padding_{
        0};  ///< Padding to make BlockHeader 16 bytes (SIMD alignment)
  };

  /// Encode a posting list with inline payloads.
  /// \param doc_ids   Sorted ascending doc_id array
  /// \param tfs       Term frequency for each doc
  /// \param doc_lens  Document length for each doc
  /// \param count     Number of entries
  /// \param df        Document frequency (used for IDF in block_max_score)
  /// \param scorer    BM25 scorer with segment stats loaded
  /// \return Serialized bitpacked posting list
  static std::string encode(const uint32_t *doc_ids, const uint32_t *tfs,
                            const uint32_t *doc_lens, size_t count, uint64_t df,
                            const BM25Scorer &scorer);

  /// Check if raw data starts with the BitPacked magic number.
  static bool is_bitpacked_format(const char *data, size_t size) {
    if (size < sizeof(uint32_t)) return false;
    uint32_t magic = 0;
    std::memcpy(&magic, data, sizeof(uint32_t));
    return magic == MAGIC;
  }

  // ---- Low-level bitpacking primitives ----

  /// Pack \p count uint32 values (each using \p bitwidth bits) into \p out.
  /// \p out must have at least ceil(bitwidth * count / 8) bytes.
  /// \p count must be <= DOCS_PER_BLOCK (128).
  static void pack_uint32(const uint32_t *in, uint8_t bitwidth, uint32_t count,
                          uint8_t *out);

  /// Unpack \p count uint32 values (each using \p bitwidth bits) from \p in.
  /// \p out must have room for \p count uint32_t values.
  static void unpack_uint32(const uint8_t *in, uint8_t bitwidth, uint32_t count,
                            uint32_t *out);

  /// Compute the minimum number of bits needed to represent \p max_value.
  /// Returns 0 if max_value == 0.
  static uint8_t bits_needed(uint32_t max_value);

  /// Compute packed byte size for \p count values at \p bitwidth bits each
  /// (scalar format, used for tail blocks with count < DOCS_PER_BLOCK).
  static size_t packed_byte_size(uint8_t bitwidth, uint32_t count) {
    return (static_cast<size_t>(bitwidth) * count + 7) / 8;
  }

  /// Compute packed byte size for a full SIMD block (128 values).
  /// SIMD format stores bitwidth __m128i values = bitwidth * 16 bytes.
  static size_t simd_packed_byte_size(uint8_t bitwidth) {
    return static_cast<size_t>(bitwidth) * 16;
  }
};

// ============================================================
// BitPacked Posting Iterator (zero-copy, block-at-a-time)
// ============================================================

/// Zero-copy iterator over a serialized BitPacked posting list.
/// Decodes one block at a time into stack-allocated arrays.
class BitPackedPostingIterator {
 public:
  static constexpr uint32_t NO_MORE_DOCS = UINT32_MAX;

  BitPackedPostingIterator() = default;

  /// Open from serialized data (zero-copy, does not own the data).
  /// \param data  Pointer to serialized bitpacked posting list
  /// \param size  Size of the serialized data in bytes
  /// \return 0 on success, -1 on error (bad magic, truncated data, etc.)
  int open(const char *data, size_t size);

  /// Advance to the next document.
  /// \return doc_id of the next document, or NO_MORE_DOCS if exhausted.
  uint32_t next_doc();

  /// Advance to the first document with doc_id >= target.
  /// Uses the skip list for O(log N_blocks) block-level seeking.
  /// \return doc_id >= target, or NO_MORE_DOCS if exhausted.
  uint32_t advance(uint32_t target);

  /// Current document ID (valid after next_doc/advance).
  uint32_t doc_id() const {
    return current_doc_id_;
  }

  /// Term frequency of the current document (valid after next_doc/advance).
  /// NOTE: non-const because lazy decode may be triggered on first access.
  uint32_t term_freq();

  /// Document length of the current document (valid after next_doc/advance).
  /// NOTE: non-const because lazy decode may be triggered on first access.
  uint32_t doc_len();

  /// Return both block_max_score and max_doc_id for the block containing
  /// \p target in a single binary search on the skip list.
  /// Does NOT move the iterator position.
  struct BlockMaxInfo {
    float block_max_score{0.0f};
    uint32_t block_last_doc{NO_MORE_DOCS};
  };
  BlockMaxInfo block_max_info_for(uint32_t target) const;

  /// Total number of documents in this posting list.
  uint64_t cost() const {
    return num_docs_;
  }

  /// Maximum block_max_score across all blocks (global upper bound).
  float max_score() const {
    return global_max_score_;
  }

 private:
  /// Decode block at index \p block_idx into the stack arrays.
  void decode_block(size_t block_idx);

  /// Lazy decode: ensure tf values are decoded before access.
  void ensure_tf_decoded();

  /// Lazy decode: ensure doc_len values are decoded before access.
  void ensure_dl_decoded();

  /// SIMD search: find first index i in block_doc_ids_[start..size)
  /// where doc_id >= target. Uses SSE4.1 for 4-wide comparison.
  size_t simd_find_first_ge(uint32_t target, size_t start) const;

  // File header fields
  uint32_t num_docs_{0};
  uint32_t num_blocks_{0};

  // Skip list (pointer into data_, not owned)
  const BitPackedPostingList::BlockMeta *skip_list_{nullptr};

  // Raw data pointer (not owned)
  const char *data_{nullptr};
  size_t data_size_{0};

  // Current block state (decoded into stack arrays)
  alignas(16) uint32_t block_doc_ids_[BitPackedPostingList::DOCS_PER_BLOCK];
  alignas(16) uint32_t block_tfs_[BitPackedPostingList::DOCS_PER_BLOCK];
  alignas(16) uint32_t block_doc_lens_[BitPackedPostingList::DOCS_PER_BLOCK];
  size_t current_block_idx_{0};
  uint32_t current_block_size_{0};
  size_t in_block_pos_{0};     ///< Position within current decoded block
  bool block_decoded_{false};  ///< Whether current block is decoded

  // Lazy decode state: tf and doc_len are decoded on first access
  bool tf_decoded_{false};
  bool dl_decoded_{false};

  // Store packed data pointers for lazy decode
  const uint8_t *packed_tf_ptr_{nullptr};
  const uint8_t *packed_dl_ptr_{nullptr};
  uint8_t current_bitwidth_tf_{0};
  uint8_t current_bitwidth_dl_{0};
  uint32_t current_block_num_docs_{0};  ///< num_docs for lazy decode dispatch
  bool current_block_is_full_{false};   ///< Whether current block is full (128)

  uint32_t current_doc_id_{NO_MORE_DOCS};
  float global_max_score_{0.0f};

  // Cached SIMD dispatch function pointers (initialized in open()).
  // Avoids repeated PLT/indirect calls through get_dispatch() on every
  // decode_block / simd_find_first_ge invocation.
  simd::PrefixSumFunc prefix_sum_fn_{nullptr};
  simd::FindFirstGeFunc find_first_ge_fn_{nullptr};
  simd::UnpackFunc unpack_fn_{nullptr};

  // Cache for block_max_info_for to avoid repeated binary searches.
  // If target falls within [cached_bmi_block_min_doc_+1, cached_bmi_last_doc_],
  // we can return the cached result directly.
  mutable uint32_t cached_bmi_last_doc_{0};
  mutable float cached_bmi_score_{0.0f};
  mutable size_t cached_bmi_block_idx_{0};
  mutable bool cached_bmi_valid_{false};
};

}  // namespace zvec::fts
