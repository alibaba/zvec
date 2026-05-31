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

#include "bitpacked_posting_list.h"
#include <bitpackinghelpers.h>
#include <cstdlib>
#include <memory>
#include <zvec/ailego/logger/logger.h>
#include "bitpacked_simd_dispatch.h"

#ifdef _MSC_VER
#include <intrin.h>
#include <malloc.h>
#endif

namespace zvec::fts {

// ============================================================
// BitPacked Posting List on-disk format
// ============================================================
//
// Encodes doc_id deltas, term frequencies, and document lengths using
// per-block bitpacking.  Each block stores up to 128 entries and carries
// a precomputed BM25 score upper bound to support Block-Max WAND pruning.
//
// File layout:
//   [Header 16B] [SkipList N*12B] [Block0] [Block1] ...
//
// Block layout:
//   [BlockHeader 16B] [packed_deltas] [packed_tfs] [packed_dlens]

namespace {

/// Round up \p value to the next multiple of \p alignment.
constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

/// Allocate 16-byte-aligned memory for \p count uint32_t values, returned as
/// a unique_ptr with a custom deleter.
inline auto make_aligned_uint32_array(size_t count) {
  const size_t num_bytes = align_up(count * sizeof(uint32_t), 16);
#ifdef _MSC_VER
  auto *ptr = static_cast<uint32_t *>(_aligned_malloc(num_bytes, 16));
  return std::unique_ptr<uint32_t[], decltype(&_aligned_free)>(ptr,
                                                               _aligned_free);
#else
  auto *ptr = static_cast<uint32_t *>(std::aligned_alloc(16, num_bytes));
  return std::unique_ptr<uint32_t[], decltype(&std::free)>(ptr, std::free);
#endif
}

}  // namespace

// ============================================================
// Low-level bitpacking primitives
// ============================================================

uint8_t BitPackedPostingList::bits_needed(uint32_t max_value) {
  if (max_value == 0) return 0;
#ifdef _MSC_VER
  unsigned long index = 0;
  _BitScanReverse(&index, max_value);
  return static_cast<uint8_t>(index + 1);
#else
  return static_cast<uint8_t>(32 - __builtin_clz(max_value));
#endif
}

void BitPackedPostingList::pack_uint32(const uint32_t *in, uint8_t bitwidth,
                                       uint32_t count, uint8_t *out) {
  if (bitwidth == 0 || count == 0) return;

  // Full block path: 128 values at once via dispatch (SIMD or scalar)
  if (count == DOCS_PER_BLOCK) {
    simd::get_dispatch().pack_uint32_128(in, bitwidth, out);
    return;
  }

  // Tail block path (count < 128): use scalar fastpack, 32 at a time
  const size_t total_bytes = packed_byte_size(bitwidth, count);
  std::memset(out, 0, total_bytes);

  uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
  uint32_t offset = 0;

  while (offset + 32 <= count) {
    FastPForLib::fastpackwithoutmask(in + offset, out32, bitwidth);
    out32 += bitwidth;
    offset += 32;
  }

  // Tail: fewer than 32 integers
  if (offset < count) {
    alignas(16) uint32_t padded_in[32] = {};
    std::memcpy(padded_in, in + offset, (count - offset) * sizeof(uint32_t));
    alignas(16) uint32_t padded_out[32] = {};
    FastPForLib::fastpackwithoutmask(padded_in, padded_out, bitwidth);
    size_t tail_bytes = packed_byte_size(bitwidth, count - offset);
    std::memcpy(out32, padded_out, tail_bytes);
  }
}

void BitPackedPostingList::unpack_uint32(const uint8_t *in, uint8_t bitwidth,
                                         uint32_t count, uint32_t *out) {
  if (bitwidth == 0 || count == 0) {
    for (uint32_t i = 0; i < count; ++i) {
      out[i] = 0;
    }
    return;
  }

  // Full block path: 128 values at once via dispatch (SIMD or scalar)
  if (count == DOCS_PER_BLOCK) {
    simd::get_dispatch().unpack_uint32_128(in, bitwidth, out);
    return;
  }

  // Tail block path (count < 128): use scalar fastunpack, 32 at a time
  const uint32_t *in32 = reinterpret_cast<const uint32_t *>(in);
  uint32_t offset = 0;

  while (offset + 32 <= count) {
    FastPForLib::fastunpack(in32, out + offset, bitwidth);
    in32 += bitwidth;
    offset += 32;
  }

  // Tail: fewer than 32 integers
  if (offset < count) {
    const size_t tail_bytes = packed_byte_size(bitwidth, count - offset);
    alignas(16) uint32_t padded_in[32] = {};
    std::memcpy(padded_in, in32, tail_bytes);
    alignas(16) uint32_t padded_out[32] = {};
    FastPForLib::fastunpack(padded_in, padded_out, bitwidth);
    std::memcpy(out + offset, padded_out, (count - offset) * sizeof(uint32_t));
  }
}

// ============================================================
// Encoder
// ============================================================

std::string BitPackedPostingList::encode(const uint32_t *doc_ids,
                                         const uint32_t *tfs,
                                         const uint32_t *doc_lens, size_t count,
                                         uint64_t df,
                                         const BM25Scorer &scorer) {
  if (count == 0) {
    // Encode an empty posting list (just the header)
    Header hdr{};
    hdr.magic = MAGIC;
    hdr.version = VERSION;
    hdr.num_docs = 0;
    hdr.num_blocks = 0;
    std::string result(HEADER_SIZE, '\0');
    std::memcpy(result.data(), &hdr, HEADER_SIZE);
    return result;
  }

  const uint32_t num_blocks =
      static_cast<uint32_t>((count + DOCS_PER_BLOCK - 1) / DOCS_PER_BLOCK);

  // ---- Phase 1: Compute delta-encoded doc_ids ----
  // Use 16-byte-aligned allocation so SIMD pack/max paths can use aligned loads
  auto deltas = make_aligned_uint32_array(count);
  deltas[0] = doc_ids[0];
  for (size_t i = 1; i < count; ++i) {
    deltas[i] = doc_ids[i] - doc_ids[i - 1];
  }

  // ---- Phase 2: Compute per-block metadata and packed sizes ----
  struct BlockInfo {
    size_t start;        // index into the arrays
    uint32_t num_docs;   // number of docs in this block
    uint8_t bw_id;       // bitwidth for doc_id deltas
    uint8_t bw_tf;       // bitwidth for tfs
    uint8_t bw_dl;       // bitwidth for doc_lens
    float max_score;     // block max BM25 score
    size_t packed_size;  // total packed data size for this block
  };

  std::vector<BlockInfo> blocks(num_blocks);

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const size_t start = static_cast<size_t>(b) * DOCS_PER_BLOCK;
    const uint32_t num_docs = static_cast<uint32_t>(
        std::min(static_cast<size_t>(DOCS_PER_BLOCK), count - start));

    // Find max values in block for bitwidth computation
    uint32_t max_delta = 0, max_tf = 0, max_dl = 0;
    float block_max = 0.0f;

    if (num_docs == DOCS_PER_BLOCK) {
      // Dispatch max for full blocks (SSE4.1 or scalar fallback)
      simd::get_dispatch().max_128(deltas.get(), tfs, doc_lens, start,
                                   DOCS_PER_BLOCK, max_delta, max_tf, max_dl);
      // block_max_score still needs scalar loop (float BM25 scoring)
      for (uint32_t i = 0; i < DOCS_PER_BLOCK; ++i) {
        float s = scorer.score(df, tfs[start + i], doc_lens[start + i]);
        block_max = std::max(block_max, s);
      }
    } else {
      // Scalar path for tail blocks
      for (uint32_t i = 0; i < num_docs; ++i) {
        max_delta = std::max(max_delta, deltas[start + i]);
        max_tf = std::max(max_tf, tfs[start + i]);
        max_dl = std::max(max_dl, doc_lens[start + i]);
        float s = scorer.score(df, tfs[start + i], doc_lens[start + i]);
        block_max = std::max(block_max, s);
      }
    }

    blocks[b].start = start;
    blocks[b].num_docs = num_docs;
    blocks[b].bw_id = bits_needed(max_delta);
    blocks[b].bw_tf = bits_needed(max_tf);
    blocks[b].bw_dl = bits_needed(max_dl);
    blocks[b].max_score = block_max;
    // Full block (128 values): use SIMD packed size; tail block: use scalar
    if (num_docs == DOCS_PER_BLOCK) {
      blocks[b].packed_size = simd_packed_byte_size(blocks[b].bw_id) +
                              simd_packed_byte_size(blocks[b].bw_tf) +
                              simd_packed_byte_size(blocks[b].bw_dl);
    } else {
      blocks[b].packed_size = packed_byte_size(blocks[b].bw_id, num_docs) +
                              packed_byte_size(blocks[b].bw_tf, num_docs) +
                              packed_byte_size(blocks[b].bw_dl, num_docs);
    }
  }

  // ---- Phase 3: Compute total size and block offsets ----
  const size_t skip_list_size = num_blocks * sizeof(BlockMeta);
  const size_t block_header_size = sizeof(BlockHeader);

  // Compute block offsets, aligning each block start to a 16-byte boundary
  // so that SIMD decode paths can use aligned loads on the packed data.
  size_t current_offset = align_up(HEADER_SIZE + skip_list_size, 16);
  std::vector<uint32_t> block_offsets(num_blocks);
  for (uint32_t b = 0; b < num_blocks; ++b) {
    block_offsets[b] = static_cast<uint32_t>(current_offset);
    current_offset = align_up(
        current_offset + block_header_size + blocks[b].packed_size, 16);
  }

  const size_t total_size = current_offset;

  // ---- Phase 4: Serialize ----
  std::string result(total_size, '\0');
  char *buf = result.data();

  // File Header
  Header hdr{};
  hdr.magic = MAGIC;
  hdr.version = VERSION;
  hdr.num_docs = static_cast<uint32_t>(count);
  hdr.num_blocks = num_blocks;
  std::memcpy(buf, &hdr, HEADER_SIZE);

  // Skip List
  BlockMeta *skip = reinterpret_cast<BlockMeta *>(buf + HEADER_SIZE);
  for (uint32_t b = 0; b < num_blocks; ++b) {
    const size_t last_idx = blocks[b].start + blocks[b].num_docs - 1;
    skip[b].max_doc_id = doc_ids[last_idx];
    skip[b].block_offset = block_offsets[b];
    skip[b].block_max_score = blocks[b].max_score;
  }

  // Blocks
  for (uint32_t b = 0; b < num_blocks; ++b) {
    char *block_ptr = buf + block_offsets[b];

    // Block Header
    BlockHeader bhdr{};
    bhdr.min_doc_id = doc_ids[blocks[b].start];
    bhdr.bitwidth_id = blocks[b].bw_id;
    bhdr.bitwidth_tf = blocks[b].bw_tf;
    bhdr.bitwidth_dl = blocks[b].bw_dl;
    bhdr.num_docs = static_cast<uint8_t>(blocks[b].num_docs);
    bhdr.block_max_score = blocks[b].max_score;
    std::memcpy(block_ptr, &bhdr, sizeof(BlockHeader));

    uint8_t *packed_ptr =
        reinterpret_cast<uint8_t *>(block_ptr + sizeof(BlockHeader));

    const bool is_full_block = (blocks[b].num_docs == DOCS_PER_BLOCK);

    // Pack doc_id deltas
    const size_t id_bytes =
        is_full_block ? simd_packed_byte_size(blocks[b].bw_id)
                      : packed_byte_size(blocks[b].bw_id, blocks[b].num_docs);
    pack_uint32(&deltas[blocks[b].start], blocks[b].bw_id, blocks[b].num_docs,
                packed_ptr);
    packed_ptr += id_bytes;

    // Pack term frequencies
    const size_t tf_bytes =
        is_full_block ? simd_packed_byte_size(blocks[b].bw_tf)
                      : packed_byte_size(blocks[b].bw_tf, blocks[b].num_docs);
    pack_uint32(&tfs[blocks[b].start], blocks[b].bw_tf, blocks[b].num_docs,
                packed_ptr);
    packed_ptr += tf_bytes;

    // Pack document lengths
    pack_uint32(&doc_lens[blocks[b].start], blocks[b].bw_dl, blocks[b].num_docs,
                packed_ptr);
  }

  return result;
}

// ============================================================
// Iterator
// ============================================================

int BitPackedPostingIterator::open(const char *data, size_t size) {
  if (!data || size < BitPackedPostingList::HEADER_SIZE) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: truncated data, "
        "size[%zu] expected_min[%zu]",
        size, BitPackedPostingList::HEADER_SIZE);
    return -1;
  }

  // Parse file header
  BitPackedPostingList::Header hdr{};
  std::memcpy(&hdr, data, sizeof(hdr));

  if (hdr.magic != BitPackedPostingList::MAGIC) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: bad magic, "
        "got[0x%x] expected[0x%x]",
        hdr.magic, BitPackedPostingList::MAGIC);
    return -1;
  }
  if (hdr.version != BitPackedPostingList::VERSION) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: unsupported version, "
        "got[%u] expected[%u]",
        hdr.version, BitPackedPostingList::VERSION);
    return -1;
  }

  num_docs_ = hdr.num_docs;
  num_blocks_ = hdr.num_blocks;
  data_ = data;
  data_size_ = size;

  if (num_docs_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return 0;
  }

  // Validate skip list fits
  const size_t skip_list_offset = BitPackedPostingList::HEADER_SIZE;
  const size_t skip_list_size =
      num_blocks_ * sizeof(BitPackedPostingList::BlockMeta);
  if (skip_list_offset + skip_list_size > size) {
    LOG_ERROR(
        "BitPackedPostingIterator open failed: skip list overruns buffer, "
        "num_blocks[%u] data_size[%zu] need[%zu]",
        num_blocks_, size, skip_list_offset + skip_list_size);
    return -1;
  }

  skip_list_ = reinterpret_cast<const BitPackedPostingList::BlockMeta *>(
      data + skip_list_offset);

  // Compute global max score
  global_max_score_ = 0.0f;
  for (uint32_t b = 0; b < num_blocks_; ++b) {
    global_max_score_ =
        std::max(global_max_score_, skip_list_[b].block_max_score);
  }

  // Initialize to before-first-block state
  current_block_idx_ = 0;
  in_block_pos_ = 0;
  current_block_size_ = 0;
  block_decoded_ = false;
  current_doc_id_ = NO_MORE_DOCS;

  // Cache SIMD dispatch function pointers to avoid PLT overhead on hot path
  const auto &dispatch = simd::get_dispatch();
  prefix_sum_fn_ = dispatch.prefix_sum_128;
  find_first_ge_fn_ = dispatch.find_first_ge;
  unpack_fn_ = dispatch.unpack_uint32_128;

  return 0;
}

void BitPackedPostingIterator::decode_block(size_t block_idx) {
  if (block_idx >= num_blocks_) {
    LOG_WARN(
        "BitPackedPostingIterator decode_block out of range: "
        "block_idx[%zu] num_blocks[%u]",
        block_idx, num_blocks_);
    current_block_size_ = 0;
    block_decoded_ = false;
    return;
  }

  const auto &meta = skip_list_[block_idx];
  const char *block_ptr = data_ + meta.block_offset;

  // Parse block header
  BitPackedPostingList::BlockHeader bhdr{};
  std::memcpy(&bhdr, block_ptr, sizeof(bhdr));

  current_block_size_ = bhdr.num_docs;
  current_block_idx_ = block_idx;
  in_block_pos_ = 0;

  const uint8_t *packed_ptr =
      reinterpret_cast<const uint8_t *>(block_ptr + sizeof(bhdr));

  const bool is_full_block =
      (bhdr.num_docs == BitPackedPostingList::DOCS_PER_BLOCK);

  // Unpack doc_id deltas
  const size_t id_bytes =
      is_full_block
          ? BitPackedPostingList::simd_packed_byte_size(bhdr.bitwidth_id)
          : BitPackedPostingList::packed_byte_size(bhdr.bitwidth_id,
                                                   bhdr.num_docs);
  alignas(16) uint32_t deltas[BitPackedPostingList::DOCS_PER_BLOCK];
  if (is_full_block) {
    // Fast path: use cached function pointer directly for full blocks
    unpack_fn_(packed_ptr, bhdr.bitwidth_id, deltas);
  } else {
    BitPackedPostingList::unpack_uint32(packed_ptr, bhdr.bitwidth_id,
                                        bhdr.num_docs, deltas);
  }
  packed_ptr += id_bytes;

  // Reconstruct absolute doc_ids from deltas using prefix-sum
  if (is_full_block) {
    prefix_sum_fn_(deltas, bhdr.min_doc_id,
                   BitPackedPostingList::DOCS_PER_BLOCK, block_doc_ids_);
  } else {
    // Scalar prefix-sum for tail block
    block_doc_ids_[0] = bhdr.min_doc_id;
    for (uint32_t i = 1; i < bhdr.num_docs; ++i) {
      block_doc_ids_[i] = block_doc_ids_[i - 1] + deltas[i];
    }
  }

  // Lazy decode: record packed data pointers and bitwidths for tf/doc_len.
  // Actual decoding is deferred until term_freq() or doc_len() is called.
  const size_t tf_bytes =
      is_full_block
          ? BitPackedPostingList::simd_packed_byte_size(bhdr.bitwidth_tf)
          : BitPackedPostingList::packed_byte_size(bhdr.bitwidth_tf,
                                                   bhdr.num_docs);
  packed_tf_ptr_ = packed_ptr;
  current_bitwidth_tf_ = bhdr.bitwidth_tf;
  packed_ptr += tf_bytes;

  packed_dl_ptr_ = packed_ptr;
  current_bitwidth_dl_ = bhdr.bitwidth_dl;

  current_block_num_docs_ = bhdr.num_docs;
  current_block_is_full_ = is_full_block;

  // Reset lazy decode flags
  tf_decoded_ = false;
  dl_decoded_ = false;

  block_decoded_ = true;
}

uint32_t BitPackedPostingIterator::next_doc() {
  if (num_docs_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // If no block is decoded yet, decode the first block
  if (!block_decoded_) {
    decode_block(0);
    if (current_block_size_ == 0) {
      current_doc_id_ = NO_MORE_DOCS;
      return NO_MORE_DOCS;
    }
    current_doc_id_ = block_doc_ids_[0];
    in_block_pos_ = 0;
    return current_doc_id_;
  }

  // Advance within current block
  ++in_block_pos_;
  if (in_block_pos_ < current_block_size_) {
    current_doc_id_ = block_doc_ids_[in_block_pos_];
    return current_doc_id_;
  }

  // Move to next block
  size_t next_block = current_block_idx_ + 1;
  if (next_block >= num_blocks_) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  decode_block(next_block);
  if (current_block_size_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  current_doc_id_ = block_doc_ids_[0];
  in_block_pos_ = 0;
  return current_doc_id_;
}

size_t BitPackedPostingIterator::simd_find_first_ge(uint32_t target,
                                                    size_t start) const {
  return find_first_ge_fn_(block_doc_ids_, current_block_size_, target, start);
}

uint32_t BitPackedPostingIterator::advance(uint32_t target) {
  if (num_docs_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // If current doc_id already >= target, return it
  if (current_doc_id_ != NO_MORE_DOCS && current_doc_id_ >= target) {
    return current_doc_id_;
  }

  // Use skip list to find the target block via binary search.
  // Find the first block whose max_doc_id >= target.
  size_t lo = 0, hi = num_blocks_;

  // If we have a current block and its max_doc_id >= target,
  // we can search within the current block first.
  if (block_decoded_ && current_block_idx_ < num_blocks_ &&
      skip_list_[current_block_idx_].max_doc_id >= target) {
    // Target might be in current block - SIMD scan from current position
    {
      size_t pos = simd_find_first_ge(target, in_block_pos_);
      if (pos < current_block_size_) {
        in_block_pos_ = pos;
        current_doc_id_ = block_doc_ids_[pos];
        return current_doc_id_;
      }
    }
    // Not found in current block (shouldn't happen if skip list is correct)
    lo = current_block_idx_ + 1;
  } else if (block_decoded_) {
    // Current block's max_doc_id < target, start search from next block
    lo = current_block_idx_ + 1;
  }

  // Binary search in skip list for the first block with max_doc_id >= target
  size_t target_block = hi;  // sentinel: no block found
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (skip_list_[mid].max_doc_id >= target) {
      target_block = mid;
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }

  if (target_block >= num_blocks_) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // Decode the target block
  decode_block(target_block);
  if (current_block_size_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // SIMD scan within the block for the first doc_id >= target
  {
    size_t pos = simd_find_first_ge(target, 0);
    if (pos < current_block_size_) {
      in_block_pos_ = pos;
      current_doc_id_ = block_doc_ids_[pos];
      return current_doc_id_;
    }
  }

  // All docs in this block are < target (shouldn't happen with correct skip
  // list), try next block
  size_t next = target_block + 1;
  if (next >= num_blocks_) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  decode_block(next);
  if (current_block_size_ == 0) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  {
    size_t pos = simd_find_first_ge(target, 0);
    if (pos < current_block_size_) {
      in_block_pos_ = pos;
      current_doc_id_ = block_doc_ids_[pos];
      return current_doc_id_;
    }
  }
  current_doc_id_ = NO_MORE_DOCS;
  return NO_MORE_DOCS;
}

void BitPackedPostingIterator::ensure_tf_decoded() {
  if (tf_decoded_) {
    return;
  }
  if (current_block_is_full_) {
    unpack_fn_(packed_tf_ptr_, current_bitwidth_tf_, block_tfs_);
  } else {
    BitPackedPostingList::unpack_uint32(packed_tf_ptr_, current_bitwidth_tf_,
                                        current_block_num_docs_, block_tfs_);
  }
  tf_decoded_ = true;
}

void BitPackedPostingIterator::ensure_dl_decoded() {
  if (dl_decoded_) {
    return;
  }
  if (current_block_is_full_) {
    unpack_fn_(packed_dl_ptr_, current_bitwidth_dl_, block_doc_lens_);
  } else {
    BitPackedPostingList::unpack_uint32(packed_dl_ptr_, current_bitwidth_dl_,
                                        current_block_num_docs_,
                                        block_doc_lens_);
  }
  dl_decoded_ = true;
}

uint32_t BitPackedPostingIterator::term_freq() {
  if (!block_decoded_ || in_block_pos_ >= current_block_size_) {
    return 0;
  }
  ensure_tf_decoded();
  return block_tfs_[in_block_pos_];
}

uint32_t BitPackedPostingIterator::doc_len() {
  if (!block_decoded_ || in_block_pos_ >= current_block_size_) {
    return 1;
  }
  ensure_dl_decoded();
  return block_doc_lens_[in_block_pos_];
}

BitPackedPostingIterator::BlockMaxInfo
BitPackedPostingIterator::block_max_info_for(uint32_t target) const {
  if (num_blocks_ == 0 || skip_list_ == nullptr) {
    return {0.0f, NO_MORE_DOCS};
  }

  // Fast path: check if target falls within the previously cached block
  if (cached_bmi_valid_ && target <= cached_bmi_last_doc_) {
    // target is in the same or earlier block as last query.
    // Check if it's still in the same block (block_idx is correct).
    if (cached_bmi_block_idx_ == 0 ||
        target > skip_list_[cached_bmi_block_idx_ - 1].max_doc_id) {
      return {cached_bmi_score_, cached_bmi_last_doc_};
    }
  }

  size_t lo = 0, hi = num_blocks_;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (skip_list_[mid].max_doc_id >= target) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  if (lo >= num_blocks_) {
    return {0.0f, NO_MORE_DOCS};
  }

  // Update cache
  cached_bmi_block_idx_ = lo;
  cached_bmi_last_doc_ = skip_list_[lo].max_doc_id;
  cached_bmi_score_ = skip_list_[lo].block_max_score;
  cached_bmi_valid_ = true;

  return {cached_bmi_score_, cached_bmi_last_doc_};
}

}  // namespace zvec::fts
