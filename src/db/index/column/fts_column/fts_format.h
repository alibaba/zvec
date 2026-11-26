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

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <zvec/db/status.h>

namespace zvec::fts::format {

using Digest = std::array<uint8_t, 32>;

// Wire primitives deliberately do not serialize host structs. Decode failures
// are sticky. A caller must check done() after decoding an entire record.
void AppendU32(uint32_t value, std::string *out);
void AppendU64(uint64_t value, std::string *out);
void AppendVarint(uint64_t value, std::string *out);
bool ValidRange(uint64_t offset, uint64_t length, uint64_t size);

class Decoder {
 public:
  explicit Decoder(std::string_view bytes) : bytes_(bytes) {}
  uint32_t u32();
  uint64_t u64();
  uint64_t varint();
  std::string_view take(uint64_t length);
  bool ok() const {
    return ok_;
  }
  bool done() const {
    return ok_ && bytes_.empty();
  }
  size_t remaining() const {
    return bytes_.size();
  }

 private:
  std::string_view bytes_;
  bool ok_{true};
};

// FIPS 180-4 SHA-256, shared by dictionary fingerprints and range integrity.
// Fixed workspace; digest() is non-destructive. No platform crypto dependency.
class Sha256 {
 public:
  Status update(std::string_view bytes);
  Digest digest() const;

 private:
  void transform(const uint8_t *block);

  std::array<uint32_t, 8> state_{{0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                  0xa54ff53a, 0x510e527f, 0x9b05688c,
                                  0x1f83d9ab, 0x5be0cd19}};
  std::array<uint8_t, 64> block_{};
  uint64_t bytes_{0};
  size_t used_{0};
};

// Canonical bytes: "ZVEC_FTS_DICTIONARY\0", u32_le(order=1),
// u32_le(encoding=1), each u64_le(length)+term bytes, u64_le(term_count).
// Caller supplies terms in unsigned-byte order. Parts may be arbitrarily
// small, including for oversized terms; no term bytes are retained here.
class DictionaryFingerprint {
 public:
  DictionaryFingerprint();
  Status begin_term(uint64_t length);
  Status append_term_bytes(std::string_view bytes);
  Result<Digest> finish() const;
  uint64_t term_count() const {
    return terms_;
  }

 private:
  Sha256 hash_;
  uint64_t terms_{0};
  uint64_t remaining_{0};
};

using Identity = std::array<uint8_t, 16>;

struct FieldMetadata {
  Identity id{};
  Identity incarnation{};
  std::string name;
  uint64_t present_count{0};
  uint64_t scoring_document_count{0};
  uint64_t total_tokens{0};
  uint64_t dictionary_generation{0};
  uint64_t term_count{0};
  Digest dictionary_fingerprint{};
  // Persisted tokenizer/filter configuration identity; never re-tokenized.
  std::string analysis_identity;
  Digest scoring_identity{};
  Digest bound_statistics_identity{};
  uint32_t bound_version{1};
};

struct SectionMetadata {
  std::string name;
  uint64_t length{0};
  uint64_t first_checksum{0};
};

// Container-local metadata, NOT a collection manifest or commit marker.
// v1 fixes parent-segment-local DocIDs, legacy_compatible statistics, unsigned
// byte term ordering, SHA-256 and codec version 1. These identities are encoded
// explicitly and rejected when unsupported rather than silently defaulted.
struct Metadata {
  uint32_t minor_version{0};
  uint64_t required_features{0};
  uint64_t optional_features{0};
  Identity base_id{};
  uint64_t base_generation{0};
  uint32_t block_id{0};
  uint64_t source_segment_id{0};
  uint64_t row_count{0};
  uint32_t layout_granule{4096};
  std::vector<FieldMetadata> fields;
  std::vector<SectionMetadata> sections;
};

constexpr size_t kMetadataBudget = 1024 * 1024;
constexpr size_t kMaxFields = 256;
constexpr size_t kMaxSections = 4096;

// Metadata carries its own SHA-256. Unknown optional records are skipped by
// length; unknown required records/features/codec versions fail closed.
Result<std::string> EncodeMetadata(const Metadata &metadata,
                                   size_t budget = kMetadataBudget);
Result<Metadata> DecodeMetadata(std::string_view bytes,
                                size_t budget = kMetadataBudget);

constexpr size_t kBlockDocs = 128;

enum class PostingEncoding : uint8_t { kVarint = 1, kPacked = 2 };

struct PostingBlock {
  uint32_t count{0};
  std::array<uint32_t, kBlockDocs> docs{};
  std::array<uint32_t, kBlockDocs> tfs{};
};

struct PositionBlock {
  uint32_t count{0};
  std::array<uint32_t, 128> values{};
};

struct DocLengthBlock {
  uint32_t count{0};
  std::array<uint32_t, kBlockDocs> lengths{};
  std::array<bool, kBlockDocs> present{};
};

struct DocLengthEntry {
  bool present{false};
  uint32_t length{0};
};

// Validates the complete block, then reads individual lengths without
// expanding its arrays. The caller retains the immutable encoded bytes.
class DocLengthBlockView {
 public:
  static Result<DocLengthBlockView> Open(std::string_view bytes);
  uint32_t count() const {
    return count_;
  }
  Result<DocLengthEntry> at(uint32_t index) const {
    if (index >= count_) {
      return tl::make_unexpected(
          Status::InvalidArgument("FTS doclen index is outside block"));
    }
    return DocLengthEntry{
        (static_cast<uint8_t>(presence_[index / 8]) & (1u << (index % 8))) != 0,
        LoadLength(lengths_.data() + index * 4)};
  }

 private:
  // The caller has validated all four bytes. Byte loads work for unaligned
  // wire arrays on either byte order; compilers can combine them on
  // little-endian CPUs.
  static uint32_t LoadLength(const char *bytes) {
    const auto *value = reinterpret_cast<const uint8_t *>(bytes);
    return uint32_t{value[0]} | (uint32_t{value[1]} << 8) |
           (uint32_t{value[2]} << 16) | (uint32_t{value[3]} << 24);
  }
  uint32_t count_{0};
  std::string_view presence_, lengths_;
};

// Fixed-size codec workspace; callers stream successive blocks. DF=1 is a
// TermMeta concern. These functions also handle the final tail of a long term.
Result<std::string> EncodePostings(const PostingBlock &block,
                                   PostingEncoding encoding);
Result<PostingBlock> DecodePostings(std::string_view bytes,
                                    PostingEncoding encoding);
Result<std::string> EncodePositions(const PositionBlock &block);
Result<PositionBlock> DecodePositions(std::string_view bytes);
Result<std::string> EncodeDocLengths(const DocLengthBlock &block);
Result<DocLengthBlock> DecodeDocLengths(std::string_view bytes);

struct ByteRange {
  uint64_t offset{0};
  uint64_t length{0};
};

struct PostingSkip {
  uint32_t last_doc{0};
  uint32_t count{0};
  float max_score{0};
  ByteRange postings;
  ByteRange position_index;
};

constexpr size_t kPostingSkipBytes = 48;
Result<std::string> EncodePostingSkip(const PostingSkip &skip);
Result<PostingSkip> DecodePostingSkip(std::string_view bytes);

struct PositionIndexBlock {
  uint32_t count{0};
  ByteRange positions;
  // Complete contiguous occupancy per document, including subblock headers.
  // Long-list directory references live in those document headers.
  std::array<uint64_t, kBlockDocs> lengths{};
};

// Explicit 64-bit checkpoint width, every 32 documents, addressing both the
// position bytes and the varint byte-length stream. Full decode validates all
// checkpoints at publication; lookup consumes at most 31 preceding lengths.
Result<std::string> EncodePositionIndex(const PositionIndexBlock &block);
Result<PositionIndexBlock> DecodePositionIndex(std::string_view bytes);
Result<ByteRange> LocatePositionBytes(std::string_view bytes, uint32_t doc);

struct DictionaryBlock {
  uint64_t first_ordinal{0};
  std::vector<std::string> terms;
  // Exactly one oversized term, stored in the terms section outside ordinary
  // blocks. This range is streamed for comparison and fingerprinting.
  std::optional<ByteRange> oversized;
};

constexpr size_t kDictionaryBlockBytes = 4096;
constexpr size_t kDictionaryBlockTerms = 64;
// The worst prefix expansion of an ordinary block is also explicitly bounded.
constexpr size_t kDictionaryDecodeBudget = 64 * 4096;
int CompareTerms(std::string_view left, std::string_view right);
Result<std::string> EncodeDictionary(const DictionaryBlock &block);
Result<DictionaryBlock> DecodeDictionary(
    std::string_view bytes, size_t budget = kDictionaryDecodeBudget);

enum class PostingTag : uint32_t { kInline = 1, kVarint = 2, kPacked = 3 };
enum class PositionTag : uint32_t { kInline = 1, kExternal = 2 };

struct TermMetadata {
  PostingTag posting_tag{PostingTag::kInline};
  PositionTag position_tag{PositionTag::kInline};
  uint64_t document_frequency{0};
  uint64_t total_frequency{0};
  float max_score{0};
  uint32_t inline_doc{0};
  uint32_t inline_tf{0};
  uint32_t inline_position{0};
  ByteRange postings;
  ByteRange skips;
  ByteRange positions;
  ByteRange position_index;
};

// This is a fixed 112-byte record; explicit tags distinguish inline values
// from external ranges. External ranges still need section-bound validation.
constexpr size_t kTermMetadataBytes = 112;
Result<std::string> EncodeTermMetadata(const TermMetadata &term);
Result<TermMetadata> DecodeTermMetadata(std::string_view bytes);

}  // namespace zvec::fts::format
