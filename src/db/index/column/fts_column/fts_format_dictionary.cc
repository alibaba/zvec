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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "fts_format.h"

namespace zvec::fts::format {
namespace {

Status Corrupt() {
  return Status::InternalError("Malformed FTS dictionary or term metadata");
}

Status BudgetExceeded() {
  return {StatusCode::RESOURCE_EXHAUSTED,
          "FTS dictionary block budget exceeded"};
}

bool ValidReference(const ByteRange &range) {
  return ValidRange(range.offset, range.length, UINT64_MAX);
}

bool ValidPostingSkip(const PostingSkip &skip) {
  return skip.count != 0 && skip.count <= kBlockDocs &&
         skip.last_doc != UINT32_MAX && std::isfinite(skip.max_score) &&
         skip.max_score >= 0 && skip.postings.length != 0 &&
         skip.position_index.length != 0 && ValidReference(skip.postings) &&
         ValidReference(skip.position_index);
}

bool EmptyReference(const ByteRange &range) {
  return range.offset == 0 && range.length == 0;
}

Status ValidateTerm(const TermMetadata &term) {
  if (term.posting_tag != PostingTag::kInline &&
      term.posting_tag != PostingTag::kVarint &&
      term.posting_tag != PostingTag::kPacked) {
    return Status::NotSupported("Unknown FTS posting tag");
  }
  if (term.position_tag != PositionTag::kInline &&
      term.position_tag != PositionTag::kExternal) {
    return Status::NotSupported("Unknown FTS position tag");
  }
  if (term.document_frequency == 0 || term.document_frequency > UINT32_MAX ||
      term.total_frequency < term.document_frequency ||
      term.total_frequency > term.document_frequency * UINT32_MAX ||
      !std::isfinite(term.max_score) || term.max_score < 0 ||
      !ValidReference(term.postings) || !ValidReference(term.skips) ||
      !ValidReference(term.positions) || !ValidReference(term.position_index)) {
    return Status::InvalidArgument("Invalid FTS term statistics or ranges");
  }
  if (term.posting_tag == PostingTag::kInline) {
    if (term.document_frequency != 1 || term.inline_doc == UINT32_MAX ||
        term.inline_tf == 0 || term.total_frequency != term.inline_tf ||
        !EmptyReference(term.postings) || !EmptyReference(term.skips)) {
      return Status::InvalidArgument("Invalid FTS inline posting");
    }
  } else {
    if (term.inline_doc != 0 || term.inline_tf != 0 ||
        term.postings.length == 0 ||
        (term.posting_tag == PostingTag::kVarint &&
         (term.document_frequency < 2 || term.document_frequency >= 128 ||
          !EmptyReference(term.skips))) ||
        (term.posting_tag == PostingTag::kPacked &&
         (term.document_frequency < 128 || term.skips.length == 0))) {
      return Status::InvalidArgument("Invalid FTS external posting");
    }
  }
  if (term.position_tag == PositionTag::kInline) {
    if (term.document_frequency != 1 || term.total_frequency != 1 ||
        !EmptyReference(term.positions) ||
        !EmptyReference(term.position_index)) {
      return Status::InvalidArgument("Invalid FTS inline position");
    }
  } else if (term.inline_position != 0 || term.positions.length == 0 ||
             (term.document_frequency > 1 && term.position_index.length == 0)) {
    return Status::InvalidArgument("Invalid FTS external positions");
  }
  return Status::OK();
}

}  // namespace

int CompareTerms(std::string_view left, std::string_view right) {
  const size_t length = std::min(left.size(), right.size());
  const int compared =
      length == 0 ? 0 : std::memcmp(left.data(), right.data(), length);
  if (compared != 0) {
    return compared;
  }
  return left.size() < right.size() ? -1 : (left.size() > right.size() ? 1 : 0);
}

Result<std::string> EncodeDictionary(const DictionaryBlock &block) {
  const size_t count = block.oversized ? 1 : block.terms.size();
  if (count == 0 || count > kDictionaryBlockTerms ||
      count > UINT64_MAX - block.first_ordinal ||
      (block.oversized && !block.terms.empty())) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS dictionary block"));
  }
  std::string out;
  AppendU64(block.first_ordinal, &out);
  AppendU32(static_cast<uint32_t>(count), &out);
  AppendU32(block.oversized ? 1 : 0, &out);
  if (block.oversized) {
    if (block.oversized->length == 0 || !ValidReference(*block.oversized)) {
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid oversized FTS term"));
    }
    AppendU64(block.oversized->offset, &out);
    AppendU64(block.oversized->length, &out);
    return out;
  }
  std::string_view previous;
  bool first = true;
  for (const auto &term : block.terms) {
    if (term.size() > kDictionaryBlockBytes) {
      return tl::make_unexpected(BudgetExceeded());
    }
    if (!first && CompareTerms(previous, term) >= 0) {
      return tl::make_unexpected(
          Status::InvalidArgument("Unsorted FTS dictionary"));
    }
    size_t prefix = 0;
    while (prefix < previous.size() && prefix < term.size() &&
           previous[prefix] == term[prefix]) {
      ++prefix;
    }
    std::string header;
    AppendVarint(prefix, &header);
    AppendVarint(term.size() - prefix, &header);
    const size_t extra = header.size() + term.size() - prefix;
    if (extra > kDictionaryBlockBytes - out.size()) {
      return tl::make_unexpected(BudgetExceeded());
    }
    out.append(header);
    out.append(term.data() + prefix, term.size() - prefix);
    previous = term;
    first = false;
  }
  return out;
}

Result<DictionaryBlock> DecodeDictionary(std::string_view bytes,
                                         size_t budget) {
  if (bytes.size() > kDictionaryBlockBytes) {
    return tl::make_unexpected(Corrupt());
  }
  Decoder decoder(bytes);
  DictionaryBlock block;
  block.first_ordinal = decoder.u64();
  const auto count = decoder.u32();
  const auto flags = decoder.u32();
  if (!decoder.ok() || count == 0 || count > kDictionaryBlockTerms ||
      count > UINT64_MAX - block.first_ordinal) {
    return tl::make_unexpected(Corrupt());
  }
  if (flags == 1) {
    ByteRange range;
    range.offset = decoder.u64();
    range.length = decoder.u64();
    if (count != 1 || range.length == 0 || !ValidReference(range) ||
        !decoder.done()) {
      return tl::make_unexpected(Corrupt());
    }
    block.oversized = range;
    return block;
  }
  if (flags != 0) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS dictionary flags"));
  }
  size_t decoded_size = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t prefix = decoder.varint();
    const uint64_t suffix_length = decoder.varint();
    const auto suffix = decoder.take(suffix_length);
    const std::string_view previous =
        i == 0 ? std::string_view{} : block.terms.back();
    if (!decoder.ok() || prefix > previous.size() ||
        suffix_length > kDictionaryBlockBytes - prefix) {
      return tl::make_unexpected(Corrupt());
    }
    const size_t term_length = static_cast<size_t>(prefix + suffix_length);
    if (term_length > budget - decoded_size) {
      return tl::make_unexpected(BudgetExceeded());
    }
    decoded_size += term_length;
    std::string term(previous.substr(0, static_cast<size_t>(prefix)));
    term.append(suffix);
    if (i != 0 && CompareTerms(previous, term) >= 0) {
      return tl::make_unexpected(Corrupt());
    }
    block.terms.push_back(std::move(term));
  }
  if (!decoder.done()) {
    return tl::make_unexpected(Corrupt());
  }
  return block;
}

Result<std::string> EncodeTermMetadata(const TermMetadata &term) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "FTS v1 requires IEEE-754 binary32");
  const auto status = ValidateTerm(term);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  std::string out;
  AppendU32(kTermMetadataBytes, &out);
  AppendU32(1, &out);
  AppendU32(static_cast<uint32_t>(term.posting_tag), &out);
  AppendU32(static_cast<uint32_t>(term.position_tag), &out);
  AppendU64(term.document_frequency, &out);
  AppendU64(term.total_frequency, &out);
  uint32_t score_bits;
  std::memcpy(&score_bits, &term.max_score, sizeof(score_bits));
  AppendU32(score_bits, &out);
  AppendU32(term.inline_doc, &out);
  AppendU32(term.inline_tf, &out);
  AppendU32(term.inline_position, &out);
  for (const auto &range :
       {term.postings, term.skips, term.positions, term.position_index}) {
    AppendU64(range.offset, &out);
    AppendU64(range.length, &out);
  }
  return out;
}

Result<TermMetadata> DecodeTermMetadata(std::string_view bytes) {
  if (bytes.size() != kTermMetadataBytes) {
    return tl::make_unexpected(Corrupt());
  }
  Decoder decoder(bytes);
  if (decoder.u32() != kTermMetadataBytes) {
    return tl::make_unexpected(Corrupt());
  }
  if (decoder.u32() != 1) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS term metadata version"));
  }
  TermMetadata term;
  term.posting_tag = static_cast<PostingTag>(decoder.u32());
  term.position_tag = static_cast<PositionTag>(decoder.u32());
  term.document_frequency = decoder.u64();
  term.total_frequency = decoder.u64();
  const uint32_t score_bits = decoder.u32();
  std::memcpy(&term.max_score, &score_bits, sizeof(score_bits));
  term.inline_doc = decoder.u32();
  term.inline_tf = decoder.u32();
  term.inline_position = decoder.u32();
  for (auto *range :
       {&term.postings, &term.skips, &term.positions, &term.position_index}) {
    range->offset = decoder.u64();
    range->length = decoder.u64();
  }
  const auto status = ValidateTerm(term);
  if (!status.ok()) {
    return tl::make_unexpected(
        status.code() == StatusCode::INVALID_ARGUMENT ? Corrupt() : status);
  }
  return term;
}

Result<std::string> EncodePostingSkip(const PostingSkip &skip) {
  if (!ValidPostingSkip(skip)) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS posting skip"));
  }
  std::string bytes;
  AppendU32(skip.last_doc, &bytes);
  AppendU32(skip.count, &bytes);
  uint32_t score;
  std::memcpy(&score, &skip.max_score, sizeof(score));
  AppendU32(score, &bytes);
  AppendU32(1, &bytes);
  AppendU64(skip.postings.offset, &bytes);
  AppendU64(skip.postings.length, &bytes);
  AppendU64(skip.position_index.offset, &bytes);
  AppendU64(skip.position_index.length, &bytes);
  return bytes;
}

Result<PostingSkip> DecodePostingSkip(std::string_view bytes) {
  if (bytes.size() != kPostingSkipBytes) {
    return tl::make_unexpected(Corrupt());
  }
  Decoder decoder(bytes);
  PostingSkip skip;
  skip.last_doc = decoder.u32();
  skip.count = decoder.u32();
  const auto score = decoder.u32();
  std::memcpy(&skip.max_score, &score, sizeof(score));
  if (decoder.u32() != 1) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS skip version"));
  }
  skip.postings = {decoder.u64(), decoder.u64()};
  skip.position_index = {decoder.u64(), decoder.u64()};
  if (!decoder.done() || !ValidPostingSkip(skip)) {
    return tl::make_unexpected(Corrupt());
  }
  return skip;
}

}  // namespace zvec::fts::format
