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

#include "db/index/column/fts_column/fts_format.h"
#include <algorithm>
#include <limits>
#include <string>
#include <gtest/gtest.h>

namespace zvec::fts::format {
namespace {

std::string Hex(const Digest &digest) {
  const char *digits = "0123456789abcdef";
  std::string out;
  for (uint8_t byte : digest) {
    out.push_back(digits[byte >> 4]);
    out.push_back(digits[byte & 15]);
  }
  return out;
}

TEST(FtsFormatTest, Sha256KnownVectorsAndStreaming) {
  Sha256 hash;
  EXPECT_EQ(Hex(hash.digest()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  ASSERT_TRUE(hash.update("a").ok());
  ASSERT_TRUE(hash.update("bc").ok());
  EXPECT_EQ(Hex(hash.digest()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  Sha256 million;
  for (size_t i = 0; i < 1000; ++i) {
    ASSERT_TRUE(million.update(std::string(1000, 'a')).ok());
  }
  EXPECT_EQ(Hex(million.digest()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(FtsFormatTest, IntegersHaveCanonicalLittleEndianEncoding) {
  std::string bytes;
  AppendU32(0x89abcdef, &bytes);
  AppendU64(0x0123456789abcdef, &bytes);
  AppendVarint(300, &bytes);
  EXPECT_EQ(bytes, std::string("\xef\xcd\xab\x89\xef\xcd\xab\x89"
                               "\x67\x45\x23\x01\xac\x02",
                               14));
  Decoder decoder(bytes);
  EXPECT_EQ(decoder.u32(), 0x89abcdefu);
  EXPECT_EQ(decoder.u64(), 0x0123456789abcdefu);
  EXPECT_EQ(decoder.varint(), 300u);
  EXPECT_TRUE(decoder.done());
  bytes.clear();
  AppendVarint(UINT64_MAX, &bytes);
  Decoder maximum(bytes);
  EXPECT_EQ(maximum.varint(), UINT64_MAX);
  EXPECT_TRUE(maximum.done());
}

TEST(FtsFormatTest, RejectsOverflowTruncationAndNoncanonicalVarints) {
  const std::string malformed[] = {
      std::string("\x80", 1), std::string("\x80\0", 2), std::string(10, '\xff'),
      std::string("\xff\xff\xff\xff\xff\xff\xff\xff\xff\x02", 10)};
  for (const auto &bytes : malformed) {
    Decoder decoder(bytes);
    decoder.varint();
    EXPECT_FALSE(decoder.ok());
  }
  Decoder truncated("abc");
  truncated.u32();
  EXPECT_FALSE(truncated.ok());
  EXPECT_FALSE(truncated.done());
  EXPECT_FALSE(ValidRange(UINT64_MAX - 1, 8, UINT64_MAX));
  EXPECT_TRUE(ValidRange(7, 0, 7));
}

TEST(FtsFormatTest, DictionaryFingerprintHasFramingAndStreamsLongTerms) {
  DictionaryFingerprint fingerprint;
  ASSERT_TRUE(fingerprint.begin_term(3).ok());
  ASSERT_TRUE(fingerprint.append_term_bytes(std::string_view("a\0", 2)).ok());
  EXPECT_FALSE(fingerprint.finish().has_value());
  EXPECT_FALSE(fingerprint.begin_term(1).ok());
  EXPECT_FALSE(fingerprint.append_term_bytes("too long").ok());
  ASSERT_TRUE(fingerprint.append_term_bytes("b").ok());
  ASSERT_TRUE(fingerprint.begin_term(1).ok());
  ASSERT_TRUE(fingerprint.append_term_bytes("z").ok());
  ASSERT_TRUE(fingerprint.begin_term(1).ok());
  ASSERT_TRUE(fingerprint.append_term_bytes("\xff").ok());
  ASSERT_TRUE(fingerprint.finish().has_value());
  EXPECT_EQ(fingerprint.term_count(), 3u);
  // Digest is frozen from Python hashlib over the specified canonical input.
  EXPECT_EQ(Hex(*fingerprint.finish()),
            "27e7b548f2867acf99200263228b67724e0fa903d0a061ca04c7deb6484e657c");
}

TEST(FtsFormatTest, MetadataPreservesIdentityStatisticsAndOptionalExtensions) {
  Metadata meta;
  meta.base_id[0] = 1;
  meta.base_generation = 17;
  meta.block_id = 3;
  meta.source_segment_id = 8;
  meta.row_count = 10;
  FieldMetadata field;
  field.id[0] = 2;
  field.incarnation[0] = 3;
  field.name = "text";
  field.present_count = 7;
  field.scoring_document_count = 10;
  field.total_tokens = 42;
  field.dictionary_generation = 5;
  field.term_count = 3;
  field.analysis_identity = "standard/lowercase";
  meta.fields.push_back(field);
  auto encoded = EncodeMetadata(meta);
  ASSERT_TRUE(encoded.has_value());
  auto decoded = DecodeMetadata(*encoded);
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->base_id, meta.base_id);
  EXPECT_EQ(decoded->base_generation, 17u);
  ASSERT_EQ(decoded->fields.size(), 1u);
  EXPECT_EQ(decoded->fields[0].incarnation, field.incarnation);
  EXPECT_EQ(decoded->fields[0].present_count, 7u);
  EXPECT_EQ(decoded->fields[0].scoring_document_count, 10u);
  EXPECT_EQ(decoded->fields[0].analysis_identity, field.analysis_identity);
  EXPECT_EQ(EncodeMetadata(*decoded).value(), *encoded);

  // Add a future length-delimited optional record before the digest.
  std::string extended = encoded->substr(0, encoded->size() - 32);
  AppendU32(200, &extended);
  AppendU32(0, &extended);
  AppendU64(3, &extended);
  extended.append("xyz");
  Sha256 digest;
  ASSERT_TRUE(digest.update(extended).ok());
  const auto checksum = digest.digest();
  extended.append(reinterpret_cast<const char *>(checksum.data()), 32);
  EXPECT_TRUE(DecodeMetadata(extended).has_value());
  extended[encoded->size() - 32 + 4] = 1;  // unknown required record
  extended.resize(extended.size() - 32);
  Sha256 required_digest;
  ASSERT_TRUE(required_digest.update(extended).ok());
  const auto required_checksum = required_digest.digest();
  extended.append(reinterpret_cast<const char *>(required_checksum.data()), 32);
  EXPECT_EQ(DecodeMetadata(extended).error().code(), StatusCode::NOT_SUPPORTED);
}

TEST(FtsFormatTest, MetadataRejectsUnknownFormatsCorruptionAndBudgetOverflow) {
  Metadata meta;
  meta.base_id[0] = 1;
  meta.required_features = UINT64_C(1) << 63;
  EXPECT_EQ(EncodeMetadata(meta).error().code(), StatusCode::NOT_SUPPORTED);
  meta.required_features = 0;
  auto encoded = EncodeMetadata(meta);
  ASSERT_TRUE(encoded.has_value());
  for (size_t i = 0; i < encoded->size(); ++i) {
    EXPECT_FALSE(DecodeMetadata(std::string_view(*encoded).substr(0, i)));
  }
  auto corrupt = *encoded;
  corrupt[50] ^= 1;
  EXPECT_EQ(DecodeMetadata(corrupt).error().code(), StatusCode::INTERNAL_ERROR);
  EXPECT_EQ(DecodeMetadata(*encoded, encoded->size() - 1).error().code(),
            StatusCode::RESOURCE_EXHAUSTED);
  meta.row_count = uint64_t{UINT32_MAX} + 1;
  EXPECT_FALSE(EncodeMetadata(meta));
}

TEST(FtsFormatTest, PostingBlocksPreserveLargeGapsAndFullTf) {
  PostingBlock block;
  block.count = 3;
  block.docs = {{0, 128, UINT32_MAX - 1}};
  block.tfs = {{1, 1, UINT32_MAX}};
  for (auto encoding : {PostingEncoding::kVarint, PostingEncoding::kPacked}) {
    auto bytes = EncodePostings(block, encoding);
    ASSERT_TRUE(bytes);
    auto decoded = DecodePostings(*bytes, encoding);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->count, 3u);
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(decoded->docs[i], block.docs[i]);
      EXPECT_EQ(decoded->tfs[i], block.tfs[i]);
    }
    for (size_t i = 0; i < bytes->size(); ++i) {
      EXPECT_FALSE(
          DecodePostings(std::string_view(*bytes).substr(0, i), encoding));
    }
  }
  block.count = 128;
  for (uint32_t i = 0; i < block.count; ++i) {
    block.docs[i] = 100 + 10 * i;
    block.tfs[i] = 1;
  }
  auto packed = EncodePostings(block, PostingEncoding::kPacked);
  ASSERT_TRUE(packed);
  EXPECT_EQ(packed->size(), 8u + 64u);  // 4-bit gaps, zero-bit tf-1
  auto decoded = DecodePostings(*packed, PostingEncoding::kPacked);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->docs, block.docs);
  EXPECT_EQ(decoded->tfs, block.tfs);
}

TEST(FtsFormatTest, PostingGoldenAndMalformedPayloads) {
  PostingBlock block;
  block.count = 2;
  block.docs = {{3, 5}};
  block.tfs = {{1, 2}};
  const auto packed = EncodePostings(block, PostingEncoding::kPacked);
  ASSERT_TRUE(packed);
  EXPECT_EQ(*packed, std::string("\x03\0\0\0\x02\x01\x02\0\x08\x02", 10));
  auto bad = *packed;
  bad[8] |= 1;  // first gap must be zero
  EXPECT_FALSE(DecodePostings(bad, PostingEncoding::kPacked));
  bad = *packed;
  bad[8] |= static_cast<char>(0x80);  // nonzero unused bits
  EXPECT_FALSE(DecodePostings(bad, PostingEncoding::kPacked));
  block.docs[1] = 3;
  EXPECT_FALSE(EncodePostings(block, PostingEncoding::kPacked));
  block.docs[1] = UINT32_MAX;
  EXPECT_FALSE(EncodePostings(block, PostingEncoding::kVarint));
  block.docs[1] = 5;
  block.tfs[1] = 0;
  EXPECT_FALSE(EncodePostings(block, PostingEncoding::kPacked));
}

TEST(FtsFormatTest, PositionsPreserveZeroDuplicatesAndUint32Maximum) {
  PositionBlock block;
  block.count = 5;
  block.values = {{0, 0, 127, 128, UINT32_MAX}};
  auto bytes = EncodePositions(block);
  ASSERT_TRUE(bytes);
  auto decoded = DecodePositions(*bytes);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->count, block.count);
  EXPECT_EQ(decoded->values, block.values);
  block.values[4] = 0;
  EXPECT_FALSE(EncodePositions(block));
  bytes->back() = 0x10;  // final gap now exceeds uint32
  EXPECT_FALSE(DecodePositions(*bytes));
}

TEST(FtsFormatTest, DocumentLengthsDistinguishMissingAndZeroTokenRows) {
  DocLengthBlock block;
  block.count = 3;
  block.lengths = {{0, 0, UINT32_MAX}};
  block.present = {{false, true, true}};
  auto bytes = EncodeDocLengths(block);
  ASSERT_TRUE(bytes);
  EXPECT_EQ(*bytes, std::string("\x03\0\0\0\x06\0\0\0\0\0\0\0\0"
                                "\xff\xff\xff\xff",
                                17));
  auto decoded = DecodeDocLengths(*bytes);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->present, block.present);
  EXPECT_EQ(decoded->lengths, block.lengths);
  block.lengths[0] = 1;
  EXPECT_FALSE(EncodeDocLengths(block));
  (*bytes)[4] = 7;  // presence may be true for a zero-token document
  EXPECT_TRUE(DecodeDocLengths(*bytes));
  (*bytes)[4] = static_cast<char>(0x86);  // unused bitmap bits must be zero
  EXPECT_FALSE(DecodeDocLengths(*bytes));
}

TEST(FtsFormatTest, DocumentLengthArrayChecksExtentAndPresence) {
  for (uint32_t count : {1u, 7u, 8u, 9u, 127u, 128u}) {
    DocLengthBlock block;
    block.count = count;
    for (uint32_t i = 0; i < count; ++i) {
      block.present[i] = i % 3 != 1;
      block.lengths[i] = block.present[i] ? UINT32_MAX - i * 1000003u : 0;
    }
    auto bytes = EncodeDocLengths(block);
    ASSERT_TRUE(bytes);
    // Deliberately unaligned wire input, including full 128-value blocks.
    const auto unaligned = "x" + *bytes;
    auto decoded = DecodeDocLengths(std::string_view(unaligned).substr(1));
    ASSERT_TRUE(decoded);
    EXPECT_EQ(block.lengths, decoded->lengths);
    EXPECT_EQ(block.present, decoded->present);
    EXPECT_FALSE(DecodeDocLengths(
        std::string_view(*bytes).substr(0, bytes->size() - 1)));
    EXPECT_FALSE(DecodeDocLengths(*bytes + "x"));
    (*bytes)[4] &= static_cast<char>(~1u);  // Missing row has nonzero length.
    EXPECT_FALSE(DecodeDocLengths(*bytes));
  }
}

TEST(FtsFormatTest, PrefixDictionaryKeepsBinaryTermsAndOrdinal) {
  DictionaryBlock block;
  block.first_ordinal = 64;
  block.terms = {std::string("a\0b", 3), "apple", "apply", "\xff"};
  auto bytes = EncodeDictionary(block);
  ASSERT_TRUE(bytes);
  auto decoded = DecodeDictionary(*bytes);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->first_ordinal, 64u);
  EXPECT_EQ(decoded->terms, block.terms);
  for (size_t i = 0; i < bytes->size(); ++i) {
    EXPECT_FALSE(DecodeDictionary(std::string_view(*bytes).substr(0, i)));
  }
  block.terms = {"b", "a"};
  EXPECT_FALSE(EncodeDictionary(block));
  block.terms = {"same", "same"};
  EXPECT_FALSE(EncodeDictionary(block));
  block.terms = {std::string(5000, 'x')};
  EXPECT_EQ(EncodeDictionary(block).error().code(),
            StatusCode::RESOURCE_EXHAUSTED);
}

TEST(FtsFormatTest, OversizedTermUsesExplicitRangeIncludingOffsetZero) {
  DictionaryBlock block;
  block.first_ordinal = 70000;
  block.oversized = ByteRange{0, 100000};
  auto bytes = EncodeDictionary(block);
  ASSERT_TRUE(bytes);
  auto decoded = DecodeDictionary(*bytes);
  ASSERT_TRUE(decoded);
  ASSERT_TRUE(decoded->oversized.has_value());
  EXPECT_EQ(decoded->oversized->offset, 0u);
  EXPECT_EQ(decoded->oversized->length, 100000u);
  EXPECT_TRUE(decoded->terms.empty());
  block.oversized = ByteRange{UINT64_MAX, 2};
  EXPECT_FALSE(EncodeDictionary(block));
}

TEST(FtsFormatTest, SingletonMetadataTagsPreserveDocTfAndPositionZero) {
  TermMetadata term;
  term.document_frequency = 1;
  term.total_frequency = 1;
  term.inline_doc = UINT32_MAX - 1;
  term.inline_tf = 1;
  term.inline_position = 0;
  term.max_score = 1.5f;
  auto encoded = EncodeTermMetadata(term);
  ASSERT_TRUE(encoded);
  auto decoded = DecodeTermMetadata(*encoded);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->posting_tag, PostingTag::kInline);
  EXPECT_EQ(decoded->position_tag, PositionTag::kInline);
  EXPECT_EQ(decoded->inline_doc, UINT32_MAX - 1);
  EXPECT_EQ(decoded->inline_position, 0u);
  EXPECT_EQ(decoded->max_score, 1.5f);
  term.inline_tf = UINT32_MAX;
  term.total_frequency = UINT32_MAX;
  term.position_tag = PositionTag::kExternal;
  term.positions = {0, 4096};
  auto high_tf = EncodeTermMetadata(term);
  ASSERT_TRUE(high_tf);
  EXPECT_EQ(DecodeTermMetadata(*high_tf)->inline_tf, UINT32_MAX);
  term.position_tag = PositionTag::kInline;
  EXPECT_FALSE(EncodeTermMetadata(term));
}

TEST(FtsFormatTest, PackedBlockThresholdsAndWidths) {
  for (uint32_t count : {1u, 2u, 127u, 128u}) {
    for (unsigned width = 0; width <= 32; ++width) {
      PostingBlock block;
      block.count = count;
      for (uint32_t i = 0; i < count; ++i) {
        block.docs[i] = i;
        block.tfs[i] = width == 32 ? UINT32_MAX : (uint32_t{1} << width);
      }
      auto bytes = EncodePostings(block, PostingEncoding::kPacked);
      ASSERT_TRUE(bytes);
      const auto actual = DecodePostings(*bytes, PostingEncoding::kPacked);
      ASSERT_TRUE(actual);
      EXPECT_EQ(actual->docs, block.docs);
      EXPECT_EQ(actual->tfs, block.tfs);
    }
  }
  PostingBlock block;
  block.count = 128;
  block.docs[0] = 0;
  block.tfs[0] = UINT32_MAX;
  for (uint32_t i = 1; i < 128; ++i) {
    block.docs[i] = 0x80000000u + i;
    block.tfs[i] = UINT32_MAX;
  }
  auto bytes = EncodePostings(block, PostingEncoding::kPacked);
  ASSERT_TRUE(bytes);
  EXPECT_EQ(bytes->size(), 1032u);
  const auto actual = DecodePostings(*bytes, PostingEncoding::kPacked);
  ASSERT_TRUE(actual);
  EXPECT_EQ(actual->docs, block.docs);
  EXPECT_EQ(actual->tfs, block.tfs);
}

TEST(FtsFormatTest, PrefixExpansionIsBudgetedAndFieldIdentitiesCannotRepeat) {
  DictionaryBlock block;
  block.terms = {std::string(3000, 'a') + "a", std::string(3000, 'a') + "b"};
  auto encoded = EncodeDictionary(block);
  ASSERT_TRUE(encoded);
  EXPECT_EQ(DecodeDictionary(*encoded, 4096).error().code(),
            StatusCode::RESOURCE_EXHAUSTED);
  Metadata meta;
  meta.base_id[0] = 1;
  FieldMetadata field;
  field.id[0] = 2;
  field.incarnation[0] = 3;
  field.name = "body";
  meta.fields = {field, field};
  EXPECT_FALSE(EncodeMetadata(meta));
}

TEST(FtsFormatTest, MetadataAndFullBlockGoldenBytes) {
  Metadata meta;
  meta.base_id[0] = 1;
  auto encoded = EncodeMetadata(meta);
  ASSERT_TRUE(encoded);
  ASSERT_EQ(encoded->size(), 148u);
  Digest checksum{};
  std::copy(encoded->end() - 32, encoded->end(), checksum.begin());
  // Independently generated with Python struct.pack and hashlib.
  EXPECT_EQ(Hex(checksum),
            "32c5b9410182322c1b90f3eb9e2617c1713ff8e2fc4bf6b6d82815ccc12c4c5d");

  PostingBlock block;
  block.count = 128;
  for (uint32_t i = 0; i < 128; ++i) {
    block.docs[i] = i;
    block.tfs[i] = 1;
  }
  auto packed = EncodePostings(block, PostingEncoding::kPacked);
  ASSERT_TRUE(packed);
  const std::string header("\0\0\0\0\x01\0\x80\x01", 8);
  EXPECT_EQ(*packed, header + std::string("\xfe", 1) + std::string(15, '\xff'));
}

TEST(FtsFormatTest, PositionDirectoryUsesDualByteCheckpoints) {
  PositionIndexBlock index;
  index.count = 128;
  index.positions.offset = uint64_t{1} << 33;
  index.lengths.fill(3);
  index.lengths[0] = (uint64_t{1} << 35) + 7;
  index.positions.length = index.lengths[0] + 381;
  auto encoded = EncodePositionIndex(index);
  ASSERT_TRUE(encoded);
  auto decoded = DecodePositionIndex(*encoded);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->lengths, index.lengths);
  for (uint32_t doc : {0u, 1u, 31u, 32u, 63u, 64u, 95u, 96u, 127u}) {
    auto located = LocatePositionBytes(*encoded, doc);
    ASSERT_TRUE(located);
    EXPECT_EQ(located->offset, doc == 0 ? index.positions.offset
                                        : index.positions.offset +
                                              index.lengths[0] + (doc - 1) * 3);
    EXPECT_EQ(located->length, doc == 0 ? index.lengths[0] : 3);
  }
  EXPECT_FALSE(LocatePositionBytes(*encoded, 128));
  std::string corrupt = *encoded;
  corrupt[48] ^= 1;  // Second checkpoint disagrees with document byte lengths.
  EXPECT_FALSE(DecodePositionIndex(corrupt));
  EXPECT_FALSE(DecodePositionIndex(
      std::string_view(*encoded).substr(0, encoded->size() - 1)));
  index.positions.offset = UINT64_MAX;
  EXPECT_FALSE(EncodePositionIndex(index));
}

TEST(FtsFormatTest, PositionDirectoryGoldenTailAndMalformedLengths) {
  PositionIndexBlock index;
  index.count = 1;
  index.positions.length = 5;
  index.lengths[0] = 5;
  auto encoded = EncodePositionIndex(index);
  ASSERT_TRUE(encoded);
  const std::string header("\x01\0\0\0\x08\0\0\0\x01\0\0\0\x20\0\0\0", 16);
  EXPECT_EQ(*encoded, header + std::string(8, '\0') +
                          std::string("\x05\0\0\0\0\0\0\0", 8) +
                          std::string(16, '\0') + std::string(1, '\x05'));
  for (uint32_t count : {1u, 31u, 32u, 33u, 127u, 128u}) {
    index.count = count;
    index.positions = {0, count};
    index.lengths.fill(1);
    encoded = EncodePositionIndex(index);
    ASSERT_TRUE(encoded);
    auto decoded = DecodePositionIndex(*encoded);
    ASSERT_TRUE(decoded);
    auto last = LocatePositionBytes(*encoded, count - 1);
    ASSERT_TRUE(last);
    EXPECT_EQ(last->offset, count - 1);
    EXPECT_EQ(last->length, 1u);
    std::string malformed = *encoded;
    malformed.back() = 0;
    EXPECT_FALSE(DecodePositionIndex(malformed));
    EXPECT_FALSE(LocatePositionBytes(malformed, count - 1));
    malformed = *encoded;
    malformed[4] = 4;
    EXPECT_EQ(DecodePositionIndex(malformed).error().code(),
              StatusCode::NOT_SUPPORTED);
    malformed = *encoded + std::string(1, '\x01');
    EXPECT_FALSE(DecodePositionIndex(malformed));
  }
}

TEST(FtsFormatTest, DocLengthViewReadsUnalignedSparseAndTailBlocks) {
  for (uint32_t count : {1u, 7u, 8u, 9u, 127u, 128u}) {
    DocLengthBlock block;
    block.count = count;
    block.present[0] = true;  // Present, zero tokens differs from missing.
    block.present[count - 1] = true;
    block.lengths[count - 1] = UINT32_MAX;
    auto encoded = EncodeDocLengths(block);
    ASSERT_TRUE(encoded);
    const std::string unaligned = "x" + *encoded;
    auto view = DocLengthBlockView::Open(std::string_view(unaligned).substr(1));
    ASSERT_TRUE(view);
    EXPECT_EQ(view->count(), count);
    for (uint32_t i = 0; i < count; ++i) {
      auto entry = view->at(i);
      ASSERT_TRUE(entry);
      EXPECT_EQ(entry->present, i == 0 || i + 1 == count);
      EXPECT_EQ(entry->length, i + 1 == count ? UINT32_MAX : 0u);
    }
    EXPECT_FALSE(view->at(count));
    EXPECT_FALSE(view->at(UINT32_MAX));
  }
  DocLengthBlock dense;
  dense.count = 128;
  dense.present.fill(true);
  dense.lengths.fill(UINT32_MAX);
  auto encoded = EncodeDocLengths(dense);
  ASSERT_TRUE(encoded);
  auto view = DocLengthBlockView::Open(*encoded);
  ASSERT_TRUE(view);
  for (uint32_t i = 0; i < 128; ++i) {
    auto entry = view->at(i);
    ASSERT_TRUE(entry);
    EXPECT_TRUE(entry->present);
    EXPECT_EQ(UINT32_MAX, entry->length);
  }
}

TEST(FtsFormatTest, DocLengthViewRejectsMalformedAndUnrequestedMissingRows) {
  const std::string valid(
      "\x03\0\0\0\x06\0\0\0\0\0\0\0\0"
      "\xff\xff\xff\xff",
      17);
  ASSERT_TRUE(DocLengthBlockView::Open(valid));
  for (size_t end = 0; end < valid.size(); ++end) {
    EXPECT_FALSE(
        DocLengthBlockView::Open(std::string_view(valid).substr(0, end)));
  }
  EXPECT_FALSE(DocLengthBlockView::Open(valid + "x"));
  auto missing = valid;
  missing[5] = 1;  // Unrequested missing row 0 must still be validated.
  EXPECT_FALSE(DocLengthBlockView::Open(missing));
  auto padding = valid;
  padding[4] = static_cast<char>(0x86);
  EXPECT_FALSE(DocLengthBlockView::Open(padding));
  for (uint8_t count : {uint8_t{0}, uint8_t{129}}) {
    auto invalid = valid;
    invalid[0] = static_cast<char>(count);
    EXPECT_FALSE(DocLengthBlockView::Open(invalid));
  }
  EXPECT_FALSE(DocLengthBlockView{}.at(0));
}

TEST(FtsFormatTest, PostingSkipGoldenAndInvalidScoreOrRange) {
  PostingSkip skip{2, 2, 1.0f, {0, 5}, {0, 50}};
  auto encoded = EncodePostingSkip(skip);
  ASSERT_TRUE(encoded);
  const std::string header("\x02\0\0\0\x02\0\0\0\0\0\x80\x3f\x01\0\0\0", 16);
  EXPECT_EQ(*encoded, header + std::string(8, '\0') +
                          std::string("\x05\0\0\0\0\0\0\0", 8) +
                          std::string(8, '\0') +
                          std::string("\x32\0\0\0\0\0\0\0", 8));
  auto decoded = DecodePostingSkip(*encoded);
  ASSERT_TRUE(decoded);
  EXPECT_FLOAT_EQ(decoded->max_score, 1.0f);
  EXPECT_EQ(decoded->position_index.length, 50u);
  const auto valid = *encoded;
  for (size_t size = 0; size < valid.size(); ++size) {
    EXPECT_FALSE(DecodePostingSkip(std::string_view(valid).substr(0, size)));
  }
  EXPECT_FALSE(DecodePostingSkip(valid + "x"));
  // Decoder validation must reject malformed fields without relying on a
  // second serialization. Keep the expected wire mutations explicit.
  for (const auto &mutation :
       std::vector<std::pair<size_t, uint32_t>>{{0, UINT32_MAX},
                                                {4, 0},
                                                {4, 129},
                                                {8, 0x7fc00000},
                                                {8, 0x7f800000},
                                                {8, 0xbf800000},
                                                {24, 0},
                                                {40, 0}}) {
    auto invalid = valid;
    std::string value;
    AppendU32(mutation.second, &value);
    invalid.replace(mutation.first, value.size(), value);
    EXPECT_FALSE(DecodePostingSkip(invalid));
  }
  for (size_t offset : {16u, 32u}) {
    auto invalid = valid;
    invalid.replace(offset, 8, std::string(8, '\xff'));
    EXPECT_FALSE(DecodePostingSkip(invalid));
  }
  (*encoded)[12] = 2;
  EXPECT_EQ(DecodePostingSkip(*encoded).error().code(),
            StatusCode::NOT_SUPPORTED);
  skip.max_score = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(EncodePostingSkip(skip));
  skip.max_score = 1;
  skip.postings.offset = UINT64_MAX;
  EXPECT_FALSE(EncodePostingSkip(skip));
  skip.postings.offset = 0;
  skip.last_doc = UINT32_MAX;
  EXPECT_FALSE(EncodePostingSkip(skip));
}

}  // namespace
}  // namespace zvec::fts::format
