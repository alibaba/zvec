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

//! Golden-byte and robustness tests for ManifestCodec.
//!
//! The byte arrays below were produced by the protobuf implementation that
//! zvec used before ManifestCodec existed (see the cross-check tests in
//! manifest_codec_test.cc, which verified both implementations agree
//! byte-for-byte). They pin the on-disk manifest format: decoding them must
//! yield the expected values and re-encoding must reproduce the exact same
//! bytes.
//!
//! These tests do not depend on libprotobuf, so they keep guarding format
//! compatibility after that dependency is dropped.
//!
//! Never regenerate these arrays to make a failing test pass: a mismatch means
//! the persisted format changed and existing collections would fail to open.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/common/manifest_codec.h"
#include "db/index/common/pb_wire.h"

using namespace zvec;

namespace {

//! Manifest of a default-constructed schema with no fields.
const uint8_t kGoldenEmpty[] = {
    0x12, 0x05, 0x18, 0x80, 0xad, 0xe2, 0x04,
};

//! Manifest with a scalar field, an HNSW vector field and one segment.
const uint8_t kGoldenSimple[] = {
    0x12, 0x31, 0x0a, 0x06, 0x73, 0x69, 0x6d, 0x70, 0x6c, 0x65, 0x12, 0x06,
    0x0a, 0x02, 0x70, 0x6b, 0x10, 0x02, 0x12, 0x1b, 0x0a, 0x05, 0x64, 0x65,
    0x6e, 0x73, 0x65, 0x10, 0x17, 0x18, 0x80, 0x01, 0x2a, 0x0d, 0x12, 0x0b,
    0x0a, 0x04, 0x08, 0x02, 0x22, 0x00, 0x10, 0x10, 0x18, 0xc8, 0x01, 0x18,
    0xa0, 0x8d, 0x06, 0x18, 0x01, 0x22, 0x0a, 0x12, 0x08, 0x10, 0x01, 0x28,
    0x0a, 0x32, 0x02, 0x70, 0x6b, 0x40, 0x01,
};

//! Manifest exercising every IndexParams oneof branch, both segment slots and
//! all scalar manifest fields.
const uint8_t kGoldenAllIndexTypes[] = {
    0x12, 0xc1, 0x02, 0x0a, 0x04, 0x72, 0x69, 0x63, 0x68, 0x12, 0x14, 0x0a,
    0x08, 0x66, 0x5f, 0x69, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x10, 0x05, 0x20,
    0x01, 0x2a, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x12, 0x24, 0x0a, 0x06, 0x66,
    0x5f, 0x68, 0x6e, 0x73, 0x77, 0x10, 0x17, 0x18, 0x80, 0x01, 0x20, 0x01,
    0x2a, 0x13, 0x12, 0x11, 0x0a, 0x08, 0x08, 0x01, 0x10, 0x02, 0x22, 0x02,
    0x08, 0x01, 0x10, 0x10, 0x18, 0xc8, 0x01, 0x20, 0x01, 0x12, 0x1a, 0x0a,
    0x06, 0x66, 0x5f, 0x66, 0x6c, 0x61, 0x74, 0x10, 0x16, 0x18, 0x40, 0x20,
    0x01, 0x2a, 0x0a, 0x1a, 0x08, 0x0a, 0x06, 0x08, 0x02, 0x10, 0x01, 0x22,
    0x00, 0x12, 0x22, 0x0a, 0x05, 0x66, 0x5f, 0x69, 0x76, 0x66, 0x10, 0x1a,
    0x18, 0x20, 0x20, 0x01, 0x2a, 0x13, 0x22, 0x11, 0x0a, 0x08, 0x08, 0x03,
    0x10, 0x03, 0x22, 0x02, 0x08, 0x01, 0x10, 0x80, 0x04, 0x18, 0x0c, 0x20,
    0x01, 0x12, 0x29, 0x0a, 0x08, 0x66, 0x5f, 0x72, 0x61, 0x62, 0x69, 0x74,
    0x71, 0x10, 0x17, 0x18, 0x80, 0x02, 0x20, 0x01, 0x2a, 0x16, 0x2a, 0x14,
    0x0a, 0x04, 0x08, 0x01, 0x10, 0x04, 0x10, 0x18, 0x18, 0x96, 0x01, 0x20,
    0x05, 0x28, 0x80, 0x08, 0x30, 0xd0, 0x86, 0x03, 0x12, 0x2b, 0x0a, 0x08,
    0x66, 0x5f, 0x76, 0x61, 0x6d, 0x61, 0x6e, 0x61, 0x10, 0x17, 0x18, 0x60,
    0x20, 0x01, 0x2a, 0x19, 0x32, 0x17, 0x0a, 0x08, 0x08, 0x01, 0x10, 0x04,
    0x22, 0x02, 0x08, 0x01, 0x10, 0x30, 0x18, 0x60, 0x25, 0xcd, 0xcc, 0xac,
    0x3f, 0x28, 0x01, 0x38, 0x01, 0x12, 0x2b, 0x0a, 0x05, 0x66, 0x5f, 0x66,
    0x74, 0x73, 0x10, 0x02, 0x20, 0x01, 0x2a, 0x1e, 0x3a, 0x1c, 0x0a, 0x05,
    0x6a, 0x69, 0x65, 0x62, 0x61, 0x12, 0x09, 0x6c, 0x6f, 0x77, 0x65, 0x72,
    0x63, 0x61, 0x73, 0x65, 0x12, 0x04, 0x73, 0x74, 0x6f, 0x70, 0x1a, 0x02,
    0x7b, 0x7d, 0x12, 0x25, 0x0a, 0x09, 0x66, 0x5f, 0x64, 0x69, 0x73, 0x6b,
    0x61, 0x6e, 0x6e, 0x10, 0x17, 0x18, 0x80, 0x04, 0x20, 0x01, 0x2a, 0x11,
    0x42, 0x0f, 0x0a, 0x06, 0x08, 0x01, 0x10, 0x02, 0x22, 0x00, 0x10, 0x40,
    0x18, 0x80, 0x01, 0x20, 0x10, 0x12, 0x0d, 0x0a, 0x07, 0x66, 0x5f, 0x70,
    0x6c, 0x61, 0x69, 0x6e, 0x10, 0x30, 0x20, 0x01, 0x18, 0x80, 0x80, 0x40,
    0x18, 0x01, 0x22, 0x16, 0x12, 0x0c, 0x10, 0x01, 0x20, 0xe7, 0x07, 0x28,
    0xe8, 0x07, 0x32, 0x02, 0x70, 0x6b, 0x22, 0x06, 0x66, 0x5f, 0x68, 0x6e,
    0x73, 0x77, 0x22, 0x1d, 0x08, 0x01, 0x12, 0x11, 0x08, 0x01, 0x10, 0x01,
    0x18, 0xe8, 0x07, 0x20, 0xcf, 0x0f, 0x28, 0xe8, 0x07, 0x32, 0x02, 0x70,
    0x6b, 0x22, 0x06, 0x66, 0x5f, 0x68, 0x6e, 0x73, 0x77, 0x2a, 0x06, 0x08,
    0x02, 0x1a, 0x02, 0x10, 0x01, 0x30, 0x05, 0x38, 0x09, 0x40, 0x0c,
};

//! IVF_RABITQ index params, encoded by hand from the wire format.
//!
//! Unlike the arrays above this one was not produced by the protobuf
//! implementation: IVF_RABITQ was added after that dependency was dropped, so
//! there is nothing left to cross-check against. The bytes were derived from
//! the field numbers of the (now deleted) proto definition instead, which is
//! what older readers expect:
//!   IndexParams.ivf_rabitq = 9
//!   IvfRabitqIndexParams { base = 1, nlist = 2, total_bits = 3,
//!                          sample_count = 4 }
//! Values: metric L2, nlist 1024, total_bits 5, sample_count 100000.
//!
//! Byte layout:
//!   4a 0f                    ivf_rabitq branch, 15 bytes of payload
//!   0a 04 08 01 10 04        base { metric = MT_L2, quantize = QT_RABITQ }
//!   10 80 08                 nlist = 1024
//!   18 05                    total_bits = 5
//!   20 a0 8d 06              sample_count = 100000
const uint8_t kGoldenIvfRabitq[] = {
    0x4a, 0x0f, 0x0a, 0x04, 0x08, 0x01, 0x10, 0x04, 0x10,
    0x80, 0x08, 0x18, 0x05, 0x20, 0xa0, 0x8d, 0x06,
};

template <size_t N>
std::string Golden(const uint8_t (&bytes)[N]) {
  return std::string(reinterpret_cast<const char *>(bytes), N);
}

//! Decodes a golden buffer and re-encodes it, expecting identical bytes.
ManifestData ExpectStableRoundTrip(const std::string &golden,
                                   const char *name) {
  ManifestData data;
  EXPECT_TRUE(ManifestCodec::Decode(golden, &data).ok()) << name;

  std::string reencoded;
  EXPECT_TRUE(ManifestCodec::Encode(data, &reencoded).ok()) << name;
  EXPECT_EQ(reencoded, golden)
      << "re-encoding " << name
      << " changed the bytes; the on-disk manifest format must stay stable";
  return data;
}

}  // namespace

TEST(ManifestCodecGolden, EmptyManifest) {
  auto data = ExpectStableRoundTrip(Golden(kGoldenEmpty), "empty");
  ASSERT_NE(data.schema, nullptr);
  EXPECT_TRUE(data.schema->name().empty());
  EXPECT_TRUE(data.schema->fields().empty());
  EXPECT_FALSE(data.enable_mmap);
  EXPECT_EQ(data.next_segment_id, 0u);
}

TEST(ManifestCodecGolden, SimpleManifest) {
  auto data = ExpectStableRoundTrip(Golden(kGoldenSimple), "simple");
  ASSERT_NE(data.schema, nullptr);
  EXPECT_EQ(data.schema->name(), "simple");
  EXPECT_EQ(data.schema->max_doc_count_per_segment(), 100000u);

  const auto fields = data.schema->fields();
  ASSERT_EQ(fields.size(), 2u);

  EXPECT_EQ(fields[0]->name(), "pk");
  EXPECT_EQ(fields[0]->data_type(), DataType::STRING);
  EXPECT_EQ(fields[0]->index_params(), nullptr);

  EXPECT_EQ(fields[1]->name(), "dense");
  EXPECT_EQ(fields[1]->data_type(), DataType::VECTOR_FP32);
  EXPECT_EQ(fields[1]->dimension(), 128u);
  const auto dense_params = fields[1]->index_params();
  ASSERT_NE(dense_params, nullptr);
  ASSERT_EQ(dense_params->type(), IndexType::HNSW);
  auto *hnsw = dynamic_cast<const HnswIndexParams *>(dense_params.get());
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->metric_type(), MetricType::IP);
  EXPECT_EQ(hnsw->m(), 16);
  EXPECT_EQ(hnsw->ef_construction(), 200);

  EXPECT_TRUE(data.enable_mmap);
  ASSERT_EQ(data.persisted_segment_metas.size(), 1u);
  EXPECT_EQ(data.persisted_segment_metas[0]->id(), 0u);
  EXPECT_EQ(data.persisted_segment_metas[0]->persisted_blocks().size(), 1u);
  EXPECT_EQ(data.next_segment_id, 1u);
}

TEST(ManifestCodecGolden, AllIndexTypes) {
  auto data =
      ExpectStableRoundTrip(Golden(kGoldenAllIndexTypes), "all_index_types");
  ASSERT_NE(data.schema, nullptr);

  const auto fields = data.schema->fields();
  ASSERT_EQ(fields.size(), 9u);

  // Every oneof branch of IndexParams must survive the round trip.
  const std::vector<IndexType> expected = {
      IndexType::INVERT, IndexType::HNSW,        IndexType::FLAT,
      IndexType::IVF,    IndexType::HNSW_RABITQ, IndexType::VAMANA,
      IndexType::FTS,    IndexType::DISKANN};
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto params = fields[i]->index_params();
    ASSERT_NE(params, nullptr) << "field " << fields[i]->name();
    EXPECT_EQ(params->type(), expected[i]) << "field " << fields[i]->name();
  }
  // The last field intentionally carries no index params.
  EXPECT_EQ(fields[8]->index_params(), nullptr);

  // Spot-check the only fixed32 field in the whole format.
  const auto vamana_params = fields[5]->index_params();
  auto *vamana = dynamic_cast<const VamanaIndexParams *>(vamana_params.get());
  ASSERT_NE(vamana, nullptr);
  EXPECT_FLOAT_EQ(vamana->alpha(), 1.35f);
  EXPECT_TRUE(vamana->saturate_graph());
  EXPECT_TRUE(vamana->use_id_map());
  EXPECT_FALSE(vamana->use_contiguous_memory());

  // Repeated strings of the FTS branch.
  const auto fts_params = fields[6]->index_params();
  auto *fts = dynamic_cast<const FtsIndexParams *>(fts_params.get());
  ASSERT_NE(fts, nullptr);
  EXPECT_EQ(fts->tokenizer_name(), "jieba");
  ASSERT_EQ(fts->filters().size(), 2u);
  EXPECT_EQ(fts->filters()[0], "lowercase");
  EXPECT_EQ(fts->filters()[1], "stop");

  EXPECT_EQ(data.persisted_segment_metas.size(), 2u);
  ASSERT_NE(data.writing_segment_meta, nullptr);
  EXPECT_EQ(data.writing_segment_meta->id(), 2u);
  EXPECT_TRUE(data.writing_segment_meta->has_writing_forward_block());
  EXPECT_EQ(data.id_map_path_suffix, 5u);
  EXPECT_EQ(data.delete_snapshot_path_suffix, 9u);
  EXPECT_EQ(data.next_segment_id, 12u);
}

TEST(ManifestCodecGolden, IvfRabitqIndexParams) {
  const std::string golden = Golden(kGoldenIvfRabitq);

  // Encoding must produce exactly the bytes an older reader expects. Note the
  // base sub-message carries no quantizer_param, matching HNSW_RABITQ.
  IvfRabitqIndexParams params(MetricType::L2, 1024, 5, 100000);
  std::string encoded;
  ManifestCodec::EncodeIndexParams(&params, &encoded);
  EXPECT_EQ(encoded, golden)
      << "the on-disk encoding of IVF_RABITQ index params must stay stable";

  // ... and decoding the golden bytes must round trip every field.
  const auto decoded = ManifestCodec::DecodeIndexParams(golden);
  ASSERT_NE(decoded, nullptr);
  ASSERT_EQ(decoded->type(), IndexType::IVF_RABITQ);
  auto *ivf_rabitq = dynamic_cast<const IvfRabitqIndexParams *>(decoded.get());
  ASSERT_NE(ivf_rabitq, nullptr);
  EXPECT_EQ(ivf_rabitq->metric_type(), MetricType::L2);
  EXPECT_EQ(ivf_rabitq->quantize_type(), QuantizeType::RABITQ);
  EXPECT_EQ(ivf_rabitq->nlist(), 1024);
  EXPECT_EQ(ivf_rabitq->total_bits(), 5);
  EXPECT_EQ(ivf_rabitq->sample_count(), 100000);
  EXPECT_EQ(*decoded, params);
}

TEST(ManifestCodecGolden, UnknownFieldsAreIgnored) {
  // Forward compatibility: a manifest written by a newer zvec may carry fields
  // this build does not know about. They must be skipped silently.
  std::string with_unknown = Golden(kGoldenSimple);
  {
    pbwire::Writer w(&with_unknown);
    w.PutVarint(99, 12345);               // unknown varint field
    w.PutMessage(100, std::string("x"));  // unknown length-delimited field
    w.PutFloat(101, 2.5f);                // unknown fixed32 field
  }

  ManifestData data;
  ASSERT_TRUE(ManifestCodec::Decode(with_unknown, &data).ok());
  ASSERT_NE(data.schema, nullptr);
  EXPECT_EQ(data.schema->name(), "simple");
  EXPECT_EQ(data.schema->fields().size(), 2u);
  EXPECT_EQ(data.next_segment_id, 1u);
}

TEST(ManifestCodecGolden, TruncatedInputDoesNotCrash) {
  const std::string golden = Golden(kGoldenAllIndexTypes);
  // Truncation at every offset must either be reported as an error or produce
  // a partial but well-formed result; it must never crash or hang.
  for (size_t len = 1; len < golden.size(); ++len) {
    ManifestData data;
    ManifestCodec::Decode(std::string_view(golden.data(), len), &data);
  }
}

TEST(ManifestCodecGolden, CorruptInputIsRejected) {
  // A length-delimited field claiming more bytes than are available.
  std::string bad_len;
  {
    pbwire::Writer w(&bad_len);
    w.PutMessage(2, std::string("payload"));
  }
  bad_len[1] = static_cast<char>(0x7F);  // overlong length
  ManifestData data;
  EXPECT_FALSE(ManifestCodec::Decode(bad_len, &data).ok());

  // A varint that never terminates.
  const std::string bad_varint(12, static_cast<char>(0xFF));
  ManifestData data2;
  EXPECT_FALSE(ManifestCodec::Decode(bad_varint, &data2).ok());

  // Field number zero is illegal.
  const std::string zero_field(1, static_cast<char>(0x00));
  ManifestData data3;
  EXPECT_FALSE(ManifestCodec::Decode(zero_field, &data3).ok());
}

TEST(ManifestCodecGolden, WireReaderRejectsGroups) {
  // Wire types 3 and 4 (deprecated groups) are never produced by zvec.proto
  // and must be treated as corrupt input rather than silently skipped.
  for (uint8_t type : {3, 4}) {
    std::string buf;
    buf.push_back(static_cast<char>((1 << 3) | type));
    pbwire::Reader r(buf);
    EXPECT_FALSE(r.Next());
    EXPECT_FALSE(r.ok());
  }
}
