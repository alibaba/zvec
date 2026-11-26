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
#include <cstring>
#include <set>
#include "fts_format.h"

namespace zvec::fts::format {
namespace {

constexpr char kMagic[] = {'Z', 'V', 'F', 'T', 'S', 'M', '0', '1'};
constexpr uint32_t kHeaderSize = 40;
constexpr uint32_t kBaseRecord = 1;
constexpr uint32_t kFieldRecord = 2;
constexpr uint32_t kSectionRecord = 3;
constexpr uint32_t kRequiredRecord = 1;

Status Corrupt() {
  return Status::InternalError("Malformed FTS metadata");
}

Status BudgetExceeded() {
  return {StatusCode::RESOURCE_EXHAUSTED, "FTS metadata budget exceeded"};
}

template <size_t N>
void PutArray(const std::array<uint8_t, N> &bytes, std::string *out) {
  out->append(reinterpret_cast<const char *>(bytes.data()), N);
}

template <size_t N>
void GetArray(Decoder *decoder, std::array<uint8_t, N> *out) {
  const auto bytes = decoder->take(N);
  if (decoder->ok()) {
    std::memcpy(out->data(), bytes.data(), N);
  }
}

bool Nonzero(const Identity &id) {
  return std::any_of(id.begin(), id.end(), [](uint8_t b) { return b != 0; });
}

void PutString(std::string_view value, std::string *out) {
  AppendU64(value.size(), out);
  out->append(value);
}

std::string GetString(Decoder *decoder) {
  const uint64_t length = decoder->u64();
  return std::string(decoder->take(length));
}

Status Validate(const Metadata &meta) {
  if (meta.required_features != 0) {
    return Status::NotSupported("Unknown required FTS feature");
  }
  if (!Nonzero(meta.base_id) || meta.row_count > UINT32_MAX ||
      meta.layout_granule < 1024 || meta.layout_granule > 65536 ||
      (meta.layout_granule & (meta.layout_granule - 1)) != 0) {
    return Status::InvalidArgument(
        "Invalid FTS base identity, row domain or granule");
  }
  if (meta.fields.size() > kMaxFields || meta.sections.size() > kMaxSections) {
    return BudgetExceeded();
  }
  std::set<std::string> names;
  std::set<Identity> field_ids;
  std::set<Identity> incarnations;
  for (const auto &field : meta.fields) {
    if (!Nonzero(field.id) || !Nonzero(field.incarnation) ||
        field.name.empty() || !names.insert(field.name).second ||
        !field_ids.insert(field.id).second ||
        !incarnations.insert(field.incarnation).second ||
        field.present_count > meta.row_count ||
        field.scoring_document_count > meta.row_count) {
      return Status::InvalidArgument(
          "Invalid FTS field identity or statistics");
    }
    if (field.bound_version != 1) {
      return Status::NotSupported("Unknown FTS bound version");
    }
  }
  names.clear();
  uint64_t checksum_end = 0;
  for (const auto &section : meta.sections) {
    if (section.name.empty() || section.name.size() > 256 ||
        section.name.find('\0') != std::string::npos ||
        section.name == "fts.meta" || section.name == "IndexVersion" ||
        section.name == "checksums" || !names.insert(section.name).second ||
        section.first_checksum != checksum_end) {
      return Status::InvalidArgument("Invalid FTS section directory");
    }
    const uint64_t count = section.length / meta.layout_granule +
                           (section.length % meta.layout_granule != 0);
    if (count > UINT64_MAX - checksum_end) {
      return Status::InvalidArgument("FTS checksum directory overflow");
    }
    checksum_end += count;
  }
  return Status::OK();
}

Status PutRecord(uint32_t type, const std::string &payload, size_t budget,
                 std::string *out) {
  // Reserve both the record header and the final SHA-256 before appending.
  if (out->size() > budget || budget - out->size() < 48 ||
      payload.size() > budget - out->size() - 48) {
    return BudgetExceeded();
  }
  AppendU32(type, out);
  AppendU32(kRequiredRecord, out);
  AppendU64(payload.size(), out);
  out->append(payload);
  return Status::OK();
}

}  // namespace

Result<std::string> EncodeMetadata(const Metadata &meta, size_t budget) {
  // Check variable input sizes before constructing any encoded payload.
  if (budget < kHeaderSize + 32 || meta.fields.size() > kMaxFields ||
      meta.sections.size() > kMaxSections) {
    return tl::make_unexpected(BudgetExceeded());
  }
  size_t variable_bytes = 0;
  auto charge = [&](size_t length) {
    if (length > budget - variable_bytes) {
      return false;
    }
    variable_bytes += length;
    return true;
  };
  for (const auto &field : meta.fields) {
    if (!charge(field.name.size()) || !charge(field.analysis_identity.size())) {
      return tl::make_unexpected(BudgetExceeded());
    }
  }
  for (const auto &section : meta.sections) {
    if (!charge(section.name.size())) {
      return tl::make_unexpected(BudgetExceeded());
    }
  }
  auto status = Validate(meta);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  std::string out(kMagic, sizeof(kMagic));
  AppendU32(1, &out);  // major
  AppendU32(meta.minor_version, &out);
  AppendU64(meta.required_features, &out);
  AppendU64(meta.optional_features, &out);
  AppendU32(1, &out);  // minimum reader version
  AppendU32(kHeaderSize, &out);

  std::string base;
  PutArray(meta.base_id, &base);
  AppendU64(meta.base_generation, &base);
  AppendU32(meta.block_id, &base);
  AppendU64(meta.source_segment_id, &base);
  AppendU64(meta.row_count, &base);
  AppendU32(1, &base);  // parent segment-local row mapping
  AppendU32(meta.layout_granule, &base);
  AppendU32(static_cast<uint32_t>(meta.fields.size()), &base);
  AppendU32(static_cast<uint32_t>(meta.sections.size()), &base);
  status = PutRecord(kBaseRecord, base, budget, &out);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  uint32_t ordinal = 0;
  for (const auto &field : meta.fields) {
    std::string payload;
    AppendU32(ordinal++, &payload);
    PutArray(field.id, &payload);
    PutArray(field.incarnation, &payload);
    PutString(field.name, &payload);
    AppendU64(meta.row_count, &payload);
    AppendU64(field.present_count, &payload);
    AppendU64(field.scoring_document_count, &payload);
    AppendU64(field.total_tokens, &payload);
    AppendU32(1, &payload);  // statistics_policy=legacy_compatible
    AppendU64(field.dictionary_generation, &payload);
    AppendU64(field.term_count, &payload);
    AppendU32(1, &payload);  // unsigned-byte term order
    AppendU32(1, &payload);  // term encoding version
    AppendU32(1, &payload);  // SHA-256
    PutArray(field.dictionary_fingerprint, &payload);
    PutString(field.analysis_identity, &payload);
    for (size_t i = 0; i < 4; ++i) {
      AppendU32(1, &payload);  // posting, position, doclen, skip codec
    }
    PutArray(field.scoring_identity, &payload);
    AppendU32(1, &payload);  // FP32 upper bounds
    PutArray(field.bound_statistics_identity, &payload);
    AppendU32(field.bound_version, &payload);
    status = PutRecord(kFieldRecord, payload, budget, &out);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  for (const auto &section : meta.sections) {
    std::string payload;
    PutString(section.name, &payload);
    AppendU64(section.length, &payload);
    AppendU64(section.first_checksum, &payload);
    status = PutRecord(kSectionRecord, payload, budget, &out);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  Sha256 hash;
  status = hash.update(out);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  PutArray(hash.digest(), &out);
  return out;
}

Result<Metadata> DecodeMetadata(std::string_view bytes, size_t budget) {
  if (bytes.size() > budget) {
    return tl::make_unexpected(BudgetExceeded());
  }
  if (bytes.size() < kHeaderSize + 32) {
    return tl::make_unexpected(Corrupt());
  }
  const auto body = bytes.substr(0, bytes.size() - 32);
  Sha256 hash;
  const auto hash_status = hash.update(body);
  if (!hash_status.ok()) {
    return tl::make_unexpected(hash_status);
  }
  const auto digest = hash.digest();
  if (std::memcmp(digest.data(), bytes.data() + body.size(), digest.size()) !=
      0) {
    return tl::make_unexpected(Corrupt());
  }
  Decoder decoder(body);
  if (decoder.take(sizeof(kMagic)) !=
      std::string_view(kMagic, sizeof(kMagic))) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS metadata magic"));
  }
  const auto major = decoder.u32();
  Metadata meta;
  meta.minor_version = decoder.u32();
  meta.required_features = decoder.u64();
  meta.optional_features = decoder.u64();
  const auto minimum_reader = decoder.u32();
  const auto header_size = decoder.u32();
  if (major != 1 || minimum_reader > 1 || meta.required_features != 0) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown required FTS format"));
  }
  if (header_size != kHeaderSize || minimum_reader == 0) {
    return tl::make_unexpected(Corrupt());
  }
  bool have_base = false;
  uint32_t field_count = 0, section_count = 0;
  while (!decoder.done()) {
    const auto type = decoder.u32();
    const auto flags = decoder.u32();
    const auto length = decoder.u64();
    const auto payload = decoder.take(length);
    if (!decoder.ok()) {
      return tl::make_unexpected(Corrupt());
    }
    if (flags & ~kRequiredRecord) {
      return tl::make_unexpected(
          Status::NotSupported("Unknown FTS record flags"));
    }
    Decoder record(payload);
    if (type == kBaseRecord) {
      if (have_base) {
        return tl::make_unexpected(Corrupt());
      }
      have_base = true;
      GetArray(&record, &meta.base_id);
      meta.base_generation = record.u64();
      meta.block_id = record.u32();
      meta.source_segment_id = record.u64();
      meta.row_count = record.u64();
      const auto mapping = record.u32();
      meta.layout_granule = record.u32();
      field_count = record.u32();
      section_count = record.u32();
      if (mapping != 1 && record.ok()) {
        return tl::make_unexpected(
            Status::NotSupported("Unknown FTS row mapping"));
      }
      if (field_count > kMaxFields || section_count > kMaxSections) {
        return tl::make_unexpected(BudgetExceeded());
      }
    } else if (type == kFieldRecord) {
      if (!have_base || meta.fields.size() >= field_count ||
          record.u32() != meta.fields.size()) {
        return tl::make_unexpected(Corrupt());
      }
      FieldMetadata field;
      GetArray(&record, &field.id);
      GetArray(&record, &field.incarnation);
      field.name = GetString(&record);
      if (record.u64() != meta.row_count) {
        return tl::make_unexpected(Corrupt());
      }
      field.present_count = record.u64();
      field.scoring_document_count = record.u64();
      field.total_tokens = record.u64();
      const auto statistics = record.u32();
      field.dictionary_generation = record.u64();
      field.term_count = record.u64();
      const auto order = record.u32();
      const auto encoding = record.u32();
      const auto fingerprint = record.u32();
      GetArray(&record, &field.dictionary_fingerprint);
      field.analysis_identity = GetString(&record);
      bool supported =
          statistics == 1 && order == 1 && encoding == 1 && fingerprint == 1;
      for (size_t i = 0; i < 4; ++i) {
        const auto codec = record.u32();
        supported = codec == 1 && supported;
      }
      GetArray(&record, &field.scoring_identity);
      supported = record.u32() == 1 && supported;
      GetArray(&record, &field.bound_statistics_identity);
      field.bound_version = record.u32();
      if ((!supported || field.bound_version != 1) && record.ok()) {
        return tl::make_unexpected(
            Status::NotSupported("Unknown FTS field codec"));
      }
      meta.fields.push_back(std::move(field));
    } else if (type == kSectionRecord) {
      if (!have_base || meta.sections.size() >= section_count) {
        return tl::make_unexpected(Corrupt());
      }
      SectionMetadata section;
      section.name = GetString(&record);
      section.length = record.u64();
      section.first_checksum = record.u64();
      meta.sections.push_back(std::move(section));
    } else {
      if (flags & kRequiredRecord) {
        return tl::make_unexpected(
            Status::NotSupported("Unknown required FTS record"));
      }
      continue;
    }
    if (!record.done()) {
      return tl::make_unexpected(Corrupt());
    }
  }
  if (!have_base || field_count != meta.fields.size() ||
      section_count != meta.sections.size()) {
    return tl::make_unexpected(Corrupt());
  }
  const auto status = Validate(meta);
  if (!status.ok()) {
    return tl::make_unexpected(
        status.code() == StatusCode::INVALID_ARGUMENT ? Corrupt() : status);
  }
  return meta;
}

}  // namespace zvec::fts::format
