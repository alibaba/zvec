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

#include "fts_dictionary.h"
#include <algorithm>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "fts_stream.h"

namespace zvec::fts {
namespace {

constexpr size_t kIndexHeader = 32;
constexpr size_t kIndexRecord = 32;
constexpr size_t kIndexPageRecords = 128;
constexpr size_t kChunk = format::kDictionaryBlockBytes;
constexpr std::string_view kIndexMagic = "ZVFTDI01";

// Unsigned, big-endian and zero-padded. Unequal prefixes order the full byte
// strings. Equal prefixes (including trailing NULs) need an exact comparison.
uint64_t FencePrefix(std::string_view term) {
  uint64_t value = 0;
  for (size_t i = 0; i < std::min(sizeof(value), term.size()); ++i) {
    value |= static_cast<uint64_t>(static_cast<uint8_t>(term[i]))
             << (56 - 8 * i);
  }
  return value;
}

Status Corrupt() {
  return Status::InternalError("Malformed FTS dictionary directory/reference");
}

Status Exhausted() {
  return {StatusCode::RESOURCE_EXHAUSTED,
          "FTS dictionary workspace/platform limit exceeded"};
}

Status NotFound() {
  return Status::NotFound("FTS dictionary term/ordinal not found");
}

Result<uint64_t> SectionLength(const FtsContainerReader &container,
                               const std::string &name) {
  for (const auto &section : container.metadata().sections) {
    if (section.name == name) {
      return section.length;
    }
  }
  return tl::make_unexpected(Corrupt());
}

}  // namespace

FtsDictionaryWriter::FtsDictionaryWriter() = default;
FtsDictionaryWriter::~FtsDictionaryWriter() = default;

Result<std::unique_ptr<FtsDictionaryWriter>> FtsDictionaryWriter::Create() {
  std::unique_ptr<FtsDictionaryWriter> writer(new FtsDictionaryWriter);
  for (auto *section : {&writer->terms_, &writer->index_, &writer->metadata_}) {
    auto spool = FtsSpool::Create();
    if (!spool) {
      return tl::make_unexpected(spool.error());
    }
    *section = std::move(*spool);
  }
  return writer;
}

Result<int> FtsDictionaryWriter::compare_previous(std::string_view term) {
  if (!previous_large_) {
    return format::CompareTerms(previous_, term);
  }
  const uint64_t common =
      std::min<uint64_t>(previous_large_->length, term.size());
  for (uint64_t offset = 0; offset < common;) {
    const auto n =
        static_cast<size_t>(std::min<uint64_t>(kChunk, common - offset));
    auto bytes = terms_->read(previous_large_->offset + offset, n);
    if (!bytes) {
      return tl::make_unexpected(bytes.error());
    }
    const int compared = format::CompareTerms(
        *bytes, term.substr(static_cast<size_t>(offset), n));
    if (compared != 0) {
      return compared;
    }
    offset += n;
  }
  return previous_large_->length < term.size()
             ? -1
             : (previous_large_->length > term.size() ? 1 : 0);
}

Status FtsDictionaryWriter::write_block(const format::DictionaryBlock &block) {
  auto encoded = format::EncodeDictionary(block);
  if (!encoded) {
    return encoded.error();
  }
  std::string index;
  format::AppendU64(block.first_ordinal, &index);
  format::AppendU32(
      block.oversized ? 1 : static_cast<uint32_t>(block.terms.size()), &index);
  format::AppendU32(0, &index);
  format::AppendU64(terms_->size(), &index);
  format::AppendU64(encoded->size(), &index);
  auto status = terms_->append(*encoded);
  if (status.ok()) {
    status = index_->append(index);
  }
  if (status.ok()) {
    ++blocks_;
  }
  return status;
}

Status FtsDictionaryWriter::flush_block() {
  if (pending_.terms.empty()) {
    return Status::OK();
  }
  auto status = write_block(pending_);
  if (status.ok()) {
    pending_ = {};
  }
  return status;
}

Status FtsDictionaryWriter::add(std::string_view term,
                                const format::TermMetadata &metadata) {
  if (!status_.ok()) {
    return status_;
  }
  if (finished_) {
    return Status::FailedPrecondition("FTS dictionary writer is finished");
  }
  if (count_ >= static_cast<uint64_t>(INT64_MAX) / format::kTermMetadataBytes) {
    return Exhausted();
  }
  auto encoded_meta = format::EncodeTermMetadata(metadata);
  if (!encoded_meta) {
    return encoded_meta.error();
  }
  if (count_ != 0) {
    auto compared = compare_previous(term);
    if (!compared) {
      return status_ = compared.error();
    }
    if (*compared >= 0) {
      return Status::InvalidArgument("FTS terms must be strictly sorted");
    }
  }
  format::DictionaryBlock singleton;
  singleton.first_ordinal = count_;
  bool oversized = term.size() > kChunk;
  if (!oversized) {
    singleton.terms.emplace_back(term);
    oversized = !format::EncodeDictionary(singleton).has_value();
  }
  if (oversized) {
    status_ = flush_block();
    if (!status_.ok()) {
      return status_;
    }
    singleton.terms.clear();
    singleton.oversized = format::ByteRange{terms_->size(), term.size()};
    status_ = terms_->append(term);
    if (status_.ok()) {
      status_ = write_block(singleton);
    }
    previous_large_ = singleton.oversized;
    previous_.clear();
  } else {
    if (pending_.terms.size() == format::kDictionaryBlockTerms) {
      status_ = flush_block();
    }
    if (!status_.ok()) {
      return status_;
    }
    if (pending_.terms.empty()) {
      pending_.first_ordinal = count_;
    }
    pending_.terms.emplace_back(term);
    if (!format::EncodeDictionary(pending_)) {
      pending_.terms.pop_back();
      status_ = flush_block();
      pending_ = std::move(singleton);
    }
    previous_.assign(term);
    previous_large_.reset();
  }
  if (status_.ok()) {
    status_ = metadata_->append(*encoded_meta);
  }
  if (status_.ok()) {
    status_ = fingerprint_.begin_term(term.size());
  }
  if (status_.ok()) {
    status_ = fingerprint_.append_term_bytes(term);
  }
  if (status_.ok()) {
    ++count_;
  }
  return status_;
}

Status FtsDictionaryWriter::add(uint64_t length, const FtsTermSource &source,
                                const format::TermMetadata &metadata) {
  if (!status_.ok()) {
    return status_;
  }
  if (finished_ || !source) {
    return Status::FailedPrecondition("Invalid streamed FTS dictionary input");
  }
  if (length > INT64_MAX ||
      count_ >= static_cast<uint64_t>(INT64_MAX) / format::kTermMetadataBytes) {
    return Exhausted();
  }
  auto read = [&](uint64_t offset, size_t count) -> Result<std::string> {
    auto bytes = source(offset, count);
    if (!bytes) {
      return bytes;
    }
    if (bytes->size() != count) {
      return tl::make_unexpected(
          Status::InternalError("Short FTS term source read"));
    }
    return bytes;
  };
  if (length <= kChunk) {
    auto bytes = read(0, static_cast<size_t>(length));
    return bytes ? add(*bytes, metadata) : (status_ = bytes.error());
  }
  auto encoded_meta = format::EncodeTermMetadata(metadata);
  if (!encoded_meta) {
    return encoded_meta.error();
  }
  if (count_ != 0) {
    const auto previous_length =
        previous_large_ ? previous_large_->length : previous_.size();
    const auto common = std::min(previous_length, length);
    int compared = 0;
    for (uint64_t offset = 0; offset < common && compared == 0;) {
      const size_t count = std::min<uint64_t>(kChunk, common - offset);
      auto current = read(offset, count);
      if (!current) {
        return status_ = current.error();
      }
      if (previous_large_) {
        auto previous = terms_->read(previous_large_->offset + offset, count);
        if (!previous) {
          return status_ = previous.error();
        }
        compared = format::CompareTerms(*previous, *current);
      } else {
        compared = format::CompareTerms(
            std::string_view(previous_).substr(offset, count), *current);
      }
      offset += count;
    }
    if (compared > 0 || (compared == 0 && previous_length >= length)) {
      return Status::InvalidArgument("FTS terms must be strictly sorted");
    }
  }
  status_ = flush_block();
  if (!status_.ok()) {
    return status_;
  }
  format::DictionaryBlock singleton;
  singleton.first_ordinal = count_;
  singleton.oversized = format::ByteRange{terms_->size(), length};
  status_ = fingerprint_.begin_term(length);
  for (uint64_t offset = 0; status_.ok() && offset < length;) {
    const size_t count = std::min<uint64_t>(kChunk, length - offset);
    auto bytes = read(offset, count);
    if (!bytes) {
      return status_ = bytes.error();
    }
    status_ = terms_->append(*bytes);
    if (status_.ok()) {
      status_ = fingerprint_.append_term_bytes(*bytes);
    }
    offset += count;
  }
  if (status_.ok()) {
    status_ = write_block(singleton);
  }
  if (status_.ok()) {
    status_ = metadata_->append(*encoded_meta);
  }
  if (status_.ok()) {
    previous_large_ = singleton.oversized;
    previous_.clear();
    ++count_;
  }
  return status_;
}

Result<FtsDictionarySummary> FtsDictionaryWriter::finish(
    FtsContainerWriter *container, uint32_t field_ordinal) {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (finished_ || !container || field_ordinal >= format::kMaxFields) {
    return tl::make_unexpected(
        Status::FailedPrecondition("Invalid FTS dictionary finish"));
  }
  auto fingerprint = fingerprint_.finish();
  if (!fingerprint) {
    return tl::make_unexpected(fingerprint.error());
  }
  status_ = flush_block();
  // Each upper level samples the first fence of every lower-level page.
  // Only one fixed-size record is resident while building the hierarchy.
  uint64_t level_offset = 0, level_count = blocks_;
  while (status_.ok() && level_count > kIndexPageRecords) {
    const uint64_t next_offset = index_->size();
    for (uint64_t i = 0; status_.ok() && i < level_count;
         i += kIndexPageRecords) {
      auto bytes = index_->read(level_offset + i * kIndexRecord, kIndexRecord);
      if (!bytes) {
        status_ = bytes.error();
      } else {
        status_ = index_->append(*bytes);
      }
    }
    level_offset = next_offset;
    level_count = (level_count - 1) / kIndexPageRecords + 1;
  }
  std::string header(kIndexMagic);
  format::AppendU32(1, &header);
  format::AppendU32(kIndexRecord, &header);
  format::AppendU64(blocks_, &header);
  format::AppendU64(count_, &header);
  const auto prefix = "field." + std::to_string(field_ordinal) + ".";
  if (status_.ok()) {
    status_ = terms_->copy_to(container, prefix + "terms");
  }
  if (status_.ok()) {
    status_ = index_->copy_to(container, prefix + "term_index", header);
  }
  if (status_.ok()) {
    status_ = metadata_->copy_to(container, prefix + "term_meta");
  }
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  finished_ = true;
  return FtsDictionarySummary{count_, *fingerprint};
}

struct FtsDictionaryReader::Record {
  uint64_t first{0};
  uint32_t count{0};
  format::ByteRange bytes;
};

struct FtsDictionaryReader::IndexPage {
  uint64_t first{UINT64_MAX};
  FtsReadRange bytes;
};

FtsDictionaryReader::~FtsDictionaryReader() {
  if (fence_bytes_) {
    fence_prefixes_.reset();
    ailego::MemoryLimitPool::get_instance().release_metadata(fence_bytes_);
  }
}

Result<FtsDictionaryReader::Pointer> FtsDictionaryReader::Open(
    FtsContainerReader::Pointer container, uint32_t field_ordinal) {
  if (!container || field_ordinal >= container->metadata().fields.size()) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS dictionary field"));
  }
  Pointer reader(new FtsDictionaryReader);
  reader->container_ = std::move(container);
  reader->field_ = field_ordinal;
  reader->prefix_ = "field." + std::to_string(field_ordinal) + ".";
  reader->count_ =
      reader->container_->metadata().fields[field_ordinal].term_count;
  auto terms = SectionLength(*reader->container_, reader->prefix_ + "terms");
  auto index =
      SectionLength(*reader->container_, reader->prefix_ + "term_index");
  auto metadata =
      SectionLength(*reader->container_, reader->prefix_ + "term_meta");
  if (!terms || !index || !metadata || *index < kIndexHeader ||
      reader->count_ > UINT64_MAX / format::kTermMetadataBytes ||
      *metadata != reader->count_ * format::kTermMetadataBytes) {
    return tl::make_unexpected(Corrupt());
  }
  reader->terms_length_ = *terms;
  auto header =
      reader->container_->read(reader->prefix_ + "term_index", 0, kIndexHeader);
  if (!header) {
    return tl::make_unexpected(header.error());
  }
  format::Decoder decoder(header->view());
  if (decoder.take(8) != kIndexMagic) {
    return tl::make_unexpected(Corrupt());
  }
  if (decoder.u32() != 1 || decoder.u32() != kIndexRecord) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS dictionary index format"));
  }
  reader->blocks_ = decoder.u64();
  if (decoder.u64() != reader->count_ || !decoder.done() ||
      reader->blocks_ > (UINT64_MAX - kIndexHeader) / kIndexRecord ||
      reader->blocks_ > reader->count_ ||
      (reader->blocks_ == 0 && (reader->count_ != 0 || *terms != 0)) ||
      (reader->count_ != 0 && reader->blocks_ == 0)) {
    return tl::make_unexpected(Corrupt());
  }
  uint64_t level_count = reader->blocks_, stride = 1;
  for (;;) {
    reader->levels_.push_back({reader->index_records_, level_count, stride});
    constexpr uint64_t limit = (UINT64_MAX - kIndexHeader) / kIndexRecord;
    if (level_count > limit - reader->index_records_) {
      return tl::make_unexpected(Corrupt());
    }
    reader->index_records_ += level_count;
    if (level_count <= kIndexPageRecords) {
      break;
    }
    level_count = (level_count - 1) / kIndexPageRecords + 1;
    stride *= kIndexPageRecords;
  }
  if (*index != kIndexHeader + reader->index_records_ * kIndexRecord) {
    return tl::make_unexpected(Corrupt());
  }
  if (reader->container_->paged_queries() && reader->blocks_ > 1) {
    // Learn separators from real lookups. Bound even very large dictionaries
    // by sampling every power-of-two stride. Charge the same pool as pages.
    auto &pool = ailego::MemoryLimitPool::get_instance();
    using Slot = std::atomic<uint64_t>;
    const size_t limit = std::min<size_t>(65536, pool.capacity() / 64);
    if (limit >= sizeof(Slot)) {
      uint64_t slots = reader->blocks_;
      while (slots > limit / sizeof(Slot)) {
        slots = (slots + 1) / 2;
        reader->fence_stride_ *= 2;
      }
      const size_t bytes = static_cast<size_t>(slots) * sizeof(Slot);
      if (pool.try_charge_metadata(bytes)) {
        reader->fence_prefixes_.reset(new (std::nothrow) Slot[slots]);
        if (reader->fence_prefixes_) {
          for (size_t i = 0; i < slots; ++i) {
            std::atomic_init(&reader->fence_prefixes_[i], uint64_t{0});
          }
          reader->fence_bytes_ = bytes;
        } else {
          pool.release_metadata(bytes);
        }
      }
    }
  }
  return reader;
}

Result<FtsDictionaryReader::Record> FtsDictionaryReader::record(
    uint64_t block, IndexPage *page) {
  return read_record(block, block, page);
}

Result<FtsDictionaryReader::Record> FtsDictionaryReader::fence(
    size_t level, uint64_t index, IndexPage *page) {
  if (level >= levels_.size() || index >= levels_[level].count) {
    return tl::make_unexpected(Corrupt());
  }
  return read_record(levels_[level].offset + index,
                     index * levels_[level].stride, page);
}

Result<FtsDictionaryReader::Record> FtsDictionaryReader::read_record(
    uint64_t location, uint64_t block, IndexPage *page) {
  if (block >= blocks_ || location >= index_records_) {
    return tl::make_unexpected(Corrupt());
  }
  const uint64_t first = location / kIndexPageRecords * kIndexPageRecords;
  if (page->first != first) {
    page->bytes = {};
    auto range = container_->read(
        prefix_ + "term_index", kIndexHeader + first * kIndexRecord,
        std::min<uint64_t>(kIndexPageRecords, index_records_ - first) *
            kIndexRecord);
    if (!range) {
      return tl::make_unexpected(range.error());
    }
    page->bytes = std::move(*range);
    page->first = first;
  }
  const auto offset = static_cast<size_t>(location - first) * kIndexRecord;
  format::Decoder decoder(page->bytes.view().substr(offset, kIndexRecord));
  Record result;
  result.first = decoder.u64();
  result.count = decoder.u32();
  if (decoder.u32() != 0) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS dictionary index flags"));
  }
  result.bytes.offset = decoder.u64();
  result.bytes.length = decoder.u64();
  if (!decoder.done() || result.count == 0 ||
      result.count > format::kDictionaryBlockTerms ||
      !format::ValidRange(result.first, result.count, count_) ||
      result.bytes.length < 16 || result.bytes.length > kChunk ||
      !format::ValidRange(result.bytes.offset, result.bytes.length,
                          terms_length_) ||
      (block == 0 && result.first != 0) ||
      (block == blocks_ - 1 && result.first + result.count != count_)) {
    return tl::make_unexpected(Corrupt());
  }
  return result;
}

Result<format::DictionaryBlock> FtsDictionaryReader::block(
    const Record &record) {
  auto bytes = container_->read(prefix_ + "terms", record.bytes.offset,
                                record.bytes.length);
  if (!bytes) {
    return tl::make_unexpected(bytes.error());
  }
  auto decoded = format::DecodeDictionary(bytes->view());
  if (!decoded) {
    return tl::make_unexpected(decoded.error());
  }
  if (decoded->first_ordinal != record.first ||
      (decoded->oversized ? 1 : decoded->terms.size()) != record.count ||
      (decoded->oversized &&
       (!format::ValidRange(decoded->oversized->offset,
                            decoded->oversized->length, terms_length_) ||
        decoded->oversized->offset + decoded->oversized->length !=
            record.bytes.offset))) {
    return tl::make_unexpected(Corrupt());
  }
  return decoded;
}

Result<FtsDictionaryEntry> FtsDictionaryReader::entry(
    const format::DictionaryBlock &block, size_t index, bool with_metadata) {
  FtsDictionaryEntry result;
  result.owner_ = shared_from_this();
  result.ordinal_ = block.first_ordinal + index;
  if (block.oversized) {
    result.external_ = block.oversized;
  } else {
    result.term_ = block.terms[index];
  }
  if (with_metadata) {
    auto bytes = container_->read(prefix_ + "term_meta",
                                  result.ordinal_ * format::kTermMetadataBytes,
                                  format::kTermMetadataBytes);
    if (!bytes) {
      return tl::make_unexpected(bytes.error());
    }
    auto metadata = format::DecodeTermMetadata(bytes->view());
    if (!metadata) {
      return tl::make_unexpected(metadata.error());
    }
    auto status = validate_metadata(*metadata);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
    result.metadata_ = *metadata;
  }
  return result;
}

Status FtsDictionaryReader::validate_metadata(
    const format::TermMetadata &term) const {
  if (term.document_frequency > container_->metadata().row_count ||
      (term.posting_tag == format::PostingTag::kInline &&
       term.inline_doc >= container_->metadata().row_count)) {
    return Corrupt();
  }
  const std::pair<const char *, format::ByteRange> ranges[] = {
      {"postings", term.postings},
      {"skips", term.skips},
      {"positions", term.positions},
      {"position_index", term.position_index}};
  for (const auto &range : ranges) {
    if (range.second.length != 0) {
      auto length = SectionLength(*container_, prefix_ + range.first);
      if (!length || !format::ValidRange(range.second.offset,
                                         range.second.length, *length)) {
        return Corrupt();
      }
    }
  }
  return Status::OK();
}

Result<std::string> FtsDictionaryEntry::read_term(uint64_t offset,
                                                  size_t length) const {
  if (!owner_ || !format::ValidRange(offset, length, term_size())) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS term range"));
  }
  if (length > kChunk) {
    return tl::make_unexpected(Exhausted());
  }
  if (!external_) {
    return term_.substr(static_cast<size_t>(offset), length);
  }
  auto bytes = owner_->container_->read(owner_->prefix_ + "terms",
                                        external_->offset + offset, length);
  if (!bytes) {
    return tl::make_unexpected(bytes.error());
  }
  return std::string(bytes->view());
}

Result<int> FtsDictionaryReader::compare(const FtsDictionaryEntry &entry,
                                         std::string_view term) {
  const uint64_t common = std::min<uint64_t>(entry.term_size(), term.size());
  for (uint64_t offset = 0; offset < common;) {
    const auto n =
        static_cast<size_t>(std::min<uint64_t>(kChunk, common - offset));
    auto bytes = entry.read_term(offset, n);
    if (!bytes) {
      return tl::make_unexpected(bytes.error());
    }
    const int compared = format::CompareTerms(
        *bytes, term.substr(static_cast<size_t>(offset), n));
    if (compared != 0) {
      return compared;
    }
    offset += n;
  }
  return entry.term_size() < term.size()
             ? -1
             : (entry.term_size() > term.size() ? 1 : 0);
}

Result<int> FtsDictionaryReader::compare(const FtsDictionaryEntry &left,
                                         const FtsDictionaryEntry &right) {
  const uint64_t common = std::min(left.term_size(), right.term_size());
  for (uint64_t offset = 0; offset < common;) {
    const auto n =
        static_cast<size_t>(std::min<uint64_t>(kChunk, common - offset));
    auto a = left.read_term(offset, n);
    if (!a) {
      return tl::make_unexpected(a.error());
    }
    auto b = right.read_term(offset, n);
    if (!b) {
      return tl::make_unexpected(b.error());
    }
    const int compared = format::CompareTerms(*a, *b);
    if (compared != 0) {
      return compared;
    }
    offset += n;
  }
  return left.term_size() < right.term_size()
             ? -1
             : (left.term_size() > right.term_size() ? 1 : 0);
}

Result<std::optional<FtsDictionaryEntry>> FtsDictionaryReader::next(
    Scan *scan) {
  if (scan->remaining_ == 0) {
    if (scan->leaf_ == blocks_) {
      if (scan->ordinal_ != count_) {
        return tl::make_unexpected(Corrupt());
      }
      return std::optional<FtsDictionaryEntry>{};
    }
    scan->term_meta_ = {};
    scan->block_ = {};
    IndexPage directory;
    auto leaf = record(scan->leaf_, &directory);
    if (!leaf || leaf->first != scan->ordinal_) {
      return tl::make_unexpected(leaf ? Corrupt() : leaf.error());
    }
    auto decoded = block(*leaf);
    if (!decoded) {
      return tl::make_unexpected(decoded.error());
    }
    auto bytes = container_->read(prefix_ + "term_meta",
                                  leaf->first * format::kTermMetadataBytes,
                                  leaf->count * format::kTermMetadataBytes);
    if (!bytes) {
      return tl::make_unexpected(bytes.error());
    }
    scan->term_meta_ = std::move(*bytes);
    scan->block_ = std::move(*decoded);
    scan->remaining_ = leaf->count;
    ++scan->leaf_;
  }
  const size_t index = scan->ordinal_ - scan->block_.first_ordinal;
  auto result = entry(scan->block_, index, false);
  if (!result) {
    return tl::make_unexpected(result.error());
  }
  auto metadata = format::DecodeTermMetadata(scan->term_meta_.view().substr(
      index * format::kTermMetadataBytes, format::kTermMetadataBytes));
  if (!metadata) {
    return tl::make_unexpected(metadata.error());
  }
  auto status = validate_metadata(*metadata);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  result->metadata_ = *metadata;
  ++scan->ordinal_;
  --scan->remaining_;
  return std::optional<FtsDictionaryEntry>{std::move(*result)};
}

Result<FtsDictionaryEntry> FtsDictionaryReader::at(uint64_t ordinal) {
  if (ordinal >= count_) {
    return tl::make_unexpected(NotFound());
  }
  IndexPage page;
  uint64_t selected = 0;
  for (size_t level = levels_.size(); level-- > 0;) {
    const uint64_t start = selected * kIndexPageRecords;
    uint64_t lo = start;
    uint64_t hi = std::min(start + kIndexPageRecords, levels_[level].count);
    while (lo < hi) {
      const uint64_t mid = lo + (hi - lo) / 2;
      auto found = fence(level, mid, &page);
      if (!found) {
        return tl::make_unexpected(found.error());
      }
      if (found->first <= ordinal) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo == start) {
      return tl::make_unexpected(Corrupt());
    }
    selected = lo - 1;
  }
  auto found = record(selected, &page);
  if (!found) {
    return tl::make_unexpected(found.error());
  }
  if (ordinal < found->first || ordinal - found->first >= found->count) {
    return tl::make_unexpected(Corrupt());
  }
  auto decoded = block(*found);
  if (!decoded) {
    return tl::make_unexpected(decoded.error());
  }
  return entry(*decoded, static_cast<size_t>(ordinal - found->first), true);
}

Result<int> FtsDictionaryReader::compare_fence(size_t level, uint64_t index,
                                               IndexPage *page,
                                               std::string_view term,
                                               uint64_t prefix) {
  const uint64_t leaf = index * levels_[level].stride;
  std::atomic<uint64_t> *slot = nullptr;
  if (fence_prefixes_ && leaf % fence_stride_ == 0) {
    slot = &fence_prefixes_[leaf / fence_stride_];
    const uint64_t cached = slot->load(std::memory_order_relaxed);
    if (cached != 0 && cached != prefix) {
      return cached < prefix ? -1 : 1;
    }
  }
  auto found = fence(level, index, page);
  if (!found) {
    return tl::make_unexpected(found.error());
  }
  auto decoded = block(*found);
  if (!decoded) {
    return tl::make_unexpected(decoded.error());
  }
  if (!decoded->oversized) {
    const auto &key = decoded->terms.front();
    if (slot) {
      slot->store(FencePrefix(key), std::memory_order_relaxed);
    }
    return format::CompareTerms(key, term);
  }
  auto first = entry(*decoded, 0, false);
  return compare(*first, term);
}

Result<FtsDictionaryEntry> FtsDictionaryReader::seek_ge(std::string_view term) {
  if (blocks_ == 0) {
    return tl::make_unexpected(NotFound());
  }
  IndexPage page;
  uint64_t selected = 0;
  const uint64_t prefix = FencePrefix(term);
  for (size_t level = levels_.size(); level-- > 0;) {
    const uint64_t start = selected * kIndexPageRecords;
    uint64_t lo = start;
    uint64_t hi = std::min(start + kIndexPageRecords, levels_[level].count);
    while (lo < hi) {
      const uint64_t mid = lo + (hi - lo) / 2;
      auto compared = compare_fence(level, mid, &page, term, prefix);
      if (!compared) {
        return tl::make_unexpected(compared.error());
      }
      if (*compared <= 0) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    selected = lo == start ? start : lo - 1;
  }
  auto found = record(selected, &page);
  if (!found) {
    return tl::make_unexpected(found.error());
  }
  auto decoded = block(*found);
  if (!decoded) {
    return tl::make_unexpected(decoded.error());
  }
  for (size_t i = 0; i < found->count; ++i) {
    auto candidate = entry(*decoded, i, false);
    auto compared = compare(*candidate, term);
    if (!compared) {
      return tl::make_unexpected(compared.error());
    }
    if (*compared >= 0) {
      return entry(*decoded, i, true);
    }
  }
  return at(found->first + found->count);
}

Result<FtsDictionaryEntry> FtsDictionaryReader::seek(std::string_view term) {
  auto found = seek_ge(term);
  if (!found) {
    return found;
  }
  auto compared = compare(*found, term);
  if (!compared) {
    return tl::make_unexpected(compared.error());
  }
  if (*compared != 0) {
    return tl::make_unexpected(NotFound());
  }
  return found;
}

Status FtsDictionaryReader::verify() {
  IndexPage page;
  for (size_t level = 1; level < levels_.size(); ++level) {
    for (uint64_t i = 0; i < levels_[level].count; ++i) {
      auto upper = fence(level, i, &page);
      if (!upper) {
        return upper.error();
      }
      auto leaf = record(i * levels_[level].stride, &page);
      if (!leaf) {
        return leaf.error();
      }
      if (upper->first != leaf->first || upper->count != leaf->count ||
          upper->bytes.offset != leaf->bytes.offset ||
          upper->bytes.length != leaf->bytes.length) {
        return Corrupt();
      }
    }
  }
  format::DictionaryFingerprint fingerprint;
  std::optional<FtsDictionaryEntry> previous;
  uint64_t ordinal = 0, end = 0;
  for (uint64_t i = 0; i < blocks_; ++i) {
    auto found = record(i, &page);
    if (!found) {
      return found.error();
    }
    auto decoded = block(*found);
    if (!decoded) {
      return decoded.error();
    }
    const uint64_t start =
        decoded->oversized ? decoded->oversized->offset : found->bytes.offset;
    if (found->first != ordinal || start != end) {
      return Corrupt();
    }
    end = found->bytes.offset + found->bytes.length;
    for (size_t j = 0; j < found->count; ++j) {
      auto current = entry(*decoded, j, true);
      if (!current) {
        return current.error();
      }
      if (previous) {
        auto compared = compare(*previous, *current);
        if (!compared) {
          return compared.error();
        }
        if (*compared >= 0) {
          return Corrupt();
        }
      }
      auto status = fingerprint.begin_term(current->term_size());
      if (!status.ok()) {
        return status;
      }
      for (uint64_t offset = 0; offset < current->term_size();) {
        const auto n = static_cast<size_t>(
            std::min<uint64_t>(kChunk, current->term_size() - offset));
        auto bytes = current->read_term(offset, n);
        if (!bytes) {
          return bytes.error();
        }
        status = fingerprint.append_term_bytes(*bytes);
        if (!status.ok()) {
          return status;
        }
        offset += n;
      }
      previous = std::move(*current);
      ++ordinal;
    }
  }
  auto digest = fingerprint.finish();
  if (!digest) {
    return digest.error();
  }
  if (ordinal != count_ || end != terms_length_ ||
      *digest != container_->metadata().fields[field_].dictionary_fingerprint) {
    return Corrupt();
  }
  return Status::OK();
}

}  // namespace zvec::fts
