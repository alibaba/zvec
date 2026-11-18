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

#include "fts_legacy_reader.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <roaring/roaring.h>
#include "posting/bitpacked_posting_list.h"
#include "fts_utils.h"

namespace zvec::fts {
namespace {

constexpr size_t kRetainedValueBudget = 64 * 1024 * 1024;

Status Corrupt() {
  return Status::InternalError("Malformed or missing legacy FTS payload");
}

Status ReadError(const rocksdb::Status &status) {
  if (status.IsCorruption() || status.IsNotFound()) {
    return Corrupt();
  }
  return {StatusCode::UNAVAILABLE,
          "Legacy FTS read failed: " + status.ToString()};
}

Status Exhausted() {
  return {StatusCode::RESOURCE_EXHAUSTED, "Legacy FTS read workspace exceeded"};
}

Status ValidatePacked(std::string_view bytes) {
  using Packed = BitPackedPostingList;
  if (bytes.size() < sizeof(Packed::Header)) {
    return Corrupt();
  }
  Packed::Header header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (header.version != Packed::VERSION) {
    return Status::NotSupported("Unknown legacy FTS posting version");
  }
  const uint64_t blocks = (uint64_t{header.num_docs} + 127) / 128;
  if (header.num_blocks != blocks ||
      blocks > (bytes.size() - sizeof(header)) / sizeof(Packed::BlockMeta)) {
    return Corrupt();
  }
  size_t end = sizeof(header) + blocks * sizeof(Packed::BlockMeta);
  uint32_t previous = 0;
  for (uint32_t i = 0; i < header.num_blocks; ++i) {
    Packed::BlockMeta meta;
    std::memcpy(&meta, bytes.data() + sizeof(header) + i * sizeof(meta),
                sizeof(meta));
    end = (end + 15) & ~size_t{15};
    if (meta.block_offset != end || end > bytes.size() ||
        sizeof(Packed::BlockHeader) > bytes.size() - end ||
        meta.max_doc_id == UINT32_MAX || !std::isfinite(meta.block_max_score) ||
        meta.block_max_score < 0) {
      return Corrupt();
    }
    Packed::BlockHeader block;
    std::memcpy(&block, bytes.data() + end, sizeof(block));
    const uint32_t count = std::min(128u, header.num_docs - i * 128);
    if (block.num_docs != count || block.bitwidth_id > 32 ||
        block.bitwidth_tf > 32 || block.bitwidth_dl > 32 ||
        block.min_doc_id > meta.max_doc_id ||
        (i != 0 && block.min_doc_id <= previous) ||
        block.block_max_score != meta.block_max_score) {
      return Corrupt();
    }
    end += sizeof(block);
    const size_t payload = Packed::packed_byte_size(block.bitwidth_id, count) +
                           Packed::packed_byte_size(block.bitwidth_tf, count) +
                           Packed::packed_byte_size(block.bitwidth_dl, count);
    if (payload > bytes.size() - end) {
      return Corrupt();
    }
    end += payload;
    previous = meta.max_doc_id;
  }
  return ((end + 15) & ~size_t{15}) == bytes.size() ? Status::OK() : Corrupt();
}

}  // namespace

struct LegacyFtsReader::SideLease {
  explicit SideLease(std::shared_ptr<const LegacyFtsReader> reader)
      : owner(std::move(reader)) {
    ++owner->side_users_;
    active = !owner->side_dropped_.load();
    if (!active) {
      --owner->side_users_;
    }
  }
  ~SideLease() {
    if (active) {
      --owner->side_users_;
    }
  }
  std::shared_ptr<const LegacyFtsReader> owner;
  bool active{false};
};

class LegacyFtsReader::Term final : public FtsReadTerm,
                                    public std::enable_shared_from_this<Term> {
 public:
  uint64_t df() const override {
    return df_;
  }
  float max_score(float boost) const override {
    // Stored legacy bounds use a different multiplication order from boosted
    // scoring. Do not turn them into potentially unsafe finite boosted bounds.
    return boost == 1.0f ? maximum_ : std::numeric_limits<float>::infinity();
  }
  Result<FtsReadPosting::Ptr> postings() const override;
  Result<FtsReadPosition::Ptr> positions(uint32_t doc) const override {
    auto result = owner_->positions(
        {std::const_pointer_cast<Term>(shared_from_this())}, doc);
    if (!result) {
      return tl::make_unexpected(result.error());
    }
    return std::move((*result)[0]);
  }
  std::shared_ptr<const LegacyFtsReader> owner_;
  std::shared_ptr<SideLease> side_;
  std::string key_;
  std::shared_ptr<rocksdb::PinnableSlice> value_;
  std::shared_ptr<roaring_bitmap_t> bitmap_;
  uint64_t df_{0};
  float maximum_{std::numeric_limits<float>::infinity()};
};

class LegacyFtsReader::Posting final : public FtsReadPosting {
 public:
  explicit Posting(std::shared_ptr<const Term> term) : term_(std::move(term)) {
    if (term_->bitmap_) {
      roaring_init_iterator(term_->bitmap_.get(), &roaring_);
    } else if (packed_.open(term_->value_->data(), term_->value_->size()) !=
               0) {
      status_ = Corrupt();
    }
  }
  Result<uint32_t> next() override {
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    if (!term_->bitmap_) {
      return doc_ = packed_.next_doc();
    }
    // Stream the bitmap itself, without materializing an N*4 DocID array.
    if (started_) {
      roaring_advance_uint32_iterator(&roaring_);
    }
    started_ = true;
    return doc_ = roaring_.has_value ? roaring_.current_value : UINT32_MAX;
  }
  Result<uint32_t> advance(uint32_t target) override {
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    if (!term_->bitmap_) {
      return doc_ = packed_.advance(target);
    }
    started_ = true;
    const bool found =
        roaring_move_uint32_iterator_equalorlarger(&roaring_, target);
    return doc_ = found ? roaring_.current_value : UINT32_MAX;
  }
  Result<FtsReadPayload> payload() override {
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    if (doc_ == UINT32_MAX) {
      return tl::make_unexpected(Corrupt());
    }
    // BitPacked's TF and exact length are inline; no side-CF I/O is needed.
    if (!term_->bitmap_) {
      return FtsReadPayload{packed_.term_freq(), packed_.doc_len()};
    }
    if (payload_doc_ == doc_) {
      return payload_;
    }
    const auto &source = term_->owner_->source_;
    auto *tf_cf = term_->side_->active ? source.frequencies : nullptr;
    auto *length_cf = term_->side_->active ? source.lengths : nullptr;
    auto tf = read_u32(tf_cf, make_doc_term_key(term_->key_, doc_));
    if (!tf) {
      return tl::make_unexpected(status_ = tf.error());
    }
    std::string key(sizeof(doc_), '\0');
    std::memcpy(key.data(), &doc_, sizeof(doc_));
    auto length = read_u32(length_cf, key);
    if (!length) {
      return tl::make_unexpected(status_ = length.error());
    }
    if (*tf == 0 || (tf_cf && length_cf && *tf > *length)) {
      return tl::make_unexpected(status_ = Corrupt());
    }
    payload_doc_ = doc_;
    return payload_ = {*tf, *length};
  }
  Result<Bound> bound(uint32_t target, float boost) override {
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    if (!term_->bitmap_ && boost == 1.0f) {
      auto info = packed_.block_max_info_for(target);
      return Bound{info.block_max_score, info.block_last_doc};
    }
    return Bound{term_->max_score(boost), UINT32_MAX};
  }

 private:
  Result<uint32_t> read_u32(rocksdb::ColumnFamilyHandle *cf,
                            const std::string &key) {
    // Preserve the existing explicit null-CF reader/transient contract.
    // An actual I/O failure is not a default value.
    if (!cf) {
      return 1u;
    }
    const auto &source = term_->owner_->source_;
    std::string value;
    auto status =
        source.context->db_->Get(source.context->read_opts_, cf, key, &value);
    if (!status.ok()) {
      return tl::make_unexpected(ReadError(status));
    }
    if (value.size() != sizeof(uint32_t)) {
      return tl::make_unexpected(Corrupt());
    }
    uint32_t result;
    std::memcpy(&result, value.data(), sizeof(result));
    return result;
  }
  std::shared_ptr<const Term> term_;
  BitPackedPostingIterator packed_;
  roaring_uint32_iterator_t roaring_{};
  bool started_{false};
  uint32_t doc_{UINT32_MAX}, payload_doc_{UINT32_MAX};
  FtsReadPayload payload_{0, 0};
  Status status_;
};

Result<FtsReadPosting::Ptr> LegacyFtsReader::Term::postings() const {
  return std::make_unique<Posting>(shared_from_this());
}

class LegacyFtsReader::Position final : public FtsReadPosition {
 public:
  Position(std::shared_ptr<const LegacyFtsReader> owner,
           std::shared_ptr<rocksdb::PinnableSlice> value)
      : owner_(std::move(owner)),
        value_(std::move(value)),
        decoder_(std::string_view(value_->data(), value_->size())) {}
  Result<std::optional<uint32_t>> next() override {
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    if (decoder_.done()) {
      current_.reset();
      return current_;
    }
    const uint64_t delta = decoder_.varint();
    if (!decoder_.ok() || delta > UINT32_MAX - previous_) {
      return tl::make_unexpected(status_ = Corrupt());
    }
    previous_ += static_cast<uint32_t>(delta);
    return current_ = previous_;
  }
  Result<std::optional<uint32_t>> seek_ge(uint32_t target) override {
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    if (current_ && *current_ >= target) {
      return current_;
    }
    while (true) {
      auto result = next();
      if (!result || !result->has_value() || **result >= target) {
        return result;
      }
    }
  }

 private:
  std::shared_ptr<const LegacyFtsReader> owner_;
  std::shared_ptr<rocksdb::PinnableSlice> value_;
  format::Decoder decoder_;
  uint32_t previous_{0};
  std::optional<uint32_t> current_;
  Status status_;
};

LegacyFtsReader::LegacyFtsReader(LegacyFtsSource source, BM25ScorerPtr scorer)
    : source_(std::move(source)), scorer_(std::move(scorer)) {}

Result<LegacyFtsReader::Ptr> LegacyFtsReader::Create(LegacyFtsSource source,
                                                     BM25ScorerPtr scorer) {
  if (!source.context || !source.context->db_ || !source.postings ||
      !source.positions || !scorer ||
      (source.owner && source.owner.get() != source.context)) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid legacy FTS source"));
  }
  return Ptr(new LegacyFtsReader(std::move(source), std::move(scorer)));
}

BM25ScorerPtr LegacyFtsReader::scorer() const {
  return scorer_;
}

Result<FtsReadTerm::Ptr> LegacyFtsReader::decode(
    std::string_view key, std::shared_ptr<rocksdb::PinnableSlice> value) const {
  if (value->empty()) {
    return tl::make_unexpected(Corrupt());
  }
  if (value->size() > kRetainedValueBudget) {
    return tl::make_unexpected(Exhausted());
  }
  auto term = std::make_shared<Term>();
  term->owner_ = shared_from_this();
  term->key_.assign(key);
  term->value_ = std::move(value);
  if (BitPackedPostingList::is_bitpacked_format(term->value_->data(),
                                                term->value_->size())) {
    // The historical iterator trusts block offsets and bit widths. Validate
    // those bounds before exposing any cursor to its unchecked SIMD decoder.
    auto status = ValidatePacked(
        std::string_view(term->value_->data(), term->value_->size()));
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
    BitPackedPostingIterator iterator;
    if (iterator.open(term->value_->data(), term->value_->size()) != 0) {
      return tl::make_unexpected(Corrupt());
    }
    term->df_ = iterator.cost();
    term->maximum_ = iterator.max_score();
  } else {
    auto *bitmap = roaring_bitmap_portable_deserialize_safe(
        term->value_->data(), term->value_->size());
    if (!bitmap) {
      return tl::make_unexpected(Corrupt());
    }
    term->bitmap_ =
        std::shared_ptr<roaring_bitmap_t>(bitmap, roaring_bitmap_free);
    term->df_ = roaring_bitmap_get_cardinality(bitmap);
    if (term->df_ != 0 && roaring_bitmap_maximum(bitmap) == UINT32_MAX) {
      return tl::make_unexpected(Corrupt());
    }
    term->side_ = std::make_shared<SideLease>(shared_from_this());
    auto *max_cf = term->side_->active ? source_.max_tf : nullptr;
    if (max_cf && source_.lengths && source_.frequencies) {
      std::string max_value;
      auto status = source_.context->db_->Get(source_.context->read_opts_,
                                              max_cf, term->key_, &max_value);
      if (!status.ok() && !status.IsNotFound()) {
        return tl::make_unexpected(ReadError(status));
      }
      if (status.ok()) {
        if (max_value.size() != sizeof(uint32_t)) {
          return tl::make_unexpected(Corrupt());
        }
        uint32_t maximum;
        std::memcpy(&maximum, max_value.data(), sizeof(maximum));
        const uint32_t length = min_length_.load(std::memory_order_relaxed);
        term->maximum_ = scorer_->score(term->df_, maximum,
                                        length == UINT32_MAX ? 1 : length);
      }
    }
  }
  return term->df_ == 0 ? FtsReadTerm::Ptr{} : std::move(term);
}

Result<FtsReadTerm::Ptr> LegacyFtsReader::lookup(std::string_view term) const {
  auto value = std::make_shared<rocksdb::PinnableSlice>();
  auto status = source_.context->db_->Get(
      source_.context->read_opts_, source_.postings,
      rocksdb::Slice(term.data(), term.size()), value.get());
  if (status.IsNotFound()) {
    return FtsReadTerm::Ptr{};
  }
  if (!status.ok()) {
    return tl::make_unexpected(ReadError(status));
  }
  return decode(term, std::move(value));
}

Result<std::vector<FtsReadTerm::Ptr>> LegacyFtsReader::lookup_many(
    const std::vector<std::string_view> &terms) const {
  std::vector<rocksdb::Slice> keys;
  keys.reserve(terms.size());
  for (auto term : terms) {
    keys.emplace_back(term.data(), term.size());
  }
  std::vector<rocksdb::ColumnFamilyHandle *> cfs(terms.size(),
                                                 source_.postings);
  std::vector<rocksdb::PinnableSlice> values(terms.size());
  std::vector<rocksdb::Status> statuses(terms.size());
  if (!terms.empty()) {
    source_.context->db_->MultiGet(source_.context->read_opts_, terms.size(),
                                   cfs.data(), keys.data(), values.data(),
                                   statuses.data());
  }
  std::vector<FtsReadTerm::Ptr> result;
  result.reserve(terms.size());
  size_t retained = 0;
  for (size_t i = 0; i < terms.size(); ++i) {
    if (statuses[i].IsNotFound()) {
      result.push_back(nullptr);
      continue;
    }
    if (!statuses[i].ok()) {
      return tl::make_unexpected(ReadError(statuses[i]));
    }
    if (values[i].size() > kRetainedValueBudget - retained) {
      return tl::make_unexpected(Exhausted());
    }
    retained += values[i].size();
    auto term =
        decode(terms[i],
               std::make_shared<rocksdb::PinnableSlice>(std::move(values[i])));
    if (!term) {
      return tl::make_unexpected(term.error());
    }
    result.push_back(std::move(*term));
  }
  return result;
}

Result<std::vector<FtsReadPosition::Ptr>> LegacyFtsReader::positions(
    const std::vector<FtsReadTerm::Ptr> &terms, uint32_t doc) const {
  // Deduplicate storage reads, but preserve an independent logical cursor for
  // each repeated phrase term. Values stay pinned; decoded vectors are not
  // built.
  std::unordered_map<std::string, size_t> seen;
  std::vector<std::string> keys;
  std::vector<size_t> slots;
  for (const auto &handle : terms) {
    auto term = std::dynamic_pointer_cast<Term>(handle);
    if (!term || term->owner_.get() != this) {
      return tl::make_unexpected(
          Status::InvalidArgument("Foreign FTS term handle"));
    }
    auto [found, inserted] = seen.emplace(term->key_, keys.size());
    if (inserted) {
      keys.push_back(make_doc_term_key(term->key_, doc));
    }
    slots.push_back(found->second);
  }
  std::vector<rocksdb::Slice> slices;
  for (const auto &key : keys) {
    slices.emplace_back(key);
  }
  std::vector<rocksdb::ColumnFamilyHandle *> cfs(keys.size(),
                                                 source_.positions);
  std::vector<rocksdb::PinnableSlice> values(keys.size());
  std::vector<rocksdb::Status> statuses(keys.size());
  if (!keys.empty()) {
    source_.context->db_->MultiGet(source_.context->read_opts_, keys.size(),
                                   cfs.data(), slices.data(), values.data(),
                                   statuses.data());
  }
  std::vector<std::shared_ptr<rocksdb::PinnableSlice>> pinned;
  size_t retained = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!statuses[i].ok()) {
      return tl::make_unexpected(ReadError(statuses[i]));
    }
    if (values[i].empty()) {
      return tl::make_unexpected(Corrupt());
    }
    if (values[i].size() > kRetainedValueBudget - retained) {
      return tl::make_unexpected(Exhausted());
    }
    retained += values[i].size();
    pinned.push_back(
        std::make_shared<rocksdb::PinnableSlice>(std::move(values[i])));
  }
  std::vector<FtsReadPosition::Ptr> result;
  result.reserve(terms.size());
  for (size_t slot : slots) {
    result.push_back(
        std::make_unique<Position>(shared_from_this(), pinned[slot]));
  }
  return result;
}

void LegacyFtsReader::observe_length(uint32_t length) {
  if (length != 0) {
    uint32_t current = min_length_.load(std::memory_order_relaxed);
    while (length < current &&
           !min_length_.compare_exchange_weak(current, length,
                                              std::memory_order_relaxed)) {
    }
  }
}

void LegacyFtsReader::reset_side_cfs() {
  side_dropped_.store(true);
  while (side_users_.load() != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace zvec::fts
