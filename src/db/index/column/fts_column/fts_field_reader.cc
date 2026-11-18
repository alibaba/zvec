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
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include "fts_field.h"

namespace zvec::fts {
namespace {

constexpr uint64_t kFullLengthBlockBytes = 4 + 16 + 128 * 4;

bool SupportsFiniteBounds() {
#if defined(__FAST_MATH__) || defined(_M_FP_FAST) || FLT_EVAL_METHOD != 0 || \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
  return false;
#else
  return std::numeric_limits<float>::is_iec559 &&
         std::numeric_limits<float>::digits == 24 &&
         std::numeric_limits<double>::digits >= 53 &&
         std::fegetround() == FE_TONEAREST;
#endif
}

Status Corrupt() {
  return Status::InternalError(
      "Malformed FTS field/posting reference or statistics");
}

bool Contains(format::ByteRange outer, format::ByteRange inner) {
  return inner.offset >= outer.offset &&
         format::ValidRange(inner.offset - outer.offset, inner.length,
                            outer.length);
}

Status LoadLengthBlock(FtsSectionReader *reader, uint64_t ordinal,
                       uint64_t rows, format::DocLengthBlockView *block,
                       uint64_t *loaded) {
  const auto count =
      static_cast<uint32_t>(std::min<uint64_t>(128, rows - ordinal * 128));
  auto bytes = reader->read(ordinal * kFullLengthBlockBytes,
                            4 + (count + 7) / 8 + count * 4);
  if (!bytes) {
    return bytes.error();
  }
  auto decoded = format::DocLengthBlockView::Open(*bytes);
  if (!decoded) {
    return decoded.error();
  }
  if (decoded->count() != count) {
    return Corrupt();
  }
  *block = *decoded;
  *loaded = ordinal;
  return Status::OK();
}

Result<FtsDocumentLength> ReadLength(FtsSectionReader *reader, uint32_t row,
                                     uint64_t rows,
                                     format::DocLengthBlockView *block,
                                     uint64_t *loaded) {
  if (row >= rows) {
    return tl::make_unexpected(
        Status::InvalidArgument("FTS row is outside segment"));
  }
  const uint64_t ordinal = row / 128;
  if (*loaded != ordinal) {
    auto status = LoadLengthBlock(reader, ordinal, rows, block, loaded);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  auto entry = block->at(row % 128);
  if (!entry) {
    return tl::make_unexpected(entry.error());
  }
  return FtsDocumentLength{entry->present, entry->length};
}

}  // namespace

Result<FtsFieldReader::Pointer> FtsFieldReader::Open(
    FtsContainerReader::Pointer container, uint32_t field, BM25Params params) {
  if (!container || field >= container->metadata().fields.size() ||
      !std::isfinite(params.k1) || params.k1 < 0 || !std::isfinite(params.b) ||
      params.b < 0 || params.b > 1) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS field reader arguments"));
  }
  const auto prefix = "field." + std::to_string(field) + ".";
  for (const char *name :
       {"postings", "skips", "positions", "position_index"}) {
    auto section = FtsSectionReader::Open(container, prefix + name);
    if (!section) {
      return tl::make_unexpected(section.error());
    }
  }
  auto lengths = FtsSectionReader::Open(container, prefix + "doclen");
  if (!lengths) {
    return tl::make_unexpected(lengths.error());
  }
  const uint64_t rows = container->metadata().row_count;
  const uint64_t tail = rows % 128;
  const uint64_t expected = rows / 128 * kFullLengthBlockBytes +
                            (tail == 0 ? 0 : 4 + (tail + 7) / 8 + tail * 4);
  if ((*lengths)->size() != expected) {
    return tl::make_unexpected(Corrupt());
  }
  auto dictionary = FtsDictionaryReader::Open(container, field);
  if (!dictionary) {
    return tl::make_unexpected(dictionary.error());
  }
  Pointer reader(new FtsFieldReader(params));
  reader->container_ = std::move(container);
  reader->dictionary_ = std::move(*dictionary);
  reader->field_ = field;
  reader->prefix_ = prefix;
  reader->scorer_.update_stats(reader->metadata().scoring_document_count,
                               reader->metadata().total_tokens);
  return reader;
}

const format::FieldMetadata &FtsFieldReader::metadata() const {
  return container_->metadata().fields[field_];
}

Result<FtsDictionaryEntry> FtsFieldReader::lookup(std::string_view term) {
  return dictionary_->seek(term);
}

Result<FtsDictionaryEntry> FtsFieldReader::term_at(uint64_t ordinal) {
  return dictionary_->at(ordinal);
}

Result<FtsDocumentLength> FtsFieldReader::document(uint32_t row) {
  auto lengths = FtsSectionReader::Open(container_, prefix_ + "doclen");
  if (!lengths) {
    return tl::make_unexpected(lengths.error());
  }
  format::DocLengthBlockView block;
  uint64_t loaded = UINT64_MAX;
  return ReadLength(lengths->get(), row, container_->metadata().row_count,
                    &block, &loaded);
}

float FtsFieldReader::score(uint64_t df, uint32_t tf, uint32_t length,
                            float boost) const {
  return scorer_.score_with_idf(scorer_.idf(df), tf, length, boost);
}

bool FtsFieldReader::usable_bounds(float boost) const {
  return bounds_verified_.load(std::memory_order_acquire) && boost == 1.0f &&
         SupportsFiniteBounds();
}

float FtsFieldReader::conservative_max_score(uint64_t df) const {
  const float infinity = std::numeric_limits<float>::infinity();
  if (!SupportsFiniteBounds()) {
    return infinity;
  }
  const auto stats = scorer_.stats();
  const float idf = scorer_.idf(df);
  if (!std::isfinite(idf)) {
    return infinity;
  }
  if (stats.total_docs == 0 || idf <= 0) {
    return 0;
  }
  const float average = stats.avg_doc_len();
  if (!(average > 0) || !std::isfinite(average)) {
    return infinity;
  }
  // For k1 >= 0, 0 <= b <= 1 and a positive average, the rounded
  // denominator is >= float(tf). Thus normalization is bounded by k1+1,
  // with four round-to-nearest factors (1+u): k1+1, numerator, division,
  // and final score multiplication. u=2^-24 is the FP32 unit roundoff,
  // not an empirical epsilon. A subnormal normalization is below 1 and
  // cannot approach this bound; add one denormal quantum for score underflow.
  // Compute each bound operation upward in double, then upward to float.
  // This uses current scorer/statistics and never trusts persisted max scores.
  const double double_infinity = std::numeric_limits<double>::infinity();
  auto upward = [&](double value) {
    return std::nextafter(value, double_infinity);
  };
  double upper = upward(static_cast<double>(scorer_.params().k1) + 1.0);
  const double factor = 1.0 + std::numeric_limits<float>::epsilon() / 2.0;
  for (int rounding = 0; rounding < 4; ++rounding) {
    upper = upward(upper * factor);
  }
  // uint32 TF rounds to at most 2^32. If its numerator could overflow,
  // infinity is required even when the real-valued saturation is finite.
  if (upper > std::numeric_limits<float>::max() / 4294967296.0) {
    return infinity;
  }
  upper = upward(upper * idf);
  upper = upward(upper + std::numeric_limits<float>::denorm_min());
  return upper > std::numeric_limits<float>::max()
             ? infinity
             : std::nextafter(static_cast<float>(upper), infinity);
}

float FtsFieldReader::max_score(const FtsDictionaryEntry &term,
                                float boost) const {
  if (term.owner_.get() != dictionary_.get() || boost != 1.0f ||
      !SupportsFiniteBounds()) {
    return std::numeric_limits<float>::infinity();
  }
  return usable_bounds(boost)
             ? term.metadata().max_score
             : conservative_max_score(term.metadata().document_frequency);
}

Result<std::unique_ptr<FtsPostingCursor>> FtsFieldReader::postings(
    const FtsDictionaryEntry &term) {
  if (term.owner_.get() != dictionary_.get()) {
    return tl::make_unexpected(Status::InvalidArgument(
        "FTS term belongs to another field/generation"));
  }
  const auto &meta = term.metadata();
  const uint64_t blocks = (meta.document_frequency + 127) / 128;
  if ((meta.posting_tag == format::PostingTag::kPacked &&
       meta.skips.length != blocks * format::kPostingSkipBytes) ||
      (meta.posting_tag == format::PostingTag::kVarint &&
       (meta.postings.length > 1288 || meta.position_index.length > 1376))) {
    return tl::make_unexpected(Corrupt());
  }
  std::unique_ptr<FtsPostingCursor> cursor(new FtsPostingCursor);
  cursor->owner_ = shared_from_this();
  cursor->term_ = term;
  cursor->block_count_ = blocks;
  cursor->conservative_max_score_ =
      conservative_max_score(meta.document_frequency);
  const std::pair<std::unique_ptr<FtsSectionReader> *, const char *>
      sections[] = {{&cursor->postings_, "postings"},
                    {&cursor->skips_, "skips"},
                    {&cursor->lengths_, "doclen"},
                    {&cursor->position_index_, "position_index"}};
  for (const auto &section : sections) {
    auto opened = FtsSectionReader::Open(container_, prefix_ + section.second);
    if (!opened) {
      return tl::make_unexpected(opened.error());
    }
    *section.first = std::move(*opened);
  }
  return cursor;
}

Result<format::PostingSkip> FtsPostingCursor::skip(uint64_t ordinal) {
  const auto &term = term_.metadata();
  if (term.posting_tag != format::PostingTag::kPacked ||
      ordinal >= block_count_) {
    return tl::make_unexpected(Corrupt());
  }
  // The current block already owns a validated skip record. Adjacent loads
  // reuse it for their previous-block continuity check.
  if (started_ && ordinal == block_ordinal_) {
    return skip_;
  }
  auto bytes =
      skips_->read(term.skips.offset + ordinal * format::kPostingSkipBytes,
                   format::kPostingSkipBytes);
  if (!bytes) {
    return tl::make_unexpected(bytes.error());
  }
  auto decoded = format::DecodePostingSkip(*bytes);
  if (!decoded) {
    return tl::make_unexpected(decoded.error());
  }
  if (decoded->last_doc >= owner_->container_->metadata().row_count ||
      decoded->count !=
          std::min<uint64_t>(128, term.document_frequency - ordinal * 128) ||
      decoded->postings.length > 1032 ||
      decoded->position_index.length > 1376 ||
      !Contains(term.postings, decoded->postings) ||
      !Contains(term.position_index, decoded->position_index) ||
      (ordinal == 0 && decoded->postings.offset != term.postings.offset) ||
      (ordinal + 1 == block_count_ &&
       decoded->postings.offset + decoded->postings.length !=
           term.postings.offset + term.postings.length)) {
    return tl::make_unexpected(Corrupt());
  }
  return decoded;
}

Status FtsPostingCursor::load(uint64_t ordinal) {
  const auto &term = term_.metadata();
  format::PostingSkip current;
  if (term.posting_tag == format::PostingTag::kInline) {
    if (ordinal != 0) {
      return Corrupt();
    }
    block_.count = 1;
    block_.docs[0] = term.inline_doc;
    block_.tfs[0] = term.inline_tf;
    current.last_doc = term.inline_doc;
    current.count = 1;
    current.max_score = term.max_score;
  } else {
    uint32_t previous_last = 0;
    if (term.posting_tag == format::PostingTag::kPacked) {
      auto found = skip(ordinal);
      if (!found) {
        return found.error();
      }
      current = *found;
      if (ordinal != 0) {
        auto previous = skip(ordinal - 1);
        if (!previous) {
          return previous.error();
        }
        if (previous->last_doc >= current.last_doc ||
            previous->postings.offset + previous->postings.length !=
                current.postings.offset ||
            previous->position_index.offset + previous->position_index.length !=
                current.position_index.offset) {
          return Corrupt();
        }
        previous_last = previous->last_doc;
      }
    } else {
      if (ordinal != 0) {
        return Corrupt();
      }
      current.count = static_cast<uint32_t>(term.document_frequency);
      current.max_score = term.max_score;
      current.postings = term.postings;
      current.position_index = term.position_index;
    }
    auto bytes = postings_->read(current.postings.offset,
                                 static_cast<size_t>(current.postings.length));
    if (!bytes) {
      return bytes.error();
    }
    auto decoded = format::DecodePostings(
        *bytes, term.posting_tag == format::PostingTag::kPacked
                    ? format::PostingEncoding::kPacked
                    : format::PostingEncoding::kVarint);
    if (!decoded) {
      return decoded.error();
    }
    if (decoded->count != current.count ||
        decoded->docs[decoded->count - 1] >=
            owner_->container_->metadata().row_count ||
        (ordinal != 0 && decoded->docs[0] <= previous_last) ||
        (term.posting_tag == format::PostingTag::kPacked &&
         decoded->docs[decoded->count - 1] != current.last_doc)) {
      return Corrupt();
    }
    block_ = *decoded;
    current.last_doc = block_.docs[block_.count - 1];
  }
  skip_ = current;
  block_ordinal_ = ordinal;
  within_ = 0;
  started_ = true;
  return Status::OK();
}

Result<std::optional<FtsPosting>> FtsPostingCursor::next() {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (eof_) {
    return std::optional<FtsPosting>{};
  }
  if (!started_) {
    status_ = load(0);
  } else if (within_ + 1 < block_.count) {
    ++within_;
  } else if (block_ordinal_ + 1 < block_count_) {
    status_ = load(block_ordinal_ + 1);
  } else {
    eof_ = true;
    return std::optional<FtsPosting>{};
  }
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  return std::optional<FtsPosting>{{block_.docs[within_], block_.tfs[within_]}};
}

Result<std::optional<FtsPosting>> FtsPostingCursor::advance(uint32_t target) {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (eof_) {
    return std::optional<FtsPosting>{};
  }
  if (!started_ &&
      term_.metadata().posting_tag != format::PostingTag::kPacked) {
    status_ = load(0);
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
  }
  if (!started_ || skip_.last_doc < target) {
    uint64_t lo = started_ ? block_ordinal_ + 1 : 0;
    uint64_t hi = block_count_;
    if (started_) {
      // WAND frequently advances just beyond the current block. First locate
      // a nearby bracket; a full-list binary search touches distant skip pages
      // even for an adjacent block, evicting useful verified pages.
      uint64_t step = 1;
      while (lo < hi) {
        const uint64_t probe = lo + std::min(step, hi - lo) - 1;
        auto found = skip(probe);
        if (!found) {
          return tl::make_unexpected(status_ = found.error());
        }
        if (found->last_doc >= target) {
          hi = probe;
          break;
        }
        lo = probe + 1;
        step = std::min(step, (block_count_ + 1) / 2) * 2;
      }
    }
    while (lo < hi) {
      const uint64_t mid = lo + (hi - lo) / 2;
      auto found = skip(mid);
      if (!found) {
        return tl::make_unexpected(status_ = found.error());
      }
      if (found->last_doc < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo == block_count_) {
      eof_ = true;
      return std::optional<FtsPosting>{};
    }
    status_ = load(lo);
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
  }
  const auto *begin = block_.docs.data();
  if (begin[within_] < target) {
    // WAND usually seeks to the current or an adjacent posting. Bracket the
    // target from the current position before searching a larger remainder.
    uint32_t lo = within_ + 1;
    uint32_t hi = block_.count - 1;  // The loaded block ends at/after target.
    uint32_t step = 1;
    while (lo < hi) {
      const uint32_t probe = lo + std::min(step - 1, hi - lo);
      if (begin[probe] >= target) {
        hi = probe;
        break;
      }
      lo = probe + 1;
      step *= 2;
    }
    within_ = static_cast<uint32_t>(
        std::lower_bound(begin + lo, begin + hi, target) - begin);
  }
  return std::optional<FtsPosting>{{block_.docs[within_], block_.tfs[within_]}};
}

Result<FtsPostingBatch> FtsPostingCursor::next_batch() {
  auto first = next();
  if (!first) {
    return tl::make_unexpected(first.error());
  }
  if (!first->has_value()) {
    return FtsPostingBatch{};
  }
  const uint32_t start = within_;
  const uint64_t rows = owner_->container_->metadata().row_count;
  uint32_t i = start;
  while (i < block_.count) {
    if (block_.docs[i] >= rows) {
      return tl::make_unexpected(
          status_ = Status::InvalidArgument("FTS row is outside segment"));
    }
    const uint64_t ordinal = block_.docs[i] / 128;
    // Sorted posting IDs let us consume this validated length view before
    // loading another block, which may replace the reader's borrowed bytes.
    if (length_block_ordinal_ != ordinal) {
      status_ = LoadLengthBlock(lengths_.get(), ordinal, rows, &length_block_,
                                &length_block_ordinal_);
      if (!status_.ok()) {
        return tl::make_unexpected(status_);
      }
    }
    do {
      auto length = length_block_.at(block_.docs[i] % 128);
      if (!length) {
        return tl::make_unexpected(status_ = length.error());
      }
      if (!length->present || block_.tfs[i] > length->length) {
        return tl::make_unexpected(status_ = Corrupt());
      }
      batch_lengths_[i - start] = length->length;
      ++i;
    } while (i < block_.count && block_.docs[i] / 128 == ordinal);
  }
  within_ = block_.count - 1;
  return FtsPostingBatch{block_.count - start, block_.docs.data() + start,
                         block_.tfs.data() + start, batch_lengths_.data()};
}

Result<uint32_t> FtsPostingCursor::doc_length() {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (!started_ || eof_) {
    return tl::make_unexpected(
        Status::FailedPrecondition("FTS posting cursor has no document"));
  }
  auto length = ReadLength(lengths_.get(), block_.docs[within_],
                           owner_->container_->metadata().row_count,
                           &length_block_, &length_block_ordinal_);
  if (!length) {
    return tl::make_unexpected(status_ = length.error());
  }
  if (!length->present || block_.tfs[within_] > length->length) {
    return tl::make_unexpected(status_ = Corrupt());
  }
  return length->length;
}

Result<format::ByteRange> FtsPostingCursor::position_range() {
  const auto &term = term_.metadata();
  if (term.posting_tag == format::PostingTag::kInline) {
    return term.positions;
  }
  auto bytes =
      position_index_->read(skip_.position_index.offset,
                            static_cast<size_t>(skip_.position_index.length));
  if (!bytes) {
    return tl::make_unexpected(bytes.error());
  }
  format::Decoder header(*bytes);
  (void)header.u32();
  (void)header.u32();
  if (header.u32() != block_.count) {
    return tl::make_unexpected(Corrupt());
  }
  auto range = format::LocatePositionBytes(*bytes, within_);
  if (!range) {
    return tl::make_unexpected(range.error());
  }
  if (!Contains(term.positions, *range)) {
    return tl::make_unexpected(Corrupt());
  }
  return range;
}

Result<std::unique_ptr<FtsPositionCursor>> FtsPostingCursor::positions() {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (!started_ || eof_) {
    return tl::make_unexpected(
        Status::FailedPrecondition("FTS posting cursor has no document"));
  }
  if (term_.metadata().position_tag == format::PositionTag::kInline) {
    return FtsPositionCursor::Inline(owner_->container_,
                                     term_.metadata().inline_position);
  }
  auto range = position_range();
  if (!range) {
    return tl::make_unexpected(status_ = range.error());
  }
  auto cursor = FtsPositionCursor::Open(owner_->container_, owner_->field_,
                                        *range, block_.tfs[within_]);
  if (!cursor) {
    return tl::make_unexpected(status_ = cursor.error());
  }
  return cursor;
}

uint32_t FtsPostingCursor::block_last_doc() const {
  return started_ && !eof_ ? skip_.last_doc : UINT32_MAX;
}

float FtsPostingCursor::block_max_score(float boost) const {
  if (!started_ || eof_ || boost != 1.0f || !SupportsFiniteBounds()) {
    return std::numeric_limits<float>::infinity();
  }
  return owner_->usable_bounds(boost) ? skip_.max_score
                                      : conservative_max_score_;
}

Result<FtsPostingBound> FtsPostingCursor::bound(uint32_t target, float boost) {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (!owner_->usable_bounds(boost)) {
    const float maximum = boost == 1.0f && SupportsFiniteBounds()
                              ? conservative_max_score_
                              : std::numeric_limits<float>::infinity();
    return FtsPostingBound{maximum, UINT32_MAX};
  }
  if (bound_cache_ && target >= bound_start_ &&
      target <= bound_cache_->last_doc) {
    return *bound_cache_;
  }
  const auto &term = term_.metadata();
  if (term.posting_tag == format::PostingTag::kInline) {
    return target > term.inline_doc
               ? FtsPostingBound{0, UINT32_MAX}
               : FtsPostingBound{term.max_score, term.inline_doc};
  }
  if (term.posting_tag == format::PostingTag::kVarint) {
    return FtsPostingBound{term.max_score, UINT32_MAX};
  }
  // Skip records are paged. Keep only the target interval, independently of
  // the decoded posting block so SHOULD positioning is never disturbed.
  uint64_t lo = 0, hi = block_count_;
  FtsPostingBound result{0, UINT32_MAX};
  while (lo < hi) {
    const uint64_t mid = lo + (hi - lo) / 2;
    auto found = skip(mid);
    if (!found) {
      return tl::make_unexpected(status_ = found.error());
    }
    if (found->last_doc < target) {
      lo = mid + 1;
    } else {
      result = {found->max_score, found->last_doc};
      hi = mid;
    }
  }
  bound_start_ = target;
  bound_cache_ = result;
  return result;
}

Status FtsFieldReader::verify() {
  bounds_verified_.store(false, std::memory_order_release);
  auto status = dictionary_->verify();
  if (!status.ok()) {
    return status;
  }
  const auto &meta = metadata();
  bool bounds = meta.scoring_identity == FtsScoringIdentity(scorer_.params()) &&
                meta.bound_statistics_identity ==
                    FtsBoundStatisticsIdentity(scorer_.params(),
                                               meta.scoring_document_count,
                                               meta.total_tokens) &&
                SupportsFiniteBounds();
  auto totals = FtsSpool::Create();
  if (!totals) {
    return totals.error();
  }
  const uint64_t rows = container_->metadata().row_count;
  const std::string zeros(8192, '\0');
  for (uint64_t row = 0; row < rows;) {
    const uint64_t count = std::min<uint64_t>(1024, rows - row);
    status = (*totals)->append(
        std::string_view(zeros).substr(0, static_cast<size_t>(count * 8)));
    if (!status.ok()) {
      return status;
    }
    row += count;
  }
  for (uint64_t ordinal = 0; ordinal < meta.term_count; ++ordinal) {
    auto term = dictionary_->at(ordinal);
    if (!term) {
      return term.error();
    }
    auto cursor = postings(*term);
    if (!cursor) {
      return cursor.error();
    }
    const auto &term_meta = term->metadata();
    const float idf = scorer_.idf(term_meta.document_frequency);
    uint64_t df = 0, tf = 0, position_end = term_meta.positions.offset;
    while (true) {
      auto posting = (*cursor)->next();
      if (!posting) {
        return posting.error();
      }
      if (!posting->has_value()) {
        break;
      }
      const auto doc = **posting;
      auto length = (*cursor)->doc_length();
      if (!length) {
        return length.error();
      }
      auto total = (*totals)->read(static_cast<uint64_t>(doc.doc) * 8, 8);
      if (!total) {
        return total.error();
      }
      format::Decoder decoder(*total);
      const uint64_t accumulated = decoder.u64();
      if (!format::ValidRange(accumulated, doc.tf, *length)) {
        return Corrupt();
      }
      std::string updated;
      format::AppendU64(accumulated + doc.tf, &updated);
      status =
          (*totals)->overwrite(static_cast<uint64_t>(doc.doc) * 8, updated);
      if (!status.ok()) {
        return status;
      }
      if (term_meta.position_tag == format::PositionTag::kExternal) {
        if ((*cursor)->within_ == 0 &&
            term_meta.posting_tag != format::PostingTag::kInline) {
          const auto range = (*cursor)->skip_.position_index;
          auto bytes = (*cursor)->position_index_->read(
              range.offset, static_cast<size_t>(range.length));
          if (!bytes) {
            return bytes.error();
          }
          auto index = format::DecodePositionIndex(*bytes);
          if (!index || index->count != (*cursor)->block_.count) {
            return index ? Corrupt() : index.error();
          }
        }
        auto range = (*cursor)->position_range();
        if (!range) {
          return range.error();
        }
        if (range->offset != position_end) {
          return Corrupt();
        }
        position_end += range->length;
      }
      auto positions = (*cursor)->positions();
      if (!positions) {
        return positions.error();
      }
      uint64_t count = 0;
      while (true) {
        auto position = (*positions)->next();
        if (!position) {
          return position.error();
        }
        if (!position->has_value()) {
          break;
        }
        ++count;
      }
      if (count != doc.tf) {
        return Corrupt();
      }
      const float value = scorer_.score_with_idf(idf, doc.tf, *length);
      if (!std::isfinite(value) || value < 0) {
        bounds = false;
      } else if (value > term_meta.max_score ||
                 value > (*cursor)->skip_.max_score) {
        bounds = false;
      }
      ++df;
      tf += doc.tf;
    }
    if (df != term_meta.document_frequency || tf != term_meta.total_frequency ||
        position_end !=
            term_meta.positions.offset + term_meta.positions.length) {
      return Corrupt();
    }
  }
  auto lengths = FtsSectionReader::Open(container_, prefix_ + "doclen");
  if (!lengths) {
    return lengths.error();
  }
  format::DocLengthBlockView block;
  uint64_t loaded = UINT64_MAX, present = 0, tokens = 0;
  for (uint64_t row = 0; row < rows; ++row) {
    auto length = ReadLength(lengths->get(), static_cast<uint32_t>(row), rows,
                             &block, &loaded);
    if (!length) {
      return length.error();
    }
    auto total = (*totals)->read(row * 8, 8);
    if (!total) {
      return total.error();
    }
    format::Decoder decoder(*total);
    if (decoder.u64() != length->length) {
      return Corrupt();
    }
    present += length->present ? 1 : 0;
    tokens += length->length;
  }
  if (present != meta.present_count || tokens != meta.total_tokens ||
      (meta.scoring_document_count == 0 && tokens != 0)) {
    return Corrupt();
  }
  bounds_verified_.store(bounds, std::memory_order_release);
  return Status::OK();
}

}  // namespace zvec::fts
