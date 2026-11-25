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

#include "fts_reader.h"

namespace zvec::fts {
namespace {

class CustomPosition final : public FtsReadPosition {
 public:
  explicit CustomPosition(std::unique_ptr<FtsPositionCursor> cursor)
      : cursor_(std::move(cursor)) {}
  Result<std::optional<uint32_t>> next() override {
    return cursor_->next();
  }
  Result<std::optional<uint32_t>> seek_ge(uint32_t target) override {
    return cursor_->seek_ge(target);
  }

 private:
  std::unique_ptr<FtsPositionCursor> cursor_;
};

class CustomPosting final : public FtsReadPosting {
 public:
  explicit CustomPosting(std::unique_ptr<FtsPostingCursor> cursor)
      : cursor_(std::move(cursor)) {}
  Result<uint32_t> next() override {
    return accept(cursor_->next());
  }
  Result<uint32_t> advance(uint32_t target) override {
    return accept(cursor_->advance(target));
  }
  Result<FtsReadPayload> payload() override {
    auto length = cursor_->doc_length();
    if (!length) {
      return tl::make_unexpected(length.error());
    }
    return FtsReadPayload{tf_, *length};
  }
  Result<Bound> bound(uint32_t target, float boost) override {
    return cursor_->bound(target, boost);
  }
  Result<FtsReadPosition::Ptr> current_positions(const FtsReadTerm &,
                                                 uint32_t) override {
    auto positions = cursor_->positions();
    if (!positions) {
      return tl::make_unexpected(positions.error());
    }
    return std::make_unique<CustomPosition>(std::move(*positions));
  }
  Result<FtsPostingBatch> next_batch() override {
    auto batch = cursor_->next_batch();
    if (batch && batch->count != 0) {
      tf_ = batch->tfs[batch->count - 1];
    }
    return batch;
  }

 private:
  Result<uint32_t> accept(Result<std::optional<FtsPosting>> result) {
    if (!result) {
      return tl::make_unexpected(result.error());
    }
    if (!result->has_value()) {
      return UINT32_MAX;
    }
    tf_ = (**result).tf;
    return (**result).doc;
  }
  std::unique_ptr<FtsPostingCursor> cursor_;
  uint32_t tf_{0};
};

class CustomTerm final : public FtsReadTerm {
 public:
  CustomTerm(FtsFieldReader::Pointer field, FtsDictionaryEntry term)
      : field_(std::move(field)), term_(std::move(term)) {}
  uint64_t df() const override {
    return term_.metadata().document_frequency;
  }
  float max_score(float boost) const override {
    return field_->max_score(term_, boost);
  }
  Result<FtsReadPosting::Ptr> postings() const override {
    auto cursor = field_->postings(term_);
    if (!cursor) {
      return tl::make_unexpected(cursor.error());
    }
    return std::make_unique<CustomPosting>(std::move(*cursor));
  }
  Result<FtsReadPosition::Ptr> positions(uint32_t doc) const override {
    auto cursor = field_->postings(term_);
    if (!cursor) {
      return tl::make_unexpected(cursor.error());
    }
    auto posting = (*cursor)->advance(doc);
    if (!posting) {
      return tl::make_unexpected(posting.error());
    }
    if (!posting->has_value() || (**posting).doc != doc) {
      return tl::make_unexpected(
          Status::InternalError("FTS phrase document has no posting"));
    }
    auto positions = (*cursor)->positions();
    if (!positions) {
      return tl::make_unexpected(positions.error());
    }
    return std::make_unique<CustomPosition>(std::move(*positions));
  }

 private:
  FtsFieldReader::Pointer field_;
  FtsDictionaryEntry term_;
};

class CustomReader final : public FtsSegmentReader {
 public:
  explicit CustomReader(FtsFieldReader::Pointer field)
      : field_(std::move(field)),
        scorer_(std::make_shared<BM25Scorer>(field_->scoring_params())) {
    scorer_->update_stats(field_->metadata().scoring_document_count,
                          field_->metadata().total_tokens);
  }
  BM25ScorerPtr scorer() const override {
    return scorer_;
  }
  bool supports_batches() const override {
    return true;
  }
  Result<FtsReadTerm::Ptr> lookup(std::string_view term) const override {
    auto found = field_->lookup(term);
    if (!found) {
      if (found.error().code() == StatusCode::NOT_FOUND) {
        return FtsReadTerm::Ptr{};
      }
      return tl::make_unexpected(found.error());
    }
    return OpenCustomFtsTerm(field_, std::move(*found));
  }

 private:
  FtsFieldReader::Pointer field_;
  BM25ScorerPtr scorer_;
};

}  // namespace

Result<FtsReadPosition::Ptr> FtsReadPosting::current_positions(
    const FtsReadTerm &term, uint32_t doc) {
  return term.positions(doc);
}

Result<std::vector<FtsReadTerm::Ptr>> FtsSegmentReader::lookup_many(
    const std::vector<std::string_view> &terms) const {
  std::vector<FtsReadTerm::Ptr> result;
  result.reserve(terms.size());
  for (auto term : terms) {
    auto found = lookup(term);
    if (!found) {
      return tl::make_unexpected(found.error());
    }
    result.push_back(std::move(*found));
  }
  return result;
}

Result<std::vector<FtsReadPosition::Ptr>> FtsSegmentReader::positions(
    const std::vector<FtsReadTerm::Ptr> &terms, uint32_t doc) const {
  std::vector<FtsReadPosition::Ptr> result;
  result.reserve(terms.size());
  for (const auto &term : terms) {
    auto positions = term->positions(doc);
    if (!positions) {
      return tl::make_unexpected(positions.error());
    }
    result.push_back(std::move(*positions));
  }
  return result;
}

Result<FtsSegmentReader::Ptr> OpenCustomFtsReader(
    FtsFieldReader::Pointer field) {
  if (!field) {
    return tl::make_unexpected(Status::InvalidArgument("Missing FTS field"));
  }
  return std::make_shared<CustomReader>(std::move(field));
}

Result<FtsReadTerm::Ptr> OpenCustomFtsTerm(FtsFieldReader::Pointer field,
                                           FtsDictionaryEntry term) {
  if (!field) {
    return tl::make_unexpected(Status::InvalidArgument("Missing FTS field"));
  }
  return std::make_shared<CustomTerm>(std::move(field), std::move(term));
}

}  // namespace zvec::fts
