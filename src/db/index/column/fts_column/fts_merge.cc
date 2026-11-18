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

#include "fts_merge.h"
#include <algorithm>

namespace zvec::fts {
namespace {

Status Corrupt() {
  return Status::InternalError("Inconsistent FTS merge source");
}

Result<int> Compare(const FtsSourceTerm &left, const FtsSourceTerm &right) {
  const uint64_t common = std::min(left.length, right.length);
  for (uint64_t offset = 0; offset < common;) {
    const size_t count = static_cast<size_t>(
        std::min<uint64_t>(format::kDictionaryBlockBytes, common - offset));
    auto a = left.bytes(offset, count);
    auto b = right.bytes(offset, count);
    if (!a || !b) {
      return tl::make_unexpected(!a ? a.error() : b.error());
    }
    if (a->size() != count || b->size() != count) {
      return tl::make_unexpected(Corrupt());
    }
    const int order = a->compare(*b);
    if (order != 0) {
      return order;
    }
    offset += count;
  }
  return left.length < right.length ? -1
                                    : (left.length == right.length ? 0 : 1);
}

struct Input {
  FtsFieldSource::Ptr source;
  uint64_t offset;
  FtsSourceTerms::Ptr terms;
  std::optional<FtsSourceTerm> head;
};

Status Advance(Input *input) {
  // Release the previous value before reading the next one. Legacy adapters
  // may retain complete values, so overlapping old/new heads is not free.
  input->head.reset();
  auto next = input->terms->next();
  if (!next) {
    return next.error();
  }
  input->head = std::move(*next);
  if (input->head && (!input->head->reader || !input->head->bytes)) {
    return Corrupt();
  }
  return Status::OK();
}

Status CopyPostings(Input &input, const roaring::Roaring &deletions,
                    FtsFieldWriter *writer, bool *started,
                    const FtsSourceTerm &term) {
  auto postings = input.head->reader->postings();
  if (!postings) {
    return postings.error();
  }
  uint64_t count = 0;
  uint32_t previous = 0;
  for (;;) {
    auto doc = (*postings)->next();
    if (!doc) {
      return doc.error();
    }
    if (*doc == UINT32_MAX) {
      break;
    }
    if (*doc >= input.source->statistics().rows ||
        (count != 0 && *doc <= previous)) {
      return Corrupt();
    }
    previous = *doc;
    ++count;
    const uint32_t scan_row = static_cast<uint32_t>(input.offset + *doc);
    if (deletions.contains(scan_row)) {
      continue;
    }
    auto payload = (*postings)->payload();
    auto document = input.source->document(*doc);
    if (!payload || !document) {
      return !payload ? payload.error() : document.error();
    }
    if (!document->present || document->length != payload->length) {
      return Corrupt();
    }
    auto positions = input.head->reader->positions(*doc);
    if (!positions) {
      return positions.error();
    }
    if (!*started) {
      auto status = writer->begin_term(term.length, term.bytes);
      if (!status.ok()) {
        return status;
      }
      *started = true;
    }
    const auto target =
        static_cast<uint32_t>(scan_row - deletions.rank(scan_row));
    auto status = writer->add_posting(target, payload->tf,
                                      [&]() { return (*positions)->next(); });
    if (!status.ok()) {
      return status;
    }
  }
  return count == input.head->reader->df() ? Status::OK() : Corrupt();
}

}  // namespace

Result<format::FieldMetadata> MergeFtsField(
    const std::vector<FtsFieldSource::Ptr> &inputs,
    const roaring::Roaring &deletions, FtsMergePolicy policy,
    format::FieldMetadata metadata, FtsContainerWriter *container,
    uint32_t field_ordinal, BM25Params params) {
  if (!container || inputs.empty() ||
      (policy == FtsMergePolicy::kRewrite &&
       (inputs.size() != 1 || !deletions.isEmpty()))) {
    return tl::make_unexpected(Status::InvalidArgument("Invalid FTS merge"));
  }
  // In addition to these heads, one legacy position value (at most 64 MiB)
  // may be retained while copying a posting. Custom positions remain paged.
  uint64_t workspace = 128 * 1024 * 1024;
  if (inputs.size() > 64) {
    return tl::make_unexpected(
        Status{StatusCode::RESOURCE_EXHAUSTED, "Too many FTS merge inputs"});
  }
  std::vector<Input> cursors;
  cursors.reserve(inputs.size());
  uint64_t rows = 0;
  for (const auto &source : inputs) {
    if (!source ||
        !format::ValidRange(rows, source->statistics().rows, UINT32_MAX)) {
      return tl::make_unexpected(Status::InvalidArgument("Invalid FTS rows"));
    }
    if (source->term_workspace_limit() > workspace) {
      return tl::make_unexpected(Status{StatusCode::RESOURCE_EXHAUSTED,
                                        "FTS merge workspace exceeded"});
    }
    workspace -= source->term_workspace_limit();
    cursors.push_back({source, rows, {}, {}});
    rows += source->statistics().rows;
  }
  if (!deletions.isEmpty() && deletions.maximum() >= rows) {
    return tl::make_unexpected(
        Status::InvalidArgument("FTS deletion outside forward row domain"));
  }
  const uint64_t target_rows = rows - deletions.cardinality();
  metadata.scoring_document_count =
      policy == FtsMergePolicy::kRewrite
          ? inputs.front()->statistics().scoring_documents
          : target_rows;
  metadata.total_tokens = metadata.present_count = 0;
  // Validate each source's exact statistics and establish target counts before
  // constructing the writer/scorer. No per-row remap or length vector exists.
  for (const auto &input : cursors) {
    uint64_t tokens = 0, present = 0;
    for (uint64_t row = 0; row < input.source->statistics().rows; ++row) {
      auto doc = input.source->document(static_cast<uint32_t>(row));
      if (!doc) {
        return tl::make_unexpected(doc.error());
      }
      if (!doc->present && doc->length != 0) {
        return tl::make_unexpected(Corrupt());
      }
      tokens += doc->length;
      present += doc->present ? 1 : 0;
      if (!deletions.contains(static_cast<uint32_t>(input.offset + row))) {
        metadata.total_tokens += doc->length;
        metadata.present_count += doc->present ? 1 : 0;
      }
    }
    if (tokens != input.source->statistics().total_tokens ||
        present != input.source->statistics().present) {
      return tl::make_unexpected(Corrupt());
    }
  }
  auto writer =
      FtsFieldWriter::Create(target_rows, std::move(metadata), params);
  if (!writer) {
    return tl::make_unexpected(writer.error());
  }
  uint32_t output_row = 0;
  for (auto &input : cursors) {
    for (uint64_t row = 0; row < input.source->statistics().rows; ++row) {
      if (deletions.contains(static_cast<uint32_t>(input.offset + row))) {
        continue;
      }
      auto doc = input.source->document(static_cast<uint32_t>(row));
      if (!doc) {
        return tl::make_unexpected(doc.error());
      }
      auto status =
          (*writer)->add_document(output_row++, doc->present, doc->length);
      if (!status.ok()) {
        return tl::make_unexpected(status);
      }
    }
    auto terms = input.source->terms();
    if (!terms) {
      return tl::make_unexpected(terms.error());
    }
    input.terms = std::move(*terms);
    auto status = Advance(&input);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  auto status = (*writer)->finish_documents();
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  for (;;) {
    size_t minimum = cursors.size();
    for (size_t i = 0; i < cursors.size(); ++i) {
      if (!cursors[i].head) {
        continue;
      }
      if (minimum == cursors.size()) {
        minimum = i;
        continue;
      }
      auto order = Compare(*cursors[i].head, *cursors[minimum].head);
      if (!order) {
        return tl::make_unexpected(order.error());
      }
      if (*order < 0) {
        minimum = i;
      }
    }
    if (minimum == cursors.size()) {
      break;
    }
    std::vector<size_t> matching;
    bool started = false;
    for (size_t i = 0; i < cursors.size(); ++i) {
      if (!cursors[i].head) {
        continue;
      }
      auto order = Compare(*cursors[i].head, *cursors[minimum].head);
      if (!order) {
        return tl::make_unexpected(order.error());
      }
      if (*order == 0) {
        matching.push_back(i);
        status = CopyPostings(cursors[i], deletions, writer->get(), &started,
                              *cursors[minimum].head);
        if (!status.ok()) {
          return tl::make_unexpected(status);
        }
      }
    }
    // A term with all postings deleted must not leave a DF=0 dictionary entry.
    if (started) {
      status = (*writer)->end_term();
      if (!status.ok()) {
        return tl::make_unexpected(status);
      }
    }
    for (size_t i : matching) {
      status = Advance(&cursors[i]);
      if (!status.ok()) {
        return tl::make_unexpected(status);
      }
    }
  }
  return (*writer)->finish(container, field_ordinal);
}

}  // namespace zvec::fts
