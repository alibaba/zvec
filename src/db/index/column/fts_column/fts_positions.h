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

#include <functional>
#include "fts_stream.h"

namespace zvec::fts {

using FtsPositionSource = std::function<Result<std::optional<uint32_t>>()>;

// TF<=128 uses one counted position block. Longer documents use fixed
// 128-position subblocks and a 32-byte document header referencing a paged,
// fixed-record directory in position_index. Posting length directories share
// that section. TF selects the representation; it is never a sentinel value.
class FtsPositionsWriter {
 public:
  static Result<std::unique_ptr<FtsPositionsWriter>> Create();
  Result<format::ByteRange> append_document(uint32_t tf,
                                            const FtsPositionSource &source);
  Result<format::ByteRange> append_block_index(
      const format::PositionIndexBlock &block);
  Status finish(FtsContainerWriter *container, uint32_t field_ordinal);

 private:
  FtsPositionsWriter() = default;
  std::unique_ptr<FtsSpool> data_, index_;
  bool finished_{false};
  Status status_;
};

// One current decoded subblock and two charged page caches, independent of TF.
// Optional-empty is EOF; UINT32_MAX is a valid position. Failures are sticky,
// and every returned value retains a live file owner through this cursor.
class FtsPositionCursor {
 public:
  static Result<std::unique_ptr<FtsPositionCursor>> Open(
      FtsContainerReader::Pointer container, uint32_t field_ordinal,
      format::ByteRange document, uint32_t tf);
  static std::unique_ptr<FtsPositionCursor> Inline(
      FtsContainerReader::Pointer owner, uint32_t position);
  Result<std::optional<uint32_t>> next();
  Result<std::optional<uint32_t>> seek_ge(uint32_t target);

 private:
  struct Record {
    uint32_t first{0};
    uint32_t last{0};
    uint32_t count{0};
    format::ByteRange bytes;
  };
  FtsPositionCursor() = default;
  Result<Record> record(uint64_t ordinal);
  Status load(uint64_t ordinal);
  std::unique_ptr<FtsSectionReader> data_, index_;
  format::ByteRange document_, directory_;
  uint32_t tf_{0};
  uint64_t block_count_{0};
  uint64_t block_ordinal_{0};
  format::PositionBlock block_;
  uint32_t within_{0};
  bool started_{false};
  bool eof_{false};
  Status status_;
  FtsContainerReader::Pointer inline_owner_;
  std::optional<uint32_t> inline_position_;
};

}  // namespace zvec::fts
