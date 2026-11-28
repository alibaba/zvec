// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "fts_topk_merge_node.h"
#include <array>
#include <numeric>
#include <arrow/compute/api.h>
#include "db/sqlengine/common/util.h"

namespace zvec::sqlengine {

FtsTopKMergeNode::FtsTopKMergeNode(std::vector<PlanInfo::Ptr> plans,
                                   std::vector<Segment::Ptr> segments,
                                   QueryInfo::Ptr output, bool descending,
                                   ailego::ThreadPool *pool)
    : plans_(std::move(plans)),
      segments_(std::move(segments)),
      output_(std::move(output)),
      candidates_(output_->query_topn(), descending),
      pool_(pool) {
  auto table = segments_.front()->fetch(
      output_->get_selected_scalar_field_names(), std::vector<int>{});
  if (!table) throw std::runtime_error("Cannot build FTS result schema");
  schema_ = Util::append_field(*table->schema(), FIELD_SCORE, arrow::float32());
  // Match the existing vector-recall output contract. FTS has scalar output.
}

arrow::Status FtsTopKMergeNode::collect(size_t segment,
                                        arrow::RecordBatchReader *reader) {
  uint64_t ordinal = 0;
  for (;;) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
    if (!batch) break;
    auto rows = batch->GetColumnByName(LOCAL_ROW_ID);
    auto scores = batch->GetColumnByName(FIELD_SCORE);
    if (!rows || !scores || rows->type_id() != arrow::Type::INT32 ||
        scores->type_id() != arrow::Type::FLOAT || rows->null_count() ||
        scores->null_count())
      return arrow::Status::Invalid("Invalid rank-only result batch");
    const auto &row_array = static_cast<const arrow::Int32Array &>(*rows);
    const auto &score_array = static_cast<const arrow::FloatArray &>(*scores);
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (row_array.Value(i) < 0)
        return arrow::Status::Invalid("Negative local row ID");
      RankedDoc row{segment, ordinal++, row_array.Value(i),
                    score_array.Value(i)};
      candidates_.offer(row);
    }
  }
  return reader->Close();
}

arrow::Status FtsTopKMergeNode::prepare() {
  if (!output_->query_topn()) return arrow::Status::OK();
  // Keep only a window of searches in flight. Each child reader has explicit
  // backpressure; completed plans are released, including their index results.
  for (size_t end = plans_.size(); end > 0;) {
    const size_t count = std::min(std::max(size_t{1}, pool_->count()), end);
    std::vector<Result<std::unique_ptr<arrow::RecordBatchReader>>> readers(
        count);
    auto start = [&](size_t i) {
      try {
        readers[i] = plans_[end - 1 - i]->execute_to_reader();
      } catch (const std::bad_alloc &) {
        readers[i] = tl::make_unexpected(Status(
            StatusCode::RESOURCE_EXHAUSTED, "FTS reader allocation failed"));
      } catch (const std::exception &error) {
        readers[i] = tl::make_unexpected(Status::InternalError(error.what()));
      }
    };
    // A single task has no parallel work to overlap; avoid a worker handoff
    // and wait for every segment in a one-thread query configuration.
    // A caller already on this pool must not wait for its own workers either.
    if (count == 1 || pool_->indexof_this() >= 0) {
      for (size_t i = 0; i < count; ++i) start(i);
    } else {
      auto group = pool_->make_group();
      try {
        for (size_t i = 0; i < count; ++i) group->execute(start, i);
      } catch (...) {
        group->wait_finish();
        throw;
      }
      group->wait_finish();
    }
    for (size_t i = 0; i < count; ++i) {
      const size_t segment = end - 1 - i;
      if (!readers[i]) {
        const auto &error = readers[i].error();
        if (error.code() == StatusCode::RESOURCE_EXHAUSTED)
          return arrow::Status::OutOfMemory(error.message());
        return arrow::Status::ExecutionError(error.message());
      }
      ARROW_RETURN_NOT_OK(collect(segment, readers[i].value().get()));
      readers[i].value().reset();
      plans_[segment].reset();
    }
    end -= count;
  }
  candidates_.finish();
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> FtsTopKMergeNode::fetch(
    size_t segment, const std::vector<int> &rows,
    std::shared_ptr<arrow::Array> scores) const {
  auto table = segments_[segment]->fetch(
      output_->get_selected_scalar_field_names(), rows);
  if (!table) return arrow::Status::ExecutionError("Cannot fetch FTS winners");
  ARROW_ASSIGN_OR_RAISE(auto batch, table->CombineChunksToBatch());
  ARROW_ASSIGN_OR_RAISE(
      batch,
      batch->AddColumn(batch->num_columns(), FIELD_SCORE, std::move(scores)));

  return batch;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
FtsTopKMergeNode::materialize() {
  auto *memory_pool = arrow::default_memory_pool();
  constexpr size_t kBatchRows = 1024;
  const auto &winners = this->winners();
  const size_t count = std::min(kBatchRows, winners.size() - offset_);
  std::array<size_t, kBatchRows> order;
  std::iota(order.begin(), order.begin() + count, 0);
  bool append_in_order =
      output_->fts_cond_info() && segments_.front()->uses_buffer_storage();
  if (append_in_order) {
    bool ascending = true, descending = true;
    for (size_t i = 1; i < count; ++i) {
      const auto previous = winners[offset_ + i - 1].segment;
      const auto current = winners[offset_ + i].segment;
      ascending &= previous <= current;
      descending &= previous >= current;
    }
    append_in_order = ascending || descending;
  }
  // Already grouped winners can be fetched in rank order without a second
  // gather. Keep the bounded grouped-fetch/reorder path for interleaved ranks.
  if (!append_in_order)
    std::sort(order.begin(), order.begin() + count, [&](size_t a, size_t b) {
      return winners[offset_ + a].segment < winners[offset_ + b].segment;
    });
  std::array<uint32_t, kBatchRows> inverse;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::vector<int> rows;
  batches.reserve(std::min(count, segments_.size()));
  rows.reserve(count);
  for (size_t begin = 0; begin < count;) {
    const size_t segment = winners[offset_ + order[begin]].segment;
    rows.clear();
    arrow::FloatBuilder scores(memory_pool);
    size_t end = begin;
    while (end < count && winners[offset_ + order[end]].segment == segment) {
      const auto &winner = winners[offset_ + order[end]];
      rows.push_back(winner.row);
      ARROW_RETURN_NOT_OK(scores.Append(winner.score));
      inverse[order[end]] = static_cast<uint32_t>(end);
      ++end;
    }
    ARROW_ASSIGN_OR_RAISE(auto score_array, scores.Finish());
    ARROW_ASSIGN_OR_RAISE(auto batch,
                          fetch(segment, rows, std::move(score_array)));

    batches.push_back(std::move(batch));
    begin = end;
  }
  ARROW_ASSIGN_OR_RAISE(auto table,
                        arrow::Table::FromRecordBatches(schema_, batches));
  if (append_in_order) {
    ARROW_ASSIGN_OR_RAISE(auto result,
                          table->CombineChunksToBatch(memory_pool));
    offset_ += count;
    return result;
  }
  arrow::UInt32Builder positions(memory_pool);
  ARROW_RETURN_NOT_OK(positions.AppendValues(inverse.data(), count));
  ARROW_ASSIGN_OR_RAISE(auto indices, positions.Finish());
  arrow::compute::ExecContext context(memory_pool);
  ARROW_ASSIGN_OR_RAISE(
      auto sorted,
      arrow::compute::Take(table, indices,
                           arrow::compute::TakeOptions::Defaults(), &context));
  ARROW_ASSIGN_OR_RAISE(auto result,
                        sorted.table()->CombineChunksToBatch(memory_pool));
  offset_ += count;
  return result;
}

arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>>
FtsTopKMergeNode::gen() {
  return [self = shared_from_this()]()
             -> arrow::Future<std::optional<arrow::compute::ExecBatch>> {
    using Batch = std::optional<arrow::compute::ExecBatch>;
    try {
      if (!self->prepared_) {
        auto status = self->prepare();
        self->plans_.clear();
        if (!status.ok()) return arrow::Future<Batch>::MakeFinished(status);
        self->prepared_ = true;
      }
      if (self->offset_ == self->winners().size())
        return arrow::Future<Batch>::MakeFinished(std::nullopt);
      auto result = self->materialize();
      if (!result.ok())
        return arrow::Future<Batch>::MakeFinished(result.status());
      return arrow::Future<Batch>::MakeFinished(
          arrow::compute::ExecBatch(**result));
    } catch (const std::bad_alloc &) {
      self->plans_.clear();
      return arrow::Future<Batch>::MakeFinished(arrow::Status::OutOfMemory(
          "Global FTS topK workspace allocation failed"));
    } catch (const std::exception &error) {
      self->plans_.clear();
      return arrow::Future<Batch>::MakeFinished(
          arrow::Status::ExecutionError(error.what()));
    }
  };
}

}  // namespace zvec::sqlengine
