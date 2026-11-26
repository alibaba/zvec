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
#include <memory>
#include <ailego/pattern/defer.h>
#include "inverted_column_indexer.h"

namespace zvec {
namespace {

constexpr uint64_t kMinBoundaryProbeWork =
    16 * InvertedRangeCache::kContainerSetupWork;

}  // namespace

RangeQueryPlan InvertedColumnIndexer::explain_range(const std::string &value,
                                                    CompareOp op) const {
  switch (op) {
    case CompareOp::LT:
      return plan_range(encode(value), true, false, range_cache());
    case CompareOp::LE:
      return plan_range(encode(value), true, true, range_cache());
    case CompareOp::GT:
      return plan_range(encode(value), false, false, range_cache());
    case CompareOp::GE:
      return plan_range(encode(value), false, true, range_cache());
    default:
      return {};
  }
}

RangeQueryPlan InvertedColumnIndexer::plan_range(
    const std::string &term, bool query_left, bool include_eq,
    const InvertedRangeCache *index) const {
  RangeQueryPlan plan;
  // Every candidate builds either L or its complement R. GT is the complement
  // of LE, and GE is the complement of LT, including at a repeated value.
  plan.left_inclusive = query_left ? include_eq : !include_eq;
  if (!index) {
    const bool complement = range_covers_most_values(
        term, query_left ? CompareOp::LT : CompareOp::GT);
    const bool build_left = query_left != complement;
    plan.path =
        build_left ? RangeQueryPath::ScanLeft : RangeQueryPath::ScanRight;
    plan.flip = complement;
    return plan;
  }

  const auto &buckets = index->buckets;
  const auto matches_left = [&](const std::string &value) {
    return cmp_lt(value.data(), value.size(), term.data(), term.size(),
                  plan.left_inclusive);
  };
  // [0,left_end) consists of complete left buckets; [right_begin,n) of
  // complete right buckets. At most one bucket needs exact term processing.
  const size_t left_end =
      std::partition_point(buckets.begin(), buckets.end(),
                           [&](const InvertedRangeCache::Bucket &bucket) {
                             return matches_left(bucket.end);
                           }) -
      buckets.begin();
  const size_t right_begin =
      std::partition_point(buckets.begin(), buckets.end(),
                           [&](const InvertedRangeCache::Bucket &bucket) {
                             return matches_left(bucket.begin);
                           }) -
      buckets.begin();
  const size_t boundary = left_end < right_begin ? left_end : SIZE_MAX;
  // A range bitmap does not tell us how many distinct point postings lie on
  // either side of a boundary. Even a small change in the number of complete
  // buckets can otherwise select thousands more point seeks/merges. Preserve
  // the existing physical boundary side for the fallback plan, except at exact
  // endpoints below. A bounded probe can admit a prefix ending at this bucket.
  const bool legacy_build_left =
      query_left != range_covers_most_values(
                        term, query_left ? CompareOp::LT : CompareOp::GT);
  const bool single_left = boundary != SIZE_MAX && plan.left_inclusive &&
                           term == buckets[boundary].begin;
  const bool single_right = boundary != SIZE_MAX && !plan.left_inclusive &&
                            term == buckets[boundary].end;
  const auto boundary_side_allowed = [&](const RangeQueryPlan &candidate) {
    const bool builds_left_boundary =
        candidate.path == RangeQueryPath::ScanLeft ||
        candidate.path == RangeQueryPath::PrefixAdd;
    return boundary == SIZE_MAX || builds_left_boundary == legacy_build_left ||
           (builds_left_boundary ? single_left : single_right);
  };
  const auto sum_work = [&](size_t begin, size_t end) {
    return (end ? buckets[end - 1].cumulative_decode_work : 0) -
           (begin ? buckets[begin - 1].cumulative_decode_work : 0);
  };
  // This is a deterministic byte-work proxy, not a calibrated latency model.
  // A boundary is conservatively charged as its complete source bucket; using
  // zero would spuriously favor scanning a bucket containing a large term.
  const uint64_t boundary_work =
      boundary != SIZE_MAX ? sum_work(boundary, boundary + 1) : 0;
  const uint64_t flip_bytes =
      ((static_cast<uint64_t>(max_id_.load()) + 64) / 64) * sizeof(uint64_t);
  auto make_plan = [&](RangeQueryPath path, size_t begin, size_t end,
                       size_t prefix_index, bool flip) {
    RangeQueryPlan candidate;
    candidate.path = path;
    candidate.prefix_index = prefix_index;
    candidate.range_begin = begin;
    candidate.range_end = end;
    candidate.boundary_bucket = boundary;
    candidate.left_inclusive = plan.left_inclusive;
    candidate.flip = flip;
    candidate.estimated_work = sum_work(begin, end) + boundary_work;
    if (prefix_index != SIZE_MAX) {
      candidate.estimated_work += index->prefixes[prefix_index].copy_bytes;
      candidate.estimated_work +=
          uint64_t(
              index->prefixes[prefix_index].bitmap->high_low_container.size) *
          InvertedRangeCache::kContainerSetupWork;
    }
    if (flip) {
      candidate.estimated_work += flip_bytes;
    }
    return candidate;
  };

  // Start from direct execution. Equal-cost ties prefer no final flip and no
  // subtraction; the proxy intentionally does not assign fitted weights.
  const bool initial_build_left =
      boundary == SIZE_MAX ? query_left : legacy_build_left;
  plan = initial_build_left ? make_plan(RangeQueryPath::ScanLeft, 0, left_end,
                                        SIZE_MAX, !query_left)
                            : make_plan(RangeQueryPath::ScanRight, right_begin,
                                        buckets.size(), SIZE_MAX, query_left);
  size_t boundary_subtract_prefix = SIZE_MAX;
  uint64_t boundary_subtract_work = 0;
  auto consider = [&](RangeQueryPlan candidate) {
    if (!boundary_side_allowed(candidate)) {
      if (candidate.path == RangeQueryPath::PrefixSubtract &&
          candidate.range_begin == candidate.range_end) {
        boundary_subtract_prefix = candidate.prefix_index;
        boundary_subtract_work = candidate.estimated_work;
      }
      return;
    }
    const auto penalty = [](const RangeQueryPlan &p) {
      return static_cast<unsigned>(p.flip) +
             static_cast<unsigned>(p.path == RangeQueryPath::PrefixSubtract);
    };
    if (candidate.estimated_work < plan.estimated_work ||
        (candidate.estimated_work == plan.estimated_work &&
         penalty(candidate) < penalty(plan))) {
      plan = candidate;
    }
  };
  consider(
      make_plan(RangeQueryPath::ScanLeft, 0, left_end, SIZE_MAX, !query_left));
  consider(make_plan(RangeQueryPath::ScanRight, right_begin, buckets.size(),
                     SIZE_MAX, query_left));

  // The prefix set is bounded. Considering all eligible prefixes also
  // accounts for a farther but substantially smaller run-compressed prefix.
  for (size_t i = 0; i < index->prefixes.size(); ++i) {
    const size_t end = index->prefixes[i].end_bucket;
    if (end <= left_end) {
      consider(
          make_plan(RangeQueryPath::PrefixAdd, end, left_end, i, !query_left));
    }
  }
  for (size_t i = 0; i < index->prefixes.size(); ++i) {
    const size_t end = index->prefixes[i].end_bucket;
    if (end >= right_begin) {
      consider(make_plan(RangeQueryPath::PrefixSubtract, right_begin, end, i,
                         !query_left));
    }
  }
  if (boundary_subtract_prefix != SIZE_MAX &&
      boundary_subtract_work < plan.estimated_work) {
    // Reserve at most a quarter of the savings for the boundary correction.
    // Only savings from eliminating complete buckets count toward the budget.
    const uint64_t saved_work =
        std::min(sum_work(plan.range_begin, plan.range_end),
                 plan.estimated_work - boundary_subtract_work);
    const uint64_t budget = saved_work / 4;
    if (budget >= kMinBoundaryProbeWork) {
      const auto &checkpoints =
          index->prefixes[boundary_subtract_prefix].suffix_checkpoints;
      const auto limit = std::upper_bound(
          checkpoints.begin(), checkpoints.end(), budget,
          [](uint64_t work, const InvertedRangeCache::SuffixCheckpoint &point) {
            return work < point.work;
          });
      if (limit != checkpoints.begin() &&
          cmp(term.data(), term.size(), (limit - 1)->first_term.data(),
              (limit - 1)->first_term.size()) >= 0) {
        plan.boundary_subtract_prefix = boundary_subtract_prefix;
        plan.boundary_subtract_budget = budget;
      }
    }
  }

  if (boundary != SIZE_MAX && plan.path == RangeQueryPath::PrefixAdd &&
      !plan.boundary_subtract_budget && plan.range_end - plan.range_begin > 1) {
    // A later prefix can build the same left side by removing the entire
    // boundary bucket along with the excess full buckets, then adding back
    // the same TERM postings as the original additive plan.
    const size_t additive_buckets = plan.range_end - plan.range_begin;
    for (size_t i = 0; i < index->prefixes.size(); ++i) {
      const auto &prefix = index->prefixes[i];
      const size_t end = prefix.end_bucket;
      if (end < right_begin) continue;
      if (end - boundary >= additive_buckets) break;
      const auto candidate =
          make_plan(RangeQueryPath::PrefixSubtractAddBoundary, boundary, end, i,
                    plan.flip);
      // Keep the additive plan when the prefix layout would already reject
      // the direct dense merge at the first bucket.
      if (candidate.estimated_work < plan.estimated_work &&
          zvec_inverted_bitmap_has_only_bitsets(prefix.bitmap.get())) {
        plan = candidate;
      }
    }
  }
  return plan;
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::execute_range_plan(
    const std::string &term, const RangeQueryPlan &plan,
    const InvertedRangeCache *index) const {
  if (!index) {
    auto result = plan.path == RangeQueryPath::ScanLeft
                      ? get_bitmap_lt_raw(term, plan.left_inclusive)
                      : get_bitmap_gt_raw(term, !plan.left_inclusive);
    if (result && plan.flip) {
      result = flip_bitmap(result.value());
    }
    return result;
  }

  const bool subtract_boundary = plan.path == RangeQueryPath::PrefixSubtract;
  const bool subtract_ranges =
      subtract_boundary ||
      plan.path == RangeQueryPath::PrefixSubtractAddBoundary;
  InvertedRangeCache::BitmapPtr bitmap;
  bitmap.reset(plan.prefix_index == SIZE_MAX
                   ? roaring_bitmap_create()
                   : roaring_bitmap_copy(
                         index->prefixes[plan.prefix_index].bitmap.get()));
  if (!bitmap) {
    return tl::make_unexpected(Status::InternalError());
  }
  bool needs_repair = false;
  InvertedIndexCodec::MergeScratch scratch;
  auto merge = [&](const rocksdb::Slice &value, bool subtract) {
    if (!subtract) {
      auto status = InvertedIndexCodec::merge_or(value.data(), value.size(),
                                                 true, bitmap.get(), &scratch);
      if (status.ok() && value[0] == 0) needs_repair = true;
      return status;
    }
    return InvertedIndexCodec::merge_andnot(value.data(), value.size(),
                                            bitmap.get(), &scratch);
  };

  if (plan.range_begin < plan.range_end) {
    zvec_inverted_bitmap_subtract deferred{};
    AILEGO_DEFER([&]() { roaring_aligned_free(deferred.array_words); });
    bool defer_subtract = subtract_boundary &&
                          plan.range_end - plan.range_begin > 1 &&
                          !zvec_inverted_bitmap_has_only_bitsets(bitmap.get());
    std::unique_ptr<rocksdb::Iterator> ranges(
        ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_));
    ranges->Seek(index->buckets[plan.range_begin].key);
    for (size_t i = plan.range_begin; i < plan.range_end; ++i) {
      if (!ranges->Valid() || ranges->key() != index->buckets[i].key) {
        return tl::make_unexpected(Status::InternalError());
      }
      const auto value = ranges->value();
      if (defer_subtract) {
        if (!value.empty() && value[0] == 0 &&
            zvec_inverted_bitmap_defer_subtract(
                &deferred, bitmap.get(), value.data() + 1, value.size() - 1)) {
          ranges->Next();
          continue;
        }
        if (!zvec_inverted_bitmap_finish_subtract(&deferred, bitmap.get())) {
          return tl::make_unexpected(Status::InternalError());
        }
        defer_subtract = false;
      }
      auto status = merge(value, subtract_ranges);
      if (!status.ok()) {
        return tl::make_unexpected(status);
      }
      ranges->Next();
    }
    if (!ranges->status().ok() ||
        !zvec_inverted_bitmap_finish_subtract(&deferred, bitmap.get())) {
      return tl::make_unexpected(Status::InternalError());
    }
  }

  if (plan.boundary_bucket != SIZE_MAX) {
    const auto &bucket = index->buckets[plan.boundary_bucket];
    const bool boundary_left =
        plan.path != RangeQueryPath::ScanRight && !subtract_boundary;
    const bool single_term = boundary_left
                                 ? plan.left_inclusive && term == bucket.begin
                                 : !plan.left_inclusive && term == bucket.end;
    if (single_term) {
      std::string value;
      const auto read_status =
          ctx_.db_->Get(ctx_.read_opts_, cf_terms_, term, &value);
      if (!read_status.ok())
        return tl::make_unexpected(Status::InternalError());
      const auto status = merge(rocksdb::Slice(value), subtract_boundary);
      if (!status.ok()) return tl::make_unexpected(status);
    } else {
      std::unique_ptr<rocksdb::Iterator> terms(
          ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_));
      if (boundary_left) {
        terms->Seek(bucket.begin);
      } else {
        terms->Seek(term);
        if (plan.left_inclusive && terms->Valid() && terms->key() == term) {
          terms->Next();
        }
      }
      while (terms->Valid()) {
        const auto key = terms->key();
        if (cmp(key.data(), key.size(), bucket.end.data(), bucket.end.size()) >
                0 ||
            (boundary_left && !cmp_lt(key.data(), key.size(), term.data(),
                                      term.size(), plan.left_inclusive))) {
          break;
        }
        auto status = merge(terms->value(), subtract_boundary);
        if (!status.ok()) {
          return tl::make_unexpected(status);
        }
        terms->Next();
      }
      if (!terms->status().ok()) {
        return tl::make_unexpected(Status::InternalError());
      }
    }
  }

  // An exact cached boundary needs only a bitmap copy. Repair cardinality
  // only when lazy OR was used.
  if (needs_repair) {
    roaring_bitmap_repair_after_lazy(bitmap.get());
  }
  if (plan.flip) {
    flip_bitmap(bitmap.get());
  }
  return bitmap.release();
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_lt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    return tl::make_unexpected(Status::InternalError());
  }
  return search_range(term, true, include_eq);
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_gt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    return tl::make_unexpected(Status::InternalError());
  }
  return search_range(term, false, include_eq);
}

}  // namespace zvec
