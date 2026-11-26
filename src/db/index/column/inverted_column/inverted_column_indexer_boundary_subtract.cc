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

#include <memory>
#include "inverted_column_indexer.h"

namespace zvec {

InvertedRangeCache::BitmapPtr InvertedColumnIndexer::try_subtract_boundary(
    const std::string &term, const RangeQueryPlan &plan,
    const InvertedRangeCache &index) const {
  const auto &bucket = index.buckets[plan.boundary_bucket];
  std::unique_ptr<rocksdb::Iterator> terms(
      ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_));
  terms->Seek(term);
  if (plan.left_inclusive && terms->Valid() && terms->key() == term) {
    terms->Next();
  }

  InvertedRangeCache::BitmapPtr correction(roaring_bitmap_create());
  if (!correction) return {};
  uint64_t remaining_work = plan.boundary_subtract_budget;
  size_t remaining_terms = InvertedRangeCache::kMaxBoundaryProbeTerms;
  for (; terms->Valid(); terms->Next()) {
    const auto key = terms->key();
    if (cmp(key.data(), key.size(), bucket.end.data(), bucket.end.size()) > 0) {
      break;
    }
    const auto value = terms->value();
    const auto work = InvertedRangeCache::short_term_work(value);
    if (!work || work > remaining_work || !remaining_terms) return {};
    remaining_work -= work;
    --remaining_terms;
    if (!InvertedIndexCodec::merge_or(value.data(), value.size(), false,
                                      correction.get())
             .ok()) {
      return {};
    }
  }
  // A partial correction or an optional read failure leaves the original plan
  // usable. Copy the prefix only once the correction has completed.
  if (!terms->status().ok()) return {};
  InvertedRangeCache::BitmapPtr bitmap(roaring_bitmap_copy(
      index.prefixes[plan.boundary_subtract_prefix].bitmap.get()));
  if (bitmap) {
    roaring_bitmap_andnot_inplace(bitmap.get(), correction.get());
  }
  return bitmap;
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::search_range(
    const std::string &term, bool query_left, bool include_eq) const {
  const auto *cache = range_cache();
  const auto plan = plan_range(term, query_left, include_eq, cache);
  if (plan.boundary_subtract_budget) {
    auto bitmap = try_subtract_boundary(term, plan, *cache);
    if (bitmap) {
      // Both plans build the left result, so they share the flip decision.
      if (plan.flip) flip_bitmap(bitmap.get());
      return bitmap.release();
    }
  }
  return execute_range_plan(term, plan, cache);
}

}  // namespace zvec
