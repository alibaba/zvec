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

#include <stddef.h>
#include <roaring/roaring.h>

#ifdef __cplusplus
extern "C" {
#endif

// Query-local storage for a read-only view of a portable bitmap. Initialize
// with zeroes and release with zvec_inverted_bitmap_view_clear. The view
// borrows the input until the merge finishes; reloading it invalidates the old
// view.
typedef struct ZvecInvertedBitmapView {
  roaring_bitmap_t bitmap;
  void *metadata;
  size_t metadata_capacity;
  void *payload;
  size_t payload_capacity;
} zvec_inverted_bitmap_view;

const roaring_bitmap_t *zvec_inverted_bitmap_view_load(
    zvec_inverted_bitmap_view *view, const char *data, size_t size);

void zvec_inverted_bitmap_view_clear(zvec_inverted_bitmap_view *view);

// Check whether a cached prefix has the destination layout used by the
// direct dense merge. The caller must still validate each portable posting.
int zvec_inverted_bitmap_has_only_bitsets(const roaring_bitmap_t *bitmap);

// Merge directly into matching, exclusively owned bitset containers. Returns
// zero without changing the destination when the general view is needed.
int zvec_inverted_bitmap_try_merge_dense(roaring_bitmap_t *bitmap,
                                         const char *data, size_t size,
                                         int subtract, int lazy);

typedef struct ZvecInvertedBitmapSubtract {
  uint64_t *array_words;
} zvec_inverted_bitmap_subtract;

// For a query-owned mixed bitmap, defer shrinking containers across matching
// portable postings. Initialize the scratch with zeroes and free array_words
// with roaring_aligned_free when discarding the query. A rejected posting
// does not change the batch. Finish before using the bitmap with Roaring.
int zvec_inverted_bitmap_defer_subtract(zvec_inverted_bitmap_subtract *scratch,
                                        roaring_bitmap_t *bitmap,
                                        const char *data, size_t size);

// Restore the portable container invariants and release the scratch. If
// allocation fails, the bitmap is only safe to free.
int zvec_inverted_bitmap_finish_subtract(zvec_inverted_bitmap_subtract *scratch,
                                         roaring_bitmap_t *bitmap);

#ifdef __cplusplus
}
#endif
