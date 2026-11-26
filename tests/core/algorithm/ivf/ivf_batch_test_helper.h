// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <gtest/gtest.h>
#include "ivf_searcher.h"

namespace zvec::core::test {

// Share query execution and structural checks; callers retain their own
// score assertions (exact integer results or independently bounded FP error).
template <typename CheckScores>
void CheckBatchedQueries(IVFSearcher &searcher, const void *queries,
                         const IndexQueryMeta &qmeta,
                         IndexContext::Pointer &batch,
                         IndexContext::Pointer &single,
                         CheckScores check_scores) {
  for (bool brute_force : {false, true}) {
    SCOPED_TRACE(brute_force);
    for (uint32_t count : {2U, 7U, 35U}) {
      SCOPED_TRACE(count);
      ASSERT_EQ(0, brute_force
                       ? searcher.search_bf_impl(queries, qmeta, count, batch)
                       : searcher.search_impl(queries, qmeta, count, batch));
      for (uint32_t q = 0; q < count; ++q) {
        SCOPED_TRACE(q);
        const void *query =
            static_cast<const char *>(queries) + q * qmeta.element_size();
        ASSERT_EQ(0, brute_force ? searcher.search_bf_impl(query, qmeta, single)
                                 : searcher.search_impl(query, qmeta, single));
        const auto &expected = single->result();
        const auto &actual = batch->result(q);
        ASSERT_EQ(expected.size(), actual.size());
        EXPECT_EQ(static_cast<IVFSearcherContext *>(single.get())
                      ->stats()
                      .dist_calced_count(),
                  static_cast<IVFSearcherContext *>(batch.get())
                      ->stats(q)
                      .dist_calced_count());
        for (size_t rank = 0; rank < expected.size(); ++rank) {
          SCOPED_TRACE(rank);
          EXPECT_EQ(expected[rank].key(), actual[rank].key());
          ASSERT_NO_FATAL_FAILURE(check_scores(q, expected[rank].key(),
                                               expected[rank].score(),
                                               actual[rank].score()));
        }
      }
    }
  }
}

}  // namespace zvec::core::test
