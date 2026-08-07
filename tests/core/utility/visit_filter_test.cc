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

#include "utility/visit_filter.h"
#include <gtest/gtest.h>

using zvec::core::VisitFilter;

TEST(VisitFilterTest, DestroyIsIdempotentAndAllowsReinitialization) {
  VisitFilter filter;
  ASSERT_EQ(0, filter.init(VisitFilter::ByteMap, 128, 128, 0.01f));
  filter.set_visited(7);
  EXPECT_TRUE(filter.visited(7));

  filter.destroy();
  filter.destroy();
  EXPECT_EQ(VisitFilter::Default, filter.get_mode());

  ASSERT_EQ(0, filter.init(VisitFilter::BitMap, 128, 128, 0.01f));
  filter.set_visited(11);
  EXPECT_TRUE(filter.visited(11));
}

TEST(VisitFilterTest, DestructorOwnsEveryModeContext) {
  for (int mode :
       {VisitFilter::BloomFilter, VisitFilter::BitMap, VisitFilter::ByteMap}) {
    VisitFilter filter;
    ASSERT_EQ(0, filter.init(mode, 1024, 512, 0.01f));
    filter.set_visited(17);
    EXPECT_TRUE(filter.visited(17));
  }
}

TEST(VisitFilterTest, RejectsInvalidModeWithoutRetainingContext) {
  VisitFilter filter;
  EXPECT_EQ(zvec::core::IndexError_InvalidArgument,
            filter.init(VisitFilter::Default, 128, 128, 0.01f));
  EXPECT_EQ(VisitFilter::Default, filter.get_mode());
  filter.destroy();
}
