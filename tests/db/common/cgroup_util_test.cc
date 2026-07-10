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

#include "db/common/cgroup_util.h"
#include <gtest/gtest.h>

using namespace zvec;

TEST(CgroupUtil, ParseCpuMaxQuota) {
  int cpu_cores = 0;

  ASSERT_TRUE(CgroupUtil::parseCpuMax("100000 100000", &cpu_cores));
  EXPECT_EQ(1, cpu_cores);

  ASSERT_TRUE(CgroupUtil::parseCpuMax("150000 100000\n", &cpu_cores));
  EXPECT_EQ(2, cpu_cores);

  ASSERT_TRUE(CgroupUtil::parseCpuMax("400000 100000", &cpu_cores));
  EXPECT_EQ(4, cpu_cores);
}

TEST(CgroupUtil, ParseCpuMaxUnlimitedOrInvalid) {
  int cpu_cores = 7;

  EXPECT_FALSE(CgroupUtil::parseCpuMax("max 100000", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parseCpuMax("100000/100000", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parseCpuMax("100000 0", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parseCpuMax("0 100000", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parseCpuMax("100000 100000 trailing", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parseCpuMax("invalid", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parseCpuMax("100000 100000", nullptr));

  EXPECT_EQ(7, cpu_cores);
}
