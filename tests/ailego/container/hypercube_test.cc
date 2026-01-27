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

#include <gtest/gtest.h>
#include <zvec/ailego/container/hypercube.h>

using namespace zvec::ailego;

TEST(Hypercube, General) {
  Hypercube hyper;
  hyper.insert("1", 1);
  hyper.insert("2", 2);
  hyper.insert("3", 3);

  EXPECT_EQ(1, hyper["1"].cast<int>());
  EXPECT_EQ(2, hyper["2"].cast<int>());
  EXPECT_EQ(3, hyper["3"].cast<int>());

  hyper.insert_or_assign("1", 11);
  hyper.insert_or_assign("2", 22);
  hyper.insert_or_assign("3", 33);
  hyper.insert_or_assign("4", 44);
  hyper.insert_or_assign("5", 55);
  EXPECT_EQ(11, hyper["1"].cast<int>());
  EXPECT_EQ(22, hyper["2"].cast<int>());
  EXPECT_EQ(33, hyper["3"].cast<int>());
  EXPECT_EQ(44, hyper["4"].cast<int>());
  EXPECT_EQ(55, hyper["5"].cast<int>());

  std::string key1("111"), key2("222");
  Cube val1(11);
  hyper.insert_or_assign(key1, val1);
  hyper.insert_or_assign(std::move(key2), val1);
  hyper.insert_or_assign("345464", 435465);
  EXPECT_FALSE(key1.empty());
  EXPECT_TRUE(key2.empty());
}
