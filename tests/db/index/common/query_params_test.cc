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

#include "zvec/db/query_params.h"
#include <gtest/gtest.h>

using namespace zvec;

TEST(QueryParamsTest, QueryParamsBaseClass) {
  // Test constructor
  QueryParams params(IndexType::HNSW);
  EXPECT_EQ(params.type(), IndexType::HNSW);

  // Test setter
  params.set_type(IndexType::IVF);
  EXPECT_EQ(params.type(), IndexType::IVF);
}

TEST(QueryParamsTest, HnswQueryParams) {
  // Test constructor
  HnswQueryParams params(100);
  EXPECT_EQ(params.type(), IndexType::HNSW);
  EXPECT_EQ(params.ef(), 100);

  // Test setter
  params.set_ef(200);
  EXPECT_EQ(params.ef(), 200);
}

TEST(QueryParamsTest, IVFQueryParams) {
  // Test constructor
  IVFQueryParams params(50);
  EXPECT_EQ(params.type(), IndexType::IVF);
  EXPECT_EQ(params.nprobe(), 50);

  // Test setter
  params.set_nprobe(75);
  EXPECT_EQ(params.nprobe(), 75);
}

TEST(QueryParamsTest, Polymorphism) {
  // Test polymorphic behavior
  QueryParams::Ptr hnsw_ptr = std::make_shared<HnswQueryParams>(100);
  QueryParams::Ptr ivf_ptr = std::make_shared<IVFQueryParams>(50);

  // Verify types
  EXPECT_EQ(hnsw_ptr->type(), IndexType::HNSW);
  EXPECT_EQ(ivf_ptr->type(), IndexType::IVF);

  // Test dynamic casting
  auto hnsw_cast = std::dynamic_pointer_cast<HnswQueryParams>(hnsw_ptr);
  auto ivf_cast = std::dynamic_pointer_cast<IVFQueryParams>(ivf_ptr);
  auto invalid_cast = std::dynamic_pointer_cast<HnswQueryParams>(ivf_ptr);

  EXPECT_NE(hnsw_cast, nullptr);
  EXPECT_NE(ivf_cast, nullptr);
  EXPECT_EQ(invalid_cast, nullptr);

  // Verify values after casting
  EXPECT_EQ(hnsw_cast->ef(), 100);
  EXPECT_EQ(ivf_cast->nprobe(), 50);
}

TEST(QueryParamsTest, VirtualDestructor) {
  // Test that virtual destructor allows proper deletion
  QueryParams *hnsw_ptr = new HnswQueryParams(100);
  QueryParams *ivf_ptr = new IVFQueryParams(50);

  // This should not cause memory issues
  delete hnsw_ptr;
  delete ivf_ptr;
}