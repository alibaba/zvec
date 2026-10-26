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

// A range filter on an indexed FLOAT or DOUBLE field must return the same
// documents whether a segment runs it on the inverted index or, after the
// optimizer moved it, as a forward filter. The first segment is sealed and
// its range covers most of its values, so the optimizer moves the filter to a
// forward scan there; the writing segment keeps the inverted index.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/collection.h>
#include <zvec/db/index_params.h>

using namespace zvec;

namespace {

size_t QueryCount(const Collection::Ptr &collection,
                  const std::string &filter) {
  SearchQuery query;
  query.filter_ = filter;
  query.topk_ = 2000;
  auto result = collection->query(query);
  EXPECT_TRUE(result.has_value()) << filter;
  if (!result.has_value()) return 0;
  std::cout << filter << " -> " << result.value().size() << " documents\n";
  return result.value().size();
}

}  // namespace

TEST(FloatRangeFilter, ForwardScanKeepsExactBound) {
  ailego::MemoryLimitPool::get_instance().init(256 * 1024 * 1024);
  const std::string path = "./collection_float_range_filter_dir";
  std::filesystem::remove_all(path);

  CollectionSchema schema("float_range_filter");
  schema.set_max_doc_count_per_segment(1000);
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      "price", DataType::FLOAT, false,
                      std::make_shared<InvertIndexParams>()))
                  .ok());
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      "weight", DataType::DOUBLE, false,
                      std::make_shared<InvertIndexParams>()))
                  .ok());
  auto opened = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  auto collection = std::move(opened.value());

  // 50 documents sit exactly on the bounds used below, 951 above them. The
  // 1001st document starts a new segment, so the first 1000 are sealed.
  std::vector<Doc> docs;
  for (int i = 0; i < 1001; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    doc.set<float>("price", i < 50 ? 1.0000001f : 2.0f);
    doc.set<double>("weight", i < 50 ? 1.0000000001 : 2.0);
    docs.push_back(std::move(doc));
  }
  for (size_t start = 0; start < docs.size(); start += 500) {
    std::vector<Doc> batch(docs.begin() + start,
                           docs.begin() + std::min(docs.size(), start + 500));
    auto inserted = collection->insert(batch);
    ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
    for (const auto &status : inserted.value()) {
      ASSERT_TRUE(status.ok()) << status.message();
    }
  }

  EXPECT_EQ(QueryCount(collection, "price > 1.0000001"), 951u);
  EXPECT_EQ(QueryCount(collection, "weight > 1.0000000001"), 951u);

  collection.reset();
  std::filesystem::remove_all(path);
}
