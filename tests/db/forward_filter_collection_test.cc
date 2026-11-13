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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"

namespace zvec::test {
namespace {
// Observe forward access preparation and stop at a deterministic point without
// adding test branches to the candidate-fetch hot path. Register before
// starting workers and restore only after all workers have joined.
class ForwardAccessProbe final : public ailego::Logger {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  void log(int, const char *, int, const char *format, va_list args) override {
    char text[4096];
    vsnprintf(text, sizeof(text), format, args);
    const std::string message(text);
    if (message.find("Forward filter view:") != std::string::npos) {
      ++views;
    } else if (message.find("Forward filter locked fetch:") !=
               std::string::npos) {
      ++locked;
    } else {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return !block; });
  }
  bool wait_entered() {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(10), [&] { return entered; });
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex);
    block = false;
    cv.notify_all();
  }
  int prepared() const {
    return locked.load() + views.load();
  }
  std::atomic<int> locked{0}, views{0};
  std::mutex mutex;
  std::condition_variable cv;
  bool entered{false}, block{false};
};
}  // namespace

class ForwardFilterCollectionTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    GlobalConfig::ConfigData config;
    config.memory_limit_bytes = MIN_MEMORY_LIMIT_BYTES;
    config.query_thread_count = 4;
    config.invert_to_forward_scan_ratio = 0;
    ASSERT_TRUE(GlobalConfig::Instance().initialize(config).ok());
    FileHelper::RemoveDirectory(path_);
    probe_ = std::make_shared<ForwardAccessProbe>();
    previous_ = ailego::LoggerBroker::Register(probe_);
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_DEBUG);
  }
  void TearDown() override {
    collection_.reset();
    ailego::LoggerBroker::Register(previous_);
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
    FileHelper::RemoveDirectory(path_);
  }
  void create(int segment_size = 1000) {
    CollectionSchema schema("forward_collection");
    schema.set_max_doc_count_per_segment(segment_size);
    schema.add_field(std::make_shared<FieldSchema>(
        "value", DataType::INT32, false,
        std::make_shared<InvertIndexParams>(true)));
    schema.add_field(std::make_shared<FieldSchema>("bitmap", DataType::INT32));
    schema.add_field(std::make_shared<FieldSchema>(
        "inverted", DataType::INT32, false,
        std::make_shared<InvertIndexParams>(false)));
    schema.add_field(std::make_shared<FieldSchema>(
        "bucket", DataType::INT32, false,
        std::make_shared<InvertIndexParams>(false)));
    schema.add_field(std::make_shared<FieldSchema>(
        "text", DataType::STRING, false, std::make_shared<FtsIndexParams>()));
    schema.add_field(std::make_shared<FieldSchema>(
        "vec", DataType::VECTOR_FP32, 4, false,
        std::make_shared<HnswIndexParams>(MetricType::L2, 8, 100)));
    auto created = Collection::CreateAndOpen(
        path_, schema, CollectionOptions{false, GetParam()});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    collection_ = created.value();
    std::vector<Doc> docs;
    const int count = segment_size == 1000 ? 2096 : 96;
    for (int i = 0; i < count; ++i) {
      auto item = doc(i);
      if (i >= 96) {
        item.set<int32_t>("value", -100);
        item.set<int32_t>("bitmap", -100);
        item.set<int32_t>("inverted", -100);
      }
      docs.push_back(std::move(item));
      if (docs.size() == 512 || i + 1 == count) {
        auto inserted = collection_->insert(docs);
        ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
        for (const auto &status : inserted.value())
          ASSERT_TRUE(status.ok()) << status.message();
        docs.clear();
      }
    }
    auto removed = collection_->delete_({"pk_60"});
    ASSERT_TRUE(removed.has_value());
    auto replacement = doc(61);
    replacement.set<int32_t>("value", -1);
    replacement.set<int32_t>("bitmap", -1);
    replacement.set<int32_t>("inverted", -1);
    std::vector<Doc> replacements{replacement};
    auto updated = collection_->upsert(replacements);
    ASSERT_TRUE(updated.has_value());
    for (const auto &status : updated.value()) ASSERT_TRUE(status.ok());
  }
  Doc doc(int i) {
    Doc value;
    value.set_pk("pk_" + std::to_string(i));
    value.set<int32_t>("value", i);
    value.set<int32_t>("bitmap", i);
    value.set<int32_t>("inverted", i);
    value.set<int32_t>("bucket", i % 4);
    value.set<std::string>("text", "common document");
    value.set<std::vector<float>>("vec", {static_cast<float>(i), 1, 2, 3});
    return value;
  }
  SearchQuery query(std::string filter = "value > 48") {
    SearchQuery q;
    q.topk_ = 8;
    q.filter_ = std::move(filter);
    q.output_fields_ = std::vector<std::string>{"value", "bucket"};
    q.target_.field_name_ = "vec";
    q.target_.query_params_ = std::make_shared<HnswQueryParams>(200);
    const std::vector<float> vector{60.25f, 1, 2, 3};
    q.target_.set_vector(
        std::string(reinterpret_cast<const char *>(vector.data()),
                    vector.size() * sizeof(float)));
    return q;
  }
  void reopen() {
    ASSERT_TRUE(collection_->close().ok());
    collection_.reset();
    auto opened = Collection::Open(path_, CollectionOptions{true, GetParam()});
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    collection_ = opened.value();
    probe_->locked = 0;
    probe_->views = 0;
  }
  std::set<std::string> keys(const Result<DocPtrList> &result) {
    EXPECT_TRUE(result.has_value());
    std::set<std::string> out;
    if (result)
      for (const auto &doc : result.value()) out.insert(doc->pk());
    return out;
  }
  const std::string path_{"./forward_filter_collection_test_data"};
  Collection::Ptr collection_;
  std::shared_ptr<ForwardAccessProbe> probe_;
  ailego::Logger::Pointer previous_;
};

TEST_P(ForwardFilterCollectionTest,
       WritableReadOnlyVectorFtsBitmapAndInvertedParity) {
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_TRUE(collection_->optimize().ok());
  auto forward = query();
  const auto baseline = keys(collection_->query(forward));
  ASSERT_EQ(baseline.size(), 8u);
  EXPECT_EQ(baseline.count("pk_60"), 0u);
  EXPECT_EQ(baseline.count("pk_61"), 0u);
  EXPECT_EQ(probe_->locked.load(), 0);
  EXPECT_GT(probe_->views.load(), 0);
  auto fts = query();
  fts.topk_ = 96;
  fts.target_.field_name_ = "text";
  fts.target_.clause_ = FtsClause{"common", ""};
  fts.target_.query_params_.reset();
  const auto fts_baseline = keys(collection_->query(fts));
  ASSERT_EQ(fts_baseline.size(), 45u);
  const auto unfiltered = keys(collection_->query(query("")));
  reopen();
  EXPECT_EQ(keys(collection_->query(forward)), baseline);
  EXPECT_GT(probe_->prepared(), 0);
  EXPECT_GT(probe_->views.load(), 0);
  EXPECT_EQ(probe_->locked.load(), 0);
  int readers = probe_->prepared();
  EXPECT_EQ(keys(collection_->query(query("bitmap > 48"))), baseline);
  std::string inverted = "inverted IN (49";
  for (int i = 50; i < 96; ++i) inverted += "," + std::to_string(i);
  inverted += ")";
  EXPECT_EQ(keys(collection_->query(query(inverted))), baseline);
  EXPECT_EQ(keys(collection_->query(query(""))), unfiltered);
  EXPECT_EQ(probe_->prepared(), readers);
  EXPECT_EQ(keys(collection_->query(fts)), fts_baseline);
  EXPECT_GT(probe_->prepared(), readers);
  for (const auto &condition :
       {"value IN (49, 50, 51, 52, 53, 54, 55, 56)",
        "value > 48 AND value NOT IN (60, 61)", "value > 48 AND value < 90"}) {
    auto result = collection_->query(query(condition));
    ASSERT_TRUE(result.has_value()) << result.error().message();
    for (const auto &item : result.value()) {
      EXPECT_NE(item->pk(), "pk_60");
      EXPECT_NE(item->pk(), "pk_61");
    }
  }
}

TEST_P(ForwardFilterCollectionTest, MultiQueryGroupByAndErrors) {
  ASSERT_NO_FATAL_FAILURE(create(10000));
  ASSERT_TRUE(collection_->optimize().ok());
  reopen();
  auto single = query();
  MultiQuery multi;
  multi.queries = {{single.target_, 8}, {single.target_, 8}};
  multi.topk = 8;
  multi.filter = single.filter_;
  multi.output_fields = single.output_fields_;
  EXPECT_EQ(keys(collection_->query(multi)), keys(collection_->query(single)));
  GroupByVectorQuery grouped;
  grouped.target_ = single.target_;
  grouped.filter_ = single.filter_;
  grouped.output_fields_ = single.output_fields_;
  grouped.group_by_field_name_ = "bucket";
  grouped.group_count_ = 4;
  grouped.topk_per_group_ = 2;
  auto groups = collection_->group_by_query(grouped);
  ASSERT_TRUE(groups.has_value()) << groups.error().message();
  ASSERT_EQ(groups->size(), 4u);
  for (const auto &group : groups.value()) {
    ASSERT_EQ(group.docs_.size(), 2u);
    for (const auto &item : group.docs_)
      EXPECT_GT(item.get<int32_t>("value").value(), 48);
  }
  EXPECT_GT(probe_->prepared(), 0);
  EXPECT_GT(probe_->views.load(), 0);
  EXPECT_EQ(probe_->locked.load(), 0);
  single.filter_ = "missing > 1";
  EXPECT_FALSE(collection_->query(single).has_value());
  multi.filter = single.filter_;
  EXPECT_FALSE(collection_->query(multi).has_value());
  EXPECT_TRUE(collection_->close().ok());
}

TEST_P(ForwardFilterCollectionTest, CloseWaitsForActiveFanout) {
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_TRUE(collection_->optimize().ok());
  reopen();
  probe_->block = true;
  auto pending = std::async(std::launch::async,
                            [&] { return collection_->query(query()); });
  const bool entered = probe_->wait_entered();
  EXPECT_TRUE(entered);
  std::promise<void> closing;
  auto closed = std::async(std::launch::async, [&] {
    closing.set_value();
    return collection_->close();
  });
  closing.get_future().wait();
  EXPECT_EQ(closed.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  probe_->release();
  EXPECT_EQ(keys(pending.get()).size(), 8u);
  EXPECT_TRUE(closed.get().ok());
  EXPECT_GT(probe_->prepared(), 0);
  EXPECT_GT(probe_->views.load(), 0);
  EXPECT_EQ(probe_->locked.load(), 0);
}

TEST_P(ForwardFilterCollectionTest, DropColumnWaitsForCapturedQuery) {
  ASSERT_NO_FATAL_FAILURE(create());
  ASSERT_TRUE(collection_->optimize().ok());
  probe_->block = true;
  auto pending = std::async(std::launch::async,
                            [&] { return collection_->query(query()); });
  EXPECT_TRUE(probe_->wait_entered());
  std::promise<void> started;
  auto dropping = std::async(std::launch::async, [&] {
    started.set_value();
    return collection_->drop_column("value");
  });
  started.get_future().wait();
  EXPECT_EQ(dropping.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  probe_->release();
  auto result = pending.get();
  auto dropped = dropping.get();
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_EQ(keys(result).size(), 8u);
  ASSERT_TRUE(dropped.ok()) << dropped.message();
  EXPECT_FALSE(collection_->query(query()).has_value());
  auto surviving = query("bitmap > 48");
  surviving.output_fields_ = std::vector<std::string>{"bitmap", "bucket"};
  EXPECT_EQ(keys(collection_->query(surviving)).size(), 8u);
  EXPECT_GT(probe_->views.load(), 0);
  EXPECT_EQ(probe_->locked.load(), 0);
}

TEST_P(ForwardFilterCollectionTest, WritableFlushSwitchOptimizeAndDdlUseViews) {
  ASSERT_NO_FATAL_FAILURE(create());
  EXPECT_EQ(keys(collection_->query(query())).size(), 8u);
  ASSERT_TRUE(collection_->flush().ok());
  EXPECT_EQ(keys(collection_->query(query())).size(), 8u);
  ASSERT_TRUE(collection_->optimize().ok());
  EXPECT_EQ(keys(collection_->query(query())).size(), 8u);
  ASSERT_TRUE(
      collection_
          ->add_column(std::make_shared<FieldSchema>("extra", DataType::INT32),
                       "value + 1")
          .ok());
  EXPECT_EQ(keys(collection_->query(query())).size(), 8u);
  EXPECT_EQ(probe_->locked.load(), 0);
  EXPECT_GT(probe_->views.load(), 0);
}

TEST_P(ForwardFilterCollectionTest, AppendAfterCaptureDoesNotEnterCandidates) {
  ASSERT_NO_FATAL_FAILURE(create(10000));
  probe_->block = true;
  auto pending = std::async(std::launch::async,
                            [&] { return collection_->query(query()); });
  EXPECT_TRUE(probe_->wait_entered());
  // The query retains its collection guard, but no segment capture lock. A
  // writer can publish a nearer vector while this query's scalar view is fixed.
  auto writing = std::async(std::launch::async, [&] {
    auto added = doc(10000);
    added.set<std::vector<float>>("vec", {60.25f, 1, 2, 3});
    std::vector<Doc> docs{added};
    return collection_->insert(docs);
  });
  EXPECT_EQ(writing.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  probe_->release();
  auto written = writing.get();
  ASSERT_TRUE(written.has_value());
  ASSERT_TRUE(written->front().ok());
  auto before = keys(pending.get());
  EXPECT_EQ(before.size(), 8u);
  EXPECT_EQ(before.count("pk_10000"), 0u);
  auto after = keys(collection_->query(query()));
  EXPECT_EQ(after.count("pk_10000"), 1u);
  EXPECT_EQ(probe_->locked.load(), 0);
}

INSTANTIATE_TEST_SUITE_P(PersistedStores, ForwardFilterCollectionTest,
                         testing::Bool());
}  // namespace zvec::test
