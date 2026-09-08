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

#include "mixed_reducer/merged_provider_index_holder.h"
#include <algorithm>
#include <atomic>
#include <functional>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "mixed_reducer/mixed_reducer_params.h"
#include "mixed_reducer/mixed_streamer_reducer.h"

namespace zvec {
namespace core {
namespace {

constexpr size_t kDimension = 2;

template <IndexMeta::DataType DT = IndexMeta::DataType::DT_FP32,
          typename T = float>
IndexProvider::Pointer MakeProvider(
    const std::vector<std::pair<uint64_t, float>> &docs,
    size_t dimension = kDimension) {
  auto provider = std::make_shared<MultiPassIndexProvider<DT>>(dimension);
  for (const auto &doc : docs) {
    ailego::NumericalVector<T> vector(dimension);
    vector[0] = doc.second;
    if (dimension > 1) {
      vector[1] = doc.second + 0.5f;
    }
    EXPECT_TRUE(provider->emplace(doc.first, std::move(vector)));
  }
  return provider;
}

struct ProviderLifetimeStats {
  size_t created_count{0};
  size_t live_count{0};
  size_t peak_live_count{0};
};

class CountingProvider final : public IndexProvider {
 public:
  CountingProvider(IndexProvider::Pointer delegate,
                   std::shared_ptr<ProviderLifetimeStats> stats)
      : delegate_(std::move(delegate)), stats_(std::move(stats)) {
    ++stats_->created_count;
    ++stats_->live_count;
    stats_->peak_live_count =
        std::max(stats_->peak_live_count, stats_->live_count);
  }

  ~CountingProvider(void) override {
    --stats_->live_count;
  }

  size_t count(void) const override {
    return delegate_->count();
  }

  size_t dimension(void) const override {
    return delegate_->dimension();
  }

  IndexMeta::DataType data_type(void) const override {
    return delegate_->data_type();
  }

  size_t element_size(void) const override {
    return delegate_->element_size();
  }

  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return delegate_->create_iterator();
  }

  const void *get_vector(uint64_t key) const override {
    return delegate_->get_vector(key);
  }

  const std::string &owner_class(void) const override {
    return delegate_->owner_class();
  }

 private:
  IndexProvider::Pointer delegate_{};
  std::shared_ptr<ProviderLifetimeStats> stats_{};
};

class TestStreamer final : public IndexStreamer {
 public:
  using ProviderFactory =
      std::function<IndexProvider::Pointer(size_t create_count)>;

  TestStreamer(ProviderFactory provider_factory,
               std::shared_ptr<ProviderLifetimeStats> stats,
               size_t dimension = kDimension,
               IndexMeta::DataType data_type = IndexMeta::DataType::DT_FP32)
      : provider_factory_(std::move(provider_factory)),
        stats_(std::move(stats)),
        meta_(data_type, dimension) {}

  int open(IndexStorage::Pointer) override {
    return 0;
  }

  int flush(uint64_t) override {
    return 0;
  }

  int close(void) override {
    return 0;
  }

  int cleanup(void) override {
    return 0;
  }

  const Stats &stats(void) const override {
    return stats_value_;
  }

  const IndexMeta &meta(void) const override {
    return meta_;
  }

  IndexProvider::Pointer create_provider(void) const override {
    auto provider = provider_factory_(provider_create_count_++);
    if (!provider) {
      return nullptr;
    }
    return std::make_shared<CountingProvider>(std::move(provider), stats_);
  }

 private:
  ProviderFactory provider_factory_{};
  std::shared_ptr<ProviderLifetimeStats> stats_{};
  Stats stats_value_{};
  IndexMeta meta_{};
  mutable size_t provider_create_count_{0};
};

IndexStreamer::Pointer MakeStreamer(
    const std::vector<std::pair<uint64_t, float>> &docs,
    const std::shared_ptr<ProviderLifetimeStats> &stats) {
  return std::make_shared<TestStreamer>(
      [docs](size_t) { return MakeProvider(docs); }, stats);
}

IndexStreamer::Pointer MakeStreamer(
    const std::vector<std::pair<uint64_t, float>> &docs) {
  return MakeStreamer(docs, std::make_shared<ProviderLifetimeStats>());
}

MergedProviderIndexHolder::Source MakeSource(
    const IndexStreamer::Pointer &owner) {
  MergedProviderIndexHolder::Source source;
  source.owner = owner;
  source.provider_meta =
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension);
  return source;
}

std::vector<std::pair<uint64_t, float>> ReadAll(
    MergedProviderIndexHolder *holder) {
  std::vector<std::pair<uint64_t, float>> docs;
  auto iter = holder->create_iterator();
  EXPECT_TRUE(iter != nullptr);
  while (iter && iter->is_valid()) {
    const float *data = static_cast<const float *>(iter->data());
    EXPECT_TRUE(data != nullptr);
    if (!data) {
      break;
    }
    docs.emplace_back(iter->key(), data[0]);
    iter->next();
  }
  return docs;
}

class ReadFailureReformer : public IndexReformer {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int load(IndexStorage::Pointer) override {
    return 0;
  }
  int unload() override {
    return 0;
  }
  int revert(const void *in, const IndexQueryMeta &meta,
             std::string *out) const override {
    if (fail && *static_cast<const float *>(in) == failed_value) {
      return IndexError_ReadData;
    }
    out->assign(static_cast<const char *>(in), meta.element_size());
    return 0;
  }
  bool fail{false};
  float failed_value{0.0F};
};

class RetainingTestBuilder : public IndexBuilder {
 public:
  explicit RetainingTestBuilder(
      const std::string &name = "SnapshotTestBuilder") {
    set_name(name);
  }
  int train(IndexThreads::Pointer, IndexHolder::Pointer) override {
    ++train_calls;
    return 0;
  }
  int build(IndexThreads::Pointer, IndexHolder::Pointer input) override {
    holder = std::move(input);
    return 0;
  }
  int cleanup() override {
    holder.reset();
    return 0;
  }
  const Stats &stats() const override {
    return stats_;
  }

  size_t train_calls{0};
  IndexHolder::Pointer holder;

 private:
  Stats stats_;
};

TEST(MergedProviderIndexHolderTest, NonIvfBuildersRetainOwnedInput) {
  for (auto type : {IndexMeta::DataType::DT_FP32, IndexMeta::DataType::DT_FP16,
                    IndexMeta::DataType::DT_INT8}) {
    SCOPED_TRACE(static_cast<int>(type));
    bool source_available = true;
    auto lifetime = std::make_shared<ProviderLifetimeStats>();
    auto source = std::make_shared<TestStreamer>(
        [&](size_t) -> IndexProvider::Pointer {
          if (!source_available) return nullptr;
          const std::vector<std::pair<uint64_t, float>> docs{
              {5, 1.0F}, {9, 2.0F}, {12, 3.0F}};
          if (type == IndexMeta::DataType::DT_FP16) {
            return MakeProvider<IndexMeta::DataType::DT_FP16, ailego::Float16>(
                docs);
          }
          if (type == IndexMeta::DataType::DT_INT8) {
            return MakeProvider<IndexMeta::DataType::DT_INT8, int8_t>(docs);
          }
          return MakeProvider(docs);
        },
        lifetime, kDimension, type);
    auto builder = std::make_shared<RetainingTestBuilder>();
    auto expected_provider = source->create_provider();
    std::vector<std::string> expected;
    for (auto key : {5u, 12u}) {
      expected.emplace_back(
          static_cast<const char *>(expected_provider->get_vector(key)),
          expected_provider->element_size());
    }
    expected_provider.reset();
    {
      ailego::ThreadPool pool(1, false);
      MixedStreamerReducer reducer;
      ailego::Params params;
      params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1);
      ASSERT_EQ(0, reducer.init(params));
      reducer.set_thread_pool(&pool);
      ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(
                       builder, source, nullptr, nullptr,
                       IndexQueryMeta(type, kDimension)));
      ASSERT_EQ(0, reducer.feed_streamer_with_reformer(source, nullptr));
      size_t filter_calls = 0;
      IndexFilter filter;
      filter.set([&](uint64_t key) {
        ++filter_calls;
        return key == 9;
      });
      ASSERT_EQ(0, reducer.reduce(filter));
      EXPECT_EQ(3u, filter_calls);
    }
    ASSERT_NE(nullptr, builder->holder);
    EXPECT_EQ(nullptr,
              dynamic_cast<MergedProviderIndexHolder *>(builder->holder.get()));
    EXPECT_EQ(type, builder->holder->data_type());
    EXPECT_EQ(2u, builder->holder->count());
    EXPECT_TRUE(builder->holder->multipass());
    const auto provider_count = lifetime->created_count;
    std::weak_ptr<TestStreamer> source_owner = source;
    source_available = false;
    source.reset();
    EXPECT_TRUE(source_owner.expired());
    EXPECT_EQ(0u, lifetime->live_count);
    // A later dump (including a repeated dump) must not revisit the source.
    for (size_t pass = 0; pass < 2; ++pass) {
      auto iter = builder->holder->create_iterator();
      ASSERT_NE(nullptr, iter);
      size_t ordinal = 0;
      for (; iter->is_valid(); iter->next(), ++ordinal) {
        ASSERT_LT(ordinal, expected.size());
        EXPECT_EQ(ordinal, iter->key());
        ASSERT_NE(nullptr, iter->data());
        EXPECT_EQ(expected[ordinal],
                  std::string(static_cast<const char *>(iter->data()),
                              builder->holder->element_size()));
      }
      EXPECT_EQ(expected.size(), ordinal);
    }
    EXPECT_EQ(provider_count, lifetime->created_count);
  }
}

TEST(MergedProviderIndexHolderTest,
     NonIvfBuildersRejectFailedInputBeforeTraining) {
  auto source = MakeStreamer({{0, 0.0F}, {1, 1.0F}});
  auto reformer = std::make_shared<ReadFailureReformer>();
  reformer->failed_value = 1.0F;
  reformer->fail = true;
  auto builder = std::make_shared<RetainingTestBuilder>();
  ailego::ThreadPool pool(1, false);
  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1);
  ASSERT_EQ(0, reducer.init(params));
  reducer.set_thread_pool(&pool);
  ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(
                   builder, source, nullptr, nullptr,
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension)));
  ASSERT_EQ(0, reducer.feed_streamer_with_reformer(source, reformer));
  EXPECT_EQ(IndexError_ReadData, reducer.reduce({}));
  EXPECT_EQ(0u, builder->train_calls);
  EXPECT_EQ(nullptr, builder->holder);
}

TEST(MergedProviderIndexHolderTest, IvfBuilderKeepsProviderBackedInput) {
  auto source = MakeStreamer({{0, 0.0F}, {1, 1.0F}});
  auto builder = std::make_shared<RetainingTestBuilder>("IVFBuilder");
  ailego::ThreadPool pool(1, false);
  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1);
  ASSERT_EQ(0, reducer.init(params));
  reducer.set_thread_pool(&pool);
  ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(
                   builder, source, nullptr, nullptr,
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension)));
  ASSERT_EQ(0, reducer.feed_streamer_with_reformer(source, nullptr));
  ASSERT_EQ(0, reducer.reduce({}));
  ASSERT_NE(nullptr, builder->holder);
  auto *merged =
      dynamic_cast<MergedProviderIndexHolder *>(builder->holder.get());
  ASSERT_NE(nullptr, merged);
  EXPECT_EQ((std::vector<std::pair<uint64_t, float>>{{0, 0.0F}, {1, 1.0F}}),
            ReadAll(merged));
}

TEST(MergedProviderIndexHolderTest, FailedReadKeepsRepeatedDataCallsSafe) {
  auto source = MakeSource(MakeStreamer({{0, 0.0F}, {1, 1.0F}}));
  auto reformer = std::make_shared<ReadFailureReformer>();
  source.reformer = reformer;
  source.need_revert = true;
  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension), {source});
  ASSERT_EQ(0, holder.init({}));
  auto iter = holder.create_iterator();
  ASSERT_NE(nullptr, iter);
  ASSERT_TRUE(iter->is_valid());
  reformer->fail = true;
  const void *data = iter->data();
  ASSERT_NE(nullptr, data);
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(IndexError_ReadData, holder.status());
  EXPECT_EQ(
      std::string(holder.element_size(), 0),
      std::string(static_cast<const char *>(data), holder.element_size()));
  EXPECT_EQ(data, iter->data());
  // An error remains terminal even though the current pointer stays safe.
  iter->next();
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(nullptr, holder.create_iterator());
}

TEST(MergedProviderIndexHolderTest, UniformUint4RejectsFailedVectorReads) {
  // A failure on the last record must not train successfully from zeros.
  for (float failed_value : {0.0F, 2.0F}) {
    SCOPED_TRACE(failed_value);
    auto source = MakeSource(MakeStreamer({{0, 0.0F}, {1, 1.0F}, {2, 2.0F}}));
    auto reformer = std::make_shared<ReadFailureReformer>();
    source.reformer = reformer;
    source.need_revert = true;
    auto holder = std::make_shared<MergedProviderIndexHolder>(
        IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
        std::vector<MergedProviderIndexHolder::Source>{source});
    ASSERT_EQ(0, holder->init({}));
    auto converter = IndexFactory::CreateConverter("UniformUint4Converter");
    ASSERT_NE(nullptr, converter);
    IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    ASSERT_EQ(0, converter->init(meta, {}));
    reformer->failed_value = failed_value;
    reformer->fail = true;
    EXPECT_EQ(IndexError_ReadData, converter->train(holder));
    EXPECT_EQ(IndexError_ReadData, holder->status());
    EXPECT_EQ(0u, converter->stats().trained_count());
  }
}

TEST(MergedProviderIndexHolderTest, FiltersRewritesIdsAndSupportsMultiPass) {
  auto lifetime = std::make_shared<ProviderLifetimeStats>();
  auto first = MakeStreamer({{0, 10.0f}, {1, 11.0f}}, lifetime);
  auto second = MakeStreamer({{0, 20.0f}, {1, 21.0f}, {2, 22.0f}}, lifetime);

  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(first));
  sources.emplace_back(MakeSource(second));

  size_t filter_calls = 0;
  IndexFilter filter;
  filter.set([&filter_calls](uint64_t logical_id) {
    ++filter_calls;
    return logical_id == 1 || logical_id == 3;
  });

  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      std::move(sources));
  ASSERT_EQ(0, holder.init(filter));
  EXPECT_EQ(3u, holder.count());
  EXPECT_EQ(2u, holder.filtered_count());
  EXPECT_EQ(5u, filter_calls);
  EXPECT_TRUE(holder.multipass());
  EXPECT_EQ(2u, lifetime->created_count);
  EXPECT_EQ(0u, lifetime->live_count);
  EXPECT_EQ(1u, lifetime->peak_live_count);

  const std::vector<std::pair<uint64_t, float>> expected = {
      {0, 10.0f}, {1, 20.0f}, {2, 22.0f}};
  EXPECT_EQ(expected, ReadAll(&holder));
  EXPECT_EQ(0u, lifetime->live_count);
  EXPECT_EQ(expected, ReadAll(&holder));
  EXPECT_EQ(6u, lifetime->created_count);
  EXPECT_EQ(0u, lifetime->live_count);
  EXPECT_EQ(1u, lifetime->peak_live_count);

  // Filter decisions are planned once and do not depend on filter lifetime or
  // side effects during later builder passes.
  EXPECT_EQ(5u, filter_calls);
  EXPECT_EQ(0, holder.status());
}

TEST(MergedProviderIndexHolderTest, RejectsUnreformedMetaMismatch) {
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(MakeStreamer({{0, 10.0f}})));

  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension + 1),
      std::move(sources));
  EXPECT_EQ(IndexError_Mismatch, holder.init(IndexFilter()));
}

TEST(MergedProviderIndexHolderTest, OrdinalReadsReuseFilterAndOneProvider) {
  auto lifetime = std::make_shared<ProviderLifetimeStats>();
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(MakeStreamer({}, lifetime)));
  sources.emplace_back(
      MakeSource(MakeStreamer({{7, 10.0f}, {19, 11.0f}}, lifetime)));
  sources.emplace_back(MakeSource(MakeStreamer({{0, 12.0f}}, lifetime)));
  sources.emplace_back(
      MakeSource(MakeStreamer({{8, 20.0f}, {99, 21.0f}}, lifetime)));
  size_t filter_calls = 0;
  IndexFilter filter;
  filter.set([&](uint64_t key) {
    ++filter_calls;
    return key == 19 || key == 2;  // Drop a document and an entire source.
  });
  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      std::move(sources));
  ASSERT_EQ(0, holder.init(filter));
  ASSERT_EQ(3u, holder.count());
  OrdinalAccessHolder::Reader::Pointer reader;
  ASSERT_EQ(0, holder.create_ordinal_reader(&reader));
  ASSERT_NE(nullptr, reader);
  EXPECT_EQ(0u, lifetime->live_count);
  const auto expected = ReadAll(&holder);
  for (size_t ordinal : {2u, 0u, 1u, 0u, 2u}) {
    uint64_t key = 99;
    const void *data = nullptr;
    ASSERT_EQ(0, reader->read(ordinal, &key, &data));
    EXPECT_EQ(expected[ordinal].first, key);
    EXPECT_FLOAT_EQ(expected[ordinal].second,
                    static_cast<const float *>(data)[0]);
    EXPECT_EQ(1u, lifetime->live_count);
  }
  uint64_t key = 0;
  const void *data = nullptr;
  EXPECT_EQ(IndexError_OutOfRange, reader->read(3, &key, &data));
  EXPECT_EQ(IndexError_InvalidArgument, reader->read(0, nullptr, &data));
  reader->reset();
  EXPECT_EQ(0u, lifetime->live_count);
  ASSERT_EQ(0, reader->read(0, &key, &data));
  reader.reset();
  EXPECT_EQ(0u, lifetime->live_count);
  EXPECT_EQ(1u, lifetime->peak_live_count);
  EXPECT_EQ(5u, filter_calls);
  EXPECT_EQ(0, holder.status());
}

TEST(MergedProviderIndexHolderTest, OrdinalReadsRejectChangedOrMissingSource) {
  // Changes while planning the key map or when reopening for dump must fail;
  // they must not silently read a different document or emit partial success.
  for (const size_t change_at : {1u, 2u}) {
    for (const int failure : {0, 1, 2, 3}) {
      auto lifetime = std::make_shared<ProviderLifetimeStats>();
      auto owner = std::make_shared<TestStreamer>(
          [change_at, failure](size_t created) -> IndexProvider::Pointer {
            if (created < change_at) {
              return MakeProvider({{7, 10.0f}});
            }
            if (failure == 0) return nullptr;
            if (failure == 1) return MakeProvider({});
            if (failure == 2) return MakeProvider({{7, 10.0f}}, kDimension + 1);
            return MakeProvider({{8, 10.0f}});  // Same count, old key missing.
          },
          lifetime);
      MergedProviderIndexHolder holder(
          IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
          {MakeSource(owner)});
      ASSERT_EQ(0, holder.init(IndexFilter()));
      OrdinalAccessHolder::Reader::Pointer reader;
      const int ret = holder.create_ordinal_reader(&reader);
      if (change_at == 1 && failure != 3) {
        EXPECT_NE(0, ret);
        EXPECT_EQ(nullptr, reader);
      } else {
        ASSERT_EQ(0, ret);
        uint64_t key = 0;
        const void *data = nullptr;
        const int read_ret = reader->read(0, &key, &data);
        // A changed key before mapping belongs to the new consistent pass;
        // losing a mapped key after mapping is an error.
        if (change_at == 1 && failure == 3) {
          EXPECT_EQ(0, read_ret);
        } else {
          EXPECT_NE(0, read_ret);
          EXPECT_EQ(nullptr, data);
          EXPECT_EQ(read_ret, holder.status());
        }
      }
      reader.reset();
      EXPECT_EQ(0u, lifetime->live_count);
      EXPECT_EQ(1u, lifetime->peak_live_count);
    }
  }
}

TEST(MergedProviderIndexHolderTest,
     OrdinalReaderHonorsCancellationAndEmptyInput) {
  std::atomic<bool> stop{false};
  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      {MakeSource(MakeStreamer({{0, 10.0f}}))});
  ASSERT_EQ(0, holder.init(IndexFilter(), &stop));
  OrdinalAccessHolder::Reader::Pointer reader;
  ASSERT_EQ(0, holder.create_ordinal_reader(&reader));
  stop = true;
  uint64_t key = 0;
  const void *data = nullptr;
  EXPECT_EQ(IndexError_Canceled, reader->read(0, &key, &data));
  EXPECT_EQ(IndexError_Canceled, holder.status());

  MergedProviderIndexHolder empty(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension), {});
  EXPECT_EQ(IndexError_InvalidArgument, empty.create_ordinal_reader(&reader));
  ASSERT_EQ(0, empty.init(IndexFilter()));
  ASSERT_EQ(0, empty.create_ordinal_reader(&reader));
  EXPECT_EQ(IndexError_OutOfRange, reader->read(0, &key, &data));
}

TEST(MergedProviderIndexHolderTest,
     OrdinalReaderDeclinesReformationBeforeAnyPass) {
  class CopyReformer : public IndexReformer {
   public:
    int init(const ailego::Params &) override {
      return 0;
    }
    int cleanup() override {
      return 0;
    }
    int load(IndexStorage::Pointer) override {
      return 0;
    }
    int unload() override {
      return 0;
    }
    int revert(const void *in, const IndexQueryMeta &meta,
               std::string *out) const override {
      ++reads;
      out->assign(static_cast<const char *>(in), meta.element_size());
      return 0;
    }
    mutable size_t reads{0};
  };
  auto lifetime = std::make_shared<ProviderLifetimeStats>();
  auto reformer = std::make_shared<CopyReformer>();
  auto source = MakeSource(MakeStreamer({{0, 10.0f}}, lifetime));
  source.need_revert = true;
  source.reformer = reformer;
  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension), {source});
  ASSERT_EQ(0, holder.init(IndexFilter()));
  OrdinalAccessHolder::Reader::Pointer reader;
  EXPECT_EQ(IndexError_NotImplemented, holder.create_ordinal_reader(&reader));
  EXPECT_EQ(nullptr, reader);
  EXPECT_EQ(1u, lifetime->created_count);
  EXPECT_EQ(1u, reformer->reads);
  EXPECT_EQ(0, holder.status());
  EXPECT_EQ((std::vector<std::pair<uint64_t, float>>{{0, 10.0f}}),
            ReadAll(&holder));
}

TEST(MergedProviderIndexHolderTest, StopsDuringFilterPlanning) {
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(MakeStreamer({{0, 10.0f}})));

  std::atomic<bool> stop{true};
  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      std::move(sources));
  EXPECT_EQ(IndexError_Canceled, holder.init(IndexFilter(), &stop));
}

TEST(MergedProviderIndexHolderTest, RetainsOwnerForLazyProviderCreation) {
  auto lifetime = std::make_shared<ProviderLifetimeStats>();
  std::weak_ptr<IndexStreamer> weak_owner;

  {
    auto owner = MakeStreamer({{0, 10.0f}}, lifetime);
    weak_owner = owner;
    std::vector<MergedProviderIndexHolder::Source> sources;
    sources.emplace_back(MakeSource(owner));

    MergedProviderIndexHolder holder(
        IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
        std::move(sources));
    owner.reset();

    EXPECT_FALSE(weak_owner.expired());
    ASSERT_EQ(0, holder.init(IndexFilter()));
    EXPECT_EQ((std::vector<std::pair<uint64_t, float>>{{0, 10.0f}}),
              ReadAll(&holder));
    EXPECT_EQ(0u, lifetime->live_count);
  }

  EXPECT_TRUE(weak_owner.expired());
}

TEST(MergedProviderIndexHolderTest, RejectsProviderCountChangeBetweenPasses) {
  auto lifetime = std::make_shared<ProviderLifetimeStats>();
  auto owner = std::make_shared<TestStreamer>(
      [](size_t create_count) {
        return create_count == 0 ? MakeProvider({{0, 10.0f}, {1, 11.0f}})
                                 : MakeProvider({{0, 10.0f}});
      },
      lifetime);
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(owner));

  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      std::move(sources));
  ASSERT_EQ(0, holder.init(IndexFilter()));

  auto iter = holder.create_iterator();
  ASSERT_NE(nullptr, iter);
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(IndexError_Mismatch, holder.status());
  EXPECT_EQ(0u, lifetime->live_count);
}

TEST(MergedProviderIndexHolderTest, RejectsProviderMetaChangeBetweenPasses) {
  auto lifetime = std::make_shared<ProviderLifetimeStats>();
  auto owner = std::make_shared<TestStreamer>(
      [](size_t create_count) {
        return create_count == 0 ? MakeProvider({{0, 10.0f}})
                                 : MakeProvider({{0, 10.0f}}, kDimension + 1);
      },
      lifetime);
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(owner));

  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      std::move(sources));
  ASSERT_EQ(0, holder.init(IndexFilter()));

  auto iter = holder.create_iterator();
  ASSERT_NE(nullptr, iter);
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(IndexError_Mismatch, holder.status());
  EXPECT_EQ(0u, lifetime->live_count);
}


}  // namespace
}  // namespace core
}  // namespace zvec
