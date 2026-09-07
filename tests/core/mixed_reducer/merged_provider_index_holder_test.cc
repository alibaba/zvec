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

namespace zvec {
namespace core {
namespace {

constexpr size_t kDimension = 2;

IndexProvider::Pointer MakeProvider(
    const std::vector<std::pair<uint64_t, float>> &docs,
    size_t dimension = kDimension) {
  auto provider =
      std::make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(
          dimension);
  for (const auto &doc : docs) {
    ailego::NumericalVector<float> vector(dimension);
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
               std::shared_ptr<ProviderLifetimeStats> stats)
      : provider_factory_(std::move(provider_factory)),
        stats_(std::move(stats)),
        meta_(IndexMeta::DataType::DT_FP32, kDimension) {}

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
