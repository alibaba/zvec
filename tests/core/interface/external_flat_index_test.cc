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

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/flat/flat_streamer_context.h"
#include "tests/test_util.h"

using namespace zvec::core_interface;
namespace core = zvec::core;

namespace {
constexpr uint32_t kDimension = 8;

// Each batch owns a temporary page. Release must precede advancing the cursor.
// Reverse scan order and noncontiguous IDs catch assumptions about row offsets.
class BatchSource : public core::VectorSource {
 public:
  std::map<uint32_t, std::vector<float>> rows;
  mutable std::atomic<uint32_t> scans{0};
  mutable std::atomic<uint32_t> batch_reads{0};
  size_t page_rows{7};
  std::set<uint32_t> null_scan_ids;
  bool supports_scan{true}, fail_scan{false}, malformed_scan{false};
  bool null_scan{false}, fail_read{false};

  explicit BatchSource(uint32_t count = 89, float offset = 0) {
    for (uint32_t i = 0; i < count; ++i) {
      auto &row = rows[i * 13 + 3];
      for (uint32_t d = 0; d < kDimension; ++d) {
        row.push_back(((i * 19 + d * 7) % 101) / 101.0f + offset);
      }
    }
  }

  const void *get_vector(uint32_t id) const override {
    if (fail_read) return nullptr;
    auto it = rows.find(id);
    return it == rows.end() ? nullptr : it->second.data();
  }

  void get_vectors(const uint32_t *ids, uint32_t count,
                   const void **out) const override {
    ++batch_reads;
    EXPECT_LE(count, 1024u);
    VectorSource::get_vectors(ids, count, out);
  }

  class Cursor : public Iterator {
   public:
    explicit Cursor(const BatchSource &source) : source_(source) {
      for (auto it = source.rows.rbegin(); it != source.rows.rend(); ++it) {
        ids_.push_back(it->first);
      }
    }
    int next_batch(uint32_t max_count, Batch *out) override {
      EXPECT_TRUE(previous_.expired());
      out->clear();
      if (source_.fail_scan && position_) return core::IndexError_ReadData;
      const size_t count = std::min(
          {size_t{max_count}, source_.page_rows, ids_.size() - position_});
      if (!count) return 0;
      auto page = std::make_shared<std::vector<float>>(count * kDimension);
      for (size_t i = 0; i < count; ++i) {
        auto id = ids_[position_++];
        std::memcpy(page->data() + i * kDimension, source_.rows.at(id).data(),
                    kDimension * sizeof(float));
        out->ids.push_back(id);
        out->vectors.push_back(source_.null_scan ||
                                       source_.null_scan_ids.count(id)
                                   ? nullptr
                                   : page->data() + i * kDimension);
      }
      out->lease = page;
      previous_ = page;
      if (source_.malformed_scan) out->vectors.pop_back();
      return 0;
    }

   private:
    const BatchSource &source_;
    std::vector<uint32_t> ids_;
    size_t position_{0};
    std::weak_ptr<const void> previous_;
  };

  Iterator::Pointer create_iterator() const override {
    ++scans;
    return supports_scan ? std::make_unique<Cursor>(*this) : nullptr;
  }
};

// An existing source that implements only the original random-read API.
class RandomAccessSource : public core::VectorSource {
 public:
  explicit RandomAccessSource(const BatchSource &source) : source_(source) {}

  const void *get_vector(uint32_t id) const override {
    reads.push_back(id);
    return source_.get_vector(id);
  }

  mutable std::vector<uint32_t> reads;

 private:
  const BatchSource &source_;
};

class ContiguousSource : public BatchSource {
 public:
  ContiguousSource(uint32_t count, bool explicit_ids, size_t padding = 0,
                   uint32_t first_id = 3)
      : BatchSource(0),
        explicit_ids_(explicit_ids),
        stride_elements_(kDimension + padding) {
    data_.resize(count * stride_elements_, -999.0f);
    for (uint32_t i = 0; i < count; ++i) {
      auto id = first_id + i * (explicit_ids ? 13 : 1);
      row_ids_.push_back(id);
      auto &row = rows[id];
      for (uint32_t d = 0; d < kDimension; ++d) {
        row.push_back(((i * 19 + d * 7) % 101) / 101.0f + i * 0.001f);
      }
      std::copy(row.begin(), row.end(), data_.begin() + i * stride_elements_);
    }
  }

  const void *get_vector(uint32_t id) const override {
    auto it = std::lower_bound(row_ids_.begin(), row_ids_.end(), id);
    if (it == row_ids_.end() || *it != id) return nullptr;
    return data_.data() + (it - row_ids_.begin()) * stride_elements_;
  }

  class Cursor : public Iterator {
   public:
    explicit Cursor(const ContiguousSource &source) : source_(source) {}
    ~Cursor() override {
      EXPECT_TRUE(previous_.expired());
    }

    int next_batch(uint32_t max_count, Batch *out) override {
      EXPECT_TRUE(previous_.expired());
      out->clear();
      if (source_.fail_scan && position_) return core::IndexError_ReadData;
      const auto count = static_cast<uint32_t>(
          std::min({size_t{max_count}, source_.page_rows,
                    source_.row_ids_.size() - position_}));
      // Exercise EOF expressed as an empty contiguous view too.
      out->contiguous = ContiguousBatch{};
      if (!count) return 0;
      ++source_.scan_batches;
      auto &view = *out->contiguous;
      view.data = source_.data_.data() + position_ * source_.stride_elements_;
      view.first_id = source_.row_ids_[position_];
      view.count = count;
      view.stride = source_.stride_elements_ * sizeof(float);
      if (source_.explicit_ids_) view.ids = source_.row_ids_.data() + position_;
      if (source_.lease_pages) {
        auto page = std::make_shared<Page>();
        const auto *begin = static_cast<const float *>(view.data);
        page->data.assign(begin, begin + count * source_.stride_elements_);
        page->ids.assign(source_.row_ids_.begin() + position_,
                         source_.row_ids_.begin() + position_ + count);
        view.data = page->data.data();
        if (view.ids) view.ids = page->ids.data();
        out->lease = page;
        previous_ = page;
      }
      if (source_.mixed_batches && batch_number_ % 2) {
        for (uint32_t i = 0; i < count; ++i) {
          out->ids.push_back(view.ids ? view.ids[i] : view.first_id + i);
          out->vectors.push_back(static_cast<const char *>(view.data) +
                                 i * view.stride);
        }
        out->contiguous.reset();
      }
      if (source_.mutate_batch) source_.mutate_batch(out);
      position_ += count;
      ++batch_number_;
      return 0;
    }

   private:
    struct Page {
      std::vector<float> data;
      std::vector<uint32_t> ids;
    };
    const ContiguousSource &source_;
    size_t position_{0};
    size_t batch_number_{0};
    std::weak_ptr<const void> previous_;
  };

  Iterator::Pointer create_iterator() const override {
    ++scans;
    return std::make_unique<Cursor>(*this);
  }

  mutable std::atomic<uint32_t> scan_batches{0};
  bool lease_pages{false}, mixed_batches{false};
  std::function<void(Batch *)> mutate_batch;
  std::vector<uint32_t> row_ids_;

 private:
  bool explicit_ids_;
  size_t stride_elements_;
  std::vector<float> data_;
};

VectorData Vector(const std::vector<float> &row) {
  return VectorData{DenseVector{row.data()}};
}

FlatIndexParam::Pointer Param(bool external,
                              MetricType metric = MetricType::kL2sq,
                              bool id_map = true) {
  return FlatIndexParamBuilder()
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_metric_type(metric)
      .with_use_external_vector(external)
      .with_use_id_map(id_map)
      .build();
}

void SameResults(const SearchResult &want, const SearchResult &got) {
  ASSERT_EQ(want.doc_list_.size(), got.doc_list_.size());
  for (size_t i = 0; i < want.doc_list_.size(); ++i) {
    EXPECT_EQ(want.doc_list_[i].key(), got.doc_list_[i].key());
    EXPECT_NEAR(want.doc_list_[i].score(), got.doc_list_[i].score(), 1e-5);
  }
}

class ExternalFlatTest : public testing::Test {
 protected:
  std::vector<Index::Pointer> opened_;
  std::vector<std::string> paths_;

  std::string test_path(const std::string &suffix) {
    std::string path =
        "test_external_flat_" +
        std::string(
            testing::UnitTest::GetInstance()->current_test_info()->name()) +
        suffix;
    paths_.push_back(path);
    zvec::test_util::RemoveTestFiles(path);
    return path;
  }

  Index::Pointer open_index(const BaseIndexParam::Pointer &param,
                            const std::string &path, bool create = true) {
    auto index = IndexFactory::CreateAndInitIndex(*param);
    EXPECT_NE(nullptr, index);
    if (!index) return index;
    int ret = index->open(path, {StorageOptions::StorageType::kMMAP, create});
    EXPECT_EQ(0, ret);
    if (!ret) opened_.push_back(index);
    return index;
  }

  void add_vectors(Index *index, const BatchSource &source,
                   bool external = true) {
    for (auto &[id, row] : source.rows) {
      ASSERT_EQ(0, external ? index->add_with_source(Vector(row), id, source)
                            : index->add(Vector(row), id));
    }
  }

  void close_index(const Index::Pointer &index) {
    EXPECT_EQ(0, index->close());
    opened_.erase(std::remove(opened_.begin(), opened_.end(), index),
                  opened_.end());
  }

  void TearDown() override {
    for (auto &index : opened_) index->close();
    for (auto &path : paths_) zvec::test_util::RemoveTestFiles(path);
  }
};

class InspectableFlatIndex : public FlatIndex {
 public:
  using Index::acquire_context;
  using Index::init;
};

TEST_F(ExternalFlatTest, RegistrationDoesNotCopyVectorPayload) {
  BatchSource source(3);
  auto index = std::make_shared<InspectableFlatIndex>();
  ASSERT_EQ(0, index->init(*Param(true)));
  ASSERT_EQ(0, index->open(test_path("index"),
                           {StorageOptions::StorageType::kMMAP, true}));
  opened_.push_back(index);
  auto &context = index->acquire_context();
  ASSERT_NE(nullptr, context);
  *context->mutable_features() = "unchanged";
  add_vectors(index.get(), source);
  ASSERT_EQ("unchanged", *context->mutable_features());
  EXPECT_EQ(source.rows.size(), index->get_doc_count());

  // Keep the generic add guards when bypassing its conversion pipeline.
  EXPECT_NE(
      0, index->add_with_source(VectorData{DenseVector{nullptr}}, 3, source));
  EXPECT_NE(0, index->add_with_source(
                   VectorData{SparseVector{0, nullptr, nullptr}}, 3, source));
  const auto vector = Vector(source.rows.at(3));
  EXPECT_NE(0, index->add_with_source(vector, UINT32_MAX, source));
  EXPECT_NE(0, index->add_with_source(vector, 99999, source));
  EXPECT_NE(0, index->add(vector, 3));
  close_index(index);
  EXPECT_NE(0, index->add_with_source(vector, 3, source));
}

TEST_F(ExternalFlatTest, FullScanAppliesSnapshotFilterBeforeReadingVectors) {
  BatchSource source(30);
  auto external = open_index(Param(true), test_path("external"));
  auto embedded = open_index(Param(false), test_path("embedded"));
  add_vectors(external.get(), source);
  add_vectors(embedded.get(), source, false);
  // Model an immutable snapshot whose invisible rows have no usable pointer.
  const std::set<uint32_t> invisible{3, 16, 42};
  source.null_scan_ids = invisible;
  source.rows[99999] = std::vector<float>(kDimension, 0.3f);
  source.null_scan_ids.insert(99999);  // Unregistered rows are also skipped.
  std::vector<float> query(kDimension, 0.3f);
  auto make_query = [&]() {
    auto qp = FlatQueryParamBuilder().with_topk(10).build();
    qp->filter = std::make_shared<IndexFilter>();
    qp->filter->set([invisible](uint64_t id) { return invisible.count(id); });
    return qp;
  };
  SearchResult expected, actual;
  ASSERT_EQ(0, embedded->search(Vector(query), make_query(), &expected));
  auto qp = make_query();
  // Reusing the query must preserve its visibility filter across requests.
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(0,
              external->search_with_source(Vector(query), qp, source, &actual));
    SameResults(expected, actual);
    EXPECT_TRUE(qp->filter->is_valid());
  }
}

TEST_F(ExternalFlatTest, ScanMatchesEmbeddedAndBorrowsReturnedVectors) {
  int case_id = 0;
  for (auto metric : {MetricType::kL2sq, MetricType::kInnerProduct}) {
    for (bool id_map : {false, true}) {
      BatchSource source;
      auto suffix = std::to_string(case_id++);
      auto external =
          open_index(Param(true, metric, id_map), test_path(suffix + "ext"));
      auto embedded =
          open_index(Param(false, metric, id_map), test_path(suffix + "base"));
      add_vectors(external.get(), source);
      add_vectors(embedded.get(), source, false);
      ASSERT_EQ(source.rows.size(), external->get_doc_count());
      source.rows[99999] = std::vector<float>(kDimension, 0.3f);
      std::vector<float> query(kDimension, 0.3f);
      auto qp =
          FlatQueryParamBuilder().with_topk(11).with_fetch_vector(true).build();
      SearchResult want, got;
      ASSERT_EQ(0, embedded->search(Vector(query), qp, &want));
      ASSERT_EQ(0,
                external->search_with_source(Vector(query), qp, source, &got));
      SameResults(want, got);
      EXPECT_EQ(1, source.scans);
      EXPECT_EQ(0, source.batch_reads);
      close_index(external);
      ASSERT_EQ(11, got.doc_list_.size());
      // External vectors are returned directly in the input layout, without
      // creating a redundant dequantized copy in reverted_vector_list_.
      EXPECT_TRUE(got.reverted_vector_list_.empty());
      for (size_t i = 0; i < got.doc_list_.size(); ++i) {
        const auto &doc = got.doc_list_[i];
        // Returned vectors borrow the stable random-access source, not the
        // transient scan pages. The source remains alive after index close.
        EXPECT_EQ(source.get_vector(doc.key()), doc.vector());
      }
    }
  }
}

TEST_F(ExternalFlatTest, FullScanWithOnlyGetVector) {
  BatchSource data(30);
  RandomAccessSource source(data);
  auto external = open_index(Param(true), test_path("external"));
  auto embedded = open_index(Param(false), test_path("embedded"));
  add_vectors(external.get(), data);
  add_vectors(embedded.get(), data, false);
  // Invisible rows must be excluded before calling the source.
  const std::set<uint32_t> invisible{3, 16, 42};
  for (auto id : invisible) data.rows.erase(id);
  data.rows[99999] = std::vector<float>(kDimension, 0.3f);
  auto qp = FlatQueryParamBuilder().with_topk(10).build();
  qp->filter = std::make_shared<IndexFilter>();
  qp->filter->set([invisible](uint64_t id) { return invisible.count(id); });
  std::vector<float> query(kDimension, 0.3f);
  SearchResult want, got;
  ASSERT_EQ(0, embedded->search(Vector(query), qp, &want));
  ASSERT_EQ(0, external->search_with_source(Vector(query), qp, source, &got));
  SameResults(want, got);
  EXPECT_EQ(27, source.reads.size());
  for (auto id : source.reads) {
    EXPECT_EQ(0, invisible.count(id));
    EXPECT_NE(99999, id);
  }
  source.reads.clear();
  qp->filter->set([](uint64_t) { return true; });
  ASSERT_EQ(0, external->search_with_source(Vector(query), qp, source, &got));
  EXPECT_TRUE(got.doc_list_.empty());
  EXPECT_TRUE(source.reads.empty());
  qp->filter.reset();
  EXPECT_NE(0, external->search_with_source(Vector(query), qp, source, &got));
  EXPECT_TRUE(got.doc_list_.empty());
}

TEST_F(ExternalFlatTest, ContiguousScanMatchesEmbeddedAndSkipsDeletedRows) {
  int case_id = 0;
  for (auto metric : {MetricType::kL2sq, MetricType::kInnerProduct}) {
    for (bool id_map : {false, true}) {
      for (bool explicit_ids : {false, true}) {
        for (size_t padding : {0, 3}) {
          ContiguousSource source(2103, explicit_ids, padding);
          source.page_rows = 1024;
          // This row is in the scan but is not registered in either index.
          source.rows.erase(source.row_ids_.back());
          auto suffix = std::to_string(case_id++);
          auto external = open_index(Param(true, metric, id_map),
                                     test_path(suffix + "ext"));
          auto embedded = open_index(Param(false, metric, id_map),
                                     test_path(suffix + "base"));
          add_vectors(external.get(), source);
          add_vectors(embedded.get(), source, false);
          auto qp = FlatQueryParamBuilder()
                        .with_topk(11)
                        .with_fetch_vector(true)
                        .build();
          qp->filter = std::make_shared<IndexFilter>();
          // The first physical batch is entirely deleted. It must not end
          // the scan, and holes in later batches must retain the correct IDs.
          qp->filter->set([last_deleted = source.row_ids_[1023]](uint64_t id) {
            return id <= last_deleted || id % 7 == 0;
          });
          std::vector<float> query(kDimension, 0.3f);
          SearchResult want, got;
          ASSERT_EQ(0, embedded->search(Vector(query), qp, &want));
          ASSERT_EQ(
              0, external->search_with_source(Vector(query), qp, source, &got));
          SameResults(want, got);
          EXPECT_EQ(3, source.scan_batches);
          EXPECT_EQ(0, source.batch_reads);
          for (const auto &doc : got.doc_list_) {
            EXPECT_EQ(source.get_vector(doc.key()), doc.vector());
            EXPECT_NE(source.row_ids_.back(), doc.key());
          }
          // Candidate lookup still uses random access, even with this source.
          qp->bf_pks = std::make_shared<std::vector<uint64_t>>(
              source.row_ids_.begin() + 1024, source.row_ids_.begin() + 1048);
          ASSERT_EQ(0, embedded->search(Vector(query), qp, &want));
          ASSERT_EQ(
              0, external->search_with_source(Vector(query), qp, source, &got));
          SameResults(want, got);
          EXPECT_EQ(1, source.scans);
          EXPECT_EQ(1, source.batch_reads);
        }
      }
    }
  }
}

TEST_F(ExternalFlatTest, ContiguousScanLeasesAndMixedBatches) {
  ContiguousSource source(30, true, 3);
  source.lease_pages = source.mixed_batches = true;
  auto external = open_index(Param(true), test_path("external"));
  auto embedded = open_index(Param(false), test_path("embedded"));
  add_vectors(external.get(), source);
  add_vectors(embedded.get(), source, false);
  auto qp =
      FlatQueryParamBuilder().with_topk(10).with_fetch_vector(true).build();
  std::vector<float> query(kDimension, 0.3f);
  SearchResult want, got;
  ASSERT_EQ(0, embedded->search(Vector(query), qp, &want));
  ASSERT_EQ(0, external->search_with_source(Vector(query), qp, source, &got));
  SameResults(want, got);
  EXPECT_EQ(5, source.scan_batches);
  for (const auto &doc : got.doc_list_) {
    EXPECT_EQ(source.get_vector(doc.key()), doc.vector());
  }
  qp->filter = std::make_shared<IndexFilter>();
  qp->filter->set([](uint64_t) { return true; });
  ASSERT_EQ(0, external->search_with_source(Vector(query), qp, source, &got));
  EXPECT_TRUE(got.doc_list_.empty());
  EXPECT_EQ(10, source.scan_batches);
  qp->filter.reset();
  source.fail_scan = true;
  EXPECT_NE(0, external->search_with_source(Vector(query), qp, source, &got));
  EXPECT_TRUE(got.doc_list_.empty());
}

TEST_F(ExternalFlatTest, RejectsMalformedContiguousBatches) {
  ContiguousSource source(10, false);
  source.lease_pages = true;
  auto index = open_index(Param(true), test_path("index"));
  add_vectors(index.get(), source);
  auto qp = FlatQueryParamBuilder().with_topk(3).build();
  std::vector<float> query(kDimension, 0.3f);
  SearchResult result;
  for (int failure = 0; failure < 8; ++failure) {
    SCOPED_TRACE(failure);
    source.mutate_batch = [failure](core::VectorSource::Batch *batch) {
      auto &view = *batch->contiguous;
      switch (failure) {
        case 0:
          batch->ids.push_back(view.first_id);
          break;
        case 1:
          batch->vectors.push_back(view.data);
          break;
        case 2:
          view.data = nullptr;
          break;
        case 3:
          view.stride = kDimension * sizeof(float) - 1;
          break;
        case 4:
          view.stride = std::numeric_limits<size_t>::max();
          break;
        case 5:
          view.count = 1025;
          break;
        case 6:
          view.first_id = UINT32_MAX - 1;
          break;
        case 7: {
          static const uint32_t kInvalidIds[7] = {UINT32_MAX};
          view.ids = kInvalidIds;
          break;
        }
      }
    };
    result.doc_list_.emplace_back(999, 0);
    EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
    EXPECT_TRUE(result.doc_list_.empty());
  }
  source.mutate_batch = nullptr;
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_EQ(3, result.doc_list_.size());
}

TEST_F(ExternalFlatTest, ContiguousIdBoundaryAndEmptySource) {
  ContiguousSource source(1, false, 0, UINT32_MAX - 1);
  auto index = open_index(Param(true), test_path("index"));
  add_vectors(index.get(), source);
  auto qp = FlatQueryParamBuilder().with_topk(1).build();
  SearchResult result;
  auto query = Vector(source.rows.begin()->second);
  ASSERT_EQ(0, index->search_with_source(query, qp, source, &result));
  ASSERT_EQ(1, result.doc_list_.size());
  EXPECT_EQ(UINT32_MAX - 1, result.doc_list_[0].key());
  ContiguousSource empty(0, false);
  ASSERT_EQ(0, index->search_with_source(query, qp, empty, &result));
  EXPECT_TRUE(result.doc_list_.empty());
}

TEST_F(ExternalFlatTest, FallbackMatchesEmbeddedUsesBatchesAndReopens) {
  int case_id = 0;
  for (auto metric : {MetricType::kL2sq, MetricType::kInnerProduct}) {
    for (bool id_map : {false, true}) {
      BatchSource source(2103);
      source.supports_scan = false;
      // Break ties in the generated rows when comparing full result lists.
      for (auto &[id, row] : source.rows) row[0] += id * 0.0001f;
      auto param = Param(true, metric, id_map);
      auto suffix = std::to_string(case_id++);
      auto path = test_path(suffix + "external");
      auto external = open_index(param, path);
      auto embedded =
          open_index(Param(false, metric, id_map), test_path(suffix + "base"));
      add_vectors(external.get(), source);
      add_vectors(embedded.get(), source, false);
      source.rows[99999] = std::vector<float>(kDimension, 0.3f);
      auto qp =
          FlatQueryParamBuilder().with_topk(11).with_fetch_vector(true).build();
      std::vector<float> query(kDimension, 0.3f);
      SearchResult want, got;
      ASSERT_EQ(0, embedded->search(Vector(query), qp, &want));
      for (int reopened = 0; reopened < 2; ++reopened) {
        source.batch_reads = 0;
        ASSERT_EQ(
            0, external->search_with_source(Vector(query), qp, source, &got));
        SameResults(want, got);
        EXPECT_GE(source.batch_reads.load(), 3u);
        for (auto &doc : got.doc_list_) {
          EXPECT_EQ(source.get_vector(doc.key()), doc.vector());
          EXPECT_NE(99999, doc.key());
        }
        if (!reopened) {
          ASSERT_EQ(0, external->flush());
          close_index(external);
          external = open_index(param, path, false);
        }
      }
      EXPECT_EQ(2, source.scans);
    }
  }
}

TEST_F(ExternalFlatTest, CandidateLookupFilterRadiusAndFetch) {
  BatchSource source;
  source.supports_scan = false;
  auto index = open_index(Param(true), test_path("index"));
  add_vectors(index.get(), source);
  auto qp =
      FlatQueryParamBuilder().with_topk(5).with_fetch_vector(true).build();
  qp->radius = 0.5f;
  qp->filter = std::make_shared<IndexFilter>();
  qp->filter->set([](uint64_t id) { return id == 3; });
  qp->bf_pks = std::make_shared<std::vector<uint64_t>>();
  for (auto &[id, row] : source.rows) qp->bf_pks->push_back(id);
  qp->bf_pks->push_back(999999);
  qp->bf_pks->push_back(1ull << 40);
  std::vector<float> query(kDimension, 0.3f);
  SearchResult result;
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_EQ(0, source.scans);
  ASSERT_FALSE(result.doc_list_.empty());
  for (auto &doc : result.doc_list_) {
    EXPECT_NE(3, doc.key());
    EXPECT_LE(doc.score(), qp->radius);
    EXPECT_EQ(source.get_vector(doc.key()), doc.vector());
  }
  VectorDataBuffer buffer;
  EXPECT_NE(0, index->fetch(3, &buffer));
  EXPECT_NE(0, index->add(Vector(query), 3));
  EXPECT_NE(0, index->search(Vector(query), qp, &result));
}

TEST_F(ExternalFlatTest, ReopenUpdatesAndRejectsWrongStorageMode) {
  BatchSource source;
  auto path = test_path("index");
  auto index = open_index(Param(true, MetricType::kL2sq, false), path);
  add_vectors(index.get(), source);
  auto *streamer =
      dynamic_cast<core::FlatStreamer<32> *>(index->index_searcher().get());
  ASSERT_NE(nullptr, streamer);
  EXPECT_EQ(ailego_align(sizeof(core::BlockHeader) + sizeof(core::DeletionMap) +
                             32 * sizeof(uint64_t),
                         32),
            streamer->entity().linear_block_size());
  source.rows[3] = std::vector<float>(kDimension, 12.0f);
  ASSERT_EQ(0, index->add_with_source(Vector(source.rows.at(3)), 3, source));
  EXPECT_EQ(89, index->get_doc_count());
  ASSERT_EQ(0, index->flush());
  close_index(index);
  index = open_index(Param(true, MetricType::kL2sq, false), path, false);
  ASSERT_EQ(89, index->get_doc_count());
  auto qp = FlatQueryParamBuilder().with_topk(1).build();
  SearchResult result;
  ASSERT_EQ(0, index->search_with_source(Vector(source.rows.at(3)), qp, source,
                                         &result));
  ASSERT_EQ(1, result.doc_list_.size());
  EXPECT_EQ(3, result.doc_list_[0].key());
  EXPECT_EQ(0, result.doc_list_[0].score());
  source.rows[1000000] = std::vector<float>(kDimension, 24.0f);
  ASSERT_EQ(0, index->add_with_source(Vector(source.rows.at(1000000)), 1000000,
                                      source));
  EXPECT_EQ(90, index->get_doc_count());
  close_index(index);
  auto wrong = IndexFactory::CreateAndInitIndex(*Param(false));
  ASSERT_NE(nullptr, wrong);
  EXPECT_NE(0, wrong->open(path, {StorageOptions::StorageType::kMMAP, false}));
  wrong.reset();
  auto embedded_path = test_path("embedded");
  auto embedded = open_index(Param(false), embedded_path);
  close_index(embedded);
  wrong = IndexFactory::CreateAndInitIndex(*Param(true));
  EXPECT_NE(0, wrong->open(embedded_path,
                           {StorageOptions::StorageType::kMMAP, false}));
}

TEST_F(ExternalFlatTest, RejectsReadFailuresAndClearsRequestSource) {
  BatchSource source;
  auto index = open_index(Param(true), test_path("index"));
  add_vectors(index.get(), source);
  auto qp = FlatQueryParamBuilder().with_topk(3).build();
  std::vector<float> query(kDimension, 0.3f);
  SearchResult result;
  for (int failure = 0; failure < 4; ++failure) {
    source.fail_scan = failure == 0;
    source.malformed_scan = failure == 1;
    source.null_scan = failure == 2;
    source.supports_scan = failure != 3;
    source.fail_read = failure == 3;
    result.doc_list_.emplace_back(999, 0);
    EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
    EXPECT_TRUE(result.doc_list_.empty());
    EXPECT_NE(0, index->search(Vector(query), qp, &result));
  }
  source.fail_scan = source.malformed_scan = source.null_scan = false;
  source.supports_scan = true;
  source.fail_read = false;
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  source.fail_read = true;
  qp->fetch_vector = true;
  EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_TRUE(result.doc_list_.empty());
  qp->fetch_vector = false;
  qp->bf_pks = std::make_shared<std::vector<uint64_t>>(
      std::initializer_list<uint64_t>{3});
  EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
  source.fail_read = false;
  source.rows.erase(3);
  EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_NE(0, index->add_with_source(Vector(query), 3, source));
}

TEST_F(ExternalFlatTest, ConcurrentRequestsKeepTheirSourceAndCursor) {
  BatchSource first(50), second(50, 2.0f);
  second.supports_scan = false;
  auto index = open_index(Param(true), test_path("index"));
  add_vectors(index.get(), first);
  std::vector<float> query(kDimension, 0.3f);
  SearchResult expected[2];
  BatchSource *sources[] = {&first, &second};
  auto qp = FlatQueryParamBuilder().with_topk(5).build();
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(0, index->search_with_source(Vector(query), qp, *sources[i],
                                           &expected[i]));
  }
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, i] {
      auto local_qp = FlatQueryParamBuilder().with_topk(5).build();
      for (int j = 0; j < 30; ++j) {
        SearchResult result;
        ASSERT_EQ(0, index->search_with_source(Vector(query), local_qp,
                                               *sources[i % 2], &result));
        SameResults(expected[i % 2], result);
      }
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(61, first.scans);
  EXPECT_EQ(61, second.scans);
  EXPECT_EQ(0, first.batch_reads);
  EXPECT_EQ(61, second.batch_reads);
}

TEST_F(ExternalFlatTest, RejectsUnsupportedFormatsAndUnboundOperations) {
  for (int i = 0; i < 6; ++i) {
    auto param = Param(true);
    if (i == 0) param->metric_type = MetricType::kCosine;
    if (i == 1)
      param->quantizer_param =
          std::make_shared<QuantizerParam>(QuantizerType::kInt8);
    if (i == 2) param->storage_data_type = DataType::DT_FP16;
    if (i == 3) param->major_order = core::IndexMeta::MO_COLUMN;
    if (i == 4) param->use_contiguous_memory = true;
    if (i == 5) param->is_sparse = true;
    EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*param));
  }
  BatchSource source(1);
  auto index = open_index(Param(true), test_path("index"));
  add_vectors(index.get(), source);
  EXPECT_NE(0, index->index_searcher()->dump(nullptr));
  auto provider = index->create_index_provider();
  ASSERT_NE(nullptr, provider);
  EXPECT_EQ(1, provider->count());
  EXPECT_EQ(nullptr, provider->create_iterator());
  EXPECT_EQ(nullptr, provider->get_vector(3));
  EXPECT_NE(0, index->merge({index}, IndexFilter{}));
  EXPECT_EQ(0, index->merge({}, IndexFilter{}));
  auto embedded = open_index(Param(false), test_path("embedded"));
  add_vectors(embedded.get(), source, false);
  EXPECT_NE(0, embedded->merge({index}, IndexFilter{}));

  auto qp = FlatQueryParamBuilder().with_topk(1).build();
  qp->group_by_param = std::make_shared<GroupByParam>();
  qp->group_by_param->group_count = 1;
  qp->group_by_param->group_topk = 1;
  qp->group_by_param->group_by = [](uint64_t) { return std::string("group"); };
  SearchResult result;
  const auto query = Vector(source.rows.at(3));
  EXPECT_EQ(core::IndexError_Unsupported,
            index->search_with_source(query, qp, source, &result));
  EXPECT_TRUE(result.group_doc_list_.empty());
  qp->group_by_param.reset();
  // The existing refine entry point has no source binding for external Flat.
  qp->refiner_param = std::make_shared<RefinerParam>();
  qp->refiner_param->scale_factor_ = 1;
  qp->refiner_param->reference_index = index;
  EXPECT_NE(0, embedded->search(query, qp, &result));
}

TEST_F(ExternalFlatTest, BufferPoolReopen) {
  BatchSource source(40);
  auto path = test_path("index");
  auto index = open_index(Param(true), path);
  add_vectors(index.get(), source);
  close_index(index);
  index = IndexFactory::CreateAndInitIndex(*Param(true));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kBufferPool,
                                  false, true}));
  opened_.push_back(index);
  auto qp =
      FlatQueryParamBuilder().with_topk(1).with_fetch_vector(true).build();
  SearchResult result;
  auto vector = Vector(source.rows.at(3));
  EXPECT_NE(0, index->add_with_source(vector, 3, source));
  EXPECT_NE(0, index->search(vector, qp, &result));
  ASSERT_EQ(0, index->search_with_source(vector, qp, source, &result));
  ASSERT_EQ(1, result.doc_list_.size());
  EXPECT_EQ(3, result.doc_list_[0].key());
  EXPECT_EQ(0, result.doc_list_[0].score());
  source.supports_scan = false;
  ASSERT_EQ(0, index->search_with_source(vector, qp, source, &result));
  ASSERT_EQ(1, result.doc_list_.size());
  EXPECT_EQ(3, result.doc_list_[0].key());
  EXPECT_EQ(0, result.doc_list_[0].score());
}

TEST_F(ExternalFlatTest, LowLevelBatchQueriesAndMetadataValidation) {
  BatchSource source(20);
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, storage->open(test_path("index"), true));
  core::IndexMeta meta;
  meta.set_meta(core::IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params{});
  zvec::ailego::Params params;
  params.set(core::PARAM_FLAT_USE_EXTERNAL_VECTOR, true);
  auto streamer = core::IndexFactory::CreateStreamer("FlatStreamer16");
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto context = streamer->create_context();
  auto *ctx = dynamic_cast<core::FlatStreamerContext<16> *>(context.get());
  ASSERT_NE(nullptr, ctx);
  ctx->set_vector_source(&source);
  ctx->set_topk(1);
  core::IndexQueryMeta qmeta(core::IndexMeta::DT_FP32, kDimension);
  for (auto &[id, row] : source.rows) {
    ASSERT_EQ(0, streamer->add_impl(id, row.data(), qmeta, context));
  }
  auto query = source.rows.at(3);
  query.insert(query.end(), source.rows.at(16).begin(),
               source.rows.at(16).end());
  ASSERT_EQ(0, streamer->search_bf_impl(query.data(), qmeta, 2, context));
  ASSERT_EQ(1, context->result(0).size());
  ASSERT_EQ(1, context->result(1).size());
  EXPECT_EQ(3, context->result(0)[0].key());
  EXPECT_EQ(16, context->result(1)[0].key());
  EXPECT_EQ(2, source.scans);
  source.supports_scan = false;
  ASSERT_EQ(0, streamer->search_bf_impl(query.data(), qmeta, 2, context));
  ASSERT_EQ(1, context->result(0).size());
  ASSERT_EQ(1, context->result(1).size());
  EXPECT_EQ(3, context->result(0)[0].key());
  EXPECT_EQ(16, context->result(1)[0].key());
  EXPECT_EQ(2, source.batch_reads);
  core::IndexQueryMeta wrong(core::IndexMeta::DT_FP32, kDimension - 1);
  EXPECT_NE(0, streamer->search_bf_impl(query.data(), wrong, 2, context));
  ctx->set_group_params(1, 1);
  EXPECT_EQ(core::IndexError_Unsupported,
            streamer->search_bf_impl(query.data(), qmeta, 2, context));
  EXPECT_EQ(core::IndexError_Unsupported,
            streamer->search_bf_by_p_keys_impl(query.data(), {{3}, {16}}, qmeta,
                                               2, context));
  ctx->set_group_params(0, 0);
  context->reset();
  EXPECT_NE(0, streamer->search_bf_impl(query.data(), qmeta, 2, context));
  EXPECT_EQ(0, streamer->close());
  context.reset();
  EXPECT_EQ(0, storage->close());
}

TEST_F(ExternalFlatTest, EmptyIndexAndEmptyCandidates) {
  BatchSource source(0);
  auto index = open_index(Param(true), test_path("index"));
  auto qp = FlatQueryParamBuilder().with_topk(3).build();
  std::vector<float> query(kDimension, 0.3f);
  SearchResult result;
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_TRUE(result.doc_list_.empty());
  source.supports_scan = false;
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_TRUE(result.doc_list_.empty());
  EXPECT_EQ(0, source.batch_reads);
  qp->bf_pks = std::make_shared<std::vector<uint64_t>>();
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_TRUE(result.doc_list_.empty());
}
}  // namespace
