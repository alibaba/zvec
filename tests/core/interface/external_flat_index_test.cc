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
#include <map>
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
  mutable std::atomic<uint32_t> scans{0}, reads{0};
  size_t page_rows{7};
  bool supports_scan{true}, fail_scan{false}, malformed_scan{false};
  bool null_scan{false}, fail_read{false}, wrong_read_ids{false};
  bool legacy_reads{false};

  explicit BatchSource(uint32_t count = 89, float offset = 0) {
    for (uint32_t i = 0; i < count; ++i) {
      auto &row = rows[i * 13 + 3];
      for (uint32_t d = 0; d < kDimension; ++d) {
        row.push_back(((i * 19 + d * 7) % 101) / 101.0f + offset);
      }
    }
  }

  const void *get_vector(uint32_t id) const override {
    auto it = rows.find(id);
    return it == rows.end() ? nullptr : it->second.data();
  }

  int get_vector_batch(const uint32_t *ids, uint32_t count,
                       Batch *out) const override {
    ++reads;
    if (legacy_reads) return VectorSource::get_vector_batch(ids, count, out);
    if (fail_read) return core::IndexError_ReadData;
    out->clear();
    auto page = std::make_shared<std::vector<float>>(count * kDimension);
    for (uint32_t i = 0; i < count; ++i) {
      const void *data = get_vector(ids[i]);
      if (!data) return core::IndexError_NoExist;
      std::memcpy(page->data() + i * kDimension, data,
                  kDimension * sizeof(float));
      out->ids.push_back(ids[i] + (wrong_read_ids ? 1 : 0));
      out->vectors.push_back(page->data() + i * kDimension);
    }
    out->lease = page;
    return 0;
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
        std::memcpy(page->data() + i * kDimension, source_.get_vector(id),
                    kDimension * sizeof(float));
        out->ids.push_back(id);
        out->vectors.push_back(
            source_.null_scan ? nullptr : page->data() + i * kDimension);
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

  std::string Path(const std::string &suffix) {
    std::string path =
        "test_external_flat_" +
        std::string(
            testing::UnitTest::GetInstance()->current_test_info()->name()) +
        suffix;
    paths_.push_back(path);
    zvec::test_util::RemoveTestFiles(path);
    return path;
  }

  Index::Pointer Open(const BaseIndexParam::Pointer &param,
                      const std::string &path, bool create = true) {
    auto index = IndexFactory::CreateAndInitIndex(*param);
    EXPECT_NE(nullptr, index);
    if (!index) return index;
    int ret = index->open(path, {StorageOptions::StorageType::kMMAP, create});
    EXPECT_EQ(0, ret);
    if (!ret) opened_.push_back(index);
    return index;
  }

  void Add(Index *index, const BatchSource &source, bool external = true) {
    for (auto &[id, row] : source.rows) {
      ASSERT_EQ(0, external ? index->add_with_source(Vector(row), id, source)
                            : index->add(Vector(row), id));
    }
  }

  void Close(const Index::Pointer &index) {
    EXPECT_EQ(0, index->close());
    opened_.erase(std::remove(opened_.begin(), opened_.end(), index),
                  opened_.end());
  }

  void TearDown() override {
    for (auto &index : opened_) index->close();
    for (auto &path : paths_) zvec::test_util::RemoveTestFiles(path);
  }
};

TEST_F(ExternalFlatTest, ScanMatchesEmbeddedAndOwnsReturnedVectors) {
  int case_id = 0;
  for (auto metric : {MetricType::kL2sq, MetricType::kInnerProduct}) {
    for (bool id_map : {false, true}) {
      BatchSource source;
      auto suffix = std::to_string(case_id++);
      auto external = Open(Param(true, metric, id_map), Path(suffix + "ext"));
      auto embedded = Open(Param(false, metric, id_map), Path(suffix + "base"));
      Add(external.get(), source);
      Add(embedded.get(), source, false);
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
      auto expected = source.rows;
      source.rows.clear();
      Close(external);
      ASSERT_EQ(11, got.doc_list_.size());
      ASSERT_EQ(11, got.reverted_vector_list_.size());
      for (size_t i = 0; i < got.doc_list_.size(); ++i) {
        const auto &doc = got.doc_list_[i];
        const auto &row = expected.at(doc.key());
        EXPECT_EQ(0, std::memcmp(row.data(), doc.vector(),
                                 kDimension * sizeof(float)));
        EXPECT_EQ(0,
                  std::memcmp(row.data(), got.reverted_vector_list_[i].data(),
                              kDimension * sizeof(float)));
      }
    }
  }
}

TEST_F(ExternalFlatTest, CandidateLookupFilterRadiusAndFetch) {
  BatchSource source;
  source.supports_scan = false;
  auto index = Open(Param(true), Path("index"));
  Add(index.get(), source);
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
    VectorDataBuffer buffer;
    ASSERT_EQ(0, index->fetch_with_source(doc.key(), source, &buffer));
    EXPECT_EQ(0,
              std::memcmp(
                  source.get_vector(doc.key()),
                  std::get<DenseVectorBuffer>(buffer.vector_buffer).data.data(),
                  kDimension * sizeof(float)));
  }
  VectorDataBuffer buffer;
  EXPECT_NE(0, index->fetch_with_source(999999, source, &buffer));
  EXPECT_NE(0, index->fetch(3, &buffer));
  EXPECT_NE(0, index->add(Vector(query), 3));
  EXPECT_NE(0, index->search(Vector(query), qp, &result));
}

TEST_F(ExternalFlatTest, ReopenUpdatesAndRejectsWrongStorageMode) {
  BatchSource source;
  auto path = Path("index");
  auto index = Open(Param(true, MetricType::kL2sq, false), path);
  Add(index.get(), source);
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
  Close(index);
  index = Open(Param(true, MetricType::kL2sq, false), path, false);
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
  Close(index);
  auto wrong = IndexFactory::CreateAndInitIndex(*Param(false));
  ASSERT_NE(nullptr, wrong);
  EXPECT_NE(0, wrong->open(path, {StorageOptions::StorageType::kMMAP, false}));
  wrong.reset();
  auto embedded_path = Path("embedded");
  auto embedded = Open(Param(false), embedded_path);
  Close(embedded);
  wrong = IndexFactory::CreateAndInitIndex(*Param(true));
  EXPECT_NE(0, wrong->open(embedded_path,
                           {StorageOptions::StorageType::kMMAP, false}));
}

TEST_F(ExternalFlatTest, RejectsReadFailuresAndClearsRequestSource) {
  BatchSource source;
  auto index = Open(Param(true), Path("index"));
  Add(index.get(), source);
  auto qp = FlatQueryParamBuilder().with_topk(3).build();
  std::vector<float> query(kDimension, 0.3f);
  SearchResult result;
  for (int failure = 0; failure < 4; ++failure) {
    source.fail_scan = failure == 0;
    source.malformed_scan = failure == 1;
    source.null_scan = failure == 2;
    source.supports_scan = failure != 3;
    result.doc_list_.emplace_back(999, 0);
    EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
    EXPECT_TRUE(result.doc_list_.empty());
    EXPECT_NE(0, index->search(Vector(query), qp, &result));
  }
  source.fail_scan = source.malformed_scan = source.null_scan = false;
  source.supports_scan = true;
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  qp->bf_pks = std::make_shared<std::vector<uint64_t>>(
      std::initializer_list<uint64_t>{3});
  source.fail_read = true;
  EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
  source.fail_read = false;
  source.wrong_read_ids = true;
  EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
  source.wrong_read_ids = false;
  source.rows.erase(3);
  EXPECT_NE(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_NE(0, index->add_with_source(Vector(query), 3, source));
}

TEST_F(ExternalFlatTest, GroupByUsesExternalVectors) {
  BatchSource source;
  auto index = Open(Param(true), Path("ext"));
  auto embedded = Open(Param(false), Path("base"));
  Add(index.get(), source);
  Add(embedded.get(), source, false);
  auto qp =
      FlatQueryParamBuilder().with_topk(6).with_fetch_vector(true).build();
  qp->group_by_param = std::make_shared<GroupByParam>();
  qp->group_by_param->group_count = 3;
  qp->group_by_param->group_topk = 2;
  qp->group_by_param->group_by = [](uint64_t id) {
    return std::to_string(id % 3);
  };
  std::vector<float> query(kDimension, 0.3f);
  SearchResult want, got;
  ASSERT_EQ(0, embedded->search(Vector(query), qp, &want));
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &got));
  ASSERT_EQ(3, got.group_doc_list_.size());
  for (size_t i = 0; i < want.group_doc_list_.size(); ++i) {
    const auto &a = want.group_doc_list_[i];
    const auto &b = got.group_doc_list_[i];
    EXPECT_EQ(a.group_id(), b.group_id());
    ASSERT_EQ(2, b.docs().size());
    for (size_t j = 0; j < b.docs().size(); ++j) {
      EXPECT_EQ(a.docs()[j].key(), b.docs()[j].key());
      EXPECT_NEAR(a.docs()[j].score(), b.docs()[j].score(), 1e-5);
      EXPECT_EQ(0,
                std::memcmp(source.get_vector(b.docs()[j].key()),
                            b.docs()[j].vector(), kDimension * sizeof(float)));
    }
  }
}

TEST_F(ExternalFlatTest, RefineBindsItsOwnSource) {
  BatchSource coarse_source(30), reference_source(30, 0.8f);
  auto coarse = Open(Param(true), Path("coarse"));
  auto reference = Open(Param(true), Path("reference"));
  Add(coarse.get(), coarse_source);
  Add(reference.get(), reference_source);
  auto qp =
      FlatQueryParamBuilder().with_topk(5).with_fetch_vector(true).build();
  std::vector<float> query(kDimension, 0.3f);
  SearchResult want, got;
  ASSERT_EQ(0, reference->search_with_source(Vector(query), qp,
                                             reference_source, &want));
  reference_source.supports_scan = false;
  qp->refiner_param = std::make_shared<RefinerParam>();
  qp->refiner_param->scale_factor_ = 6;
  qp->refiner_param->reference_index = reference;
  qp->refiner_param->reference_vector_source = &reference_source;
  ASSERT_EQ(0,
            coarse->search_with_source(Vector(query), qp, coarse_source, &got));
  SameResults(want, got);
  for (auto &doc : got.doc_list_) {
    EXPECT_EQ(0, std::memcmp(reference_source.get_vector(doc.key()),
                             doc.vector(), kDimension * sizeof(float)));
  }
  qp->refiner_param->reference_vector_source = nullptr;
  EXPECT_NE(0,
            coarse->search_with_source(Vector(query), qp, coarse_source, &got));
  qp->refiner_param.reset();
  EXPECT_NE(0, reference->search(Vector(query), qp, &got));
}

TEST_F(ExternalFlatTest, HnswExternalRefinesWithExternalFlat) {
  BatchSource source(30);
  auto original = std::move(source.rows);
  uint32_t id = 0;
  for (auto &entry : original) source.rows[id++] = entry.second;
  auto coarse_param = HNSWIndexParamBuilder()
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_metric_type(MetricType::kL2sq)
                          .with_use_external_vector(true)
                          .build();
  auto coarse = Open(coarse_param, Path("coarse"));
  auto reference = Open(Param(true), Path("reference"));
  Add(coarse.get(), source);
  Add(reference.get(), source);
  std::vector<float> query(kDimension, 0.3f);
  auto flat_qp =
      FlatQueryParamBuilder().with_topk(5).with_fetch_vector(true).build();
  SearchResult want, got;
  ASSERT_EQ(
      0, reference->search_with_source(Vector(query), flat_qp, source, &want));
  auto qp = HNSWQueryParamBuilder()
                .with_topk(5)
                .with_ef_search(64)
                .with_fetch_vector(true)
                .build();
  qp->refiner_param = std::make_shared<RefinerParam>();
  qp->refiner_param->scale_factor_ = 6;
  qp->refiner_param->reference_index = reference;
  qp->refiner_param->reference_vector_source = &source;
  source.supports_scan = false;
  ASSERT_EQ(0, coarse->search_with_source(Vector(query), qp, source, &got));
  SameResults(want, got);
}

TEST_F(ExternalFlatTest, ConcurrentRequestsKeepTheirSourceAndCursor) {
  BatchSource first(50), second(50, 2.0f);
  auto index = Open(Param(true), Path("index"));
  Add(index.get(), first);
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
  auto index = Open(Param(true), Path("index"));
  Add(index.get(), source);
  EXPECT_NE(0, index->index_searcher()->dump(nullptr));
  auto provider = index->create_index_provider();
  ASSERT_NE(nullptr, provider);
  EXPECT_EQ(1, provider->count());
  EXPECT_EQ(nullptr, provider->create_iterator());
  EXPECT_EQ(nullptr, provider->get_vector(3));
  EXPECT_NE(0, index->merge({index}, IndexFilter{}));
}

TEST_F(ExternalFlatTest, LegacyRandomAccessAndBufferPoolReopen) {
  BatchSource source(40);
  source.legacy_reads = true;
  auto path = Path("index");
  auto index = Open(Param(true), path);
  Add(index.get(), source);
  Close(index);
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
}

TEST_F(ExternalFlatTest, LowLevelBatchQueriesAndMetadataValidation) {
  BatchSource source(20);
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, storage->open(Path("index"), true));
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
  core::IndexQueryMeta wrong(core::IndexMeta::DT_FP32, kDimension - 1);
  EXPECT_NE(0, streamer->search_bf_impl(query.data(), wrong, 2, context));
  context->reset();
  EXPECT_NE(0, streamer->search_bf_impl(query.data(), qmeta, 2, context));
  EXPECT_EQ(0, streamer->close());
  context.reset();
  EXPECT_EQ(0, storage->close());
}

TEST_F(ExternalFlatTest, EmptyIndexAndEmptyCandidates) {
  BatchSource source(0);
  auto index = Open(Param(true), Path("index"));
  auto qp = FlatQueryParamBuilder().with_topk(3).build();
  std::vector<float> query(kDimension, 0.3f);
  SearchResult result;
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_TRUE(result.doc_list_.empty());
  source.supports_scan = false;
  qp->bf_pks = std::make_shared<std::vector<uint64_t>>();
  ASSERT_EQ(0, index->search_with_source(Vector(query), qp, source, &result));
  EXPECT_TRUE(result.doc_list_.empty());
}
}  // namespace
