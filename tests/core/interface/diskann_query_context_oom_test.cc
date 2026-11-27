// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#if DISKANN_SUPPORTED
#include <array>
#include <numeric>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/diskann/diskann_context.h"
#include "algorithm/diskann/diskann_searcher_entity.h"
#include "query_context_oom_fixture.h"

namespace zvec {
namespace {
namespace ci = core_interface;

ci::BaseIndexParam::Pointer diskann_parameters(uint32_t dimension) {
  return ci::DiskAnnIndexParamBuilder()
      .with_metric_type(ci::MetricType::kL2sq)
      .with_data_type(ci::DataType::DT_FP32)
      .with_dimension(dimension)
      .with_max_degree(8)
      .with_list_size(32)
      .with_pq_chunk_num(1)
      .build();
}

void build_diskann_file(const std::string &path,
                        const ci::BaseIndexParam &param,
                        std::vector<float> &rows) {
  rows.resize(512 * param.dimension);
  auto writer = ci::IndexFactory::CreateAndInitIndex(param);
  ASSERT_NE(nullptr, writer);
  ASSERT_EQ(0,
            writer->open(path, {ci::StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t id = 0; id < 512; ++id) {
    auto *row = rows.data() + id * param.dimension;
    std::fill_n(row, param.dimension, static_cast<float>(id));
    ASSERT_EQ(0, writer->add(ci::VectorData{ci::DenseVector{row}}, id));
  }
  ASSERT_EQ(0, writer->train());
  ASSERT_EQ(0, writer->close());
}

TEST_P(QueryContextOomTest, DiskAnnCloneFailurePreservesNoMemoryAndCanRetry) {
  const auto path = (directory_ / "clone.index").string();
  const auto param = diskann_parameters(1);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_diskann_file(path, *param, rows));
  auto index = ci::IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(path,
                     {GetParam() ? ci::StorageOptions::StorageType::kBufferPool
                                 : ci::StorageOptions::StorageType::kMMAP,
                      false, true}));
  auto refuse_clone = [&](auto operation) {
    refused_allocation_size = sizeof(core::DiskAnnSearcherEntity);
    const auto reset =
        ailego::ScopeGuard::Make([&] { refused_allocation_size = 0; });
    const int status = operation();
    const bool refused = refused_allocation_size == 0;
    refused_allocation_size = 0;
    EXPECT_TRUE(refused);
    EXPECT_EQ(core::IndexError_NoMemory, status);
  };

  auto query = std::make_shared<ci::DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 32;
  const float value = 256.0F;
  const ci::VectorData input{ci::DenseVector{&value}};
  ci::SearchResult result;
  refuse_clone([&] { return index->search(input, query, &result); });
  int64_t id = -999;
  float score = -999.0F;
  refuse_clone(
      [&] { return index->search_internal_ids(input, query, &id, &score); });
  EXPECT_EQ(-999, id);
  EXPECT_EQ(-999.0F, score);
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(256u, result.doc_list_[0].key());

  auto streamer = index->index_searcher();
  core::IndexContext::Pointer context;
  refuse_clone([&] { return streamer->create_context_with_status(context); });
  EXPECT_EQ(nullptr, context);
  core::IndexStorage::MemoryBlock block;
  refuse_clone([&] { return streamer->get_vector_by_id(256, block); });
  ci::VectorDataBuffer fetched;
  refuse_clone([&] { return index->fetch(256, &fetched); });
  ASSERT_EQ(0, index->fetch(256, &fetched));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(&value), sizeof(value)),
            std::get<ci::DenseVectorBuffer>(fetched.vector_buffer).data);

  core::IndexStorage::Pointer storage;
  ASSERT_NO_FATAL_FAILURE(open_read_storage(path, &storage));
  auto searcher = core::IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(nullptr, searcher);
  ASSERT_EQ(0, searcher->init(ailego::Params{}));
  ASSERT_EQ(0, searcher->load(storage, nullptr));
  // A context from another reader must be replaced through the same clone
  // path, and a failed replacement must leave it available for a retry.
  context = streamer->create_context();
  ASSERT_NE(nullptr, context);
  context->set_topk(1);
  const core::IndexQueryMeta meta(core::IndexMeta::DataType::DT_FP32, 1);
  refuse_clone([&] { return searcher->search_impl(&value, meta, context); });
  ASSERT_EQ(0, searcher->search_impl(&value, meta, context));
  ASSERT_EQ(1u, context->result().size());
  EXPECT_EQ(256u, context->result()[0].key());
  ASSERT_EQ(0, index->close());
}

TEST_P(QueryContextOomTest, DiskAnnIdQueriesDoNotAllocateVectorPayloads) {
  constexpr uint32_t dimension = 384;
  const auto path = (directory_ / "ids.index").string();
  const auto param = diskann_parameters(dimension);
  std::vector<float> rows;
  ASSERT_NO_FATAL_FAILURE(build_diskann_file(path, *param, rows));
  auto index = ci::IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(path,
                     {GetParam() ? ci::StorageOptions::StorageType::kBufferPool
                                 : ci::StorageOptions::StorageType::kMMAP,
                      false, true}));
  // Match this standard library's allocation size for one full vector string.
  const size_t payload_size =
      std::string(dimension * sizeof(float), '\0').capacity() + 1;
  auto without_payload_allocation = [&](auto operation) {
    refused_allocation_size = payload_size;
    const auto reset =
        ailego::ScopeGuard::Make([&] { refused_allocation_size = 0; });
    const int status = operation();
    const bool refused = refused_allocation_size == 0;
    refused_allocation_size = 0;
    EXPECT_FALSE(refused);
    EXPECT_EQ(0, status);
  };
  auto query = std::make_shared<ci::DiskAnnQueryParam>();
  query->topk = 8;
  query->list_size = 32;
  auto keys = std::make_shared<std::vector<uint64_t>>(32);
  std::iota(keys->begin(), keys->end(), 240);
  // Keep every distance distinct: an integer query ties symmetric rows, whose
  // order can change with asynchronous I/O completion order.
  std::array<float, dimension> query_row;
  query_row.fill(256.25F);
  const auto *row = query_row.data();
  const ci::VectorData input{ci::DenseVector{row}};
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    query->is_linear = mode == 1;
    query->bf_pks = mode == 2 ? keys : nullptr;
    query->fetch_vector = true;
    ci::SearchResult expected;
    ASSERT_EQ(0, index->search(input, query, &expected));
    ASSERT_EQ(query->topk, expected.doc_list_.size());
    for (size_t i = 1; i < expected.doc_list_.size(); ++i) {
      ASSERT_LT(expected.doc_list_[i - 1].score(),
                expected.doc_list_[i].score());
    }
    query->fetch_vector = false;
    ci::SearchResult result;
    without_payload_allocation(
        [&] { return index->search(input, query, &result); });
    ASSERT_EQ(expected.doc_list_.size(), result.doc_list_.size());
    for (size_t i = 0; i < result.doc_list_.size(); ++i) {
      EXPECT_EQ(expected.doc_list_[i].key(), result.doc_list_[i].key());
      EXPECT_EQ(expected.doc_list_[i].score(), result.doc_list_[i].score());
      EXPECT_TRUE(result.doc_list_[i].vector_string().empty());
      EXPECT_EQ(dimension * sizeof(float),
                expected.doc_list_[i].vector_string().size());
    }
    std::array<int64_t, 8> ids{};
    std::array<float, 8> scores{};
    without_payload_allocation([&] {
      return index->search_internal_ids(input, query, ids.data(),
                                        scores.data());
    });
    for (size_t i = 0; i < ids.size(); ++i) {
      EXPECT_EQ(expected.doc_list_[i].key(), static_cast<uint64_t>(ids[i]));
      EXPECT_EQ(expected.doc_list_[i].score(), scores[i]);
    }

    auto streamer = index->index_searcher();
    auto context = streamer->create_context();
    ASSERT_NE(nullptr, context);
    context->set_group_params(2, 2);
    context->set_group_by([](uint64_t key) { return std::to_string(key % 2); });
    const core::IndexQueryMeta meta(core::IndexMeta::DataType::DT_FP32,
                                    dimension);
    auto grouped_search = [&] {
      if (mode == 1) return streamer->search_bf_impl(row, meta, context);
      if (mode == 2)
        return streamer->search_bf_by_p_keys_impl(row, {*keys}, meta, context);
      return streamer->search_impl(row, meta, context);
    };
    without_payload_allocation(grouped_search);
    ASSERT_EQ(2u, context->group_result().size());
    for (const auto &group : context->group_result()) {
      ASSERT_EQ(2u, group.docs().size());
      for (const auto &doc : group.docs())
        EXPECT_TRUE(doc.vector_string().empty());
    }
    context->set_fetch_vector(true);
    ASSERT_EQ(0, grouped_search());
    for (const auto &group : context->group_result()) {
      for (const auto &doc : group.docs()) {
        const std::string vector(
            reinterpret_cast<const char *>(rows.data() + doc.key() * dimension),
            dimension * sizeof(float));
        EXPECT_EQ(vector, doc.vector_string());
      }
    }
  }
  ASSERT_EQ(0, index->close());
}

TEST_P(QueryContextOomTest, DiskAnnReadersReleaseFailedContexts) {
  namespace ci = core_interface;
  const auto path = (directory_ / "diskann.index").string();
  auto param = ci::DiskAnnIndexParamBuilder()
                   .with_metric_type(ci::MetricType::kL2sq)
                   .with_data_type(ci::DataType::DT_FP32)
                   .with_dimension(1)
                   .with_max_degree(8)
                   .with_list_size(32)
                   .with_pq_chunk_num(1)
                   .build();
  {
    auto writer = ci::IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, writer);
    ASSERT_EQ(
        0, writer->open(path, {ci::StorageOptions::StorageType::kMMAP, true}));
    for (uint32_t id = 0; id < 512; ++id) {
      const auto value = static_cast<float>(id);
      ASSERT_EQ(0, writer->add(ci::VectorData{ci::DenseVector{&value}}, id));
    }
    ASSERT_EQ(0, writer->train());
    ASSERT_EQ(0, writer->close());
  }

  core::IndexStorage::Pointer searcher_storage;
  core::IndexStorage::Pointer streamer_storage;
  ASSERT_NO_FATAL_FAILURE(open_read_storage(path, &searcher_storage));
  ASSERT_NO_FATAL_FAILURE(open_read_storage(path, &streamer_storage));
  auto searcher = core::IndexFactory::CreateSearcher("DiskAnnSearcher");
  auto streamer = core::IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(nullptr, searcher);
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, searcher->init(ailego::Params{}));
  ASSERT_EQ(0, searcher->load(searcher_storage, nullptr));
  core::IndexMeta meta(core::IndexMeta::DataType::DT_FP32, 1);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  ASSERT_EQ(0, streamer->init(meta, ailego::Params{}));
  ASSERT_EQ(0, streamer->open(streamer_storage));

  const core::IndexQueryMeta query_meta(core::IndexMeta::DataType::DT_FP32, 1);
  const float query = 256.0F;
  auto check_reader = [&](auto &reader) {
    ASSERT_NO_FATAL_FAILURE(
        check_context_failure(reader, sizeof(core::DiskAnnContext)));
    auto context = reader.create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(1);
    ASSERT_EQ(0, reader.search_bf_impl(&query, query_meta, context));
    ASSERT_EQ(1u, context->result().size());
    EXPECT_EQ(256u, context->result()[0].key());
  };
  ASSERT_NO_FATAL_FAILURE(check_reader(*searcher));
  ASSERT_NO_FATAL_FAILURE(check_reader(*streamer));
}
TEST_P(QueryContextOomTest, DiskAnnKeepsIdsWhenScoreNormalizationFails) {
  namespace ci = core_interface;
  using StorageType = ci::StorageOptions::StorageType;
  const auto path = (directory_ / "fp16.index").string();
  const auto param =
      ci::DiskAnnIndexParamBuilder()
          .with_metric_type(ci::MetricType::kL2sq)
          .with_data_type(ci::DataType::DT_FP32)
          .with_quantizer_param(ci::QuantizerParam(ci::QuantizerType::kFP16))
          .with_dimension(8)
          .with_max_degree(8)
          .with_list_size(32)
          .with_pq_chunk_num(1)
          .build();
  auto index = ci::IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageType::kMMAP, true}));
  for (uint32_t id = 0; id < 512; ++id) {
    std::array<float, 8> values;
    values.fill(static_cast<float>(id));
    ASSERT_EQ(0,
              index->add(ci::VectorData{ci::DenseVector{values.data()}}, id));
  }
  ASSERT_EQ(0, index->train());
  ASSERT_EQ(0, index->close());
  ASSERT_EQ(0, index->open(path, {GetParam() ? StorageType::kBufferPool
                                             : StorageType::kMMAP,
                                  false, true}));

  constexpr size_t topk = 3;
  auto query = std::make_shared<ci::DiskAnnQueryParam>();
  query->topk = topk;
  query->list_size = 64;
  std::array<float, 8> values;
  values.fill(256.25F);
  const ci::VectorData input{ci::DenseVector{values.data()}};
  std::array<int64_t, topk> ids;
  std::array<float, topk> scores;
  ASSERT_EQ(
      0, index->search_internal_ids(input, query, ids.data(), scores.data()));
  const auto expected_ids = ids;
  const auto expected_scores = scores;
  ids.fill(-999);
  scores.fill(-999.0F);
  const auto previous_ids = ids;
  const auto previous_scores = scores;

  // The reformer's normalization reserves one IndexDocument per match.
  refused_allocation_size = topk * sizeof(core::IndexDocument);
  const int status =
      index->search_internal_ids(input, query, ids.data(), scores.data());
  const bool refused = refused_allocation_size == 0;
  refused_allocation_size = 0;
  ASSERT_TRUE(refused);
  EXPECT_EQ(core::IndexError_NoMemory, status);
  EXPECT_EQ(previous_ids, ids);
  EXPECT_EQ(previous_scores, scores);
  ASSERT_EQ(
      0, index->search_internal_ids(input, query, ids.data(), scores.data()));
  EXPECT_EQ(expected_ids, ids);
  EXPECT_EQ(expected_scores, scores);
  ASSERT_EQ(0, index->close());
}
INSTANTIATE_TEST_SUITE_P(StorageBackends, QueryContextOomTest,
                         ::testing::Bool());
}  // namespace
}  // namespace zvec
#endif
