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

#include <cstring>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tests/test_util.h"
#if RABITQ_SUPPORTED
#include "core/algorithm/hnsw_rabitq/rabitq_converter.h"
#include "zvec/core/framework/index_provider.h"
#endif
#include "zvec/core/interface/index.h"
#include "zvec/core/interface/index_factory.h"
#include "zvec/core/interface/index_param.h"
#include "zvec/core/interface/index_param_builders.h"

using namespace zvec::core_interface;

namespace {

constexpr uint32_t kDimension = 4;
constexpr uint32_t kNumDocs = 12;
constexpr uint32_t kNumGroups = 3;
constexpr uint32_t kGroupTopk = 2;
constexpr uint32_t kSearchTopk = 100;

struct GroupByCase {
  std::string name;
  BaseIndexParam::Pointer index_param;
  BaseIndexQueryParam::Pointer query_param;
  bool is_sparse = false;
  uint32_t dimension = kDimension;
  bool with_refiner = false;
  int expected_error = 0;
};

std::shared_ptr<std::vector<uint64_t>> AllPks() {
  auto pks = std::make_shared<std::vector<uint64_t>>();
  pks->reserve(kNumDocs);
  for (uint32_t i = 0; i < kNumDocs; ++i) {
    pks->push_back(i);
  }
  return pks;
}

void AttachGroupBy(const BaseIndexQueryParam::Pointer &query_param) {
  query_param->group_by_param = std::make_shared<GroupByParam>();
  query_param->group_by_param->group_count = kNumGroups;
  query_param->group_by_param->group_topk = kGroupTopk;
  query_param->group_by_param->group_by = [](uint64_t key) {
    return std::to_string(key % kNumGroups);
  };
}

BaseIndexParam::Pointer DenseFlatParam(uint32_t dimension = kDimension) {
  return FlatIndexParamBuilder()
      .with_metric_type(MetricType::kInnerProduct)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(dimension)
      .with_is_sparse(false)
      .build();
}

BaseIndexParam::Pointer SparseFlatParam() {
  return FlatIndexParamBuilder()
      .with_metric_type(MetricType::kInnerProduct)
      .with_data_type(DataType::DT_FP32)
      .with_is_sparse(true)
      .build();
}

BaseIndexParam::Pointer DenseHnswParam(uint32_t dimension = kDimension) {
  return HNSWIndexParamBuilder()
      .with_metric_type(MetricType::kInnerProduct)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(dimension)
      .with_is_sparse(false)
      .with_ef_construction(100)
      .build();
}

BaseIndexParam::Pointer SparseHnswParam() {
  return HNSWIndexParamBuilder()
      .with_metric_type(MetricType::kInnerProduct)
      .with_data_type(DataType::DT_FP32)
      .with_is_sparse(true)
      .with_ef_construction(100)
      .build();
}

BaseIndexQueryParam::Pointer FlatQuery(bool fetch_vector = false) {
  return FlatQueryParamBuilder()
      .with_topk(kSearchTopk)
      .with_fetch_vector(fetch_vector)
      .build();
}

BaseIndexQueryParam::Pointer FlatQuery(bool fetch_vector, bool is_linear,
                                       bool with_bf_pks) {
  auto builder = FlatQueryParamBuilder()
                     .with_topk(kSearchTopk)
                     .with_fetch_vector(fetch_vector)
                     .with_is_linear(is_linear);
  if (with_bf_pks) {
    builder.with_bf_pks(AllPks());
  }
  return builder.build();
}

BaseIndexQueryParam::Pointer HnswQuery(bool fetch_vector = false,
                                       bool is_linear = false,
                                       bool with_bf_pks = false) {
  auto builder = HNSWQueryParamBuilder()
                     .with_topk(kSearchTopk)
                     .with_ef_search(kSearchTopk)
                     .with_fetch_vector(fetch_vector)
                     .with_is_linear(is_linear);
  if (with_bf_pks) {
    builder.with_bf_pks(AllPks());
  }
  return builder.build();
}

#if RABITQ_SUPPORTED
BaseIndexParam::Pointer DenseHnswRabitqParam(uint32_t dimension) {
  using namespace zvec::ailego;
  using namespace zvec::core;

  constexpr size_t kTrainCount = 500;
  auto holder =
      std::make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(
          dimension);
  for (size_t i = 0; i < kTrainCount; ++i) {
    NumericalVector<float> vec(dimension, static_cast<float>(i));
    EXPECT_TRUE(holder->emplace(i, vec));
  }

  auto index_meta =
      std::make_shared<IndexMeta>(IndexMeta::DataType::DT_FP32, dimension);
  index_meta->set_metric("InnerProduct", 0, Params());
  RabitqConverter converter;
  EXPECT_EQ(0, converter.init(*index_meta, Params()));
  EXPECT_EQ(0, converter.train(holder));

  std::shared_ptr<IndexReformer> reformer;
  EXPECT_EQ(0, converter.to_reformer(&reformer));

  return HNSWRabitqIndexParamBuilder()
      .with_metric_type(MetricType::kInnerProduct)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(dimension)
      .with_is_sparse(false)
      .with_ef_construction(100)
      .with_provider(holder)
      .with_reformer(reformer)
      .build();
}

BaseIndexQueryParam::Pointer HnswRabitqQuery(bool fetch_vector = false,
                                             bool is_linear = false,
                                             bool with_bf_pks = false) {
  auto builder = HNSWRabitqQueryParamBuilder()
                     .with_topk(kSearchTopk)
                     .with_ef_search(kSearchTopk)
                     .with_fetch_vector(fetch_vector)
                     .with_is_linear(is_linear);
  if (with_bf_pks) {
    builder.with_bf_pks(AllPks());
  }
  return builder.build();
}

BaseIndexParam::Pointer DenseIvfRabitqParam(uint32_t dimension) {
  return IVFRabitqIndexParamBuilder()
      .with_metric_type(MetricType::kInnerProduct)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(dimension)
      .with_is_sparse(false)
      .with_n_list(4)
      .with_total_bits(7)
      .build();
}

BaseIndexQueryParam::Pointer IvfRabitqQuery(bool is_linear = false,
                                            bool with_bf_pks = false,
                                            bool fetch_vector = false) {
  auto query = std::make_shared<IVFRabitqQueryParam>();
  query->topk = kSearchTopk;
  query->nprobe = 4;
  query->is_linear = is_linear;
  query->fetch_vector = fetch_vector;
  if (with_bf_pks) {
    query->bf_pks = AllPks();
  }
  return query;
}
#endif

#if DISKANN_SUPPORTED
BaseIndexParam::Pointer DenseDiskAnnParam(uint32_t dimension = kDimension) {
  return DiskAnnIndexParamBuilder()
      .with_metric_type(MetricType::kInnerProduct)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(dimension)
      .with_is_sparse(false)
      .with_max_degree(32)
      .with_list_size(kSearchTopk)
      .with_pq_chunk_num(0)
      .build();
}

BaseIndexQueryParam::Pointer DiskAnnQuery(bool fetch_vector = false,
                                          bool is_linear = false,
                                          bool with_bf_pks = false) {
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = kSearchTopk;
  query->list_size = kSearchTopk;
  query->fetch_vector = fetch_vector;
  query->is_linear = is_linear;
  if (with_bf_pks) {
    query->bf_pks = AllPks();
  }
  return query;
}
#endif

class GroupByInterfaceTest : public ::testing::Test {
 protected:
  void RunOk(const GroupByCase &test_case) {
    Run(test_case, /*expect_error=*/false);
  }

  void RunRejected(const GroupByCase &test_case) {
    Run(test_case, /*expect_error=*/true);
  }

 private:
  struct QueryHolder {
    std::vector<float> values;
    std::vector<uint32_t> indices;
    VectorData data;
  };

  void Run(const GroupByCase &test_case, bool expect_error) {
    const std::string index_name = "test_groupby_" + test_case.name;
    const std::string source_index_name = index_name + "_source";
    zvec::test_util::RemoveTestFiles(index_name + "*");
    zvec::test_util::RemoveTestFiles(source_index_name + "*");

    auto source = IndexFactory::CreateAndInitIndex(*FlatSourceParam(test_case));
    ASSERT_NE(nullptr, source) << test_case.name;
    ASSERT_EQ(0, source->open(source_index_name,
                              {StorageOptions::StorageType::kMMAP, true}))
        << test_case.name;

    for (uint32_t i = 0; i < kNumDocs; ++i) {
      AddDoc(source, i, test_case);
    }
    ASSERT_EQ(0, source->train()) << test_case.name;

    auto index = IndexFactory::CreateAndInitIndex(*test_case.index_param);
    ASSERT_NE(nullptr, index) << test_case.name;
    ASSERT_EQ(
        0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}))
        << test_case.name;
    // Unsupported combinations are rejected by Index::search before training
    // or backend-specific search setup.  Do not turn this API validation into
    // an integration build of every rejected backend (notably DiskAnn).
    if (!expect_error) {
      ASSERT_EQ(0, index->merge({source}, IndexFilter())) << test_case.name;
    }

    auto query_param = test_case.query_param->clone();
    AttachGroupBy(query_param);
    if (test_case.with_refiner) {
      query_param->refiner_param = std::make_shared<RefinerParam>();
      query_param->refiner_param->scale_factor_ = 1.0f;
      query_param->refiner_param->reference_index = source;
    }
    auto query = MakeQuery(test_case);

    SearchResult result;
    const int ret = index->search(query.data, query_param, &result);
    if (expect_error) {
      if (test_case.expected_error != 0) {
        ASSERT_EQ(test_case.expected_error, ret) << test_case.name;
      } else {
        ASSERT_NE(0, ret) << test_case.name;
      }
    } else {
      ASSERT_EQ(0, ret) << test_case.name;
      AssertGroupedResult(result, query_param, test_case);
    }

    ASSERT_EQ(0, index->close()) << test_case.name;
    ASSERT_EQ(0, source->close()) << test_case.name;
    zvec::test_util::RemoveTestFiles(index_name + "*");
    zvec::test_util::RemoveTestFiles(source_index_name + "*");
  }

  BaseIndexParam::Pointer FlatSourceParam(const GroupByCase &test_case) {
    if (test_case.is_sparse) {
      return SparseFlatParam();
    }
    return DenseFlatParam(test_case.dimension);
  }

  void AddDoc(const Index::Pointer &index, uint32_t key,
              const GroupByCase &test_case) {
    std::vector<float> values(test_case.dimension, static_cast<float>(key));
    if (test_case.is_sparse) {
      std::vector<uint32_t> indices(test_case.dimension);
      std::iota(indices.begin(), indices.end(), 0u);
      VectorData data{
          SparseVector{test_case.dimension, indices.data(), values.data()}};
      ASSERT_EQ(0, index->add(data, key)) << key;
      return;
    }
    VectorData data{DenseVector{values.data()}};
    ASSERT_EQ(0, index->add(data, key)) << key;
  }

  QueryHolder MakeQuery(const GroupByCase &test_case) {
    QueryHolder holder;
    holder.values.assign(test_case.dimension, 1.0f);
    if (test_case.is_sparse) {
      holder.indices.resize(test_case.dimension);
      std::iota(holder.indices.begin(), holder.indices.end(), 0u);
      holder.data = VectorData{SparseVector{
          test_case.dimension, holder.indices.data(), holder.values.data()}};
    } else {
      holder.data = VectorData{DenseVector{holder.values.data()}};
    }
    return holder;
  }

  void AssertGroupedResult(const SearchResult &result,
                           const BaseIndexQueryParam::Pointer &query_param,
                           const GroupByCase &test_case) {
    ASSERT_TRUE(result.doc_list_.empty());
    ASSERT_EQ(kNumGroups, result.group_doc_list_.size());

    std::set<std::string> group_ids;
    for (const auto &group : result.group_doc_list_) {
      group_ids.insert(group.group_id());
      ASSERT_LE(group.docs().size(), kGroupTopk);
      ASSERT_GE(group.docs().size(), 1u);

      const uint32_t expected_mod = std::stoul(group.group_id());
      for (const auto &doc : group.docs()) {
        ASSERT_EQ(expected_mod, doc.key() % kNumGroups);
      }
      for (size_t i = 1; i < group.docs().size(); ++i) {
        ASSERT_GE(group.docs()[i - 1].score(), group.docs()[i].score());
      }
    }
    for (uint32_t group = 0; group < kNumGroups; ++group) {
      ASSERT_TRUE(group_ids.count(std::to_string(group)) > 0);
    }

    if (!query_param->fetch_vector) {
      return;
    }
    if (test_case.index_param->index_type == IndexType::kIVFRabitq) {
      ASSERT_TRUE(result.group_reverted_vector_list_.empty());
      for (const auto &group : result.group_doc_list_) {
        for (const auto &doc : group.docs()) {
          ASSERT_EQ(nullptr, doc.vector());
          ASSERT_TRUE(doc.vector_string().empty());
        }
      }
      return;
    }
    if (test_case.is_sparse) {
      AssertSparseVectorsFetched(result, test_case.dimension);
    } else {
      AssertDenseVectorsFetched(result, test_case.dimension, test_case.name);
    }
  }

  void AssertDenseVectorsFetched(const SearchResult &result, uint32_t dimension,
                                 const std::string &case_name = "") {
    const bool has_reverted = !result.group_reverted_vector_list_.empty();
    if (has_reverted) {
      ASSERT_EQ(result.group_doc_list_.size(),
                result.group_reverted_vector_list_.size());
    }
    for (size_t group_idx = 0; group_idx < result.group_doc_list_.size();
         ++group_idx) {
      const auto &group = result.group_doc_list_[group_idx];
      const std::vector<std::string> *group_vectors = nullptr;
      if (has_reverted) {
        group_vectors = &result.group_reverted_vector_list_[group_idx];
        ASSERT_EQ(group.docs().size(), group_vectors->size());
      }
      for (size_t doc_idx = 0; doc_idx < group.docs().size(); ++doc_idx) {
        const auto &doc = group.docs()[doc_idx];
        const float expected = static_cast<float>(doc.key());
        const float *vector = nullptr;
        if (has_reverted) {
          vector =
              reinterpret_cast<const float *>((*group_vectors)[doc_idx].data());
        } else if (doc.vector() != nullptr) {
          vector = reinterpret_cast<const float *>(doc.vector());
        } else {
          // DiskAnn stores fetched vectors in vector_string_ rather than
          // the raw pointer field.
          ASSERT_FALSE(doc.vector_string().empty())
              << case_name << " key=" << doc.key();
          vector = reinterpret_cast<const float *>(doc.vector_string().data());
        }
        for (uint32_t i = 0; i < dimension; ++i) {
          ASSERT_FLOAT_EQ(expected, vector[i])
              << case_name << " key=" << doc.key() << " i=" << i;
        }
      }
    }
  }

  void AssertSparseVectorsFetched(const SearchResult &result,
                                  uint32_t dimension) {
    const bool has_reverted =
        !result.group_reverted_sparse_values_list_.empty();
    if (has_reverted) {
      ASSERT_EQ(result.group_doc_list_.size(),
                result.group_reverted_sparse_values_list_.size());
    }
    for (size_t group_idx = 0; group_idx < result.group_doc_list_.size();
         ++group_idx) {
      const auto &group = result.group_doc_list_[group_idx];
      const std::vector<std::string> *group_sparse_values = nullptr;
      if (has_reverted) {
        group_sparse_values =
            &result.group_reverted_sparse_values_list_[group_idx];
        ASSERT_EQ(group.docs().size(), group_sparse_values->size());
      }
      for (size_t doc_idx = 0; doc_idx < group.docs().size(); ++doc_idx) {
        const auto &doc = group.docs()[doc_idx];
        const auto &sparse = doc.sparse_doc();
        ASSERT_EQ(dimension, sparse.sparse_count());
        const auto *indices =
            reinterpret_cast<const uint32_t *>(sparse.sparse_indices().data());
        const float *values = nullptr;
        if (has_reverted) {
          values = reinterpret_cast<const float *>(
              (*group_sparse_values)[doc_idx].data());
        } else {
          values =
              reinterpret_cast<const float *>(sparse.sparse_values().data());
        }
        const float expected = static_cast<float>(doc.key());
        for (uint32_t i = 0; i < dimension; ++i) {
          ASSERT_EQ(i, indices[i]);
          ASSERT_FLOAT_EQ(expected, values[i]);
        }
      }
    }
  }
};

}  // namespace

TEST_F(GroupByInterfaceTest, Dense) {
  std::vector<GroupByCase> cases{
      {"dense_flat_graph", DenseFlatParam(), FlatQuery()},
      {"dense_flat_linear", DenseFlatParam(),
       FlatQuery(/*fetch_vector=*/false, /*is_linear=*/true,
                 /*with_bf_pks=*/false)},
      {"dense_flat_bf_pks", DenseFlatParam(),
       FlatQuery(/*fetch_vector=*/false, /*is_linear=*/false,
                 /*with_bf_pks=*/true)},
      {"dense_flat_fetch_vector", DenseFlatParam(),
       FlatQuery(/*fetch_vector=*/true, /*is_linear=*/false,
                 /*with_bf_pks=*/false)},
      {"dense_hnsw_graph", DenseHnswParam(), HnswQuery()},
      {"dense_hnsw_linear", DenseHnswParam(),
       HnswQuery(/*fetch_vector=*/false, /*is_linear=*/true)},
      {"dense_hnsw_bf_pks", DenseHnswParam(),
       HnswQuery(/*fetch_vector=*/false, /*is_linear=*/false,
                 /*with_bf_pks=*/true)},
      {"dense_hnsw_fetch_vector", DenseHnswParam(),
       HnswQuery(/*fetch_vector=*/true)},
#if RABITQ_SUPPORTED
      {"dense_hnsw_rabitq_graph", DenseHnswRabitqParam(64), HnswRabitqQuery(),
       /*is_sparse=*/false, /*dimension=*/64},
      {"dense_hnsw_rabitq_linear", DenseHnswRabitqParam(64),
       HnswRabitqQuery(/*fetch_vector=*/false, /*is_linear=*/true),
       /*is_sparse=*/false, /*dimension=*/64},
      {"dense_hnsw_rabitq_bf_pks", DenseHnswRabitqParam(64),
       HnswRabitqQuery(/*fetch_vector=*/false, /*is_linear=*/false,
                       /*with_bf_pks=*/true),
       /*is_sparse=*/false, /*dimension=*/64},
      {"dense_ivf_rabitq_graph", DenseIvfRabitqParam(64), IvfRabitqQuery(),
       /*is_sparse=*/false, /*dimension=*/64},
      {"dense_ivf_rabitq_linear", DenseIvfRabitqParam(64),
       IvfRabitqQuery(/*is_linear=*/true),
       /*is_sparse=*/false, /*dimension=*/64},
      {"dense_ivf_rabitq_bf_pks", DenseIvfRabitqParam(64),
       IvfRabitqQuery(/*is_linear=*/false, /*with_bf_pks=*/true),
       /*is_sparse=*/false, /*dimension=*/64},
      {"dense_ivf_rabitq_fetch_vector_ignored", DenseIvfRabitqParam(64),
       IvfRabitqQuery(/*is_linear=*/false, /*with_bf_pks=*/false,
                      /*fetch_vector=*/true),
       /*is_sparse=*/false, /*dimension=*/64},
      {"dense_ivf_rabitq_large_nprobe", DenseIvfRabitqParam(64),
       [] {
         auto query = IvfRabitqQuery();
         std::dynamic_pointer_cast<IVFRabitqQueryParam>(query)->nprobe = 1025;
         return query;
       }(),
       /*is_sparse=*/false, /*dimension=*/64},
  // IVF RaBitQ ignores fetch_vector; DB queries fetch original vectors from
  // the accompanying Flat index after recall.

#endif
  };

  for (const auto &test_case : cases) {
    RunOk(test_case);
  }
}

TEST_F(GroupByInterfaceTest, Sparse) {
  std::vector<GroupByCase> cases{
      {"sparse_flat_graph", SparseFlatParam(), FlatQuery(),
       /*is_sparse=*/true},
      {"sparse_hnsw_graph", SparseHnswParam(), HnswQuery(),
       /*is_sparse=*/true},
      {"sparse_hnsw_linear", SparseHnswParam(),
       HnswQuery(/*fetch_vector=*/false, /*is_linear=*/true),
       /*is_sparse=*/true},
      {"sparse_hnsw_bf_pks", SparseHnswParam(),
       HnswQuery(/*fetch_vector=*/false, /*is_linear=*/false,
                 /*with_bf_pks=*/true),
       /*is_sparse=*/true},
      {"sparse_hnsw_fetch_vector", SparseHnswParam(),
       HnswQuery(/*fetch_vector=*/true), /*is_sparse=*/true},
  };

  for (const auto &test_case : cases) {
    RunOk(test_case);
  }
}

TEST_F(GroupByInterfaceTest, UnsupportedIndexTypes) {
  std::vector<GroupByCase> cases{
      {"unsupported_vamana",
       VamanaIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_max_degree(32)
           .with_search_list_size(100)
           .with_alpha(1.2f)
           .build(),
       VamanaQueryParamBuilder()
           .with_topk(kSearchTopk)
           .with_ef_search(kSearchTopk)
           .build()},
      {"unsupported_ivf",
       IVFIndexParamBuilder()
           .with_metric_type(MetricType::kInnerProduct)
           .with_data_type(DataType::DT_FP32)
           .with_dimension(kDimension)
           .with_is_sparse(false)
           .with_n_list(4)
           .build(),
       IVFQueryParamBuilder().with_topk(kSearchTopk).build()},
      {"unsupported_refiner", DenseHnswParam(), HnswQuery(),
       /*is_sparse=*/false,
       /*dimension=*/kDimension,
       /*with_refiner=*/true},
#if RABITQ_SUPPORTED
      {"unsupported_ivf_rabitq_zero_nprobe", DenseIvfRabitqParam(64),
       [] {
         auto query = IvfRabitqQuery();
         std::dynamic_pointer_cast<IVFRabitqQueryParam>(query)->nprobe = 0;
         return query;
       }(),
       /*is_sparse=*/false, /*dimension=*/64},
#endif
#if DISKANN_SUPPORTED
      {"unsupported_diskann_graph", DenseDiskAnnParam(), DiskAnnQuery(),
       /*is_sparse=*/false, /*dimension=*/kDimension,
       /*with_refiner=*/false},
      {"unsupported_diskann_linear", DenseDiskAnnParam(),
       DiskAnnQuery(/*fetch_vector=*/false, /*is_linear=*/true),
       /*is_sparse=*/false, /*dimension=*/kDimension,
       /*with_refiner=*/false},
      {"unsupported_diskann_bf_pks", DenseDiskAnnParam(),
       DiskAnnQuery(/*fetch_vector=*/false, /*is_linear=*/false,
                    /*with_bf_pks=*/true),
       /*is_sparse=*/false, /*dimension=*/kDimension,
       /*with_refiner=*/false},
      {"unsupported_diskann_fetch_vector", DenseDiskAnnParam(),
       DiskAnnQuery(/*fetch_vector=*/true),
       /*is_sparse=*/false, /*dimension=*/kDimension,
       /*with_refiner=*/false},
#endif
  };

  for (const auto &test_case : cases) {
    RunRejected(test_case);
  }
}

#if DISKANN_SUPPORTED
TEST(DiskAnnInterfaceTest, PropagatesCommonQueryOptions) {
  const std::string source_name = "test_diskann_query_options_source";
  const std::string index_name = "test_diskann_query_options";
  zvec::test_util::RemoveTestFiles(source_name + "*");
  zvec::test_util::RemoveTestFiles(index_name + "*");

  auto source_param = FlatIndexParamBuilder()
                          .with_metric_type(MetricType::kL2sq)
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_is_sparse(false)
                          .build();
  auto source = IndexFactory::CreateAndInitIndex(*source_param);
  ASSERT_NE(source, nullptr);
  ASSERT_EQ(
      0, source->open(source_name, {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t key = 0; key < kNumDocs; ++key) {
    std::vector<float> values(kDimension, static_cast<float>(key));
    ASSERT_EQ(0, source->add(VectorData{DenseVector{values.data()}}, key));
  }
  ASSERT_EQ(0, source->train());

  auto diskann_param = DiskAnnIndexParamBuilder()
                           .with_metric_type(MetricType::kL2sq)
                           .with_data_type(DataType::DT_FP32)
                           .with_dimension(kDimension)
                           .with_is_sparse(false)
                           .with_max_degree(32)
                           .with_list_size(kSearchTopk)
                           .with_pq_chunk_num(0)
                           .build();
  auto index = IndexFactory::CreateAndInitIndex(*diskann_param);
  ASSERT_NE(index, nullptr);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, index->merge({source}, IndexFilter()));

  std::vector<float> query_values(kDimension, 3.0f);
  VectorData query{DenseVector{query_values.data()}};

  auto filtered_query = std::make_shared<DiskAnnQueryParam>();
  filtered_query->topk = 6;
  filtered_query->list_size = kSearchTopk;
  filtered_query->fetch_vector = true;
  filtered_query->filter = std::make_shared<IndexFilter>();
  filtered_query->filter->set([](uint64_t key) { return key == 3; });

  SearchResult filtered_result;
  ASSERT_EQ(0, index->search(query, filtered_query, &filtered_result));
  ASSERT_FALSE(filtered_result.doc_list_.empty());
  for (const auto &doc : filtered_result.doc_list_) {
    EXPECT_NE(3UL, doc.key());
    ASSERT_EQ(kDimension * sizeof(float), doc.vector_string().size());
    for (uint32_t i = 0; i < kDimension; ++i) {
      float vector_value = 0.0f;
      std::memcpy(&vector_value,
                  doc.vector_string().data() + i * sizeof(vector_value),
                  sizeof(vector_value));
      EXPECT_FLOAT_EQ(static_cast<float>(doc.key()), vector_value);
    }
  }

  auto radius_query = std::make_shared<DiskAnnQueryParam>();
  radius_query->topk = kNumDocs;
  radius_query->list_size = kSearchTopk;
  radius_query->radius = 1.0f;

  SearchResult radius_result;
  ASSERT_EQ(0, index->search(query, radius_query, &radius_result));
  ASSERT_FALSE(radius_result.doc_list_.empty());
  for (const auto &doc : radius_result.doc_list_) {
    EXPECT_LE(doc.score(), radius_query->radius);
  }

  ASSERT_EQ(0, index->close());
  ASSERT_EQ(0, source->close());

  // The thread-local context pool is shared by all DiskAnn indexes. Reusing
  // the prior L2 context for an InnerProduct index must copy the caller-facing
  // radius and denormalize it with the new metric.
  const std::string ip_source_name = "test_diskann_query_options_ip_source";
  const std::string ip_index_name = "test_diskann_query_options_ip";
  zvec::test_util::RemoveTestFiles(ip_source_name + "*");
  zvec::test_util::RemoveTestFiles(ip_index_name + "*");

  auto ip_source_param = FlatIndexParamBuilder()
                             .with_metric_type(MetricType::kInnerProduct)
                             .with_data_type(DataType::DT_FP32)
                             .with_dimension(kDimension)
                             .with_is_sparse(false)
                             .build();
  auto ip_source = IndexFactory::CreateAndInitIndex(*ip_source_param);
  ASSERT_NE(ip_source, nullptr);
  ASSERT_EQ(0, ip_source->open(ip_source_name,
                               {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t key = 0; key < kNumDocs; ++key) {
    std::vector<float> values(kDimension, static_cast<float>(key));
    ASSERT_EQ(0, ip_source->add(VectorData{DenseVector{values.data()}}, key));
  }
  ASSERT_EQ(0, ip_source->train());

  auto ip_diskann_param = DiskAnnIndexParamBuilder()
                              .with_metric_type(MetricType::kInnerProduct)
                              .with_data_type(DataType::DT_FP32)
                              .with_dimension(kDimension)
                              .with_is_sparse(false)
                              .with_max_degree(32)
                              .with_list_size(kNumDocs)
                              .with_pq_chunk_num(0)
                              .build();
  auto ip_index = IndexFactory::CreateAndInitIndex(*ip_diskann_param);
  ASSERT_NE(ip_index, nullptr);
  ASSERT_EQ(0, ip_index->open(ip_index_name,
                              {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, ip_index->merge({ip_source}, IndexFilter()));

  std::vector<float> ip_query_values(kDimension, 1.0f);
  VectorData ip_query{DenseVector{ip_query_values.data()}};
  auto ip_radius_query = std::make_shared<DiskAnnQueryParam>();
  ip_radius_query->topk = kNumDocs;
  ip_radius_query->list_size = kNumDocs;
  ip_radius_query->is_linear = true;
  ip_radius_query->radius = static_cast<float>(kDimension * 8);

  SearchResult ip_radius_result;
  ASSERT_EQ(0, ip_index->search(ip_query, ip_radius_query, &ip_radius_result));
  ASSERT_EQ(kNumDocs - 8, ip_radius_result.doc_list_.size());
  for (const auto &doc : ip_radius_result.doc_list_) {
    EXPECT_GE(doc.key(), 8UL);
    EXPECT_GE(doc.score(), ip_radius_query->radius);
  }

  ASSERT_EQ(0, ip_index->close());
  ASSERT_EQ(0, ip_source->close());
  zvec::test_util::RemoveTestFiles(source_name + "*");
  zvec::test_util::RemoveTestFiles(index_name + "*");
  zvec::test_util::RemoveTestFiles(ip_source_name + "*");
  zvec::test_util::RemoveTestFiles(ip_index_name + "*");
}
#endif
