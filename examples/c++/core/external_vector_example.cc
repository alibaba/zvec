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

/// @file external_vector_example.cc
/// @brief Demonstrates using HNSW index in external-vector mode.
///
/// In external-vector mode the index does NOT store raw vectors internally.
/// Instead, a user-provided VectorSource is passed on every Add/Search call
/// so the index can fetch vectors on demand.  This is useful when vectors are
/// already stored elsewhere (e.g. a columnar store, mmap file, remote storage)
/// and you want to avoid duplicating them inside the index.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/core/interface/vector_source.h>

using namespace zvec::core_interface;

// ---------------------------------------------------------------------------
// A simple VectorSource backed by an in-memory float matrix.
// In production this could be backed by mmap, a database, or remote storage.
// ---------------------------------------------------------------------------
class InMemoryVectorSource : public zvec::core::VectorSource {
 public:
  /// @param base     Pointer to a contiguous float array of shape [n, dim].
  /// @param dim      Dimensionality of each vector.
  InMemoryVectorSource(const float *base, uint32_t dim)
      : base_(base), dim_(dim) {}

  const void *get_vector(uint32_t node_id) const override {
    return base_ + static_cast<size_t>(node_id) * dim_;
  }

 private:
  const float *base_;
  uint32_t dim_;
};

// ---------------------------------------------------------------------------
// Helper: generate random float vectors in [0, 1)
// ---------------------------------------------------------------------------
static std::vector<float> generate_random_vectors(uint32_t count,
                                                  uint32_t dim) {
  std::vector<float> data(static_cast<size_t>(count) * dim);
  for (auto &v : data) {
    v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Helper: brute-force kNN (L2) for recall verification
// ---------------------------------------------------------------------------
static std::vector<uint32_t> brute_force_knn(const float *base, uint32_t n,
                                             uint32_t dim, const float *query,
                                             uint32_t topk) {
  std::vector<std::pair<float, uint32_t>> dists(n);
  for (uint32_t i = 0; i < n; ++i) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
      float diff = base[static_cast<size_t>(i) * dim + d] - query[d];
      dist += diff * diff;
    }
    dists[i] = {dist, i};
  }
  std::partial_sort(dists.begin(), dists.begin() + topk, dists.end());
  std::vector<uint32_t> result(topk);
  for (uint32_t i = 0; i < topk; ++i) {
    result[i] = dists[i].second;
  }
  return result;
}

int main() {
  constexpr uint32_t kDimension = 32;
  constexpr uint32_t kDocCount = 200;
  constexpr uint32_t kTopK = 5;
  const std::string index_path = "external_vector_example.index";

  // Clean up any previous run
  std::filesystem::remove_all(index_path);

  // ------ Step 1: Generate random vector data (simulating external storage)
  std::srand(42);
  auto vectors = generate_random_vectors(kDocCount, kDimension);

  // Wrap data in our VectorSource
  InMemoryVectorSource source(vectors.data(), kDimension);

  // ------ Step 2: Build HNSW index with external-vector mode enabled
  auto param = HNSWIndexParamBuilder()
                   .WithMetricType(MetricType::kL2sq)
                   .WithDataType(DataType::DT_FP32)
                   .WithDimension(kDimension)
                   .WithIsSparse(false)
                   .WithUseExternalVector(true)  // <-- key setting
                   .Build();

  auto index = IndexFactory::CreateAndInitIndex(*param);
  if (!index) {
    std::cerr << "Failed to create index." << std::endl;
    return 1;
  }

  int ret = index->Open(
      index_path, StorageOptions{StorageOptions::StorageType::kMMAP, true});
  if (ret != 0) {
    std::cerr << "Failed to open index." << std::endl;
    return 1;
  }

  // ------ Step 3: Add vectors using AddWithSource
  for (uint32_t i = 0; i < kDocCount; ++i) {
    VectorData vd;
    vd.vector =
        DenseVector{vectors.data() + static_cast<size_t>(i) * kDimension};
    ret = index->AddWithSource(vd, i, source);
    if (ret != 0) {
      std::cerr << "Failed to add doc " << i << std::endl;
      return 1;
    }
  }
  std::cout << "[OK] Added " << kDocCount << " vectors in external mode."
            << std::endl;

  // ------ Step 4: Search using SearchWithSource
  auto query_param =
      HNSWQueryParamBuilder().with_topk(kTopK).with_ef_search(64).build();

  // Use the first vector as query
  const float *query_vec = vectors.data();
  VectorData query;
  query.vector = DenseVector{query_vec};

  SearchResult result;
  ret = index->SearchWithSource(query, query_param, source, &result);
  if (ret != 0) {
    std::cerr << "Search failed." << std::endl;
    return 1;
  }

  std::cout << "[OK] Search returned " << result.doc_list_.size() << " results."
            << std::endl;

  // The closest vector to vectors[0] should be itself (doc_id=0)
  if (!result.doc_list_.empty() && result.doc_list_[0].key() == 0) {
    std::cout << "[OK] Nearest neighbor is doc_id=0 (self), score="
              << result.doc_list_[0].score() << std::endl;
  }

  // ------ Step 5: Verify recall against brute-force
  auto gt =
      brute_force_knn(vectors.data(), kDocCount, kDimension, query_vec, kTopK);
  uint32_t hits = 0;
  for (const auto &doc : result.doc_list_) {
    if (std::find(gt.begin(), gt.end(), static_cast<uint32_t>(doc.key())) !=
        gt.end()) {
      ++hits;
    }
  }
  float recall = static_cast<float>(hits) / static_cast<float>(kTopK);
  std::cout << "[OK] Recall@" << kTopK << " = " << recall * 100.0f << "%"
            << std::endl;

  // ------ Step 6: Reopen index and search again (persistence verification)
  index->Close();
  std::cout << "[OK] Index closed." << std::endl;

  // Must re-create index instance with same params before reopening
  index = IndexFactory::CreateAndInitIndex(*param);
  if (!index) {
    std::cerr << "Failed to re-create index for reopen." << std::endl;
    return 1;
  }

  ret = index->Open(index_path,
                    StorageOptions{StorageOptions::StorageType::kMMAP, false});
  if (ret != 0) {
    std::cerr << "Failed to reopen index." << std::endl;
    return 1;
  }

  SearchResult result2;
  ret = index->SearchWithSource(query, query_param, source, &result2);
  if (ret != 0) {
    std::cerr << "Search after reopen failed." << std::endl;
    return 1;
  }

  std::cout << "[OK] After reopen: search returned " << result2.doc_list_.size()
            << " results, top1 doc_id=" << result2.doc_list_[0].key()
            << std::endl;

  // Cleanup
  index->Close();
  std::filesystem::remove_all(index_path);
  std::cout << "\n=== External Vector Example Complete ===" << std::endl;
  return 0;
}
