#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>

using namespace zvec::core_interface;

namespace {

constexpr uint32_t kDimension = 32;
constexpr uint32_t kNumDocuments = 10000;
constexpr uint32_t kQueryDocId = 7777;
const std::string kIndexPath = "omega_example.index";

std::vector<float> MakeRandomUnitVector(std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> values(kDimension, 0.0f);
  float norm_sq = 0.0f;
  for (auto &value : values) {
    value = dist(rng);
    norm_sq += value * value;
  }

  const float norm = std::sqrt(norm_sq);
  if (norm > 0.0f) {
    for (auto &value : values) {
      value /= norm;
    }
  }
  return values;
}

BaseIndexParam::Pointer CreateOmegaParam() {
  return OmegaIndexParamBuilder()
      .WithMetricType(MetricType::kInnerProduct)
      .WithDataType(DataType::DT_FP32)
      .WithDimension(kDimension)
      .WithIsSparse(false)
      .WithM(32)
      .WithEFConstruction(500)
      .WithMinVectorThreshold(10000)
      .WithNumTrainingQueries(1000)
      .WithEFTraining(500)
      .WithEFGroundTruth(1000)
      .Build();
}

}  // namespace

int main() {
  std::filesystem::remove_all(kIndexPath);
  std::filesystem::remove_all("omega_model");

  auto index = IndexFactory::CreateAndInitIndex(*CreateOmegaParam());
  if (!index) {
    std::cerr << "failed to create omega index" << std::endl;
    return 1;
  }

  if (index->Open(kIndexPath, StorageOptions{StorageOptions::StorageType::kMMAP,
                                             true}) != 0) {
    std::cerr << "failed to open omega index" << std::endl;
    return 1;
  }

  std::mt19937 rng(42);
  std::vector<std::vector<float>> dataset;
  dataset.reserve(kNumDocuments);
  for (uint32_t doc_id = 0; doc_id < kNumDocuments; ++doc_id) {
    dataset.push_back(MakeRandomUnitVector(rng));
    VectorData vector_data;
    vector_data.vector = DenseVector{dataset.back().data()};
    if (index->Add(vector_data, doc_id) != 0) {
      std::cerr << "failed to add document " << doc_id << std::endl;
      return 1;
    }
  }

  if (index->Train() != 0) {
    std::cerr << "failed to train omega index" << std::endl;
    return 1;
  }
  if (!std::filesystem::exists("omega_model/model.txt")) {
    std::cerr << "omega model was not generated" << std::endl;
    return 1;
  }

  VectorData query{DenseVector{dataset[kQueryDocId].data()}};

  auto query_param = OmegaQueryParamBuilder()
                         .with_topk(3)
                         .with_fetch_vector(true)
                         .with_ef_search(32)
                         .with_target_recall(0.95f)
                         .build();

  SearchResult result;
  if (index->Search(query, query_param, &result) != 0) {
    std::cerr << "failed to search omega index" << std::endl;
    return 1;
  }

  std::cout << "omega results: " << result.doc_list_.size() << std::endl;
  if (result.doc_list_.empty()) {
    std::cerr << "omega example returned no results" << std::endl;
    return 1;
  }

  std::cout << "top result key=" << result.doc_list_[0].key()
            << " score=" << result.doc_list_[0].score() << std::endl;
  if (result.doc_list_[0].key() != kQueryDocId) {
    std::cerr << "unexpected top result key" << std::endl;
    return 1;
  }
  if (index->Close() != 0) {
    std::cerr << "failed to close omega index" << std::endl;
    return 1;
  }

  return 0;
}
