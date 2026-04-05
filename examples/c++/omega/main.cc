#include <filesystem>
#include <iostream>
#include <vector>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>

using namespace zvec::core_interface;

namespace {

constexpr uint32_t kDimension = 32;
const std::string kIndexPath = "omega_example.index";

BaseIndexParam::Pointer CreateOmegaParam() {
  auto param = HNSWIndexParamBuilder()
                   .WithMetricType(MetricType::kInnerProduct)
                   .WithDataType(DataType::DT_FP32)
                   .WithDimension(kDimension)
                   .WithIsSparse(false)
                   .WithM(8)
                   .WithEFConstruction(64)
                   .Build();
  param->index_type = IndexType::kOMEGA;
  return param;
}

}  // namespace

int main() {
  std::filesystem::remove_all(kIndexPath);

  auto index = IndexFactory::CreateAndInitIndex(*CreateOmegaParam());
  if (!index) {
    std::cerr << "failed to create omega index" << std::endl;
    return 1;
  }

  if (index->Open(kIndexPath,
                  StorageOptions{StorageOptions::StorageType::kMMAP, true}) !=
      0) {
    std::cerr << "failed to open omega index" << std::endl;
    return 1;
  }

  for (uint32_t doc_id = 0; doc_id < 6; ++doc_id) {
    std::vector<float> values(kDimension, static_cast<float>(doc_id) / 10.0f);
    values[0] = 1.0f + static_cast<float>(doc_id);
    VectorData vector_data;
    vector_data.vector = DenseVector{values.data()};
    if (index->Add(vector_data, doc_id) != 0) {
      std::cerr << "failed to add document " << doc_id << std::endl;
      return 1;
    }
  }

  if (index->Train() != 0) {
    std::cerr << "failed to train omega index" << std::endl;
    return 1;
  }

  std::vector<float> query_values(kDimension, 0.0f);
  query_values[0] = 1.0f;
  VectorData query{DenseVector{query_values.data()}};

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
  if (index->Close() != 0) {
    std::cerr << "failed to close omega index" << std::endl;
    return 1;
  }

  return 0;
}
