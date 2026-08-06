#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>

using namespace zvec::core_interface;

namespace {

constexpr uint32_t kDimension = 16;
constexpr uint32_t kDocCount = 128;
constexpr uint32_t kExpectedKey = 42;
const std::string kIndexPath{"diskann-core-example.index"};

int Fail(const std::string &message, const Index::Pointer &index = nullptr) {
  std::cerr << message << std::endl;
  if (index) {
    index->Close();
  }
  std::filesystem::remove(kIndexPath);
  return EXIT_FAILURE;
}

}  // namespace

int main() {
  std::filesystem::remove(kIndexPath);

  auto param = DiskAnnIndexParamBuilder()
                   .WithMetricType(MetricType::kL2sq)
                   .WithDataType(DataType::DT_FP32)
                   .WithDimension(kDimension)
                   .WithIsSparse(false)
                   .WithMaxDegree(32)
                   .WithListSize(64)
                   .WithPqChunkNum(0)
                   .Build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  if (!index) {
    return Fail("Failed to create the DiskANN index.");
  }

  int ret = index->Open(kIndexPath, {StorageOptions::StorageType::kMMAP, true});
  if (ret != 0) {
    return Fail("Failed to open the DiskANN index.", index);
  }

  for (uint32_t key = 0; key < kDocCount; ++key) {
    std::vector<float> values(kDimension, static_cast<float>(key));
    ret = index->Add(VectorData{DenseVector{values.data()}}, key);
    if (ret != 0) {
      return Fail("Failed to add a vector to the DiskANN index.", index);
    }
  }

  if (index->Train() != 0) {
    return Fail("Failed to build the DiskANN index.", index);
  }

  std::vector<float> query_values(kDimension, static_cast<float>(kExpectedKey));
  VectorData query{DenseVector{query_values.data()}};
  auto query_param = std::make_shared<DiskAnnQueryParam>();
  query_param->topk = 5;
  query_param->list_size = 64;

  SearchResult result;
  if (index->Search(query, query_param, &result) != 0) {
    return Fail("Failed to query the DiskANN index.", index);
  }

  const auto expected =
      std::find_if(result.doc_list_.begin(), result.doc_list_.end(),
                   [](const auto &doc) { return doc.key() == kExpectedKey; });
  if (expected == result.doc_list_.end()) {
    return Fail("DiskANN query did not return the expected document.", index);
  }

  if (index->Close() != 0) {
    return Fail("Failed to close the DiskANN index.");
  }
  std::filesystem::remove(kIndexPath);
  std::cout << "DiskANN core-only integration passed." << std::endl;
  return EXIT_SUCCESS;
}
