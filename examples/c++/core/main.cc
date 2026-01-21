#include <cstdlib>
#include <iostream>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include "zvec/core/interface/index_param_builders.h"

using namespace zvec::core_interface;

constexpr uint32_t kDimension = 64;
const std::string index_name{"test.index"};

Index::Pointer create_index(const BaseIndexParam::Pointer &param,
                            int doc_num = 10) {
  auto index = IndexFactory::CreateAndInitIndex(*param);
  if (!index) {
    std::cout << "Failed to create index." << std::endl;
    return nullptr;
  }

  int ret = index->Open(
      index_name, StorageOptions{StorageOptions::StorageType::kMMAP, true});
  if (ret != 0) {
    std::cout << "Failed to open index." << std::endl;
    return nullptr;
  }

  for (int i = 0; i < doc_num; ++i) {
    std::vector<float> vector(kDimension, i / 10.0f + 0.1f);
    VectorData vector_data;
    vector_data.vector = DenseVector{vector.data()};
    ret = index->Add(vector_data, i);
    if (ret != 0) {
      std::cout << "Failed to add to index." << std::endl;
      return nullptr;
    }
  }

  ret = index->Train();
  if (ret != 0) {
    std::cout << "Failed to train index." << std::endl;
    return nullptr;
  }

  return index;
}

int main() {
  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_name.c_str());
  system(cmd_buf);

  auto param = HNSWIndexParamBuilder()
                   .WithMetricType(MetricType::kInnerProduct)
                   .WithDataType(DataType::DT_FP32)
                   .WithDimension(kDimension)
                   .WithIsSparse(false)
                   .Build();
  auto index = create_index(param, 1);
  std::cout << "index stats: " << index->GetDocCount() << std::endl;

  // query
  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(10)
                         .with_fetch_vector(true)
                         .with_ef_search(20)
                         .build();

  SearchResult result;
  VectorData query;
  std::vector<float> vector(kDimension, 0.1f);
  query.vector = DenseVector{vector.data()};
  int ret = index->Search(query, query_param, &result);
  if (ret != 0) {
    std::cout << "Failed to search index." << std::endl;
    return -1;
  }

  std::cout << "query results: " << result.doc_list_.size() << std::endl;
  if (result.doc_list_.size() == 0) {
    std::cout << "No results found." << std::endl;
    return -1;
  }

  std::cout << "key: " << result.doc_list_[0].key()
            << ", score: " << result.doc_list_[0].score() << std::endl;

  return 0;
}