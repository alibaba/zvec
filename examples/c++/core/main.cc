#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>

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

  int ret = index->open(
      index_name, StorageOptions{StorageOptions::StorageType::kMMAP, true});
  if (ret != 0) {
    std::cout << "Failed to open index." << std::endl;
    return nullptr;
  }

  for (int i = 0; i < doc_num; ++i) {
    std::vector<float> vector(kDimension, i / 10.0f + 0.1f);
    VectorData vector_data;
    vector_data.vector = DenseVector{vector.data()};
    ret = index->add(vector_data, i);
    if (ret != 0) {
      std::cout << "Failed to add to index." << std::endl;
      return nullptr;
    }
  }

  ret = index->train();
  if (ret != 0) {
    std::cout << "Failed to train index." << std::endl;
    return nullptr;
  }

  return index;
}

int main() {
  std::filesystem::remove(index_name);

  auto param = HNSWIndexParamBuilder()
                   .with_metric_type(MetricType::kInnerProduct)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_is_sparse(false)
                   .build();
  auto index = create_index(param, 1);
  std::cout << "index stats: " << index->get_doc_count() << std::endl;

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
  int ret = index->search(query, query_param, &result);
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