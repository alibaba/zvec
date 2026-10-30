// Exercises the *advanced* engine headers, which the package carries only when
// it was configured with -DZVEC_INSTALL_ENGINE_HEADERS=ON; the default package
// does not, and this test is skipped there.
//
// The engine API is served by libzvec -- or by the standalone libzvec_core when
// the package was built with -DZVEC_INSTALL_ENGINE_SHARED_LIBRARIES=ON -- so
// shipping those headers without symbols that actually resolve would be a silent
// regression the stable-API smoke test cannot catch.  This builds a one-doc HNSW
// index through the public engine API to prove both the headers and the exports.
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>

namespace {

constexpr uint32_t kDimension = 8;

int Fail(const std::string &message) {
  std::cerr << "core_smoke: " << message << std::endl;
  return 1;
}

}  // namespace

int main() {
  using namespace zvec::core_interface;

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "zvec-sdk-core-smoke.index";
  std::filesystem::remove_all(path);

  const auto param = HNSWIndexParamBuilder()
                         .with_metric_type(MetricType::kInnerProduct)
                         .with_data_type(DataType::DT_FP32)
                         .with_dimension(kDimension)
                         .with_is_sparse(false)
                         .build();

  Index::Pointer index = IndexFactory::CreateAndInitIndex(*param);
  if (!index) {
    return Fail("IndexFactory::CreateAndInitIndex returned null");
  }

  if (index->open(path.string(),
                  StorageOptions{StorageOptions::StorageType::kMMAP, true}) !=
      0) {
    return Fail("Index::open failed");
  }

  const std::vector<float> vector(kDimension, 0.5f);
  VectorData vector_data;
  vector_data.vector = DenseVector{vector.data()};
  if (index->add(vector_data, 0) != 0) {
    return Fail("Index::add failed");
  }
  if (index->train() != 0) {
    return Fail("Index::train failed");
  }
  if (index->get_doc_count() != 1) {
    return Fail("unexpected doc count: " +
                std::to_string(index->get_doc_count()));
  }

  index->close();
  std::filesystem::remove_all(path);
  std::cout << "core_smoke: engine API OK" << std::endl;
  return 0;
}
