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

// External Flat scans logical batches supplied by the source. A disk-backed
// source may read physical blocks and attach page pins to Batch::lease.
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>

using namespace zvec::core_interface;

class ExternalRows : public zvec::core::VectorSource {
 public:
  static constexpr uint32_t kDimension = 4;
  struct Row {
    uint32_t id;
    std::array<float, kDimension> vector;
  };
  std::vector<Row> rows;

  const void *get_vector(uint32_t id) const override {
    for (const auto &row : rows) {
      if (row.id == id) return row.vector.data();
    }
    return nullptr;
  }

  class Cursor : public Iterator {
   public:
    explicit Cursor(const ExternalRows &source) : source_(source) {}
    int next_batch(uint32_t max_count, Batch *out) override {
      if (!out || !max_count) return zvec::core::IndexError_InvalidArgument;
      out->clear();
      const size_t end = std::min(source_.rows.size(), position_ + max_count);
      for (; position_ < end; ++position_) {
        const auto &row = source_.rows[position_];
        out->ids.push_back(row.id);
        out->vectors.push_back(row.vector.data());
      }
      // No lease is needed: this example keeps rows immutable throughout
      // every request. Empty output signals end of scan.
      return 0;
    }

   private:
    const ExternalRows &source_;
    size_t position_{0};
  };

  Iterator::Pointer create_iterator() const override {
    return std::make_unique<Cursor>(*this);
  }
};

int main() {
  const std::string path = "external_flat_example.index";
  if (std::filesystem::exists(path)) {
    std::cerr << "Remove " << path << " before running this example.\n";
    return 1;
  }
  ExternalRows source;
  source.rows = {{42, {1, 0, 0, 0}}, {7, {0, 1, 0, 0}}, {105, {0, 0, 1, 0}}};
  auto param = FlatIndexParamBuilder()
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(ExternalRows::kDimension)
                   .with_metric_type(MetricType::kL2sq)
                   .with_use_external_vector(true)
                   .build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  if (!index || index->open(path, {StorageOptions::StorageType::kMMAP, true}))
    return 1;
  // Register IDs only; source owns vector data and subsequent updates.
  for (const auto &row : source.rows) {
    if (index->add_with_source(VectorData{DenseVector{row.vector.data()}},
                               row.id, source))
      return 1;
  }
  VectorData query{DenseVector{source.rows[0].vector.data()}};
  auto qp =
      FlatQueryParamBuilder().with_topk(1).with_fetch_vector(true).build();
  SearchResult result;
  if (index->search_with_source(query, qp, source, &result) ||
      result.doc_list_.size() != 1 || result.doc_list_[0].key() != 42 ||
      result.doc_list_[0].score() != 0)
    return 1;
  std::cout << "Full scan: id=42, distance=0\n";
  if (index->close()) return 1;

  // Reopen the persisted ID registry and bind the source on the next request.
  index = IndexFactory::CreateAndInitIndex(*param);
  if (!index || index->open(path, {StorageOptions::StorageType::kMMAP, false}))
    return 1;
  qp->bf_pks = std::make_shared<std::vector<uint64_t>>(
      std::initializer_list<uint64_t>{105, 42});
  if (index->search_with_source(query, qp, source, &result) ||
      result.doc_list_.size() != 1 || result.doc_list_[0].key() != 42)
    return 1;
  std::cout << "Candidate search after reopen: id=42\n";
  VectorDataBuffer fetched;
  if (index->fetch_with_source(42, source, &fetched) || index->close())
    return 1;
  std::filesystem::remove(path);
  return 0;
}
