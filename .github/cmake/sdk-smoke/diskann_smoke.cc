#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/query.h>
#include <zvec/db/query_params.h>
#include <zvec/db/schema.h>

namespace {

int Fail(const std::string &message) {
  std::cerr << "diskann_smoke: " << message << std::endl;
  return 1;
}

}  // namespace

int main() {
  using namespace zvec;

  const auto path =
      std::filesystem::temp_directory_path() / "zvec-sdk-diskann-smoke";
  std::filesystem::remove_all(path);
  constexpr size_t kDimension = 16;
  constexpr size_t kDocCount = 48;
  constexpr size_t kExpectedKey = 17;

  CollectionSchema schema("sdk_diskann_smoke");
  auto status = schema.add_field(std::make_shared<FieldSchema>(
      "embedding", DataType::VECTOR_FP32, kDimension, false,
      std::make_shared<DiskAnnIndexParams>(MetricType::L2, 16, 64, 2)));
  if (!status.ok()) {
    return Fail("DiskAnn schema rejected: " + status.message());
  }
  CollectionOptions options{false, true};
  auto created = Collection::CreateAndOpen(path.string(), schema, options);
  if (!created.has_value()) {
    return Fail("CreateAndOpen failed: " + created.error().message());
  }
  auto collection = std::move(created).value();

  std::vector<Doc> docs;
  std::vector<float> query_vector;
  for (size_t key = 0; key < kDocCount; ++key) {
    std::vector<float> vector(kDimension);
    double norm = 0;
    for (size_t dim = 0; dim < kDimension; ++dim) {
      vector[dim] = static_cast<float>(
          std::sin(static_cast<double>((key + 1) * (dim + 3))));
      norm += vector[dim] * vector[dim];
    }
    for (auto &value : vector) {
      value = static_cast<float>(value / std::sqrt(norm));
    }
    if (key == kExpectedKey) {
      query_vector = vector;
    }
    Doc doc;
    doc.set_pk(std::to_string(key));
    doc.set<std::vector<float>>("embedding", vector);
    docs.emplace_back(std::move(doc));
  }
  auto inserted = collection->insert(docs);
  if (!inserted.has_value() || inserted->size() != docs.size()) {
    return Fail("insert failed");
  }
  for (const auto &write : *inserted) {
    if (!write.ok()) {
      return Fail("a document failed to insert: " + write.message());
    }
  }
  if (!collection->flush().ok() || !collection->optimize().ok()) {
    return Fail("flush/optimize failed");
  }
  auto stats = collection->stats();
  if (!stats.has_value() || stats->index_completeness["embedding"] != 1) {
    return Fail("DiskAnn index was not built");
  }

  SearchQuery query;
  query.topk_ = 5;
  query.target_.field_name_ = "embedding";
  query.target_.query_params_ = std::make_shared<DiskAnnQueryParams>(64);
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(query_vector.data()),
                  query_vector.size() * sizeof(float)));
  auto matches = [&query](const Collection::Ptr &coll) {
    auto result = coll->query(query);
    return result.has_value() && !result->empty() &&
           result->front()->pk() == std::to_string(kExpectedKey) &&
           std::isfinite(result->front()->score());
  };
  if (!matches(collection)) {
    return Fail("query did not return the expected document");
  }
  if (!collection->close().ok()) {
    return Fail("close failed");
  }
  collection.reset();

  options.read_only_ = true;
  auto reopened = Collection::Open(path.string(), options);
  if (!reopened.has_value()) {
    return Fail("read-only reopen failed: " + reopened.error().message());
  }
  collection = std::move(reopened).value();
  if (!matches(collection)) {
    return Fail("query after read-only reopen returned unexpected documents");
  }
  if (!collection->close().ok()) {
    return Fail("read-only close failed");
  }
  collection.reset();
  std::filesystem::remove_all(path);
  std::cout
      << "diskann_smoke: built, queried, and reopened DiskAnn from the SDK"
      << std::endl;
  return 0;
}
