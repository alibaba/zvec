#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/query.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>

namespace {

int Fail(const std::string &message) {
  std::cerr << "cpp_smoke: " << message << std::endl;
  return 1;
}

}  // namespace

int main() {
  using namespace zvec;

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "zvec-sdk-smoke-db";
  std::filesystem::remove_all(path);

  CollectionSchema schema("sdk_smoke");
  schema.add_field(std::make_shared<FieldSchema>(
      "embedding", DataType::VECTOR_FP32, 4, false,
      std::make_shared<HnswIndexParams>(MetricType::IP)));

  CollectionOptions options{false, true};
  auto create_result =
      Collection::CreateAndOpen(path.string(), schema, options);
  if (!create_result.has_value()) {
    return Fail("CreateAndOpen failed: " + create_result.error().message());
  }
  auto collection = std::move(create_result).value();

  std::vector<Doc> docs;
  for (int i = 0; i < 3; ++i) {
    Doc doc;
    doc.set_pk("doc_" + std::to_string(i));
    doc.set<std::vector<float>>("embedding",
                                std::vector<float>{0.1F + i, 0.2F, 0.3F, 0.4F});
    docs.emplace_back(std::move(doc));
  }

  auto insert_result = collection->insert(docs);
  if (!insert_result.has_value()) {
    return Fail("insert failed: " + insert_result.error().message());
  }
  if (!collection->flush().ok()) {
    return Fail("flush failed");
  }
  if (!collection->optimize().ok()) {
    return Fail("optimize failed");
  }

  SearchQuery query;
  query.topk_ = 3;
  query.target_.field_name_ = "embedding";
  const std::vector<float> query_vector{0.1F, 0.2F, 0.3F, 0.4F};
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(query_vector.data()),
                  query_vector.size() * sizeof(float)));
  auto query_result = collection->query(query);
  if (!query_result.has_value() || query_result.value().empty()) {
    return Fail("query returned no documents");
  }

  if (!collection->close().ok()) {
    return Fail("close failed");
  }
  collection.reset();

  options.read_only_ = true;
  auto reopen_result = Collection::Open(path.string(), options);
  if (!reopen_result.has_value()) {
    return Fail("read-only reopen failed: " + reopen_result.error().message());
  }
  auto reopened = std::move(reopen_result).value();
  auto reopened_query_result = reopened->query(query);
  if (!reopened_query_result.has_value() ||
      reopened_query_result.value().empty()) {
    return Fail("query after reopen returned no documents");
  }
  if (!reopened->close().ok()) {
    return Fail("read-only close failed");
  }
  reopened.reset();

  std::filesystem::remove_all(path);
  const char *message = GetDefaultMessage(StatusCode::OK);
  if (!message) {
    return Fail("GetDefaultMessage() returned null");
  }
  std::cout << "cpp_smoke: inserted, optimized, queried, and reopened 3 docs"
            << std::endl;
  std::cout << "cpp_smoke: StatusCode::OK -> " << message << std::endl;
  std::cout << "cpp_smoke: OK" << std::endl;
  return 0;
}
