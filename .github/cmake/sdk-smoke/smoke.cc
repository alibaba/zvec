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
  schema.add_field(std::make_shared<FieldSchema>(
      "sparse_embedding", DataType::SPARSE_VECTOR_FP32, 0, false,
      std::make_shared<HnswIndexParams>(MetricType::IP)));
  schema.add_field(
      std::make_shared<FieldSchema>("content", DataType::STRING, false,
                                    std::make_shared<FtsIndexParams>("jieba")));

  CollectionOptions options{false, true};
  auto create_result =
      Collection::CreateAndOpen(path.string(), schema, options);
  if (!create_result.has_value()) {
    return Fail("CreateAndOpen failed: " + create_result.error().message());
  }
  auto collection = std::move(create_result).value();

  const std::vector<std::string> contents{
      u8"\u6211\u7231\u5317\u4eac\u5929\u5b89\u95e8",
      u8"\u5357\u4eac\u5e02\u957f\u6c5f\u5927\u6865",
      u8"\u81ea\u7136\u8bed\u8a00\u5904\u7406"};
  std::vector<Doc> docs;
  for (int i = 0; i < 3; ++i) {
    Doc doc;
    doc.set_pk("doc_" + std::to_string(i));
    doc.set<std::vector<float>>("embedding",
                                std::vector<float>{0.1F + i, 0.2F, 0.3F, 0.4F});
    doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
        "sparse_embedding", std::make_pair(std::vector<uint32_t>{1, 3},
                                           std::vector<float>{1.0F + i, 0.5F}));
    doc.set<std::string>("content", contents[i]);
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
    return Fail("dense query returned no documents");
  }

  SearchQuery sparse_query;
  sparse_query.topk_ = 3;
  sparse_query.target_.field_name_ = "sparse_embedding";
  const std::vector<uint32_t> sparse_query_indices{1, 3};
  const std::vector<float> sparse_query_values{1.0F, 0.5F};
  sparse_query.target_.set_sparse_vector(
      std::string(reinterpret_cast<const char *>(sparse_query_indices.data()),
                  sparse_query_indices.size() * sizeof(uint32_t)),
      std::string(reinterpret_cast<const char *>(sparse_query_values.data()),
                  sparse_query_values.size() * sizeof(float)));
  auto sparse_query_result = collection->query(sparse_query);
  if (!sparse_query_result.has_value() || sparse_query_result.value().empty()) {
    return Fail("sparse query returned no documents");
  }

  // The default dictionary and HMM model must work from the packaged
  // library alone, including after the collection is reopened.
  SearchQuery fts_query;
  fts_query.topk_ = 3;
  fts_query.target_.field_name_ = "content";
  FtsClause fts_clause;
  fts_clause.match_string_ = u8"\u5929\u5b89\u95e8";
  fts_query.target_.clause_ = std::move(fts_clause);
  auto fts_query_result = collection->query(fts_query);
  if (!fts_query_result.has_value()) {
    return Fail("Chinese FTS query failed: " +
                fts_query_result.error().message());
  }
  if (fts_query_result.value().size() != 1 ||
      fts_query_result.value()[0]->pk() != "doc_0" ||
      fts_query_result.value()[0]->score() <= 0.0F) {
    return Fail("Chinese FTS query did not return the expected document");
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
    return Fail("dense query after reopen returned no documents");
  }
  auto reopened_sparse_query_result = reopened->query(sparse_query);
  if (!reopened_sparse_query_result.has_value() ||
      reopened_sparse_query_result.value().empty()) {
    return Fail("sparse query after reopen returned no documents");
  }
  auto reopened_fts_query_result = reopened->query(fts_query);
  if (!reopened_fts_query_result.has_value()) {
    return Fail("Chinese FTS query after reopen failed: " +
                reopened_fts_query_result.error().message());
  }
  if (reopened_fts_query_result.value().size() != 1 ||
      reopened_fts_query_result.value()[0]->pk() != "doc_0" ||
      reopened_fts_query_result.value()[0]->score() <= 0.0F) {
    return Fail("Chinese FTS query after reopen returned unexpected documents");
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
  std::cout << "cpp_smoke: inserted, optimized, queried dense/sparse/Chinese "
               "FTS, and reopened 3 docs"
            << std::endl;
  std::cout << "cpp_smoke: StatusCode::OK -> " << message << std::endl;
  std::cout << "cpp_smoke: OK" << std::endl;
  return 0;
}
