#include <cstdlib>
#include <string>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>

using namespace zvec;

Doc create_doc(const uint64_t doc_id, const CollectionSchema &schema,
               std::string pk = "") {
  Doc new_doc;
  if (pk.empty()) {
    pk = "pk_" + std::to_string(doc_id);
  }
  new_doc.set_pk(pk);

  for (auto &field : schema.fields()) {
    switch (field->data_type()) {
      case DataType::BINARY: {
        std::string binary_str("binary_" + std::to_string(doc_id));
        new_doc.set<std::string>(field->name(), binary_str);
        break;
      }
      case DataType::BOOL:
        new_doc.set<bool>(field->name(), doc_id % 10 == 0);
        break;
      case DataType::INT32:
        new_doc.set<int32_t>(field->name(), (int32_t)doc_id);
        break;
      case DataType::INT64:
        new_doc.set<int64_t>(field->name(), (int64_t)doc_id);
        break;
      case DataType::UINT32:
        new_doc.set<uint32_t>(field->name(), (uint32_t)doc_id);
        break;
      case DataType::UINT64:
        new_doc.set<uint64_t>(field->name(), (uint64_t)doc_id);
        break;
      case DataType::FLOAT:
        new_doc.set<float>(field->name(), (float)doc_id);
        break;
      case DataType::DOUBLE:
        new_doc.set<double>(field->name(), (double)doc_id);
        break;
      case DataType::STRING:
        new_doc.set<std::string>(field->name(),
                                 "value_" + std::to_string(doc_id));
        break;
      case DataType::ARRAY_BINARY: {
        std::vector<std::string> bin_vec;
        for (size_t i = 0; i < (doc_id % 10); i++) {
          bin_vec.push_back("bin_" + std::to_string(i));
        }
        new_doc.set<std::vector<std::string>>(field->name(), bin_vec);
        break;
      }
      case DataType::ARRAY_BOOL:
        new_doc.set<std::vector<bool>>(field->name(),
                                       std::vector<bool>(10, doc_id % 10 == 0));
        break;
      case DataType::ARRAY_INT32:
        new_doc.set<std::vector<int32_t>>(
            field->name(), std::vector<int32_t>(10, (int32_t)doc_id));
        break;
      case DataType::ARRAY_INT64:
        new_doc.set<std::vector<int64_t>>(
            field->name(), std::vector<int64_t>(10, (int64_t)doc_id));
        break;
      case DataType::ARRAY_UINT32:
        new_doc.set<std::vector<uint32_t>>(
            field->name(), std::vector<uint32_t>(10, (uint32_t)doc_id));
        break;
      case DataType::ARRAY_UINT64:
        new_doc.set<std::vector<uint64_t>>(
            field->name(), std::vector<uint64_t>(10, (uint64_t)doc_id));
        break;
      case DataType::ARRAY_FLOAT:
        new_doc.set<std::vector<float>>(field->name(),
                                        std::vector<float>(10, (float)doc_id));
        break;
      case DataType::ARRAY_DOUBLE:
        new_doc.set<std::vector<double>>(
            field->name(), std::vector<double>(10, (double)doc_id));
        break;
      case DataType::ARRAY_STRING:
        new_doc.set<std::vector<std::string>>(
            field->name(),
            std::vector<std::string>(10, "value_" + std::to_string(doc_id)));
        break;
      case DataType::VECTOR_BINARY32:
        new_doc.set<std::vector<uint32_t>>(
            field->name(),
            std::vector<uint32_t>(field->dimension(), uint32_t(doc_id + 0.1)));
        break;
      case DataType::VECTOR_BINARY64:
        new_doc.set<std::vector<uint64_t>>(
            field->name(),
            std::vector<uint64_t>(field->dimension(), uint64_t(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP32:
        new_doc.set<std::vector<float>>(
            field->name(),
            std::vector<float>(field->dimension(), float(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP64:
        new_doc.set<std::vector<double>>(
            field->name(),
            std::vector<double>(field->dimension(), double(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP16:
        new_doc.set<std::vector<float16_t>>(
            field->name(), std::vector<float16_t>(
                               field->dimension(),
                               static_cast<float16_t>(float(doc_id + 0.1))));
        break;
      case DataType::VECTOR_INT8:
        new_doc.set<std::vector<int8_t>>(
            field->name(),
            std::vector<int8_t>(field->dimension(), (int8_t)doc_id));
        break;
      case DataType::VECTOR_INT16:
        new_doc.set<std::vector<int16_t>>(
            field->name(),
            std::vector<int16_t>(field->dimension(), (int16_t)doc_id));
        break;
      case DataType::SPARSE_VECTOR_FP16: {
        std::vector<uint32_t> indices;
        std::vector<float16_t> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float16_t(float(doc_id + 0.1)));
        }
        std::pair<std::vector<uint32_t>, std::vector<float16_t>>
            sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
            field->name(), sparse_float_vec);
        break;
      }
      case DataType::SPARSE_VECTOR_FP32: {
        std::vector<uint32_t> indices;
        std::vector<float> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float(doc_id + 0.1));
        }
        std::pair<std::vector<uint32_t>, std::vector<float>> sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
            field->name(), sparse_float_vec);
        break;
      }
      default:
        std::cout << "Unsupported data type: " << field->name() << std::endl;
        throw std::runtime_error("Unsupported vector data type");
    }
  }

  return new_doc;
}

CollectionSchema::Ptr create_schema() {
  auto schema = std::make_shared<CollectionSchema>("demo");
  schema->set_max_doc_count_per_segment(1000);

  schema->add_field(std::make_shared<FieldSchema>(
      "id", DataType::INT64, false, std::make_shared<InvertIndexParams>(true)));
  schema->add_field(std::make_shared<FieldSchema>(
      "name", DataType::STRING, false,
      std::make_shared<InvertIndexParams>(false)));
  schema->add_field(
      std::make_shared<FieldSchema>("weight", DataType::FLOAT, true));

  schema->add_field(std::make_shared<FieldSchema>(
      "dense", DataType::VECTOR_FP32, 128, false,
      std::make_shared<HnswIndexParams>(MetricType::IP)));
  schema->add_field(std::make_shared<FieldSchema>(
      "sparse", DataType::SPARSE_VECTOR_FP32, 0, false,
      std::make_shared<HnswIndexParams>(MetricType::IP)));

  return schema;
}

int main() {
  std::string path = "./demo";
  std::string rm_cmd = "rm -rf " + path;
  system(rm_cmd.c_str());

  auto schema = create_schema();
  CollectionOptions options{false, true};

  auto result = Collection::CreateAndOpen(path, *schema, options);
  if (!result.has_value()) {
    std::cout << result.error().message() << std::endl;
    return -1;
  }

  std::cout << "init stats: " << result.value()->Stats().value().to_string()
            << std::endl;

  auto coll = std::move(result).value();

  // insert docs
  {
    auto doc1 = create_doc(0, *schema);
    std::vector<Doc> docs{doc1};
    auto res = coll->Insert(docs);
    if (!res.has_value()) {
      std::cout << res.error().message() << std::endl;
      return -1;
    }
    std::cout << "after insert stats " << coll->Stats().value().to_string()
              << std::endl;
  }

  // optimize
  {
    auto res = coll->Optimize();
    if (!res.ok()) {
      std::cout << res.message() << std::endl;
      return -1;
    }
    std::cout << "after optimize stats " << coll->Stats().value().to_string()
              << std::endl;
  }

  // query
  {
    VectorQuery query;
    query.topk_ = 10;
    query.field_name_ = "dense";
    query.include_vector_ = true;
    std::vector<float> query_vector = std::vector<float>(128, 0.1);
    query.query_vector_.assign((char *)query_vector.data(),
                               query_vector.size() * sizeof(float));
    auto res = coll->Query(query);
    if (!res.has_value()) {
      std::cout << res.error().message() << std::endl;
      return -1;
    }
    std::cout << "query result: doc_count[" << res.value().size() << "]"
              << std::endl;
    std::cout << "first doc: " << res.value()[0]->to_detail_string()
              << std::endl;
  }

  // close and reopen
  coll.reset();
  options.read_only_ = true;
  result = Collection::Open(path, options);
  if (!result.has_value()) {
    std::cout << result.error().message() << std::endl;
    return -1;
  }
  std::cout << "reopen stats: " << result.value()->Stats().value().to_string()
            << std::endl;

  return 0;
}