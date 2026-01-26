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
#pragma once

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/db/query_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>

namespace zvec {

using float16_t = ailego::Float16;

class Doc {
 public:
  using Value = std::variant<
      std::monostate,  // 0 - represents null value
      bool, int32_t, uint32_t, int64_t, uint64_t, float, double,  // 1~7
      std::string,                                                // 8
      std::vector<bool>,                                          // 9
      std::vector<int8_t>,                                        // 10
      std::vector<int16_t>,                                       // 11
      std::vector<int32_t>,                                       // 12
      std::vector<int64_t>,                                       // 13
      std::vector<uint32_t>,                                      // 14
      std::vector<uint64_t>,                                      // 15
      std::vector<float16_t>,                                     // 16
      std::vector<float>,                                         // 17
      std::vector<double>,                                        // 18
      std::vector<std::string>,                                   // 19
      std::pair<std::vector<uint32_t>, std::vector<float>>,       // 20
      std::pair<std::vector<uint32_t>, std::vector<float16_t>>>;  // 21

  using Ptr = std::shared_ptr<Doc>;

  Doc() = default;
  ~Doc() = default;

  Doc(const Doc &) = default;
  Doc &operator=(const Doc &) = default;
  Doc(Doc &&) = default;
  Doc &operator=(Doc &&) = default;

 public:
  void set_pk(std::string pk) {
    pk_ = std::move(pk);
  }

  std::string pk() const {
    return pk_;
  }

  void set_score(float score) {
    score_ = score;
  }

  float score() const {
    return score_;
  }

  void set_doc_id(uint64_t doc_id) {
    doc_id_ = doc_id;
  }

  uint64_t doc_id() const {
    return doc_id_;
  }

  std::vector<std::string> field_names() const {
    std::vector<std::string> names;
    names.reserve(fields_.size());

    for (const auto &[name, _] : fields_) {
      names.emplace_back(name);
    }

    return names;
  }

  void set_operator(const Operator op) {
    op_ = op;
  }

  Operator get_operator() {
    return op_;
  }

  // Set field value
  template <typename T>
  bool set(const std::string &field_name, T value) {
    // TODO: support char*
    static_assert(is_valid_type_v<T>, "Unsupported type");
    fields_[field_name] = std::move(value);
    return true;
  }

  // Set field to null
  void set_null(const std::string &field_name) {
    fields_[field_name] = std::monostate{};
  }

  // Check if field exists
  bool has(const std::string &field_name) const {
    return fields_.find(field_name) != fields_.end();
  }

  // Check if field exists and is not null
  bool has_value(const std::string &field_name) const {
    auto it = fields_.find(field_name);
    if (it == fields_.end()) {
      return false;
    }
    return !std::holds_alternative<std::monostate>(it->second);
  }

  // Check if field is null
  bool is_null(const std::string &field_name) const {
    auto it = fields_.find(field_name);
    if (it == fields_.end()) {
      return false;  // Field does not exist is not equal to null
    }
    return std::holds_alternative<std::monostate>(it->second);
  }

  // Check if fields is empty
  bool is_empty() const {
    return fields_.empty();
  }

  // Field get status enumeration
  enum class FieldGetStatus {
    SUCCESS,       // Successfully got value
    NOT_FOUND,     // Field does not exist
    IS_NULL,       // Field exists but is null
    TYPE_MISMATCH  // Field exists but type mismatch
  };

  // Field get result template class
  template <typename T>
  class FieldGetResult {
   public:
    // Constructor - success case
    explicit FieldGetResult(const T &value)
        : status_(FieldGetStatus::SUCCESS), value_(value) {}

    // Constructor - error case
    explicit FieldGetResult(FieldGetStatus status) : status_(status) {
      if (status == FieldGetStatus::SUCCESS) {
        throw std::invalid_argument("Use value constructor for SUCCESS status");
      }
    }

    // Get status
    FieldGetStatus status() const {
      return status_;
    }

    // Get value (only available when successful)
    const T &value() const {
      if (status_ != FieldGetStatus::SUCCESS) {
        throw std::runtime_error("No value available");
      }
      return value_;
    }

    // Check if successful
    bool ok() const {
      return status_ == FieldGetStatus::SUCCESS;
    }

    // Convert to optional
    operator std::optional<T>() const {
      if (status_ == FieldGetStatus::SUCCESS) {
        return value_;
      }
      return std::nullopt;
    }

   private:
    FieldGetStatus status_;
    T value_;
  };


  // Get field value, distinguish between not found, null and type mismatch
  // cases
  template <typename T>
  typename Doc::template FieldGetResult<T> get_field(
      const std::string &field_name) const {
    static_assert(is_valid_type_v<T>, "Unsupported type");

    auto it = fields_.find(field_name);
    if (it == fields_.end()) {
      return FieldGetResult<T>(FieldGetStatus::NOT_FOUND);
    }

    if (std::holds_alternative<std::monostate>(it->second)) {
      return FieldGetResult<T>(FieldGetStatus::IS_NULL);
    }

    try {
      return FieldGetResult<T>(std::get<T>(it->second));
    } catch (const std::bad_variant_access &) {
      return FieldGetResult<T>(FieldGetStatus::TYPE_MISMATCH);
    }
  }

  template <typename T>
  std::optional<T> get(const std::string &field_name) const {
    auto result = get_field<T>(field_name);
    if (result.status() == FieldGetStatus::SUCCESS) {
      return result.value();
    }
    return std::nullopt;
  }

  void remove(const std::string &field_name) {
    fields_.erase(field_name);
  }

  Status validate(const CollectionSchema::Ptr &schema,
                  bool is_update = false) const;

  size_t memory_usage() const;

  void clear() {
    pk_.clear();
    score_ = 0.0f;
    doc_id_ = 0;
    fields_.clear();
  }

  const std::string to_string() const {
    std::stringstream ss;
    ss << "[op:" << (uint32_t)op_ << ", doc_id: " << doc_id_
       << ", score: " << score_ << ", pk: " << pk_
       << ", fields: " << fields_.size() << "]";
    return ss.str();
  }

  std::string to_detail_string() const;

  bool operator==(const Doc &other) const;

  bool operator!=(const Doc &other) const {
    return !(*this == other);
  }

 public:
  std::vector<uint8_t> serialize() const;

  static Doc::Ptr deserialize(const uint8_t *data, size_t size);
  static Doc::Ptr deserialize(const std::vector<uint8_t> &data) {
    return deserialize(data.data(), data.size());
  }

 public:
  void merge(const Doc &other) {
    pk_ = other.pk_;
    score_ = other.score_;
    doc_id_ = other.doc_id_;
    op_ = other.op_;
    for (const auto &[field_name, value] : other.fields_) {
      fields_[field_name] = value;
    }
  }

 private:
  static void serialize_value(std::vector<uint8_t> &buffer, const Value &value);

  static Value deserialize_value(const uint8_t *&data, uint8_t type);
  static Value deserialize_value(const uint8_t *&data);

  static void write_to_buffer(std::vector<uint8_t> &buffer, const void *src,
                              size_t size);

  static void read_from_buffer(const uint8_t *&data, void *dest, size_t size);

 private:
  std::string pk_;
  float score_{0.0f};
  uint64_t doc_id_;
  Operator op_;

  template <typename T>
  static constexpr bool is_valid_type_v =
      std::is_same_v<T, std::monostate> ||            // 0 - Added null support
      std::is_same_v<T, bool> ||                      // 1
      std::is_same_v<T, int32_t> ||                   // 2
      std::is_same_v<T, uint32_t> ||                  // 3
      std::is_same_v<T, int64_t> ||                   // 4
      std::is_same_v<T, uint64_t> ||                  // 5
      std::is_same_v<T, float> ||                     // 6
      std::is_same_v<T, double> ||                    // 7
      std::is_same_v<T, std::string> ||               // 8
      std::is_same_v<T, std::vector<bool>> ||         // 9
      std::is_same_v<T, std::vector<int8_t>> ||       // 10
      std::is_same_v<T, std::vector<int16_t>> ||      // 11
      std::is_same_v<T, std::vector<int32_t>> ||      // 12
      std::is_same_v<T, std::vector<uint32_t>> ||     // 13
      std::is_same_v<T, std::vector<int64_t>> ||      // 14
      std::is_same_v<T, std::vector<uint64_t>> ||     // 15
      std::is_same_v<T, std::vector<float16_t>> ||    // 16
      std::is_same_v<T, std::vector<float>> ||        // 17
      std::is_same_v<T, std::vector<double>> ||       // 18
      std::is_same_v<T, std::vector<std::string>> ||  // 19
      std::is_same_v<
          T, std::pair<std::vector<uint32_t>, std::vector<float16_t>>> ||  // 20
      std::is_same_v<
          T, std::pair<std::vector<uint32_t>, std::vector<float>>>;  // 21

  std::unordered_map<std::string, Value> fields_;
};

std::string get_value_type_name(const Doc::Value &value, bool is_vector);

using DocPtrList = std::vector<Doc::Ptr>;

using DocPtrMap = std::unordered_map<std::string, Doc::Ptr>;

using WriteResults = std::vector<Status>;

struct VectorQuery {
  int topk_;
  std::string field_name_;
  std::string query_vector_;  // fp16, void *
  std::string query_sparse_indices_;
  std::string query_sparse_values_;
  std::string filter_;
  bool include_vector_{false};
  bool include_doc_id_{false};
  // select * by default, select no field if output_fields_ is empty, select
  // specific fields if output_fields_ is not empty
  std::optional<std::vector<std::string>> output_fields_;
  QueryParams::Ptr query_params_;

  Status validate(const FieldSchema *schema) const;
};

struct GroupByVectorQuery {
  std::string field_name_;
  std::string query_vector_;
  std::string query_sparse_indices_;
  std::string query_sparse_values_;
  std::string filter_;
  bool include_vector_;
  // select * by default, select no field if output_fields_ is empty, select
  // specific fields if output_fields_ is not empty
  std::optional<std::vector<std::string>> output_fields_;
  std::string group_by_field_name_;
  uint32_t group_count_ = 2;
  uint32_t group_topk_ = 3;
  QueryParams::Ptr query_params_;
};

struct GroupResult {
  std::string group_by_value_;
  std::vector<Doc> docs_;
};

using GroupResults = std::vector<GroupResult>;

}  // namespace zvec
