// SPDX-License-Identifier: Apache-2.0
#include "db/sqlengine/scalar_jit/scalar_predicate_cache.h"
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "arrow/scalar.h"
#include "arrow/type.h"
#include "gandiva/condition.h"
#include "gandiva/node.h"
#include "gandiva/node_visitor.h"
#include "gandiva/tree_expr_builder.h"

namespace zvec::sqlengine::scalar_jit {
namespace {

// Length-prefix every token, including names and binary IN constants. This is
// structural serialization, not parsing or textual replacement in ToString().
void Token(std::string *key, const std::string &value) {
  key->append(std::to_string(value.size()));
  key->push_back(':');
  key->append(value);
}

arrow::Status AddType(std::string *key, const gandiva::DataTypePtr &type) {
  ARROW_RETURN_IF(!type, arrow::Status::Invalid("Expression has a null type"));
  switch (type->id()) {
    case arrow::Type::BOOL:
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
    case arrow::Type::STRING:
    case arrow::Type::BINARY:
      Token(key, std::to_string(type->id()));
      return arrow::Status::OK();
    default:
      return arrow::Status::NotImplemented("Cached scalar type: ",
                                           type->ToString());
  }
}

arrow::Result<std::shared_ptr<arrow::Scalar>> LiteralValue(
    const gandiva::LiteralNode &node) {
  if (node.is_null()) return arrow::MakeNullScalar(node.return_type());
#define SCALAR_LITERAL_CASE(ID, CPP, SCALAR)        \
  case arrow::Type::ID:                             \
    return std::static_pointer_cast<arrow::Scalar>( \
        std::make_shared<arrow::SCALAR>(std::get<CPP>(node.holder())))
  switch (node.return_type()->id()) {
    SCALAR_LITERAL_CASE(BOOL, bool, BooleanScalar);
    SCALAR_LITERAL_CASE(INT8, int8_t, Int8Scalar);
    SCALAR_LITERAL_CASE(INT16, int16_t, Int16Scalar);
    SCALAR_LITERAL_CASE(INT32, int32_t, Int32Scalar);
    SCALAR_LITERAL_CASE(INT64, int64_t, Int64Scalar);
    SCALAR_LITERAL_CASE(UINT8, uint8_t, UInt8Scalar);
    SCALAR_LITERAL_CASE(UINT16, uint16_t, UInt16Scalar);
    SCALAR_LITERAL_CASE(UINT32, uint32_t, UInt32Scalar);
    SCALAR_LITERAL_CASE(UINT64, uint64_t, UInt64Scalar);
    SCALAR_LITERAL_CASE(FLOAT, float, FloatScalar);
    SCALAR_LITERAL_CASE(DOUBLE, double, DoubleScalar);
    SCALAR_LITERAL_CASE(STRING, std::string, StringScalar);
    SCALAR_LITERAL_CASE(BINARY, std::string, BinaryScalar);
    default:
      return arrow::Status::NotImplemented("Unsupported scalar literal");
  }
#undef SCALAR_LITERAL_CASE
}

// The hit path only extracts parameters and a key. A second traversal builds
// the parameterized tree on a cache miss; no temporary AST is built on hits.
class Analyzer : public gandiva::NodeVisitor {
 public:
  Analyzer(gandiva::SchemaPtr schema, bool materialize)
      : schema_(std::move(schema)), materialize_(materialize) {
    Token(&key, "gandiva21-scalar-parameter-v1");
  }

  arrow::Status analyze(const gandiva::NodePtr &root) {
    ARROW_RETURN_IF(!root, arrow::Status::Invalid("Null expression node"));
    return root->Accept(*this);
  }

  arrow::Status Visit(const gandiva::FieldNode &node) override {
    ARROW_RETURN_IF(!node.field(), arrow::Status::Invalid("Null field"));
    const int index = schema_->GetFieldIndex(node.field()->name());
    ARROW_RETURN_IF(index < 0 || !schema_->field(index)->Equals(node.field()),
                    arrow::Status::Invalid("Field does not match data schema: ",
                                           node.field()->name()));
    Token(&key, "field");
    Token(&key, node.field()->name());
    Token(&key, std::to_string(index));
    Token(&key, node.field()->nullable() ? "nullable" : "nonnullable");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    if (materialize_)
      result = gandiva::TreeExprBuilder::MakeField(node.field());
    return arrow::Status::OK();
  }

  arrow::Status Visit(const gandiva::LiteralNode &node) override {
    Token(&key, "parameter");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    const auto ordinal = parameters.size();
    Token(&key, std::to_string(ordinal));
    ARROW_ASSIGN_OR_RAISE(auto value, LiteralValue(node));
    parameters.push_back(std::move(value));
    if (materialize_) {
      std::string name =
          "__gandiva_scalar_parameter_" + std::to_string(ordinal);
      while (schema_->GetFieldIndex(name) >= 0) name.push_back('_');
      auto field = arrow::field(name, node.return_type(), true);
      parameter_fields.push_back(field);
      result = gandiva::TreeExprBuilder::MakeField(std::move(field));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const gandiva::FunctionNode &node) override {
    ARROW_RETURN_IF(!node.descriptor(),
                    arrow::Status::Invalid("Null function descriptor"));
    Token(&key, "function");
    Token(&key, node.descriptor()->name());
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    Token(&key, std::to_string(node.children().size()));
    gandiva::NodeVector children;
    for (const auto &child : node.children()) {
      ARROW_RETURN_NOT_OK(analyze(child));
      if (materialize_) children.push_back(result);
    }
    if (materialize_) {
      result = gandiva::TreeExprBuilder::MakeFunction(
          node.descriptor()->name(), children, node.return_type());
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const gandiva::IfNode &node) override {
    Token(&key, "if");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    ARROW_RETURN_NOT_OK(analyze(node.condition()));
    auto condition = result;
    ARROW_RETURN_NOT_OK(analyze(node.then_node()));
    auto then_node = result;
    ARROW_RETURN_NOT_OK(analyze(node.else_node()));
    if (materialize_) {
      result = gandiva::TreeExprBuilder::MakeIf(condition, then_node, result,
                                                node.return_type());
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const gandiva::BooleanNode &node) override {
    ARROW_RETURN_IF(node.expr_type() != gandiva::BooleanNode::AND &&
                        node.expr_type() != gandiva::BooleanNode::OR,
                    arrow::Status::Invalid("Unknown boolean node kind"));
    Token(&key, node.expr_type() == gandiva::BooleanNode::AND ? "and" : "or");
    Token(&key, std::to_string(node.children().size()));
    gandiva::NodeVector children;
    for (const auto &child : node.children()) {
      ARROW_RETURN_NOT_OK(analyze(child));
      if (materialize_) children.push_back(result);
    }
    if (materialize_) {
      result = node.expr_type() == gandiva::BooleanNode::AND
                   ? gandiva::TreeExprBuilder::MakeAnd(children)
                   : gandiva::TreeExprBuilder::MakeOr(children);
    }
    return arrow::Status::OK();
  }

  template <class T>
  arrow::Status visit_in(const gandiva::InExpressionNode<T> &node,
                         const char *type_name) {
    Token(&key, "in");
    Token(&key, type_name);
    // Include an explicit IN type when one was supplied, since validation and
    // native helper selection depend on it, not just on the C++ set type.
    Token(&key, node.type() ? "explicit-type" : "default-type");
    if (node.type()) ARROW_RETURN_NOT_OK(AddType(&key, node.type()));
    std::vector<std::string> values;
    values.reserve(node.values().size());
    for (const auto &value : node.values()) {
      values.emplace_back(reinterpret_cast<const char *>(&value), sizeof(T));
    }
    std::sort(values.begin(), values.end());
    Token(&key, std::to_string(values.size()));
    for (const auto &value : values) Token(&key, value);
    ARROW_RETURN_NOT_OK(analyze(node.eval_expr()));
    if (materialize_) {
      result = std::make_shared<gandiva::InExpressionNode<T>>(
          result, node.values(), node.type());
    }
    return arrow::Status::OK();
  }
  arrow::Status Visit(const gandiva::InExpressionNode<int32_t> &node) override {
    return visit_in(node, "i32");
  }
  arrow::Status Visit(const gandiva::InExpressionNode<int64_t> &node) override {
    return visit_in(node, "i64");
  }
  arrow::Status Visit(const gandiva::InExpressionNode<float> &node) override {
    return visit_in(node, "f32");
  }
  arrow::Status Visit(const gandiva::InExpressionNode<double> &node) override {
    return visit_in(node, "f64");
  }
  arrow::Status Visit(
      const gandiva::InExpressionNode<gandiva::DecimalScalar128> &) override {
    return arrow::Status::NotImplemented("Cached decimal IN");
  }
  arrow::Status Visit(const gandiva::InExpressionNode<std::string> &) override {
    return arrow::Status::NotImplemented("Cached string IN");
  }

  std::string key;
  std::vector<std::shared_ptr<arrow::Scalar>> parameters;
  gandiva::FieldVector parameter_fields;
  gandiva::NodePtr result;

 private:
  gandiva::SchemaPtr schema_;
  bool materialize_;
};

struct CacheKey {
  gandiva::SchemaPtr schema;
  std::string shape;
  bool operator==(const CacheKey &other) const {
    // Do not rely on schema ToString() being a collision-free encoding. This
    // equality matches bind's semantic schema equality (metadata excluded).
    return shape == other.shape && schema->Equals(*other.schema);
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey &key) const {
    // Equal schemas necessarily have the same hash: schema text is deliberately
    // not hashed. Differing schemas sharing a shape are separated by equality.
    return std::hash<std::string>{}(key.shape);
  }
};

}  // namespace

class ScalarPredicateCache::Impl {
 public:
  explicit Impl(size_t capacity) : capacity(capacity) {
    if (capacity == 0)
      throw std::invalid_argument("Cache capacity must be positive");
  }

  struct Entry {
    std::condition_variable ready;
    bool complete = false;
    uint64_t last_use = 0;
    arrow::Result<std::shared_ptr<ScalarPredicate>> result =
        arrow::Status::Invalid("Compilation pending");
  };

  // LRU selection is O(capacity) only on eviction. Hits update a timestamp
  // without allocating list nodes; neither path is used inside evaluate.
  std::shared_ptr<Entry> evict() {
    // One completed entry was just added under the mutex, so at most one
    // eviction is needed. Destroy the retired LLVM engine after unlocking.
    if (stats.entries <= capacity) return {};
    auto oldest = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
      if (!it->second->complete) continue;
      if (oldest == entries.end() ||
          it->second->last_use < oldest->second->last_use)
        oldest = it;
    }
    auto retired = std::move(oldest->second);
    entries.erase(oldest);
    --stats.entries;
    ++stats.evictions;
    return retired;
  }

  size_t capacity;
  mutable std::mutex mutex;
  uint64_t clock = 0;
  ScalarPredicateCacheStats stats;
  std::unordered_map<CacheKey, std::shared_ptr<Entry>, CacheKeyHash> entries;
};

ScalarPredicateCache::ScalarPredicateCache(size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}
ScalarPredicateCache::~ScalarPredicateCache() = default;

arrow::Result<std::shared_ptr<BoundScalarPredicate>>
PreparedScalarPredicate::bind(std::shared_ptr<arrow::RecordBatch> batch) const {
  ARROW_RETURN_IF(!compiled,
                  arrow::Status::Invalid("Prepared predicate has no code"));
  return compiled->bind(std::move(batch), parameters, query_owners);
}

arrow::Result<PreparedScalarPredicate> ScalarPredicateCache::prepare(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<gandiva::Condition> condition,
    std::vector<std::shared_ptr<const void>> query_owners) {
  ARROW_RETURN_IF(
      !schema || !condition || !condition->root(),
      arrow::Status::Invalid("Schema, condition and root are required"));
  ARROW_RETURN_IF(
      !condition->root()->return_type() ||
          condition->root()->return_type()->id() != arrow::Type::BOOL,
      arrow::Status::Invalid("Condition must return bool"));
  std::unordered_set<std::string> names;
  for (const auto &field : schema->fields()) {
    ARROW_RETURN_IF(
        !names.insert(field->name()).second,
        arrow::Status::Invalid("Duplicate schema field: ", field->name()));
  }
  Analyzer analysis(schema, false);
  try {
    ARROW_RETURN_NOT_OK(analysis.analyze(condition->root()));
  } catch (const std::exception &error) {
    return arrow::Status::Invalid("Expression analysis failed: ", error.what());
  }
  CacheKey key{schema, std::move(analysis.key)};
  std::shared_ptr<Impl::Entry> entry;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    ++impl_->stats.lookups;
    auto found = impl_->entries.find(key);
    if (found != impl_->entries.end()) {
      ++impl_->stats.hits;
      entry = found->second;
      if (!entry->complete) {
        ++impl_->stats.waits;
        entry->ready.wait(lock, [&entry] { return entry->complete; });
      }
      entry->last_use = ++impl_->clock;
      ARROW_RETURN_NOT_OK(entry->result.status());
      return PreparedScalarPredicate{entry->result.ValueOrDie(),
                                     std::move(analysis.parameters), true,
                                     std::move(query_owners)};
    }
    ++impl_->stats.misses;
    ++impl_->stats.compilations;
    entry = std::make_shared<Impl::Entry>();
    impl_->entries.emplace(key, entry);
  }

  // Always publish success or failure and wake waiters, including C++
  // exceptions thrown by an unsupported/malformed tree or by the compiler.
  // Failures are not retained, so a future call can retry rather than finding a
  // poisoned entry.
  arrow::Result<std::shared_ptr<ScalarPredicate>> program =
      arrow::Status::UnknownError("Compilation did not run");
  try {
    Analyzer lowered(schema, true);
    const auto status = lowered.analyze(condition->root());
    if (!status.ok()) {
      program = status;
    } else {
      program = ScalarPredicate::MakeParameterized(
          schema, gandiva::TreeExprBuilder::MakeCondition(lowered.result),
          arrow::schema(lowered.parameter_fields));
    }
  } catch (const std::exception &error) {
    program = arrow::Status::UnknownError("Scalar compilation failed: ",
                                          error.what());
  } catch (...) {
    program = arrow::Status::UnknownError("Scalar compilation failed");
  }
  std::shared_ptr<Impl::Entry> retired;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    entry->result = program;
    entry->complete = true;
    entry->last_use = ++impl_->clock;
    if (program.ok()) {
      ++impl_->stats.entries;
      retired = impl_->evict();
    } else {
      ++impl_->stats.failed_compilations;
      impl_->entries.erase(key);
    }
  }
  entry->ready.notify_all();
  ARROW_RETURN_NOT_OK(program.status());
  return PreparedScalarPredicate{program.ValueOrDie(),
                                 std::move(analysis.parameters), false,
                                 std::move(query_owners)};
}

ScalarPredicateCacheStats ScalarPredicateCache::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

}  // namespace zvec::sqlengine::scalar_jit
