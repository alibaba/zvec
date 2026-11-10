// SPDX-License-Identifier: Apache-2.0
#include "gandiva/scalar_predicate_cache.h"

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

namespace gandiva {
namespace {

// Length-prefix every token, including names and binary IN constants. This is
// structural serialization, not parsing or textual replacement in ToString().
void Token(std::string* key, const std::string& value) {
  key->append(std::to_string(value.size()));
  key->push_back(':');
  key->append(value);
}

arrow::Status AddType(std::string* key, const DataTypePtr& type) {
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
      Token(key, std::to_string(type->id()));
      return arrow::Status::OK();
    default:
      return arrow::Status::NotImplemented("Cached scalar type: ", type->ToString());
  }
}

arrow::Result<std::shared_ptr<arrow::Scalar>> LiteralValue(const LiteralNode& node) {
  if (node.is_null()) return arrow::MakeNullScalar(node.return_type());
#define SCALAR_LITERAL_CASE(ID, CPP, SCALAR) \
  case arrow::Type::ID: \
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
    default:
      return arrow::Status::NotImplemented("Unsupported scalar literal");
  }
#undef SCALAR_LITERAL_CASE
}

// The hit path only extracts parameters and a key. A second traversal builds
// the parameterized tree on a cache miss; no temporary AST is built on hits.
class Analyzer : public NodeVisitor {
 public:
  Analyzer(SchemaPtr schema, bool materialize)
      : schema_(std::move(schema)), materialize_(materialize) {
    Token(&key, "gandiva21-scalar-parameter-v1");
  }

  Status Analyze(const NodePtr& root) {
    ARROW_RETURN_IF(!root, Status::Invalid("Null expression node"));
    return root->Accept(*this);
  }

  Status Visit(const FieldNode& node) override {
    ARROW_RETURN_IF(!node.field(), Status::Invalid("Null field"));
    const int index = schema_->GetFieldIndex(node.field()->name());
    ARROW_RETURN_IF(index < 0 || !schema_->field(index)->Equals(node.field()),
                    Status::Invalid("Field does not match data schema: ",
                                    node.field()->name()));
    Token(&key, "field");
    Token(&key, node.field()->name());
    Token(&key, std::to_string(index));
    Token(&key, node.field()->nullable() ? "nullable" : "nonnullable");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    if (materialize_) result = TreeExprBuilder::MakeField(node.field());
    return Status::OK();
  }

  Status Visit(const LiteralNode& node) override {
    Token(&key, "parameter");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    const auto ordinal = parameters.size();
    Token(&key, std::to_string(ordinal));
    ARROW_ASSIGN_OR_RAISE(auto value, LiteralValue(node));
    parameters.push_back(std::move(value));
    if (materialize_) {
      std::string name = "__gandiva_scalar_parameter_" + std::to_string(ordinal);
      while (schema_->GetFieldIndex(name) >= 0) name.push_back('_');
      auto field = arrow::field(name, node.return_type(), true);
      parameter_fields.push_back(field);
      result = TreeExprBuilder::MakeField(std::move(field));
    }
    return Status::OK();
  }

  Status Visit(const FunctionNode& node) override {
    ARROW_RETURN_IF(!node.descriptor(), Status::Invalid("Null function descriptor"));
    Token(&key, "function");
    Token(&key, node.descriptor()->name());
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    Token(&key, std::to_string(node.children().size()));
    NodeVector children;
    for (const auto& child : node.children()) {
      ARROW_RETURN_NOT_OK(Analyze(child));
      if (materialize_) children.push_back(result);
    }
    if (materialize_) {
      result = TreeExprBuilder::MakeFunction(node.descriptor()->name(), children,
                                            node.return_type());
    }
    return Status::OK();
  }

  Status Visit(const IfNode& node) override {
    Token(&key, "if");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type()));
    ARROW_RETURN_NOT_OK(Analyze(node.condition()));
    auto condition = result;
    ARROW_RETURN_NOT_OK(Analyze(node.then_node()));
    auto then_node = result;
    ARROW_RETURN_NOT_OK(Analyze(node.else_node()));
    if (materialize_) {
      result = TreeExprBuilder::MakeIf(condition, then_node, result, node.return_type());
    }
    return Status::OK();
  }

  Status Visit(const BooleanNode& node) override {
    ARROW_RETURN_IF(node.expr_type() != BooleanNode::AND &&
                       node.expr_type() != BooleanNode::OR,
                    Status::Invalid("Unknown boolean node kind"));
    Token(&key, node.expr_type() == BooleanNode::AND ? "and" : "or");
    Token(&key, std::to_string(node.children().size()));
    NodeVector children;
    for (const auto& child : node.children()) {
      ARROW_RETURN_NOT_OK(Analyze(child));
      if (materialize_) children.push_back(result);
    }
    if (materialize_) {
      result = node.expr_type() == BooleanNode::AND ? TreeExprBuilder::MakeAnd(children)
                                                   : TreeExprBuilder::MakeOr(children);
    }
    return Status::OK();
  }

  template <class T>
  Status In(const InExpressionNode<T>& node, const char* type_name) {
    Token(&key, "in");
    Token(&key, type_name);
    // Include an explicit IN type when one was supplied, since validation and
    // native helper selection depend on it, not just on the C++ set type.
    Token(&key, node.type() ? "explicit-type" : "default-type");
    if (node.type()) ARROW_RETURN_NOT_OK(AddType(&key, node.type()));
    std::vector<std::string> values;
    values.reserve(node.values().size());
    for (const auto& value : node.values()) {
      values.emplace_back(reinterpret_cast<const char*>(&value), sizeof(T));
    }
    std::sort(values.begin(), values.end());
    Token(&key, std::to_string(values.size()));
    for (const auto& value : values) Token(&key, value);
    ARROW_RETURN_NOT_OK(Analyze(node.eval_expr()));
    if (materialize_) {
      result = std::make_shared<InExpressionNode<T>>(result, node.values(), node.type());
    }
    return Status::OK();
  }
  Status Visit(const InExpressionNode<int32_t>& node) override { return In(node, "i32"); }
  Status Visit(const InExpressionNode<int64_t>& node) override { return In(node, "i64"); }
  Status Visit(const InExpressionNode<float>& node) override { return In(node, "f32"); }
  Status Visit(const InExpressionNode<double>& node) override { return In(node, "f64"); }
  Status Visit(const InExpressionNode<DecimalScalar128>&) override {
    return Status::NotImplemented("Cached decimal IN");
  }
  Status Visit(const InExpressionNode<std::string>&) override {
    return Status::NotImplemented("Cached string IN");
  }

  std::string key;
  std::vector<std::shared_ptr<arrow::Scalar>> parameters;
  FieldVector parameter_fields;
  NodePtr result;

 private:
  SchemaPtr schema_;
  bool materialize_;
};

struct CacheKey {
  SchemaPtr schema;
  std::string shape;
  bool operator==(const CacheKey& other) const {
    // Do not rely on schema ToString() being a collision-free encoding. This
    // equality matches Bind's semantic schema equality (metadata excluded).
    return shape == other.shape && schema->Equals(*other.schema);
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& key) const {
    // Equal schemas necessarily have the same hash: schema text is deliberately
    // not hashed. Differing schemas sharing a shape are separated by equality.
    return std::hash<std::string>{}(key.shape);
  }
};

}  // namespace

class ScalarPredicateCache::Impl {
 public:
  explicit Impl(size_t capacity) : capacity(capacity) {
    if (capacity == 0) throw std::invalid_argument("Cache capacity must be positive");
  }

  struct Entry {
    std::condition_variable ready;
    bool complete = false;
    uint64_t last_use = 0;
    arrow::Result<std::shared_ptr<ScalarPredicate>> result =
        arrow::Status::Invalid("Compilation pending");
  };

  // LRU selection is O(capacity) only on eviction. Hits update a timestamp
  // without allocating list nodes; neither path is used inside Evaluate.
  std::shared_ptr<Entry> Evict() {
    // One completed entry was just added under the mutex, so at most one
    // eviction is needed. Destroy the retired LLVM engine after unlocking.
    if (stats.entries <= capacity) return {};
    auto oldest = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
      if (!it->second->complete) continue;
      if (oldest == entries.end() || it->second->last_use < oldest->second->last_use)
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

arrow::Result<std::shared_ptr<BoundScalarPredicate>> PreparedScalarPredicate::Bind(
    std::shared_ptr<arrow::RecordBatch> batch) const {
  ARROW_RETURN_IF(!compiled, arrow::Status::Invalid("Prepared predicate has no code"));
  return compiled->Bind(std::move(batch), parameters);
}

arrow::Result<PreparedScalarPredicate> ScalarPredicateCache::Prepare(
    std::shared_ptr<arrow::Schema> schema, std::shared_ptr<Condition> condition) {
  ARROW_RETURN_IF(!schema || !condition || !condition->root(),
                  arrow::Status::Invalid("Schema, condition and root are required"));
  ARROW_RETURN_IF(!condition->root()->return_type() ||
                     condition->root()->return_type()->id() != arrow::Type::BOOL,
                  arrow::Status::Invalid("Condition must return bool"));
  std::unordered_set<std::string> names;
  for (const auto& field : schema->fields()) {
    ARROW_RETURN_IF(!names.insert(field->name()).second,
                    arrow::Status::Invalid("Duplicate schema field: ", field->name()));
  }
  Analyzer analysis(schema, false);
  try {
    ARROW_RETURN_NOT_OK(analysis.Analyze(condition->root()));
  } catch (const std::exception& error) {
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
                                      std::move(analysis.parameters), true};
    }
    ++impl_->stats.misses;
    ++impl_->stats.compilations;
    entry = std::make_shared<Impl::Entry>();
    impl_->entries.emplace(key, entry);
  }

  // Always publish success or failure and wake waiters, including C++ exceptions
  // thrown by an unsupported/malformed tree or by the compiler. Failures are not
  // retained, so a future call can retry rather than finding a poisoned entry.
  arrow::Result<std::shared_ptr<ScalarPredicate>> program =
      arrow::Status::UnknownError("Compilation did not run");
  try {
    Analyzer lowered(schema, true);
    const auto status = lowered.Analyze(condition->root());
    if (!status.ok()) {
      program = status;
    } else {
      program = ScalarPredicate::MakeParameterized(
          schema, TreeExprBuilder::MakeCondition(lowered.result),
          arrow::schema(lowered.parameter_fields));
    }
  } catch (const std::exception& error) {
    program = arrow::Status::UnknownError("Scalar compilation failed: ", error.what());
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
      retired = impl_->Evict();
    } else {
      ++impl_->stats.failed_compilations;
      impl_->entries.erase(key);
    }
  }
  entry->ready.notify_all();
  ARROW_RETURN_NOT_OK(program.status());
  return PreparedScalarPredicate{program.ValueOrDie(),
                                  std::move(analysis.parameters), false};
}

ScalarPredicateCacheStats ScalarPredicateCache::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

}  // namespace gandiva
