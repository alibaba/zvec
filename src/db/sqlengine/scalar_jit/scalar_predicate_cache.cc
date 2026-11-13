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

#include "db/sqlengine/scalar_jit/scalar_predicate_cache.h"
#include <algorithm>
#include <chrono>
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

arrow::Status AddType(std::string *key, const gandiva::DataTypePtr &type,
                      size_t limit) {
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
    case arrow::Type::BINARY: {
      const auto value = std::to_string(type->id());
      const auto bytes = std::to_string(value.size()).size() + 1 + value.size();
      if (key->size() > limit || bytes > limit - key->size())
        return Unavailable("expression_limit");
      Token(key, value);
    }
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
  Analyzer(gandiva::SchemaPtr schema, bool materialize,
           const ExpressionLimits &limits)
      : schema_(std::move(schema)), materialize_(materialize), limits_(limits) {
    token("gandiva21-scalar-parameter-v1");
  }

  arrow::Status analyze(const gandiva::NodePtr &root) {
    ARROW_RETURN_IF(!root, arrow::Status::Invalid("Null expression node"));
    if (++nodes_ > limits_.nodes || depth_ >= limits_.depth)
      return Unavailable("expression_limit");
    ++depth_;
    struct Pop {
      size_t &depth;
      ~Pop() {
        --depth;
      }
    } pop{depth_};
    ARROW_RETURN_NOT_OK(root->Accept(*this));
    if (key.size() > limits_.key_bytes) return Unavailable("expression_limit");
    return arrow::Status::OK();
  }

  arrow::Status Visit(const gandiva::FieldNode &node) override {
    ARROW_RETURN_IF(!node.field(), arrow::Status::Invalid("Null field"));
    const int index = schema_->GetFieldIndex(node.field()->name());
    ARROW_RETURN_IF(index < 0 || !schema_->field(index)->Equals(node.field()),
                    arrow::Status::Invalid("Field does not match data schema: ",
                                           node.field()->name()));
    token("field");
    token(node.field()->name());
    token(std::to_string(index));
    token(node.field()->nullable() ? "nullable" : "nonnullable");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type(), limits_.key_bytes));
    if (materialize_)
      result = gandiva::TreeExprBuilder::MakeField(node.field());
    return arrow::Status::OK();
  }

  arrow::Status Visit(const gandiva::LiteralNode &node) override {
    token("parameter");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type(), limits_.key_bytes));
    const auto ordinal = parameters.size();
    token(std::to_string(ordinal));
    size_t bytes = sizeof(gandiva::ScalarParameterSlot);
    if (!node.is_null() && (node.return_type()->id() == arrow::Type::STRING ||
                            node.return_type()->id() == arrow::Type::BINARY))
      bytes += std::get<std::string>(node.holder()).size();
    ARROW_RETURN_NOT_OK(input(bytes));
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
    token("function");
    token(node.descriptor()->name());
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type(), limits_.key_bytes));
    token(std::to_string(node.children().size()));
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
    token("if");
    ARROW_RETURN_NOT_OK(AddType(&key, node.return_type(), limits_.key_bytes));
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
    token(node.expr_type() == gandiva::BooleanNode::AND ? "and" : "or");
    token(std::to_string(node.children().size()));
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
    token("in");
    token(type_name);
    // Include an explicit IN type when one was supplied, since validation and
    // native helper selection depend on it, not just on the C++ set type.
    token(node.type() ? "explicit-type" : "default-type");
    if (node.type())
      ARROW_RETURN_NOT_OK(AddType(&key, node.type(), limits_.key_bytes));
    if (node.values().size() > limits_.in_values)
      return Unavailable("expression_limit");
    ARROW_RETURN_NOT_OK(input(node.values().size() * sizeof(T)));
    std::vector<std::string> values;
    values.reserve(node.values().size());
    for (const auto &value : node.values()) {
      values.emplace_back(reinterpret_cast<const char *>(&value), sizeof(T));
    }
    std::sort(values.begin(), values.end());
    token(std::to_string(values.size()));
    for (const auto &value : values) token(value);
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
  ExpressionLimits limits_;
  size_t nodes_ = 0;
  size_t depth_ = 0;
  size_t input_bytes_ = 0;

  void token(const std::string &value) {
    const auto prefix = std::to_string(value.size()).size() + 1;
    if (key.size() > limits_.key_bytes ||
        prefix > limits_.key_bytes - key.size() ||
        value.size() > limits_.key_bytes - key.size() - prefix)
      throw std::length_error("expression_limit");
    Token(std::addressof(key), value);
  }
  arrow::Status input(size_t bytes) {
    if (bytes > limits_.input_bytes - input_bytes_)
      return Unavailable("expression_limit");
    input_bytes_ += bytes;
    return arrow::Status::OK();
  }
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

arrow::Status ValidateSchema(const std::shared_ptr<arrow::Schema> &schema,
                             const ExpressionLimits &limits) {
  if (!schema) return arrow::Status::Invalid("Schema is required");
  if (static_cast<size_t>(schema->num_fields()) > limits.nodes)
    return Unavailable("expression_limit");
  size_t bytes = 0;
  std::unordered_set<std::string> names;
  for (const auto &field : schema->fields()) {
    if (!field) return arrow::Status::Invalid("Null schema field");
    if (field->name().size() > limits.key_bytes - bytes)
      return Unavailable("expression_limit");
    bytes += field->name().size();
    if (!names.insert(field->name()).second)
      return Unavailable("unsupported_expression");
  }
  return arrow::Status::OK();
}
}  // namespace

arrow::Status ValidatePredicateInputs(
    const std::shared_ptr<arrow::Schema> &schema,
    const std::shared_ptr<gandiva::Condition> &condition,
    const ExpressionLimits &limits) {
  if (!schema || !condition || !condition->root())
    return arrow::Status::Invalid("Schema, condition and root are required");
  try {
    ARROW_RETURN_NOT_OK(ValidateSchema(schema, limits));
    Analyzer analyzer(schema, false, limits);
    return analyzer.analyze(condition->root());
  } catch (const std::length_error &) {
    return Unavailable("expression_limit");
  } catch (const std::bad_alloc &) {
    return arrow::Status::OutOfMemory("Scalar expression analysis");
  } catch (const std::exception &) {
    return arrow::Status::Invalid("Malformed scalar expression");
  }
}

class ScalarPredicateCache::Impl {
 public:
  explicit Impl(ScalarPredicateCacheOptions value) : options(std::move(value)) {
    if (!options.capacity || !options.negative_capacity ||
        !options.max_compilations || !options.max_live_programs)
      throw std::invalid_argument("Scalar JIT cache limits must be positive");
    if (!options.compiler)
      options.compiler = ScalarPredicate::MakeParameterized;
    if (!options.now) options.now = std::chrono::steady_clock::now;
  }

  struct Entry {
    bool complete = false;
    uint64_t last_use = 0;
    uint32_t failures = 0;
    std::chrono::steady_clock::time_point retry_at;
    arrow::Result<std::shared_ptr<ScalarPredicate>> result =
        arrow::Status::Invalid("Compilation pending");
  };
  struct LiveProgram {
    std::weak_ptr<ScalarPredicate> program;
    MemoryUsage bytes;
  };

  std::shared_ptr<Entry> evict(bool negative) {
    const auto count = negative ? stats.negative_entries : stats.entries;
    const auto capacity =
        negative ? options.negative_capacity : options.capacity;
    if (count <= capacity) return {};
    auto oldest = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
      if (!it->second->complete || it->second->result.ok() == negative)
        continue;
      if (oldest == entries.end() ||
          it->second->last_use < oldest->second->last_use)
        oldest = it;
    }
    auto retired = std::move(oldest->second);
    entries.erase(oldest);
    if (negative)
      --stats.negative_entries;
    else
      --stats.entries;
    ++stats.evictions;
    return retired;  // Destroy LLVM outside the cache mutex.
  }

  size_t prune_live() {
    live.erase(std::remove_if(
                   live.begin(), live.end(),
                   [](const LiveProgram &p) { return p.program.expired(); }),
               live.end());
    return live.size();
  }

  ScalarPredicateCacheOptions options;
  mutable std::mutex mutex;
  uint64_t clock = 0;
  ScalarPredicateCacheStats stats;
  std::vector<LiveProgram> live;
  std::unordered_map<CacheKey, std::shared_ptr<Entry>, CacheKeyHash> entries;
};

ScalarPredicateCache::ScalarPredicateCache(size_t capacity)
    : ScalarPredicateCache([capacity] {
        ScalarPredicateCacheOptions options;
        options.capacity = capacity;
        return options;
      }()) {}
ScalarPredicateCache::ScalarPredicateCache(ScalarPredicateCacheOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
ScalarPredicateCache::~ScalarPredicateCache() {
  if (!InOriginalProcess()) (void)impl_.release();
}

arrow::Result<std::shared_ptr<BoundScalarPredicate>>
PreparedScalarPredicate::bind(std::shared_ptr<arrow::RecordBatch> batch) const {
  ARROW_RETURN_IF(!compiled,
                  arrow::Status::Invalid("Prepared predicate has no code"));
  return compiled->bind(std::move(batch), parameters, query_owners);
}

arrow::Result<PreparedScalarPredicate> ScalarPredicateCache::prepare(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<gandiva::Condition> condition,
    std::vector<std::shared_ptr<const void>> query_owners) try {
  const auto unavailable = UnavailableReason();
  if (!unavailable.empty()) return Unavailable(unavailable.c_str());
  const auto &options = impl_->options;
  ARROW_RETURN_NOT_OK(ValidateSchema(schema, options.compile.limits));
  if (!condition || !condition->root())
    return Unavailable("unsupported_expression");
  if (!condition->root()->return_type() ||
      condition->root()->return_type()->id() != arrow::Type::BOOL)
    return arrow::Status::Invalid("Condition must return bool");
  Analyzer analysis(schema, false, options.compile.limits);
  ARROW_RETURN_NOT_OK(analysis.analyze(condition->root()));
  CacheKey key{schema, std::move(analysis.key)};
  const auto now = options.now();
  std::shared_ptr<Impl::Entry> entry;
  struct Reservation {
    Impl *impl;
    const CacheKey &key;
    bool active = false;
    ~Reservation() {
      if (!active) return;
      std::lock_guard<std::mutex> lock(impl->mutex);
      --impl->stats.active_compilations;
      impl->entries.erase(key);
    }
  } reservation{impl_.get(), key};
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ++impl_->stats.lookups;
    auto found = impl_->entries.find(key);
    if (found != impl_->entries.end()) {
      entry = found->second;
      entry->last_use = ++impl_->clock;
      if (!entry->complete) {
        ++impl_->stats.in_flight_bypasses;
        return Unavailable("compilation_busy");
      }
      if (entry->result.ok()) {
        ++impl_->stats.hits;
        return PreparedScalarPredicate{entry->result.ValueOrDie(),
                                       std::move(analysis.parameters), true,
                                       std::move(query_owners)};
      }
      if (entry->retry_at == std::chrono::steady_clock::time_point::max())
        return entry->result.status();
      if (now < entry->retry_at) {
        ++impl_->stats.backoff_hits;
        return Unavailable("compile_backoff");
      }
    }
    ++impl_->stats.misses;
    if (impl_->stats.active_compilations >= options.max_compilations) {
      ++impl_->stats.admission_rejections;
      return Unavailable("compilation_busy");
    }
    const auto live = impl_->prune_live();
    if (live >= options.max_live_programs ||
        impl_->stats.active_compilations >= options.max_live_programs - live) {
      ++impl_->stats.admission_rejections;
      return Unavailable("live_program_limit");
    }
    // Allocate bookkeeping before reserving a compilation slot. Publication
    // cannot allocate even when multiple admitted compilers finish together.
    impl_->live.reserve(live + impl_->stats.active_compilations + 1);
    if (entry) {
      --impl_->stats.negative_entries;
      entry->complete = false;
    } else {
      entry = std::make_shared<Impl::Entry>();
      impl_->entries.emplace(key, entry);
    }
    ++impl_->stats.active_compilations;
    ++impl_->stats.compilations;
    reservation.active = true;
  }

  arrow::Result<std::shared_ptr<ScalarPredicate>> program =
      arrow::Status::UnknownError("Compilation did not run");
  try {
    Analyzer lowered(schema, true, options.compile.limits);
    const auto status = [&] {
      StageTimer lowering("lowering");
      return lowered.analyze(condition->root());
    }();
    if (!status.ok())
      program = status;
    else
      program = options.compiler(
          schema, gandiva::TreeExprBuilder::MakeCondition(lowered.result),
          arrow::schema(lowered.parameter_fields), options.compile);
    if (program.ok() && !*program)
      program = arrow::Status::Invalid("Compiler returned an empty program");
  } catch (const std::bad_alloc &) {
    program =
        arrow::Status::OutOfMemory("Scalar compilation allocation failed");
  } catch (...) {
    program = arrow::Status::UnknownError("Scalar compilation failed");
  }
  auto finished = now;
  try {
    finished = options.now();
  } catch (...) {
    program = arrow::Status::UnknownError("Scalar clock failed");
  }
  const auto reason =
      program.ok() ? std::string{} : FailureReason(program.status());
  const bool rejected =
      reason == "compilation_busy" || reason == "live_program_limit" ||
      reason == "expression_limit" || reason == "initialization_failed";
  if (!program.ok() && !rejected &&
      (program.status().IsInvalid() || program.status().IsNotImplemented()))
    program = Unavailable("unsupported_expression");
  std::shared_ptr<Impl::Entry> retired;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    --impl_->stats.active_compilations;
    reservation.active = false;
    entry->result = program;
    entry->complete = true;
    entry->last_use = ++impl_->clock;
    if (program.ok()) {
      ++impl_->stats.entries;
      impl_->live.push_back({*program, (*program)->memory_usage()});
      retired = impl_->evict(false);
    } else if (rejected) {
      ++impl_->stats.admission_rejections;
      --impl_->stats.compilations;
      impl_->entries.erase(key);
    } else {
      ++impl_->stats.failed_compilations;
      ++impl_->stats.negative_entries;
      const auto delay =
          std::min<uint32_t>(60, 1u << std::min<uint32_t>(entry->failures, 6));
      entry->failures = std::min<uint32_t>(entry->failures + 1, 7);
      entry->retry_at =
          program.status().IsInvalid() || program.status().IsNotImplemented()
              ? std::chrono::steady_clock::time_point::max()
              : finished + std::chrono::seconds(delay);
      retired = impl_->evict(true);
    }
  }
  ARROW_RETURN_NOT_OK(program.status());
  return PreparedScalarPredicate{*program, std::move(analysis.parameters),
                                 false, std::move(query_owners)};
} catch (const std::length_error &) {
  return Unavailable("expression_limit");
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("Scalar cache allocation");
} catch (...) {
  return arrow::Status::UnknownError("Scalar cache preparation failed");
}

ScalarPredicateCacheStats ScalarPredicateCache::stats() const {
  if (!InOriginalProcess()) return {};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto result = impl_->stats;
  auto add = [](MemoryUsage &sum, MemoryUsage bytes) {
    sum.object_bytes += bytes.object_bytes;
    sum.ir_bytes += bytes.ir_bytes;
    sum.owned_bytes += bytes.owned_bytes;
  };
  for (const auto &p : impl_->live) {
    if (p.program.expired()) continue;
    ++result.live_programs;
    add(result.live_memory, p.bytes);
  }
  for (const auto &entry : impl_->entries)
    if (entry.second->complete && entry.second->result.ok())
      add(result.resident_memory, (*entry.second->result)->memory_usage());
  return result;
}

}  // namespace zvec::sqlengine::scalar_jit
