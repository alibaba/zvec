// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

#include "db/sqlengine/stile/compile_checks.h"
#include <limits>
#include <unordered_map>
#include <arrow/array/validate.h>
#include <arrow/compute/api.h>
#include <arrow/compute/function.h>
#include <arrow/compute/registry.h>
#include "db/sqlengine/stile/validation.h"

namespace zvec::stile {
namespace {
namespace cp = arrow::compute;
using Function = std::shared_ptr<cp::Function>;

// Throwing out of a local static's constructor leaves initialization
// retryable. Preserve resource errors returned by Arrow during the snapshot.
struct InitializationFailure {
  arrow::Status status;
};

// Arrow 21 checks its LIKE regexes on first execution using a plain static
// flag per input type. Finish that first execution before publishing the
// captured built-ins, including for hosts that use their own Arrow fallback.
arrow::Status PrepareLike(const Function &like) {
  if (!like) return arrow::Status::OK();  // Arrow built without this function.
  const cp::MatchSubstringOptions options("x%");
  for (const auto &type : {arrow::utf8(), arrow::large_utf8(), arrow::binary(),
                           arrow::large_binary()}) {
    ARROW_ASSIGN_OR_RAISE(auto scalar,
                          arrow::MakeScalar(type, std::string("x")));
    const auto result =
        like->Execute({arrow::Datum(std::move(scalar))}, &options, nullptr);
    if (!result.ok() && !result.status().IsNotImplemented())
      return result.status();
  }
  return arrow::Status::OK();
}

// Keep the original functions alive even if the host later replaces a registry
// entry.
struct Builtins {
  std::unordered_map<std::string, Function> named;
  std::unordered_map<arrow::Type::type, Function> casts;
  Builtins() {
    auto *registry = cp::GetFunctionRegistry();
    for (const char *name :
         {"equal", "not_equal", "less", "less_equal", "greater",
          "greater_equal", "and", "or", "and_kleene", "or_kleene", "invert",
          "is_null", "is_valid", "is_in", "match_like", "list_value_length",
          "cast"}) {
      auto function = registry->GetFunction(name);
      if (!function.ok() && !function.status().IsKeyError())
        throw InitializationFailure{function.status()};
      named.emplace(name, std::move(function).ValueOr(nullptr));
    }
    auto ready = PrepareLike(named.at("match_like"));
    if (!ready.ok()) throw InitializationFailure{std::move(ready)};
    // Arrow binds the cast meta-function to a different scalar function per
    // target type. Bind one field using Arrow's public API to capture each.
    auto schema = arrow::schema({arrow::field("x", arrow::int8())});
    for (auto target :
         {arrow::int8(), arrow::int16(), arrow::int32(), arrow::int64(),
          arrow::uint8(), arrow::uint16(), arrow::uint32(), arrow::uint64(),
          arrow::float32(), arrow::float64()}) {
      auto expr =
          cp::call("cast", {cp::field_ref(0)}, cp::CastOptions::Safe(target));
      auto bound = expr.Bind(*schema);
      if (!bound.ok() && !bound.status().IsNotImplemented() &&
          !bound.status().IsKeyError())
        throw InitializationFailure{bound.status()};
      if (bound.ok() && bound->call())
        casts.emplace(target->id(), bound->call()->function);
    }
  }
};

const Builtins &KnownFunctions() {
  static const Builtins known;
  return known;
}

arrow::Status Charge(uint64_t n, size_t *remaining, const char *what) {
  if (n > *remaining) return arrow::Status::CapacityError("stile: ", what);
  *remaining -= static_cast<size_t>(n);
  return arrow::Status::OK();
}

bool IsBinaryLogic(const cp::Expression::Call &call) {
  const auto &name = call.function_name;
  return call.arguments.size() == 2 &&
         (name == "and" || name == "or" || name == "and_kleene" ||
          name == "or_kleene");
}

arrow::Status CheckKernel(const cp::Expression::Call &call) {
  const auto &known = KnownFunctions();
  const auto entry = known.named.find(call.function_name);
  if (entry == known.named.end())
    return arrow::Status::OK();  // Compiler checks descriptors.
  auto *expected = entry->second.get();
  auto unsupported = [&] {
    return arrow::Status::NotImplemented(
        "stile: unrecognized bound kernel or options for ", call.function_name);
  };
  const auto *options = call.options.get();
  if (call.function_name == "cast") {
    auto *cast = dynamic_cast<const cp::CastOptions *>(options);
    if (!cast || !cast->to_type.type || call.type != cast->to_type)
      return unsupported();
    const auto it = known.casts.find(call.type.id());
    if (it == known.casts.end()) return unsupported();
    expected = it->second.get();
  } else if (call.function_name == "is_null") {
    if (options && !dynamic_cast<const cp::NullOptions *>(options))
      return unsupported();
  } else if (call.function_name == "is_in") {
    if (!dynamic_cast<const cp::SetLookupOptions *>(options))
      return unsupported();
  } else if (call.function_name == "match_like") {
    if (!dynamic_cast<const cp::MatchSubstringOptions *>(options))
      return unsupported();
  } else if (options) {
    return unsupported();
  }
  if (!expected || call.function.get() != expected) return unsupported();
  std::vector<arrow::TypeHolder> types;
  types.reserve(call.arguments.size());
  for (const auto &arg : call.arguments) {
    if (!arg.is_valid() || !arg.type()) return unsupported();
    types.emplace_back(arg.type());
  }
  auto kernel = expected->DispatchExact(types);
  if (!kernel.ok()) {
    if (kernel.status().IsNotImplemented()) return unsupported();
    return kernel.status();
  }
  if (*kernel != call.kernel) return unsupported();
  return arrow::Status::OK();
}
}  // namespace

arrow::Status Initialize() try {
  ARROW_RETURN_NOT_OK(cp::Initialize());
  (void)KnownFunctions();
  return arrow::Status::OK();
} catch (InitializationFailure &failure) {
  return std::move(failure.status);
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: initializing built-ins");
}

namespace internal {
arrow::Status CompileChecks::check_tree(const cp::Expression &root) {
  if (options_.max_depth < 0)
    return arrow::Status::Invalid("Negative max_depth");
  struct Pending {
    const cp::Expression *expr;
    unsigned depth;
    size_t logic_root;
    size_t operands = 1;
  };
  size_t nodes = options_.max_nodes;
  ARROW_RETURN_NOT_OK(Charge(1, &nodes, "expression node budget"));
  std::vector<Pending> tree{{&root, 0, 0}};
  bool exceeds_boolean_depth = false;
  // Finish the bounded structural check before any semantic fallback. A
  // shallow unsupported sibling must not hide a hard node or depth limit.
  for (size_t i = 0; i < tree.size(); ++i) {
    const auto *expr = tree[i].expr;
    const unsigned depth = tree[i].depth;
    const size_t group = tree[i].logic_root;
    if (depth > 256)
      return arrow::Status::CapacityError("stile: physical depth exceeds 256");
    // A comparison may have a cast, list length and field below its leaf.
    if (depth > static_cast<uint64_t>(options_.max_depth) + 3)
      exceeds_boolean_depth = true;
    if (!expr->is_valid()) continue;
    if (const auto *call = expr->call()) {
      ARROW_RETURN_NOT_OK(
          Charge(call->arguments.size(), &nodes, "expression node budget"));
      // A binary operator adds one leaf to its flattened group. Track group
      // roots by index because push_back can relocate the traversal.
      const bool logic = IsBinaryLogic(*call);
      if (logic && ++tree[group].operands > options_.max_operands)
        return arrow::Status::CapacityError("stile: flattened operand budget");
      for (const auto &arg : call->arguments) {
        const auto *child = arg.is_valid() ? arg.call() : nullptr;
        const bool same = logic && child && child->arguments.size() == 2 &&
                          child->function_name == call->function_name;
        tree.push_back({&arg, depth + 1, same ? group : tree.size()});
      }
    }
  }
  // A soft decline in one subtree must not hide an over-budget literal or
  // set in another. Describe each distinct application call once, charging
  // repeated appearances of its set just like repeated built-in sets.
  auto fallback = arrow::Status::OK();
  for (const auto &item : tree) {
    auto status = check_node(*item.expr);
    if (!status.ok()) {
      if (!status.IsNotImplemented()) return status;
      if (fallback.ok()) fallback = std::move(status);
    }
  }
  if (exceeds_boolean_depth)
    return arrow::Status::NotImplemented("stile: expression depth");
  return fallback;
}

arrow::Status CompileChecks::check_node(const cp::Expression &expr) {
  if (!expr.is_valid() || !expr.type())
    return arrow::Status::NotImplemented("stile: unbound expression");
  if (const auto *call = expr.call()) {
    if (!call->kernel || !call->function)
      return arrow::Status::NotImplemented("stile: unbound call");
    ARROW_RETURN_NOT_OK(CheckKernel(*call));
    if (call->function_name == "is_in") {
      const auto *in =
          dynamic_cast<const cp::SetLookupOptions *>(call->options.get());
      if (in && in->value_set.is_arraylike())
        ARROW_RETURN_NOT_OK(charge_set(in->value_set));
    }
    if (const auto *like = dynamic_cast<const cp::MatchSubstringOptions *>(
            call->options.get()))
      ARROW_RETURN_NOT_OK(
          Charge(like->pattern.size(), &bytes_, "literal byte budget"));
    if (options_.functions && call->arguments.size() == 1 &&
        expr.type()->id() == arrow::Type::BOOL) {
      const auto *describe =
          options_.functions->find_list_membership(call->function_name);
      if (describe && !KnownFunctions().named.count(call->function_name))
        return check_membership(*call, *describe);
    }
  } else if (const auto *literal = expr.literal()) {
    if (!literal->is_scalar())
      return arrow::Status::NotImplemented("stile: non-scalar literal");
    const auto &scalar = *literal->scalar();
    if (const auto *binary =
            dynamic_cast<const arrow::BaseBinaryScalar *>(&scalar)) {
      if (binary->value) {
        if (binary->value->size() < 0)
          return arrow::Status::Invalid("Invalid binary literal buffer");
        ARROW_RETURN_NOT_OK(
            Charge(binary->value->size(), &bytes_, "literal byte budget"));
        if (!binary->value->is_cpu())
          return arrow::Status::NotImplemented("stile: non-CPU literal");
        if (binary->value->size() && !binary->value->data())
          return arrow::Status::Invalid("Missing binary literal data");
      }
    } else if (auto *fixed = dynamic_cast<const arrow::FixedWidthType *>(
                   scalar.type.get())) {
      ARROW_RETURN_NOT_OK(
          Charge((fixed->bit_width() + 7) / 8, &bytes_, "literal byte budget"));
    } else {
      return arrow::Status::NotImplemented("stile: literal type");
    }
    if (options_.validate_full) ARROW_RETURN_NOT_OK(scalar.ValidateFull());
  }
  return arrow::Status::OK();
}

const ListMembership *CompileChecks::membership_for(
    const cp::Expression::Call &call) const {
  const auto found = memberships_.find(&call);
  return found == memberships_.end() ? nullptr : &found->second;
}

arrow::Status CompileChecks::check_membership(
    const cp::Expression::Call &call,
    const FunctionRegistry::DescribeListMembership &describe) {
  auto found = memberships_.find(&call);
  if (found == memberships_.end()) {
    auto membership = describe(call);
    if (!membership.ok()) {
      if (membership.status().IsNotImplemented())
        memberships_.emplace(&call, ListMembership{});
      return membership.status();
    }
    found =
        memberships_.emplace(&call, std::move(membership).ValueOrDie()).first;
  }
  if (!found->second.values)
    return arrow::Status::NotImplemented(
        "stile: unavailable membership description");
  return charge_set(arrow::Datum(found->second.values));
}

arrow::Status CompileChecks::charge_set(const arrow::Datum &set) {
  const int64_t length = set.length();
  const size_t chunks = set.is_array() ? 1 : set.chunked_array()->num_chunks();
  if (length < 0) return arrow::Status::Invalid("Invalid set length");
  if (static_cast<uint64_t>(length) > std::numeric_limits<int32_t>::max())
    return arrow::Status::CapacityError(
        "stile: set exceeds member index capacity");
  ARROW_RETURN_NOT_OK(
      Charge(static_cast<size_t>(length), &values_, "set cardinality budget"));
  ARROW_RETURN_NOT_OK(Charge(chunks, &chunks_, "set chunk budget"));
  auto charge_buffers = [&](const arrow::ArrayData &chunk) {
    for (const auto &buffer : chunk.buffers) {
      if (!buffer) continue;
      if (buffer->size() < 0)
        return arrow::Status::Invalid("Negative set buffer size");
      ARROW_RETURN_NOT_OK(
          Charge(buffer->size(), &bytes_, "literal byte budget"));
    }
    return arrow::Status::OK();
  };
  if (set.is_array()) return charge_buffers(*set.array());
  for (const auto &chunk : set.chunked_array()->chunks()) {
    if (!chunk) return arrow::Status::Invalid("Missing set chunk");
    ARROW_RETURN_NOT_OK(charge_buffers(*chunk->data()));
  }
  return arrow::Status::OK();
}

arrow::Status CompileChecks::check_set_chunk(
    const arrow::ArrayData &chunk) const {
  ARROW_RETURN_NOT_OK(CheckReadable(chunk));
  return options_.validate_full ? arrow::internal::ValidateArrayFull(chunk)
                                : arrow::internal::ValidateArray(chunk);
}
}  // namespace internal
}  // namespace zvec::stile
