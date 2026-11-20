// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

#include "db/sqlengine/stile/predicate.h"
#include <algorithm>
#include <functional>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include <arrow/compute/api_scalar.h>
#include <arrow/compute/cast.h>
#include "db/sqlengine/stile/compile_checks.h"
#include "db/sqlengine/stile/nodes.h"

namespace zvec::stile {
namespace internal {

arrow::Status Unsupported(const std::string &what) {
  return arrow::Status::NotImplemented("stile: ", what);
}

namespace {

namespace cp = arrow::compute;

enum class CompareOp { kEq, kNe, kLt, kLe, kGt, kGe };

bool ParseCompare(const std::string &name, CompareOp *op) {
  static const std::pair<const char *, CompareOp> kNames[] = {
      {"equal", CompareOp::kEq},   {"not_equal", CompareOp::kNe},
      {"less", CompareOp::kLt},    {"less_equal", CompareOp::kLe},
      {"greater", CompareOp::kGt}, {"greater_equal", CompareOp::kGe}};
  for (const auto &[text, value] : kNames) {
    if (name == text) {
      *op = value;
      return true;
    }
  }
  return false;
}

// literal OP value == value FLIPPED(OP) literal.
CompareOp Flip(CompareOp op) {
  switch (op) {
    case CompareOp::kLt:
      return CompareOp::kGt;
    case CompareOp::kLe:
      return CompareOp::kGe;
    case CompareOp::kGt:
      return CompareOp::kLt;
    case CompareOp::kGe:
      return CompareOp::kLe;
    default:
      return op;
  }
}

template <typename ColumnT>
NodePtr MakeCompare(CompareOp op, ColumnT column,
                    typename ColumnT::Literal literal) {
  switch (op) {
    case CompareOp::kEq:
      return std::make_unique<Compare<ColumnT, std::equal_to<>>>(
          std::move(column), std::move(literal));
    case CompareOp::kNe:
      return std::make_unique<Compare<ColumnT, std::not_equal_to<>>>(
          std::move(column), std::move(literal));
    case CompareOp::kLt:
      return std::make_unique<Compare<ColumnT, std::less<>>>(
          std::move(column), std::move(literal));
    case CompareOp::kLe:
      return std::make_unique<Compare<ColumnT, std::less_equal<>>>(
          std::move(column), std::move(literal));
    case CompareOp::kGt:
      return std::make_unique<Compare<ColumnT, std::greater<>>>(
          std::move(column), std::move(literal));
    case CompareOp::kGe:
      break;
  }
  return std::make_unique<Compare<ColumnT, std::greater_equal<>>>(
      std::move(column), std::move(literal));
}

class Compiler {
 public:
  Compiler(const TableIndex &table, const CompileOptions &options)
      : table_(table),
        options_(options),
        checks_(options),
        fusion_pairs_(options.max_fusion_pairs) {}

  arrow::Status check_tree(const cp::Expression &expression) {
    return checks_.check_tree(expression);
  }

  NodeResult compile(const cp::Expression &expression, int depth) {
    if (depth > options_.max_depth) return Unsupported("expression depth");
    if (!expression.type() || expression.type()->id() != arrow::Type::BOOL) {
      return Unsupported("non-boolean node");
    }
    if (const auto *literal = expression.literal()) {
      if (!literal->is_scalar()) return Unsupported("non-scalar literal");
      const auto &scalar = *literal->scalar();
      if (!scalar.is_valid) return std::make_unique<Constant>(kTfNull);
      return std::make_unique<Constant>(
          ToTf(static_cast<const arrow::BooleanScalar &>(scalar).value));
    }
    if (expression.field_ref()) {
      const ColumnIndex *column = field_of(expression);
      if (!column) return Unsupported("field reference");
      return VisitColumn(*column, [](auto typed) -> NodeResult {
        using ColumnT = decltype(typed);
        if constexpr (std::is_same_v<typename ColumnT::Value, bool>) {
          return std::make_unique<BooleanColumn<ColumnT>>(std::move(typed));
        } else {
          return Unsupported("non-boolean field");
        }
      });
    }
    const auto *call = expression.call();
    if (!call) return Unsupported("expression node");
    const auto &name = call->function_name;
    const auto &args = call->arguments;

    CompareOp op;
    if (ParseCompare(name, &op) && args.size() == 2) {
      return compare_call(op, args[0], args[1]);
    }
    if (name == "and_kleene" || name == "or_kleene" || name == "and" ||
        name == "or") {
      if (args.size() != 2) return Unsupported(name);
      return logic_call(*call, depth);
    }
    if (name == "invert" && args.size() == 1) {
      ARROW_ASSIGN_OR_RAISE(auto child, compile(args[0], depth + 1));
      return std::make_unique<Not>(std::move(child));
    }
    if ((name == "is_null" || name == "is_valid") && args.size() == 1) {
      return null_test(*call);
    }
    if (name == "match_like" && args.size() == 1) return like_call(*call);
    if (name == "is_in" && args.size() == 1) return in_call(*call);
    if (const auto *membership = checks_.membership_for(*call))
      return membership_call(args[0], *membership);
    return Unsupported("function " + name);
  }

  // One AND / OR over the operands of directly nested calls to the same
  // function. NOT of an operand becomes a negated child, and a lower and an
  // upper bound on the same value under AND become one Between.
  NodeResult logic_call(const cp::Expression::Call &call, int depth) {
    const auto &name = call.function_name;
    const Logic::Kind kind = name == "and_kleene"  ? Logic::Kind::kKleeneAnd
                             : name == "or_kleene" ? Logic::Kind::kKleeneOr
                             : name == "and"       ? Logic::Kind::kPlainAnd
                                                   : Logic::Kind::kPlainOr;
    std::vector<std::pair<const cp::Expression *, int>> operands;
    ARROW_RETURN_NOT_OK(flatten(call, depth, &operands));
    std::vector<Logic::Child> children;
    std::vector<bool> fused(operands.size(), false);
    if (fusion_pairs_ &&
        (kind == Logic::Kind::kKleeneAnd || kind == Logic::Kind::kPlainAnd)) {
      std::vector<std::pair<size_t, Bound>> lowers, uppers;
      for (size_t i = 0; i < operands.size(); ++i) {
        Bound bound;
        if (AsBound(*operands[i].first, &bound))
          (bound.lower ? lowers : uppers).emplace_back(i, bound);
      }
      for (const auto &[i, low] : lowers) {
        if (!fusion_pairs_) break;
        for (const auto &[j, high] : uppers) {
          if (!fusion_pairs_) break;
          --fusion_pairs_;
          if (fused[j] || !low.value->Equals(*high.value)) continue;
          auto node = between_call(low, high);
          if (!node.ok()) {
            if (node.status().IsNotImplemented()) continue;
            return node.status();
          }
          children.push_back({node.MoveValueUnsafe(), false});
          fused[i] = fused[j] = true;
          break;
        }
      }
    }
    for (size_t i = 0; i < operands.size(); ++i) {
      if (fused[i]) continue;
      const cp::Expression *operand = operands[i].first;
      int operand_depth = operands[i].second;
      bool negate = false;
      for (const cp::Expression::Call *inner = operand->call();
           inner && inner->function_name == "invert" &&
           inner->arguments.size() == 1;
           inner = operand->call()) {
        if (++operand_depth > options_.max_depth) {
          return Unsupported("expression depth");
        }
        operand = &inner->arguments[0];
        negate = !negate;
      }
      ARROW_ASSIGN_OR_RAISE(auto node, compile(*operand, operand_depth));
      children.push_back({std::move(node), negate});
    }
    if (children.size() == 1) {
      auto node = std::move(children[0].node);
      if (children[0].negate) return std::make_unique<Not>(std::move(node));
      return node;
    }
    return std::make_unique<Logic>(kind, std::move(children));
  }

  arrow::Status flatten(
      const cp::Expression::Call &call, int depth,
      std::vector<std::pair<const cp::Expression *, int>> *operands) {
    // Both operands are one level deeper, including any fused into a range.
    if (depth >= options_.max_depth) return Unsupported("expression depth");
    for (const auto &argument : call.arguments) {
      const auto *inner = argument.call();
      if (inner && inner->function_name == call.function_name &&
          inner->arguments.size() == 2) {
        ARROW_RETURN_NOT_OK(flatten(*inner, depth + 1, operands));
      } else {
        operands->emplace_back(&argument, depth + 1);
      }
    }
    return arrow::Status::OK();
  }

  // `value > literal` (or >=, <, <=, either operand order) with a valid
  // literal of the value's type.
  struct Bound {
    const cp::Expression *value;
    const arrow::Scalar *literal;
    bool lower, inclusive;
  };
  static bool AsBound(const cp::Expression &expression, Bound *bound) {
    const auto *call = expression.call();
    CompareOp op;
    if (!call || call->arguments.size() != 2 ||
        !ParseCompare(call->function_name, &op)) {
      return false;
    }
    const cp::Expression *value = &call->arguments[0];
    const arrow::Datum *literal = call->arguments[1].literal();
    if (!literal) {
      value = &call->arguments[1];
      literal = call->arguments[0].literal();
      op = Flip(op);
    }
    if (!literal || !literal->is_scalar() || value->literal() ||
        !literal->scalar()->is_valid || !value->type() ||
        !literal->scalar()->type->Equals(*value->type())) {
      return false;
    }
    // Pair only expressions VisitValue can lower. Comparing arbitrary Arrow
    // trees here could traverse large function options on each pairing trial.
    auto *base = value;
    if (auto *cast = base->call();
        cast && cast->function_name == "cast" && cast->arguments.size() == 1)
      base = &cast->arguments.front();
    if (auto *length = base->call();
        length && length->function_name == "list_value_length" &&
        length->arguments.size() == 1)
      base = &length->arguments.front();
    if (!base->field_ref()) return false;
    bound->value = value;
    bound->literal = literal->scalar().get();
    switch (op) {
      case CompareOp::kGt:
      case CompareOp::kGe:
        bound->lower = true;
        break;
      case CompareOp::kLt:
      case CompareOp::kLe:
        bound->lower = false;
        break;
      default:
        return false;
    }
    bound->inclusive = op == CompareOp::kGe || op == CompareOp::kLe;
    return true;
  }

  NodeResult between_call(const Bound &low, const Bound &high) {
    return visit_value(*low.value, [&](auto typed) -> NodeResult {
      using ColumnT = decltype(typed);
      typename ColumnT::Literal low_value, high_value;
      ScalarValue(*low.literal, &low_value);
      ScalarValue(*high.literal, &high_value);
      return std::make_unique<Between<ColumnT>>(
          std::move(typed), std::move(low_value), low.inclusive,
          std::move(high_value), high.inclusive);
    });
  }

  // A column whose chunk boundaries every referenced column shares, so a
  // row is located once per evaluation; null when they differ.
  const ColumnIndex *shared_layout() {
    if (referenced_.empty()) return nullptr;
    const auto *first = referenced_.front();
    const auto &ends = first->chunk_ends();
    if (ends.empty()) return nullptr;
    // Check each distinct layout once even for very wide generated expressions.
    if (referenced_.size() > 2) {
      std::sort(referenced_.begin(), referenced_.end(),
                std::less<const ColumnIndex *>{});
      referenced_.erase(std::unique(referenced_.begin(), referenced_.end()),
                        referenced_.end());
    }
    for (const auto *column : referenced_) {
      if (column != first && column->chunk_ends() != ends) return nullptr;
    }
    return first;
  }

 private:
  // The table column of a top-level field reference, if its type matches.
  const ColumnIndex *field_of(const cp::Expression &expression) {
    const auto *parameter = expression.parameter();
    if (!parameter || parameter->indices.size() != 1) return nullptr;
    const int index = parameter->indices[0];
    if (index < 0 || index >= table_.num_columns()) return nullptr;
    const auto &column = *table_.column(index);
    if (!parameter->type.type ||
        !column.type()->Equals(*parameter->type.type)) {
      return nullptr;
    }
    // Bound by name against another schema would resolve another column.
    const auto *ref_name = parameter->ref.name();
    if (ref_name && *ref_name != table_.schema()->field(index)->name())
      return nullptr;
    if (column.layout() != Layout::kRowNumber) referenced_.push_back(&column);
    return &column;
  }

  // A field, or list_value_length of a list field.
  template <typename Make>
  NodeResult visit_base(const cp::Expression &expression, Make &&make) {
    if (const ColumnIndex *column = field_of(expression)) {
      return VisitColumn(*column, make);
    }
    const auto *call = expression.call();
    if (call && call->function_name == "list_value_length" &&
        call->arguments.size() == 1) {
      if (const ColumnIndex *column = field_of(call->arguments[0])) {
        return VisitList(*column, [&](auto lists) -> NodeResult {
          return make(ListLength<typename decltype(lists)::Array>(*column));
        });
      }
    }
    return Unsupported("value expression");
  }

  // VisitBase, optionally under one lossless cast.
  template <typename Make>
  NodeResult visit_value(const cp::Expression &expression, Make &&make) {
    const auto *call = expression.call();
    if (call && call->function_name == "cast" && call->arguments.size() == 1) {
      if (!dynamic_cast<const cp::CastOptions *>(call->options.get()) ||
          !expression.type()) {
        return Unsupported("cast");
      }
      const auto to = expression.type()->id();
      return visit_base(call->arguments[0], [&](auto inner) -> NodeResult {
        return WidenTo(to, std::move(inner), make);
      });
    }
    return visit_base(expression, make);
  }

  NodeResult compare_call(CompareOp op, const cp::Expression &left,
                          const cp::Expression &right) {
    const cp::Expression *value = &left;
    const arrow::Datum *literal = right.literal();
    if (!literal) {
      value = &right;
      literal = left.literal();
      op = Flip(op);
    }
    if (!literal || !literal->is_scalar() || value->literal()) {
      return Unsupported("comparison operands");
    }
    const auto &scalar = *literal->scalar();
    if (!value->type() || !scalar.type->Equals(*value->type())) {
      return Unsupported("comparison types");
    }
    const size_t previous_references = referenced_.size();
    return visit_value(*value, [&](auto typed) -> NodeResult {
      // Arrow evaluates the value operand even when the other operand is NULL.
      if (!scalar.is_valid) {
        // The folded constant does not need to locate the operand's column.
        referenced_.resize(previous_references);
        return std::make_unique<Constant>(kTfNull);
      }
      using ColumnT = decltype(typed);
      typename ColumnT::Literal literal_value;
      ScalarValue(scalar, &literal_value);
      return MakeCompare(op, std::move(typed), std::move(literal_value));
    });
  }

  NodeResult null_test(const cp::Expression::Call &call) {
    const bool is_null = call.function_name == "is_null";
    bool nan_is_null = false;
    if (is_null) {
      const auto *options =
          dynamic_cast<const cp::NullOptions *>(call.options.get());
      nan_is_null = options && options->nan_is_null;
    }
    auto make = [&](auto typed) -> NodeResult {
      using ColumnT = decltype(typed);
      if (!is_null) return std::make_unique<IsValid<ColumnT>>(std::move(typed));
      return std::make_unique<IsNull<ColumnT>>(std::move(typed), nan_is_null);
    };
    const auto &argument = call.arguments[0];
    const auto *column = field_of(argument);
    if (column && (column->layout() == Layout::kList ||
                   column->layout() == Layout::kLargeList)) {
      return VisitList(*column, make);
    }
    return visit_value(argument, make);
  }

  NodeResult like_call(const cp::Expression::Call &call) {
    const ColumnIndex *column = field_of(call.arguments[0]);
    const auto *options =
        dynamic_cast<const cp::MatchSubstringOptions *>(call.options.get());
    if (!column || !options) return Unsupported("LIKE operands");
    const auto type = column->type()->id();
    const bool utf8 =
        type == arrow::Type::STRING || type == arrow::Type::LARGE_STRING;
    ARROW_ASSIGN_OR_RAISE(auto matcher, LikeMatcher::Make(*options, utf8));
    return VisitColumn(*column, [&](auto typed) -> NodeResult {
      using ColumnT = decltype(typed);
      if constexpr (std::is_same_v<typename ColumnT::Value, std::string_view>) {
        return std::make_unique<Like<ColumnT>>(std::move(typed), matcher);
      } else {
        return Unsupported("LIKE on a non-binary column");
      }
    });
  }

  NodeResult in_call(const cp::Expression::Call &call) {
    const ColumnIndex *column = field_of(call.arguments[0]);
    const auto *options =
        dynamic_cast<const cp::SetLookupOptions *>(call.options.get());
    if (!column || !options || !options->value_set.is_arraylike()) {
      return Unsupported("IN operands");
    }
    // Arrow casts a differently typed set; keep that path in Arrow.
    if (!options->value_set.type()->Equals(*column->type())) {
      return Unsupported("IN set type");
    }
    const auto &set = options->value_set;
    const auto behavior = options->GetNullMatchingBehavior();
    const auto max_values = static_cast<size_t>(options->value_set.length());
    return VisitColumn(*column, [&](auto typed) -> NodeResult {
      using ColumnT = decltype(typed);
      if (set.is_array()) {
        ARROW_RETURN_NOT_OK(checks_.check_set_chunk(*set.array()));
      } else {
        for (const auto &chunk : set.chunked_array()->chunks()) {
          if (!chunk || !chunk->type()->Equals(*column->type()))
            return arrow::Status::TypeError("Inconsistent IN set chunk type");
          ARROW_RETURN_NOT_OK(checks_.check_set_chunk(*chunk->data()));
        }
      }
      auto node =
          std::make_unique<In<ColumnT>>(std::move(typed), behavior, max_values);
      if (set.is_array())
        node->add(*set.make_array());
      else
        for (const auto &chunk : set.chunked_array()->chunks())
          node->add(*chunk);
      node->finish();
      return node;
    });
  }

  NodeResult membership_call(const cp::Expression &argument,
                             const ListMembership &membership) {
    const ColumnIndex *column = field_of(argument);
    if (!column || !membership.values) {
      return Unsupported("list membership operands");
    }
    const bool all = membership.mode == ListMembership::Mode::kAll;
    return VisitList(*column, [&](auto lists) -> NodeResult {
      const auto &element = column->type()->field(0)->type();
      if (!membership.values->type()->Equals(*element)) {
        return Unsupported("list membership value type");
      }
      ARROW_RETURN_NOT_OK(checks_.check_set_chunk(*membership.values->data()));
      using ListArrayT = typename decltype(lists)::Array;
      auto make = [&](const auto *values) -> NodeResult {
        using ElementArrayT =
            std::remove_cv_t<std::remove_pointer_t<decltype(values)>>;
        if (all) {
          return std::make_unique<Membership<ListArrayT, ElementArrayT, true>>(
              lists, *values, membership.nulls);
        }
        return std::make_unique<Membership<ListArrayT, ElementArrayT, false>>(
            lists, *values, membership.nulls);
      };
      const auto &values = *membership.values;
      switch (element->id()) {
#define STILE_ELEMENT(ID, ARRAY) \
  case arrow::Type::ID:          \
    return make(static_cast<const arrow::ARRAY *>(&values));
        STILE_ELEMENT(INT8, Int8Array)
        STILE_ELEMENT(INT16, Int16Array)
        STILE_ELEMENT(INT32, Int32Array)
        STILE_ELEMENT(INT64, Int64Array)
        STILE_ELEMENT(UINT8, UInt8Array)
        STILE_ELEMENT(UINT16, UInt16Array)
        STILE_ELEMENT(UINT32, UInt32Array)
        STILE_ELEMENT(UINT64, UInt64Array)
        STILE_ELEMENT(FLOAT, FloatArray)
        STILE_ELEMENT(DOUBLE, DoubleArray)
        STILE_ELEMENT(BOOL, BooleanArray)
        STILE_ELEMENT(STRING, StringArray)
        STILE_ELEMENT(LARGE_STRING, LargeStringArray)
        STILE_ELEMENT(BINARY, BinaryArray)
        STILE_ELEMENT(LARGE_BINARY, LargeBinaryArray)
#undef STILE_ELEMENT
        default:
          return Unsupported("list element type " + element->ToString());
      }
    });
  }

  const TableIndex &table_;
  const CompileOptions &options_;
  CompileChecks checks_;
  size_t fusion_pairs_;
  std::vector<const ColumnIndex *> referenced_;
};

}  // namespace
}  // namespace internal

Predicate::Predicate(TableIndex::Ptr table, std::unique_ptr<detail::Node> root,
                     const ColumnIndex *layout)
    : table_(std::move(table)),
      root_(std::move(root)),
      layout_(layout),
      num_rows_(table_->num_rows()) {}

Predicate::~Predicate() = default;

namespace {
// Rows evaluated together: enough independent loads to keep the memory
// system busy, few enough that they stay in L1 until evaluated.
constexpr size_t kBlock = 32;
static_assert(kBlock <= detail::Node::kMaxBatch);
}  // namespace

void Predicate::eval_block(const int64_t *rows, size_t count, bool checked,
                           detail::Tf *out) const {
  if (checked && num_rows_ == 0) {
    std::fill(out, out + count, detail::kTfNull);
    return;
  }
  detail::Row located[detail::Node::kMaxBatch];
  for (size_t i = 0; i < count; ++i) {
    int64_t row = rows[i];
    assert(checked || (row >= 0 && row < num_rows_));
    // Out-of-range rows evaluate row 0 and are discarded below.
    if (checked &&
        static_cast<uint64_t>(row) >= static_cast<uint64_t>(num_rows_)) {
      row = 0;
    }
    located[i] = locate(row);
  }
  root_->prefetch(located, count);
  root_->eval_batch(located, count, out);
  if (!checked) return;
  for (size_t i = 0; i < count; ++i) {
    if (static_cast<uint64_t>(rows[i]) >= static_cast<uint64_t>(num_rows_)) {
      out[i] = detail::kTfNull;
    }
  }
}

size_t Predicate::match_many(const int64_t *rows, size_t count,
                             bool *matches) const {
  assert(!count || (rows && matches));
  size_t total = 0;
  detail::Tf values[kBlock];
  for (size_t begin = 0; begin < count; begin += kBlock) {
    const size_t n = std::min(kBlock, count - begin);
    eval_block(rows + begin, n, true, values);
    for (size_t i = 0; i < n; ++i) {
      const bool match = (values[i] & detail::kTfTrue) != 0;
      matches[begin + i] = match;
      total += match;
    }
  }
  return total;
}

void Predicate::evaluate_many(const int64_t *rows, size_t count,
                              Truth *out) const {
  assert(!count || (rows && out));
  constexpr Truth kTruth[3] = {Truth::kNull, Truth::kTrue, Truth::kFalse};
  detail::Tf values[kBlock];
  for (size_t begin = 0; begin < count; begin += kBlock) {
    const size_t n = std::min(kBlock, count - begin);
    eval_block(rows + begin, n, false, values);
    for (size_t i = 0; i < n; ++i) out[begin + i] = kTruth[values[i]];
  }
}

arrow::Result<std::unique_ptr<Predicate>> Predicate::Compile(
    const arrow::compute::Expression &bound, TableIndex::Ptr table,
    const CompileOptions &options) try {
  if (!table) return arrow::Status::Invalid("stile: missing table");
  ARROW_RETURN_NOT_OK(stile::Initialize());
  internal::Compiler compiler(*table, options);
  ARROW_RETURN_NOT_OK(compiler.check_tree(bound));
  ARROW_ASSIGN_OR_RAISE(auto root, compiler.compile(bound, 0));
  const ColumnIndex *layout = compiler.shared_layout();
  return std::unique_ptr<Predicate>(
      new Predicate(std::move(table), std::move(root), layout));
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: compiling a predicate");
} catch (const std::length_error &error) {
  return arrow::Status::CapacityError("stile: ", error.what());
} catch (const std::exception &error) {
  return arrow::Status::UnknownError("stile: compiling a predicate: ",
                                     error.what());
}

}  // namespace zvec::stile
