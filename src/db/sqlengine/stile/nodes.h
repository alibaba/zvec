// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

// Evaluation nodes. Each reproduces one Arrow function on a single row and
// returns its truth in the two-bit encoding of detail::Tf.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <arrow/array.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/type_traits.h>
#include "db/sqlengine/stile/accessors.h"
#include "db/sqlengine/stile/functions.h"
#include "db/sqlengine/stile/like.h"
#include "db/sqlengine/stile/value_index.h"

namespace zvec::stile::internal {

using detail::kTfFalse;
using detail::kTfNull;
using detail::kTfTrue;
using detail::Node;
using detail::Tf;

STILE_INLINE Tf ToTf(bool value) {
  return static_cast<Tf>(2 - static_cast<int>(value));
}
// NULL unless valid; without a branch.
STILE_INLINE Tf ToTf(bool valid, bool value) {
  return static_cast<Tf>(static_cast<unsigned>(valid) << (value ? 0 : 1));
}
STILE_INLINE Tf NotTf(Tf value) {
  return static_cast<Tf>(((value & 1) << 1) | (value >> 1));
}
inline Tf ToTf(Truth truth) {
  constexpr Tf kTf[3] = {kTfFalse, kTfTrue, kTfNull};
  return kTf[static_cast<int>(truth)];
}

// Costs: a column read is 1; logic sums its children.
constexpr int kCheapCost = 4;

// A leaf: one value per row. Derived defines `Tf eval_one(const Row &)` and
// optionally `void prefetch_one(const Row &)` and `int cost()`; the batch
// loops call them without virtual dispatch.
template <typename Derived>
class Leaf : public Node {
 public:
  Tf eval(const Row &row) const final {
    return self().eval_one(row);
  }
  void eval_batch(const Row *rows, size_t count, Tf *out) const final {
    for (size_t i = 0; i < count; ++i) out[i] = self().eval_one(rows[i]);
  }
  void prefetch(const Row *rows, size_t count) const final {
    for (size_t i = 0; i < count; ++i) self().prefetch_one(rows[i]);
  }
  STILE_INLINE void prefetch_one(const Row &) const {}

 private:
  const Derived &self() const {
    return static_cast<const Derived &>(*this);
  }
};

class Constant final : public Leaf<Constant> {
 public:
  explicit Constant(Tf value) : value_(value) {}
  STILE_INLINE Tf eval_one(const Row &) const {
    return value_;
  }
  int cost() const override {
    return 0;
  }

 private:
  Tf value_;
};

template <typename ColumnT>
class BooleanColumn final : public Leaf<BooleanColumn<ColumnT>> {
 public:
  explicit BooleanColumn(ColumnT column) : column_(std::move(column)) {}
  STILE_INLINE Tf eval_one(const Row &row) const {
    bool value = false;
    const bool valid = column_.load(row, &value);
    return ToTf(valid, value);
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    column_.prefetch(row);
  }

 private:
  ColumnT column_;
};

// Costs of reading one value: fixed-width data is one load, variable-width
// data an offset and the bytes, lists go through Arrow's list arrays.
template <typename ColumnT>
constexpr int ReadCost() {
  return std::is_same_v<typename ColumnT::Value, std::string_view> ? 3 : 1;
}

// Arrow compares numbers with the C++ operators (IEEE NaN semantics),
// booleans as false < true and binary values bytewise as unsigned chars.
template <typename ColumnT, typename Op>
class Compare final : public Leaf<Compare<ColumnT, Op>> {
 public:
  using Value = typename ColumnT::Value;
  Compare(ColumnT column, typename ColumnT::Literal literal)
      : column_(std::move(column)), literal_(std::move(literal)) {}
  STILE_INLINE Tf eval_one(const Row &row) const {
    Value value{};
    const bool valid = column_.load(row, &value);
    return ToTf(valid, Op{}(value, static_cast<Value>(literal_)));
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    column_.prefetch(row);
  }
  int cost() const override {
    return ReadCost<ColumnT>();
  }

 private:
  ColumnT column_;
  typename ColumnT::Literal literal_;
};

// `low <(=) value AND value <(=) high` on one value: two comparisons of the
// same column fused so the value is read once.
template <typename ColumnT>
class Between final : public Leaf<Between<ColumnT>> {
 public:
  using Value = typename ColumnT::Value;
  using Literal = typename ColumnT::Literal;
  Between(ColumnT column, Literal low, bool low_inclusive, Literal high,
          bool high_inclusive)
      : column_(std::move(column)),
        low_(std::move(low)),
        high_(std::move(high)),
        low_inclusive_(low_inclusive),
        high_inclusive_(high_inclusive) {}
  STILE_INLINE Tf eval_one(const Row &row) const {
    Value value{};
    const bool valid = column_.load(row, &value);
    const auto low = static_cast<Value>(low_);
    const auto high = static_cast<Value>(high_);
    const bool above = low_inclusive_ ? value >= low : value > low;
    const bool below = high_inclusive_ ? value <= high : value < high;
    return ToTf(valid, above && below);
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    column_.prefetch(row);
  }
  int cost() const override {
    return ReadCost<ColumnT>();
  }

 private:
  ColumnT column_;
  Literal low_, high_;
  bool low_inclusive_, high_inclusive_;
};

template <typename ColumnT>
class IsNull final : public Leaf<IsNull<ColumnT>> {
 public:
  IsNull(ColumnT column, bool nan_is_null)
      : column_(std::move(column)), nan_is_null_(nan_is_null) {}
  STILE_INLINE Tf eval_one(const Row &row) const {
    using Value = typename ColumnT::Value;
    if constexpr (std::is_floating_point_v<Value>) {
      if (nan_is_null_) {
        Value value{};
        const bool valid = column_.load(row, &value);
        return ToTf(!valid || std::isnan(value));
      }
    }
    return ToTf(!column_.valid(row));
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    column_.prefetch(row);
  }

 private:
  ColumnT column_;
  bool nan_is_null_;
};

template <typename ColumnT>
class IsValid final : public Leaf<IsValid<ColumnT>> {
 public:
  explicit IsValid(ColumnT column) : column_(std::move(column)) {}
  STILE_INLINE Tf eval_one(const Row &row) const {
    return ToTf(column_.valid(row));
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    column_.prefetch(row);
  }

 private:
  ColumnT column_;
};

class Not final : public Node {
 public:
  explicit Not(NodePtr child) : child_(std::move(child)) {}
  Tf eval(const Row &row) const override {
    return NotTf(child_->eval(row));
  }
  void eval_batch(const Row *rows, size_t count, Tf *out) const override {
    child_->eval_batch(rows, count, out);
    for (size_t i = 0; i < count; ++i) out[i] = NotTf(out[i]);
  }
  void prefetch(const Row *rows, size_t count) const override {
    child_->prefetch(rows, count);
  }
  int cost() const override {
    return child_->cost();
  }

 private:
  NodePtr child_;
};

// AND / OR over any number of children, each optionally negated.
//
// Kleene: a definite false (AND) or true (OR) decides. Plain: NULL whenever
// any child is NULL. Both are commutative, so children run cheapest first.
// Cheap children are combined without branches, which avoids mispredicted
// short-circuit branches on unpredictable data; costlier ones are skipped
// once the result is decided.
class Logic final : public Node {
 public:
  enum class Kind { kKleeneAnd, kKleeneOr, kPlainAnd, kPlainOr };
  struct Child {
    NodePtr node;
    bool negate = false;
  };

  Logic(Kind kind, std::vector<Child> children)
      : and_(kind == Kind::kKleeneAnd || kind == Kind::kPlainAnd),
        plain_(kind == Kind::kPlainAnd || kind == Kind::kPlainOr),
        children_(std::move(children)) {
    std::stable_sort(children_.begin(), children_.end(),
                     [](const Child &a, const Child &b) {
                       return a.node->cost() < b.node->cost();
                     });
    while (cheap_ < children_.size() &&
           children_[cheap_].node->cost() <= kCheapCost) {
      ++cheap_;
    }
    for (const auto &child : children_) cost_ += child.node->cost();
    pair_ = children_.size() == 2 && cheap_ == 2;
  }

  Tf eval(const Row &row) const override {
    if (pair_) {  // The common shape: two cheap children, no loop.
      State state;
      state.add(child_value(0, row), and_);
      state.add(child_value(1, row), and_);
      return state.result(and_, plain_);
    }
    // AND: all_ tracks "every child true", any_ "some child false"; OR the
    // reverse. known_ tracks "no child NULL" for plain logic.
    State state;
    for (size_t i = 0; i < cheap_; ++i) state.add(child_value(i, row), and_);
    for (size_t i = cheap_; i < children_.size(); ++i) {
      if (decided(state)) break;
      state.add(child_value(i, row), and_);
    }
    return state.result(and_, plain_);
  }

  void eval_batch(const Row *rows, size_t count, Tf *out) const override {
    State states[kMaxBatch];
    Tf values[kMaxBatch];
    for (size_t i = 0; i < cheap_; ++i) {
      children_[i].node->eval_batch(rows, count, values);
      negate(i, values, count);
      for (size_t r = 0; r < count; ++r) states[r].add(values[r], and_);
    }
    Row pending[kMaxBatch];
    uint8_t where[kMaxBatch];
    for (size_t i = cheap_; i < children_.size(); ++i) {
      size_t open = 0;
      for (size_t r = 0; r < count; ++r) {
        if (decided(states[r])) continue;
        pending[open] = rows[r];
        where[open++] = static_cast<uint8_t>(r);
      }
      if (open == 0) break;
      children_[i].node->eval_batch(pending, open, values);
      negate(i, values, open);
      for (size_t k = 0; k < open; ++k) states[where[k]].add(values[k], and_);
    }
    for (size_t r = 0; r < count; ++r) out[r] = states[r].result(and_, plain_);
  }

  void prefetch(const Row *rows, size_t count) const override {
    for (const auto &child : children_) child.node->prefetch(rows, count);
  }
  int cost() const override {
    return cost_;
  }

 private:
  struct State {
    // Bit 0 of `all` and bit 1 of `any`, as in Tf; `known` in bit 0.
    uint8_t all = kTfTrue | kTfFalse, any = 0, known = 1;
    void add(Tf value, bool is_and) {
      // AND keeps "all true" and "any false"; OR "any true" and "all false".
      const uint8_t truth = is_and ? value : NotTf(value);
      all &= truth;
      any |= truth;
      known &= static_cast<uint8_t>((value | (value >> 1)) & 1);
    }
    Tf result(bool is_and, bool plain) const {
      const Tf kleene = static_cast<Tf>((all & kTfTrue) | (any & kTfFalse));
      const Tf result = is_and ? kleene : NotTf(kleene);
      return plain ? static_cast<Tf>(result & (0u - known)) : result;
    }
  };

  Tf child_value(size_t i, const Row &row) const {
    const Tf value = children_[i].node->eval(row);
    return children_[i].negate ? NotTf(value) : value;
  }
  void negate(size_t i, Tf *values, size_t count) const {
    if (!children_[i].negate) return;
    for (size_t r = 0; r < count; ++r) values[r] = NotTf(values[r]);
  }
  // Kleene logic is decided by one deciding child; plain logic only by a
  // NULL child, since a later NULL turns any result into NULL.
  bool decided(const State &state) const {
    return plain_ ? state.known == 0 : (state.any & kTfFalse) != 0;
  }

  bool and_, plain_, pair_ = false;
  std::vector<Child> children_;
  size_t cheap_ = 0;
  int cost_ = 0;
};

template <typename ColumnT>
class Like final : public Leaf<Like<ColumnT>> {
 public:
  Like(ColumnT column, LikeMatcher matcher)
      : column_(std::move(column)), matcher_(std::move(matcher)) {}
  STILE_INLINE Tf eval_one(const Row &row) const {
    std::string_view value;
    if (!column_.read(row, &value)) return kTfNull;
    return ToTf(matcher_.match(value));
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    column_.prefetch(row);
  }
  int cost() const override {
    return matcher_.kind() == LikeMatcher::Kind::kSubstring ? 8 : 4;
  }

 private:
  ColumnT column_;
  LikeMatcher matcher_;
};

// is_in. Arrow's set lookup hashes the bit pattern of numbers, so distinct
// NaN payloads and +0/-0 are distinct members; binary values compare
// bytewise.
template <typename ColumnT>
class In final : public Leaf<In<ColumnT>> {
 public:
  using Value = typename ColumnT::Value;
  using Behavior = arrow::compute::SetLookupOptions::NullMatchingBehavior;
  static constexpr bool kBinary = std::is_same_v<Value, std::string_view>;

  In(ColumnT column, Behavior behavior, size_t max_values)
      : column_(std::move(column)), behavior_(behavior), set_(max_values) {}

  void add(const arrow::Array &values) {
    for (int64_t i = 0; i < values.length(); ++i) {
      if (values.IsNull(i)) {
        set_has_null_ |= behavior_ != Behavior::SKIP;
      } else if constexpr (kBinary) {
        set_.insert(static_cast<const typename ColumnT::Access::Array &>(values)
                        .GetView(i));
      } else {
        using ArrayType = typename arrow::TypeTraits<
            typename arrow::CTypeTraits<Value>::ArrowType>::ArrayType;
        set_.insert(Bits(static_cast<const ArrayType &>(values).Value(i)));
      }
    }
  }

  void finish() {
    set_.finish();
  }

  STILE_INLINE Tf eval_one(const Row &row) const {
    Value value;
    if (!column_.read(row, &value)) {
      if (behavior_ == Behavior::MATCH) return ToTf(set_has_null_);
      return behavior_ == Behavior::SKIP ? kTfFalse : kTfNull;
    }
    bool found;
    if constexpr (kBinary) {
      found = set_.contains(value);
    } else {
      found = set_.contains(Bits(value));
    }
    if (found) return kTfTrue;
    return behavior_ == Behavior::INCONCLUSIVE && set_has_null_ ? kTfNull
                                                                : kTfFalse;
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    column_.prefetch(row);
  }
  int cost() const override {
    return kBinary ? 8 : 3;
  }

 private:
  ColumnT column_;
  Behavior behavior_;
  bool set_has_null_ = false;
  IndexFor<Value> set_;
};

// ListMembership (see stile/functions.h): operator== on numbers, so one key
// for +0/-0 and none for NaN; bytewise on binary values.
template <typename ListArrayT, typename ElementArrayT, bool kAll>
class Membership final
    : public Leaf<Membership<ListArrayT, ElementArrayT, kAll>> {
 public:
  using Element =
      std::decay_t<decltype(std::declval<const ElementArrayT &>().Value(0))>;
  static constexpr bool kBinary = std::is_same_v<Element, std::string_view>;

  Membership(ListColumn<ListArrayT> lists, const ElementArrayT &values,
             ListMembership::Nulls nulls)
      : lists_(std::move(lists)),
        skip_nulls_(nulls == ListMembership::Nulls::kSkip),
        set_(static_cast<size_t>(values.length())) {
    for (int64_t i = 0; i < values.length(); ++i) {
      if (skip_nulls_ && values.IsNull(i)) continue;
      uint64_t key;
      if constexpr (kBinary) {
        set_.insert(values.GetView(i));
      } else if (Key(values.Value(i), &key)) {
        set_.insert(key);
      } else {
        unmatchable_ = true;  // NaN equals nothing.
      }
    }
    mask_ = set_.size() >= 64 ? ~uint64_t{0} : (uint64_t{1} << set_.size()) - 1;
    set_.finish();
  }

  STILE_INLINE Tf eval_one(const Row &row) const {
    int64_t local;
    const ListArrayT *list = lists_.list(row, &local);
    if (!list) return kTfNull;
    const auto &values = static_cast<const ElementArrayT &>(*list->values());
    const int64_t begin = list->value_offset(local);
    const int64_t end = begin + list->value_length(local);
    if constexpr (kAll) {
      if (unmatchable_) return kTfFalse;
      if (set_.size() == 0) return kTfTrue;
      if (end - begin < set_.size()) return kTfFalse;
      if (set_.size() <= 64) {
        uint64_t found = 0;
        for (int64_t i = begin; i < end && found != mask_; ++i) {
          const int32_t index = find(values, i);
          if (index >= 0) found |= uint64_t{1} << index;
        }
        return ToTf(found == mask_);
      }
      // Use a small stack frame for typical sets. The larger frame processes
      // every set within the default compilation budget in one pass.
      if (set_.size() <= 256) return match_windows<4>(values, begin, end);
      return match_windows<1024>(values, begin, end);
    } else {
      for (int64_t i = begin; i < end; ++i) {
        if (contains(values, i)) return kTfTrue;
      }
      return kTfFalse;
    }
  }
  STILE_INLINE void prefetch_one(const Row &row) const {
    lists_.prefetch(row);
  }
  int cost() const override {
    return 16;
  }

 private:
  template <size_t Words>
  Tf match_windows(const ElementArrayT &values, int64_t begin,
                   int64_t end) const {
    uint64_t seen[Words];
    constexpr int64_t width = Words * 64;
    for (int64_t base = 0; base < set_.size(); base += width) {
      auto remaining = std::min<int64_t>(width, set_.size() - base);
      std::fill_n(seen, (remaining + 63) / 64, uint64_t{0});
      for (auto i = begin; i < end && remaining; ++i) {
        const auto bit = static_cast<uint64_t>(find(values, i) - base);
        if (bit >= static_cast<uint64_t>(width)) continue;
        auto &word = seen[bit >> 6];
        const auto mask = uint64_t{1} << (bit & 63);
        remaining -= (word & mask) == 0;
        word |= mask;
      }
      if (remaining) return kTfFalse;
    }
    return kTfTrue;
  }
  static bool Key(Element value, uint64_t *key) {
    if constexpr (std::is_floating_point_v<Element>) {
      if (std::isnan(value)) return false;
      if (value == 0) value = 0;
    }
    *key = Bits(value);
    return true;
  }
  bool contains(const ElementArrayT &values, int64_t i) const {
    if (skip_nulls_ && values.IsNull(i)) return false;
    if constexpr (kBinary) {
      return set_.contains(values.GetView(i));
    } else {
      uint64_t key;
      return Key(values.Value(i), &key) && set_.contains(key);
    }
  }
  int32_t find(const ElementArrayT &values, int64_t i) const {
    if (skip_nulls_ && values.IsNull(i)) return -1;
    if constexpr (kBinary) {
      return set_.find(values.GetView(i));
    } else {
      uint64_t key;
      return Key(values.Value(i), &key) ? set_.find(key) : -1;
    }
  }

  ListColumn<ListArrayT> lists_;
  bool skip_nulls_;
  IndexFor<Element> set_;
  bool unmatchable_ = false;
  uint64_t mask_ = 0;
};

}  // namespace zvec::stile::internal
