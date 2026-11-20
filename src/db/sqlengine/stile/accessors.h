// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

// Typed per-row access to ColumnIndex data. Every accessor has `Value`,
// `Literal` (how a comparison literal is stored), `valid(row)`,
// `read(row, &value)`, which returns false for NULL, `load(row, &value)`,
// which returns validity like read() but may also fill `value` for a NULL
// (fixed-width data is read without a branch), and `prefetch(row)`.
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <arrow/array.h>
#include <arrow/scalar.h>
#include <arrow/type_traits.h>
#include <arrow/util/bit_util.h>
#include "db/sqlengine/stile/predicate.h"
#include "db/sqlengine/stile/table_index.h"

#if defined(__GNUC__) || defined(__clang__)
#define STILE_PREFETCH(address) __builtin_prefetch(address)
// Per-row accessors sit on the hot path of every node; compilers otherwise
// decline to inline them across the many template instantiations.
#define STILE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define STILE_PREFETCH(address) ((void)(address))
#define STILE_INLINE __forceinline
#else
#define STILE_PREFETCH(address) ((void)(address))
#define STILE_INLINE inline
#endif

namespace zvec::stile::internal {

using detail::Row;
using Layout = ColumnIndex::Layout;
using Span = ColumnIndex::Span;
using NodePtr = std::unique_ptr<detail::Node>;
using NodeResult = arrow::Result<NodePtr>;

arrow::Status Unsupported(const std::string &what);

// Resolves rows of one column, or uses the shared location in Row.
class Rows {
 public:
  explicit Rows(const ColumnIndex &column)
      : column_(&column),
        spans_(column.spans().data()),
        arrays_(column.arrays().data()) {}
  STILE_INLINE size_t chunk(const Row &row, int64_t *local) const {
    if (row.chunk != Row::kUnresolved) {
      *local = row.local;
      return row.chunk;
    }
    return column_->locate(row.id, local);
  }
  STILE_INLINE const Span &span(const Row &row, int64_t *local) const {
    return spans_[chunk(row, local)];
  }
  template <typename ArrayT>
  STILE_INLINE const ArrayT &array(const Row &row, int64_t *local) const {
    return static_cast<const ArrayT &>(*arrays_[chunk(row, local)]);
  }

 private:
  const ColumnIndex *column_;
  // Owned by the column, which the predicate's table keeps alive.
  const Span *spans_;
  const arrow::Array *const *arrays_;
};

STILE_INLINE bool ValidAt(const Span &span, int64_t local) {
  return !span.validity ||
         arrow::bit_util::GetBit(span.validity, span.offset + local);
}

STILE_INLINE void PrefetchValidity(const Span &span, int64_t local) {
  if (span.validity)
    STILE_PREFETCH(span.validity + ((span.offset + local) >> 3));
}

// kUnconditional: every slot, NULL or not, holds a readable value.
template <typename T>
struct FixedAccess {
  using Value = T;
  using Literal = T;
  static constexpr bool kUnconditional = true;
  STILE_INLINE static T Get(const Span &span, int64_t local) {
    return reinterpret_cast<const T *>(span.row_data)[local];
  }
  STILE_INLINE static void prefetch(const Span &span, int64_t local) {
    STILE_PREFETCH(reinterpret_cast<const T *>(span.row_data) + local);
  }
};

struct BitAccess {
  using Value = bool;
  using Literal = bool;
  static constexpr bool kUnconditional = true;
  STILE_INLINE static bool Get(const Span &span, int64_t local) {
    return arrow::bit_util::GetBit(span.values, span.offset + local);
  }
  STILE_INLINE static void prefetch(const Span &span, int64_t local) {
    STILE_PREFETCH(span.values + ((span.offset + local) >> 3));
  }
};

template <typename Offset>
struct BinaryAccess {
  using Value = std::string_view;
  using Literal = std::string;
  using Array = std::conditional_t<sizeof(Offset) == 4, arrow::BinaryArray,
                                   arrow::LargeBinaryArray>;
  // Offsets of NULL slots are only range-checked by full validation.
  static constexpr bool kUnconditional = false;
  STILE_INLINE static void prefetch(const Span &span, int64_t local) {
    STILE_PREFETCH(reinterpret_cast<const Offset *>(span.row_data) + local);
  }
  STILE_INLINE static std::string_view Get(const Span &span, int64_t local) {
    const auto *offsets = reinterpret_cast<const Offset *>(span.row_data);
    return std::string_view(
        reinterpret_cast<const char *>(span.values) + offsets[local],
        static_cast<size_t>(offsets[local + 1] - offsets[local]));
  }
};

// Flat columns read raw spans.
template <typename AccessT>
class SpanColumn {
 public:
  using Access = AccessT;
  using Value = typename Access::Value;
  using Literal = typename Access::Literal;
  explicit SpanColumn(const ColumnIndex &column) : rows_(column) {}
  STILE_INLINE bool valid(const Row &row) const {
    int64_t local;
    const Span &span = rows_.span(row, &local);
    return ValidAt(span, local);
  }
  STILE_INLINE bool read(const Row &row, Value *value) const {
    int64_t local;
    const Span &span = rows_.span(row, &local);
    if (!ValidAt(span, local)) return false;
    *value = Access::Get(span, local);
    return true;
  }
  STILE_INLINE bool load(const Row &row, Value *value) const {
    if constexpr (Access::kUnconditional) {
      int64_t local;
      const Span &span = rows_.span(row, &local);
      *value = Access::Get(span, local);
      return ValidAt(span, local);
    } else {
      return read(row, value);
    }
  }
  STILE_INLINE void prefetch(const Row &row) const {
    int64_t local;
    const Span &span = rows_.span(row, &local);
    Access::prefetch(span, local);
    PrefetchValidity(span, local);
  }

 private:
  Rows rows_;
};

// Virtual row numbers: the value is the row, never NULL.
class RowNumberColumn {
 public:
  using Value = uint64_t;
  using Literal = uint64_t;
  STILE_INLINE bool valid(const Row &) const {
    return true;
  }
  STILE_INLINE bool read(const Row &row, uint64_t *value) const {
    *value = static_cast<uint64_t>(row.id);
    return true;
  }
  STILE_INLINE bool load(const Row &row, uint64_t *value) const {
    return read(row, value);
  }
  STILE_INLINE void prefetch(const Row &) const {}
};

// Lists go through Arrow's list arrays; only validity is a scalar property.
template <typename ListArrayT>
class ListColumn {
 public:
  using Array = ListArrayT;
  using Value = std::nullptr_t;
  explicit ListColumn(const ColumnIndex &column) : rows_(column) {}
  STILE_INLINE bool valid(const Row &row) const {
    int64_t local;
    return rows_.array<ListArrayT>(row, &local).IsValid(local);
  }
  // The row's list array and its position there, or null for a NULL list.
  STILE_INLINE const ListArrayT *list(const Row &row, int64_t *local) const {
    const auto &list = rows_.array<ListArrayT>(row, local);
    return list.IsValid(*local) ? &list : nullptr;
  }
  STILE_INLINE void prefetch(const Row &row) const {
    int64_t local;
    const auto &list = rows_.array<ListArrayT>(row, &local);
    STILE_PREFETCH(list.raw_value_offsets() + local);
    if (const uint8_t *validity = list.null_bitmap_data()) {
      STILE_PREFETCH(validity + ((list.offset() + local) >> 3));
    }
  }

 private:
  Rows rows_;
};

// list_value_length: NULL for a NULL list.
template <typename ListArrayT>
class ListLength {
 public:
  using Value = typename ListArrayT::offset_type;
  using Literal = Value;
  explicit ListLength(const ColumnIndex &column) : lists_(column) {}
  STILE_INLINE bool valid(const Row &row) const {
    return lists_.valid(row);
  }
  STILE_INLINE bool read(const Row &row, Value *value) const {
    int64_t local;
    const auto *list = lists_.list(row, &local);
    if (!list) return false;
    *value = list->value_length(local);
    return true;
  }
  STILE_INLINE bool load(const Row &row, Value *value) const {
    return read(row, value);
  }
  STILE_INLINE void prefetch(const Row &row) const {
    lists_.prefetch(row);
  }

 private:
  ListColumn<ListArrayT> lists_;
};

// Arrow casts that preserve every value. Bind inserts them when a literal
// does not fit the column type.
template <typename From, typename To>
constexpr bool kLossless =
    std::is_arithmetic_v<From> && std::is_arithmetic_v<To> &&
    !std::is_same_v<From, bool> && !std::is_same_v<To, bool> &&
    (std::is_floating_point_v<To>
         ? (std::is_floating_point_v<From>
                ? sizeof(From) <= sizeof(To)
                : std::numeric_limits<From>::digits <=
                      std::numeric_limits<To>::digits)
         : (std::is_integral_v<From> &&
            (std::is_signed_v<To>
                 ? (std::is_signed_v<From> ? sizeof(From) <= sizeof(To)
                                           : sizeof(From) < sizeof(To))
                 : !std::is_signed_v<From> && sizeof(From) <= sizeof(To))));

template <typename Inner, typename To>
class Widen {
 public:
  using Value = To;
  using Literal = To;
  explicit Widen(Inner inner) : inner_(std::move(inner)) {}
  STILE_INLINE bool valid(const Row &row) const {
    return inner_.valid(row);
  }
  STILE_INLINE bool read(const Row &row, To *value) const {
    typename Inner::Value inner;
    if (!inner_.read(row, &inner)) return false;
    *value = static_cast<To>(inner);
    return true;
  }
  // Lossless casts are defined for any source value, NULL slots included.
  STILE_INLINE bool load(const Row &row, To *value) const {
    typename Inner::Value inner{};
    const bool valid = inner_.load(row, &inner);
    *value = static_cast<To>(inner);
    return valid;
  }
  STILE_INLINE void prefetch(const Row &row) const {
    inner_.prefetch(row);
  }

 private:
  Inner inner_;
};

template <typename Make>
NodeResult VisitColumn(const ColumnIndex &column, Make &&make) {
  const auto layout = column.layout();
  if (layout == Layout::kRowNumber) {
    if (column.type()->id() != arrow::Type::UINT64) {
      return Unsupported("row number type");
    }
    return make(RowNumberColumn{});
  }
  switch (layout) {
    case Layout::kBitmap:
      return make(SpanColumn<BitAccess>(column));
    case Layout::kBinary:
      return make(SpanColumn<BinaryAccess<int32_t>>(column));
    case Layout::kLargeBinary:
      return make(SpanColumn<BinaryAccess<int64_t>>(column));
    case Layout::kFixedWidth:
      break;
    default:
      return Unsupported("column layout of " + column.type()->ToString());
  }
  switch (column.type()->id()) {
#define STILE_FIXED(ID, CTYPE) \
  case arrow::Type::ID:        \
    return make(SpanColumn<FixedAccess<CTYPE>>(column));
    STILE_FIXED(INT8, int8_t)
    STILE_FIXED(INT16, int16_t)
    STILE_FIXED(INT32, int32_t)
    STILE_FIXED(INT64, int64_t)
    STILE_FIXED(UINT8, uint8_t)
    STILE_FIXED(UINT16, uint16_t)
    STILE_FIXED(UINT32, uint32_t)
    STILE_FIXED(UINT64, uint64_t)
    STILE_FIXED(FLOAT, float)
    STILE_FIXED(DOUBLE, double)
#undef STILE_FIXED
    default:
      // Temporal, decimal, half-float and fixed-size binary keep Arrow.
      return Unsupported("column type " + column.type()->ToString());
  }
}

template <typename Make>
NodeResult VisitList(const ColumnIndex &column, Make &&make) {
  switch (column.layout()) {
    case Layout::kList:
      return make(ListColumn<arrow::ListArray>(column));
    case Layout::kLargeList:
      return make(ListColumn<arrow::LargeListArray>(column));
    default:
      return Unsupported("list operation on " + column.type()->ToString());
  }
}

template <typename Inner, typename Make>
NodeResult WidenTo(arrow::Type::type to, Inner inner, Make &&make) {
  using From = typename Inner::Value;
  switch (to) {
#define STILE_WIDEN(ID, CTYPE)                            \
  case arrow::Type::ID:                                   \
    if constexpr (kLossless<From, CTYPE>) {               \
      return make(Widen<Inner, CTYPE>(std::move(inner))); \
    } else {                                              \
      return Unsupported("cast that may change values");  \
    }
    STILE_WIDEN(INT8, int8_t)
    STILE_WIDEN(INT16, int16_t)
    STILE_WIDEN(INT32, int32_t)
    STILE_WIDEN(INT64, int64_t)
    STILE_WIDEN(UINT8, uint8_t)
    STILE_WIDEN(UINT16, uint16_t)
    STILE_WIDEN(UINT32, uint32_t)
    STILE_WIDEN(UINT64, uint64_t)
    STILE_WIDEN(FLOAT, float)
    STILE_WIDEN(DOUBLE, double)
#undef STILE_WIDEN
    default:
      return Unsupported("cast target");
  }
}

// A scalar's value in the accessor's literal representation.
template <typename T>
void ScalarValue(const arrow::Scalar &scalar, T *out) {
  using ScalarType = typename arrow::TypeTraits<
      typename arrow::CTypeTraits<T>::ArrowType>::ScalarType;
  *out = static_cast<const ScalarType &>(scalar).value;
}
inline void ScalarValue(const arrow::Scalar &scalar, std::string *out) {
  const auto &value =
      static_cast<const arrow::BaseBinaryScalar &>(scalar).value;
  out->assign(value ? reinterpret_cast<const char *>(value->data()) : "",
              value ? static_cast<size_t>(value->size()) : 0);
}

}  // namespace zvec::stile::internal
