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

#include "predicate_helpers.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gandiva/configuration.h>
#include <gandiva/function_registry.h>
#include <gandiva/native_function.h>
#include <re2/re2.h>

namespace gandiva {
// Arrow 21 exposes this factory as a friend in FunctionRegistry, but omits the
// namespace-scope declaration needed for ordinary lookup. Re-declare that same
// function so the scalar registry can be isolated from Gandiva's global one.
arrow::Result<std::shared_ptr<FunctionRegistry>> MakeDefaultFunctionRegistry();
}  // namespace gandiva

namespace zvec::sqlengine::scalar_jit {
namespace {

struct LikeContext {
  MemoryTicket memory;
  enum class Kind { kSubstring, kPrefix, kSuffix, kRegex };
  Kind kind = Kind::kRegex;
  std::string literal;
  std::unique_ptr<RE2> regex;

  bool Match(std::string_view value) const {
    switch (kind) {
      case Kind::kSubstring:
        return value.find(literal) != std::string_view::npos;
      case Kind::kPrefix:
        return value.size() >= literal.size() &&
               value.substr(0, literal.size()) == literal;
      case Kind::kSuffix:
        return value.size() >= literal.size() &&
               value.substr(value.size() - literal.size()) == literal;
      case Kind::kRegex:
        return RE2::PartialMatch(re2::StringPiece(value.data(), value.size()),
                                 *regex);
    }
    return false;
  }
};

// Keep the conversion and literal fast paths equivalent to Arrow 21's
// compute/kernels/scalar_string_ascii.cc::MatchLike. In particular, Arrow's
// literal paths do not unescape backslashes, while its regex path ignores a
// trailing escape. Gandiva's built-in LIKE has different escape semantics.
std::string LikeRegex(const std::string &pattern) {
  std::string result = "(?s:^";
  result.reserve(pattern.size() + 7);
  bool escaped = false;
  for (char value : pattern) {
    if (!escaped && value == '%') {
      result.append(".*");
    } else if (!escaped && value == '_') {
      result.push_back('.');
    } else if (!escaped && value == '\\') {
      escaped = true;
    } else {
      if (std::string_view(".?+*^$\\[{()|").find(value) !=
          std::string_view::npos) {
        result.push_back('\\');
      }
      result.push_back(value);
      escaped = false;
    }
  }
  result.append("$)");
  return result;
}

bool Like(const char *value, int32_t length, int64_t context) {
  return reinterpret_cast<const LikeContext *>(context)->Match(
      std::string_view(value ? value : "", length));
}

struct InContext {
  MemoryTicket memory;
  bool has_null = false;
  std::unordered_set<uint64_t> numbers;
  std::vector<std::string> storage;
  std::unordered_set<std::string_view> strings;
};

template <typename T>
uint64_t Bits(T value) {
  // Arrow's scalar set lookup hashes the unsigned representation, including
  // floating point NaN payloads and the distinct +0/-0 representations.
  uint64_t result = 0;
  std::memcpy(&result, &value, sizeof(value));
  return result;
}

template <typename T>
bool InNumber(T value, int64_t context) {
  const auto &set = *reinterpret_cast<const InContext *>(context);
  return set.numbers.find(Bits(value)) != set.numbers.end();
}

bool InString(const char *value, int32_t length, int64_t context) {
  const auto &set = *reinterpret_cast<const InContext *>(context);
  return set.strings.find(std::string_view(value ? value : "", length)) !=
         set.strings.end();
}

template <typename ArrowType>
void AddNumbers(const arrow::Array &array, InContext *context) {
  const auto &values =
      static_cast<const arrow::NumericArray<ArrowType> &>(array);
  for (int64_t row = 0; row < values.length(); ++row) {
    if (values.IsNull(row))
      context->has_null = true;
    else
      context->numbers.insert(Bits(values.Value(row)));
  }
}

arrow::Status AddValues(const arrow::Array &array, InContext *context) {
  ARROW_RETURN_NOT_OK(array.ValidateFull());
  switch (array.type_id()) {
#define ZVEC_IN_TYPE(ID, TYPE)               \
  case arrow::Type::ID:                      \
    AddNumbers<arrow::TYPE>(array, context); \
    break
    ZVEC_IN_TYPE(INT8, Int8Type);
    ZVEC_IN_TYPE(UINT8, UInt8Type);
    ZVEC_IN_TYPE(INT16, Int16Type);
    ZVEC_IN_TYPE(UINT16, UInt16Type);
    ZVEC_IN_TYPE(INT32, Int32Type);
    ZVEC_IN_TYPE(UINT32, UInt32Type);
    ZVEC_IN_TYPE(INT64, Int64Type);
    ZVEC_IN_TYPE(UINT64, UInt64Type);
    ZVEC_IN_TYPE(FLOAT, FloatType);
    ZVEC_IN_TYPE(DOUBLE, DoubleType);
#undef ZVEC_IN_TYPE
    case arrow::Type::STRING: {
      const auto &values = static_cast<const arrow::StringArray &>(array);
      for (int64_t row = 0; row < values.length(); ++row) {
        if (values.IsNull(row))
          context->has_null = true;
        else
          context->storage.emplace_back(values.GetView(row));
      }
      break;
    }
    default:
      return arrow::Status::NotImplemented("Scalar IN type: ",
                                           array.type()->ToString());
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<gandiva::FunctionRegistry>> MakeRegistry() {
  ARROW_ASSIGN_OR_RAISE(auto registry, gandiva::MakeDefaultFunctionRegistry());
  ARROW_RETURN_NOT_OK(registry->Register(
      gandiva::NativeFunction("zvec_scalar_like", {},
                              {arrow::utf8(), arrow::int64()}, arrow::boolean(),
                              gandiva::kResultNullIfNull,
                              "zvec_scalar_like_utf8"),
      reinterpret_cast<void *>(&Like)));
  const auto add_in = [&](const std::shared_ptr<arrow::DataType> &type,
                          const std::string &name, void *function) {
    return registry->Register(
        gandiva::NativeFunction("zvec_scalar_in", {}, {type, arrow::int64()},
                                arrow::boolean(), gandiva::kResultNullIfNull,
                                "zvec_scalar_in_" + name),
        function);
  };
#define ZVEC_REGISTER_IN(TYPE, CTYPE)              \
  ARROW_RETURN_NOT_OK(add_in(arrow::TYPE(), #TYPE, \
                             reinterpret_cast<void *>(&InNumber<CTYPE>)))
  ZVEC_REGISTER_IN(int8, int8_t);
  ZVEC_REGISTER_IN(uint8, uint8_t);
  ZVEC_REGISTER_IN(int16, int16_t);
  ZVEC_REGISTER_IN(uint16, uint16_t);
  ZVEC_REGISTER_IN(int32, int32_t);
  ZVEC_REGISTER_IN(uint32, uint32_t);
  ZVEC_REGISTER_IN(int64, int64_t);
  ZVEC_REGISTER_IN(uint64, uint64_t);
  ZVEC_REGISTER_IN(float32, float);
  ZVEC_REGISTER_IN(float64, double);
#undef ZVEC_REGISTER_IN
  ARROW_RETURN_NOT_OK(
      add_in(arrow::utf8(), "utf8", reinterpret_cast<void *>(&InString)));
  return registry;
}

}  // namespace

arrow::Result<std::shared_ptr<gandiva::Configuration>>
ScalarPredicateConfiguration() {
  if (!InOriginalProcess()) return Unavailable("forked_process");
  struct RegistryState {
    std::mutex mutex;
    std::shared_ptr<gandiva::FunctionRegistry> functions;
  };
  static auto *state = new RegistryState;
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->functions) {
    ARROW_ASSIGN_OR_RAISE(state->functions, MakeRegistry());
  }
  return std::make_shared<gandiva::Configuration>(true, state->functions);
}

arrow::Result<size_t> InInputBytes(
    const arrow::compute::SetLookupOptions &options,
    const ExpressionLimits &limits) {
  if (!options.value_set.is_arraylike() || !options.value_set.type())
    return Unavailable("unsupported_expression");
  const auto length = options.value_set.length();
  if (length < 0 || static_cast<uint64_t>(length) > limits.in_values ||
      static_cast<uint64_t>(length) > limits.input_bytes / sizeof(uint64_t))
    return Unavailable("expression_limit");
  size_t bytes = static_cast<size_t>(length) * sizeof(uint64_t);
  auto check =
      [&](const std::shared_ptr<arrow::ArrayData> &data) -> arrow::Status {
    if (!data) return Unavailable("unsupported_expression");
    if (data->type->id() == arrow::Type::STRING) {
      for (const auto &buffer : data->buffers) {
        if (!buffer) continue;
        if (buffer->size() < 0 ||
            static_cast<uint64_t>(buffer->size()) > limits.input_bytes - bytes)
          return Unavailable("expression_limit");
        bytes += buffer->size();
      }
    }
    return arrow::Status::OK();
  };
  if (options.value_set.is_array()) {
    ARROW_RETURN_NOT_OK(check(options.value_set.array()));
  } else {
    const auto &chunks = options.value_set.chunked_array()->chunks();
    if (chunks.size() > limits.nodes) return Unavailable("expression_limit");
    for (const auto &chunk : chunks) ARROW_RETURN_NOT_OK(check(chunk->data()));
  }
  return bytes;
}

arrow::Result<std::shared_ptr<const void>> MakeLikeContext(
    const arrow::compute::MatchSubstringOptions &options) {
  if (!InOriginalProcess()) return Unavailable("forked_process");
  const auto &limits = GetRuntimeConfig().limits;
  if (limits.input_bytes < 7 ||
      options.pattern.size() > (limits.input_bytes - 7) / 2)
    return Unavailable("expression_limit");
  auto context = std::make_shared<LikeContext>();
  RE2::Options re_options(RE2::Quiet);
  re_options.set_encoding(RE2::Options::EncodingUTF8);
  re_options.set_case_sensitive(!options.ignore_case);
  if (!options.ignore_case) {
    static const RE2 substring(R"(%+([^%_]*[^\\%_])?%+)", RE2::Quiet);
    static const RE2 prefix(R"(([^%_]*[^\\%_])?%+)", RE2::Quiet);
    static const RE2 suffix(R"(%+([^%_]*))", RE2::Quiet);
    if (RE2::FullMatch(options.pattern, substring, &context->literal))
      context->kind = LikeContext::Kind::kSubstring;
    else if (RE2::FullMatch(options.pattern, prefix, &context->literal))
      context->kind = LikeContext::Kind::kPrefix;
    else if (RE2::FullMatch(options.pattern, suffix, &context->literal))
      context->kind = LikeContext::Kind::kSuffix;
  }
  if (context->kind == LikeContext::Kind::kRegex) {
    context->regex =
        std::make_unique<RE2>(LikeRegex(options.pattern), re_options);
    if (!context->regex->ok())
      return arrow::Status::Invalid("Invalid scalar LIKE pattern: ",
                                    context->regex->error());
  }
  MemoryUsage bytes;
  bytes.owned_bytes = sizeof(LikeContext) + context->literal.capacity();
  if (context->regex)
    bytes.owned_bytes += sizeof(RE2) + context->regex->pattern().capacity();
  context->memory.Set("helper", bytes);
  return std::static_pointer_cast<const void>(context);
}

arrow::Result<std::shared_ptr<const void>> MakeInContext(
    const arrow::compute::SetLookupOptions &options, bool *has_null) {
  if (!InOriginalProcess()) return Unavailable("forked_process");
  ARROW_RETURN_NOT_OK(
      InInputBytes(options, GetRuntimeConfig().limits).status());
  if (options.GetNullMatchingBehavior() !=
      arrow::compute::SetLookupOptions::INCONCLUSIVE)
    return arrow::Status::NotImplemented(
        "Scalar IN requires INCONCLUSIVE null handling");
  if (!options.value_set.is_arraylike())
    return arrow::Status::Invalid(
        "Scalar IN requires an array or chunked array");
  auto context = std::make_shared<InContext>();
  const auto size = static_cast<size_t>(options.value_set.length());
  if (options.value_set.type()->id() == arrow::Type::STRING)
    context->storage.reserve(size);
  else
    context->numbers.reserve(size);
  if (options.value_set.is_array()) {
    ARROW_RETURN_NOT_OK(
        AddValues(*options.value_set.make_array(), context.get()));
  } else {
    for (const auto &chunk : options.value_set.chunked_array()->chunks()) {
      ARROW_RETURN_NOT_OK(AddValues(*chunk, context.get()));
    }
  }
  // Create views only after backing strings have stopped moving, including
  // strings small enough to use std::string's inline storage.
  context->strings.reserve(context->storage.size());
  for (const auto &value : context->storage) context->strings.insert(value);
  if (has_null) *has_null = context->has_null;
  MemoryUsage bytes;
  bytes.owned_bytes = sizeof(InContext) +
                      context->storage.capacity() * sizeof(std::string) +
                      context->numbers.size() * sizeof(uint64_t) +
                      context->strings.size() * sizeof(std::string_view);
  for (const auto &value : context->storage)
    bytes.owned_bytes += value.capacity();
  context->memory.Set("helper", bytes);
  return std::static_pointer_cast<const void>(context);
}

}  // namespace zvec::sqlengine::scalar_jit
