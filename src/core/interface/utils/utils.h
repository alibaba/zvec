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

#include <magic_enum/magic_enum.hpp>
#include <zvec/ailego/encoding/json.h>
#include <zvec/ailego/logger/logger.h>

namespace zvec {
namespace core_interface {

template <typename EnumType>
constexpr bool extract_enum_from_json(const ailego::JsonObject &json_obj,
                                      const char *key, EnumType &enum_value,
                                      ailego::JsonValue &tmp_json_value) {
  if (json_obj.has(key)) {
    if (json_obj.get(key, &tmp_json_value); tmp_json_value.is_string()) {
      auto optional_enum_value =
          magic_enum::enum_cast<EnumType>(tmp_json_value.as_stl_string());
      if (optional_enum_value.has_value()) {
        enum_value = optional_enum_value.value();
      } else {
        LOG_ERROR("Invalid enum value for key: %s, value: %s", key,
                  tmp_json_value.as_c_string());
        return false;
      }
    } else {
      LOG_ERROR("Invalid json field type for key: %s", key);
      return false;
    }
  }
  return true;
}

template <typename T>
constexpr bool extract_value_from_json(const ailego::JsonObject &json_obj,
                                       const char *key, T &value,
                                       ailego::JsonValue &tmp_json_value) {
  if (json_obj.has(key)) {
    json_obj.get(key, &tmp_json_value);
    if constexpr (std::is_same_v<T, bool>) {
      if (tmp_json_value.is_boolean()) {
        value = tmp_json_value.as_bool();
      } else {
        LOG_ERROR("Invalid json field type for key: %s; expected: boolean",
                  key);
        return false;
      }
    } else if constexpr (std::is_floating_point_v<T>) {
      if (tmp_json_value.is_float() || tmp_json_value.is_integer()) {
        value = static_cast<T>(tmp_json_value.as_float());
      } else {
        LOG_ERROR("Invalid json field type for key: %s; expected: float", key);
        return false;
      }
    } else if constexpr (std::is_integral_v<T>) {
      if (tmp_json_value.is_integer()) {
        value = static_cast<T>(tmp_json_value.as_integer());
      } else {
        LOG_ERROR("Invalid json field type for key: %s; expected: integer",
                  key);
        return false;
      }
    } else {
      abort();
    }
  }
  return true;
}

#define DESERIALIZE_ENUM_FIELD(json_obj, field_name, EnumType)               \
  {                                                                          \
    ailego::JsonValue tmp_json_value;                                        \
    if (!extract_enum_from_json<EnumType>(json_obj, #field_name, field_name, \
                                          tmp_json_value)) {                 \
      LOG_ERROR("Error when deserialize json - field:%s", #field_name);      \
      return false;                                                          \
    }                                                                        \
  }


#define DESERIALIZE_VALUE_FIELD(json_obj, field_name)                   \
  {                                                                     \
    ailego::JsonValue tmp_json_value;                                   \
    if (!extract_value_from_json(json_obj, #field_name, field_name,     \
                                 tmp_json_value)) {                     \
      LOG_ERROR("Error when deserialize json - field:%s", #field_name); \
      return false;                                                     \
    }                                                                   \
  }
}  // namespace core_interface
}  // namespace zvec