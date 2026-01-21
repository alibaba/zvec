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

#include <string>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {

// 索引的工厂类
class IndexFactory {
 public:
  static Index::Pointer CreateAndInitIndex(const BaseIndexParam &param);

  static BaseIndexParam::Pointer DeserializeIndexParamFromJson(
      const std::string &json_str);


  static std::string QueryParamSerializeToJson(
      const BaseIndexQueryParam &param);


  template <
      typename QueryParamType,
      std::enable_if_t<std::is_base_of_v<BaseIndexQueryParam, QueryParamType>,
                       bool> = true>
  static std::string QueryParamSerializeToJson(const QueryParamType &param,
                                               bool omit_empty_value = false);

  template <
      typename QueryParamType,
      std::enable_if_t<std::is_base_of_v<BaseIndexQueryParam, QueryParamType>,
                       bool> = true>
  static typename QueryParamType::Pointer QueryParamDeserializeFromJson(
      const std::string &json_str);

  // register() -- Index class should have a `create` interface
};


}  // namespace zvec::core_interface