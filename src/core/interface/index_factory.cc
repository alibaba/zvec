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

#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include "core/interface/utils/utils.h"

namespace zvec::core_interface {


Index::Pointer IndexFactory::CreateAndInitIndex(const BaseIndexParam &param) {
  Index::Pointer ptr = nullptr;
  // if (param.index_type == IndexType::kIVF) {
  //   const IVFIndexParam *_param = dynamic_cast<const IVFIndexParam
  //   *>(&param); ptr = std::make_shared<IVFIndex>(param);

  //   if (_param->l1Index) {
  //     // TODO: create l1 index
  //   }
  //   if (_param->l2Index) {
  //     // TODO: create l2 index
  //   }
  // }
  // if (param.index_type == IndexType::kHNSW) {
  //   ptr = std::make_shared<HNSWIndex>(param);
  // }
  if (param.index_type == IndexType::kFlat) {
    // ptr = std::make_shared<FlatIndex>(param);
    ptr = std::make_shared<FlatIndex>();
  } else if (param.index_type == IndexType::kHNSW) {
    ptr = std::make_shared<HNSWIndex>();
  } else if (param.index_type == IndexType::kIVF) {
    ptr = std::make_shared<IVFIndex>();
  } else if (param.index_type == IndexType::kHNSWRabitq) {
    ptr = std::make_shared<HNSWRabitqIndex>();
  } else {
    LOG_ERROR("Unsupported index type: ");
    return nullptr;
  }

  if (!ptr) {
    LOG_ERROR("Failed to create index");
    return nullptr;
  }
  if (0 != ptr->Init(param)) {
    LOG_ERROR("Failed to init index");
    return nullptr;
  }
  return ptr;
}

BaseIndexParam::Pointer IndexFactory::DeserializeIndexParamFromJson(
    const std::string &json_str) {
  ailego::JsonValue json_value;
  if (!json_value.parse(json_str)) {
    LOG_ERROR("Failed to parse json string: %s", json_str.c_str());
    return nullptr;
  }
  ailego::JsonObject json_obj = json_value.as_object();
  ailego::JsonValue tmp_json_value;

  IndexType index_type;

  if (!extract_enum_from_json<IndexType>(json_obj, "index_type", index_type,
                                         tmp_json_value)) {
    LOG_ERROR("Failed to deserialize index type");
    return nullptr;
  }

  switch (index_type) {
    case IndexType::kFlat: {
      FlatIndexParam::Pointer param = std::make_shared<FlatIndexParam>();
      if (!param->DeserializeFromJson(json_str)) {
        LOG_ERROR("Failed to deserialize flat index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kHNSW: {
      HNSWIndexParam::Pointer param = std::make_shared<HNSWIndexParam>();
      if (!param->DeserializeFromJson(json_str)) {
        LOG_ERROR("Failed to deserialize hnsw index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kIVF: {
      IVFIndexParam::Pointer param = std::make_shared<IVFIndexParam>();
      if (!param->DeserializeFromJson(json_str)) {
        LOG_ERROR("Failed to deserialize hnsw index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kHNSWRabitq: {
      HNSWRabitqIndexParam::Pointer param =
          std::make_shared<HNSWRabitqIndexParam>();
      if (!param->DeserializeFromJson(json_str)) {
        LOG_ERROR("Failed to deserialize hnsqrabitq index param");
        return nullptr;
      }
      return param;
    }
    default:
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
  }
}

template <typename QueryParamType,
          std::enable_if_t<
              std::is_base_of_v<BaseIndexQueryParam, QueryParamType>, bool> >
std::string IndexFactory::QueryParamSerializeToJson(const QueryParamType &param,
                                                    bool omit_empty_value) {
  ailego::JsonObject json_obj;

  // BaseIndexQueryParam
  // omit filter & bf_pks
  if (!omit_empty_value || param.topk != 0) {
    json_obj.set("topk", ailego::JsonValue(param.topk));
  }
  if (!omit_empty_value || param.fetch_vector) {
    json_obj.set("fetch_vector", ailego::JsonValue(param.fetch_vector));
  }
  if (!omit_empty_value || param.radius != 0.0f) {
    json_obj.set("radius", ailego::JsonValue(param.radius));
  }
  if (!omit_empty_value || param.is_linear) {
    json_obj.set("is_linear", ailego::JsonValue(param.is_linear));
  }

  IndexType index_type{IndexType::kNone};
  if constexpr (std::is_same_v<QueryParamType, FlatQueryParam>) {
    // index_type
    index_type = IndexType::kFlat;
  } else if constexpr (std::is_same_v<QueryParamType, HNSWQueryParam>) {
    if (!omit_empty_value || param.ef_search != 0) {
      json_obj.set("ef_search", ailego::JsonValue(param.ef_search));
    }
    index_type = IndexType::kHNSW;
  } else if constexpr (std::is_same_v<QueryParamType, IVFQueryParam>) {
    if (!omit_empty_value || param.nprobe != 0) {
      json_obj.set("nprobe", ailego::JsonValue(param.nprobe));
    }
    index_type = IndexType::kIVF;
    // json_obj.set("l1QueryParam",
    // ailego::JsonValue(QueryParamSerializeToJson(param.l1QueryParam)));
    // json_obj.set("l2QueryParam",
    // ailego::JsonValue(QueryParamSerializeToJson(param.l2QueryParam)));
  } else if constexpr (std::is_same_v<QueryParamType, HNSWRabitqQueryParam>) {
    if (!omit_empty_value || param.ef_search != 0) {
      json_obj.set("ef_search", ailego::JsonValue(param.ef_search));
    }
    index_type = IndexType::kHNSWRabitq;
  }

  json_obj.set("index_type",
               ailego::JsonValue(magic_enum::enum_name(index_type).data()));

  return ailego::JsonValue(json_obj).as_json_string().as_stl_string();
}

template std::string
IndexFactory::QueryParamSerializeToJson<BaseIndexQueryParam>(
    const BaseIndexQueryParam &param, bool omit_empty_value);
template std::string IndexFactory::QueryParamSerializeToJson<FlatQueryParam>(
    const FlatQueryParam &param, bool omit_empty_value);
template std::string IndexFactory::QueryParamSerializeToJson<HNSWQueryParam>(
    const HNSWQueryParam &param, bool omit_empty_value);
template std::string IndexFactory::QueryParamSerializeToJson<IVFQueryParam>(
    const IVFQueryParam &param, bool omit_empty_value);

template <typename QueryParamType,
          std::enable_if_t<
              std::is_base_of_v<BaseIndexQueryParam, QueryParamType>, bool> >
typename QueryParamType::Pointer IndexFactory::QueryParamDeserializeFromJson(
    const std::string &json_str) {
  ailego::JsonValue tmp_json_value;
  if (!tmp_json_value.parse(json_str)) {
    LOG_ERROR("Failed to parse json string: %s", json_str.c_str());
    return nullptr;
  }
  ailego::JsonObject json_obj = tmp_json_value.as_object();

  auto parse_common_fields = [&](auto &param) -> bool {
    if (!extract_value_from_json(json_obj, "topk", param->topk,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize topk");
      return false;
    }

    if (!extract_value_from_json(json_obj, "fetch_vector", param->fetch_vector,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize fetch_vector");
      return false;
    }

    if (!extract_value_from_json(json_obj, "radius", param->radius,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize radius");
      return false;
    }

    if (!extract_value_from_json(json_obj, "is_linear", param->is_linear,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize is_linear");
      return false;
    }
    return true;
  };

  IndexType index_type;

  if (!extract_enum_from_json<IndexType>(json_obj, "index_type", index_type,
                                         tmp_json_value)) {
    LOG_ERROR("Failed to deserialize index type");
    return nullptr;
  }

  if constexpr (std::is_same_v<QueryParamType, BaseIndexQueryParam>) {
    if (index_type == IndexType::kFlat) {
      auto param = std::make_shared<FlatQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kHNSW) {
      auto param = std::make_shared<HNSWQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kIVF) {
      auto param = std::make_shared<IVFQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kHNSWRabitq) {
      auto param = std::make_shared<HNSWRabitqQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      return param;
    } else {
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
    }
  } else {
    auto param = std::make_shared<QueryParamType>();
    if (!parse_common_fields(param)) {
      return nullptr;
    }
    if constexpr (std::is_same_v<QueryParamType, FlatQueryParam>) {
    } else if constexpr (std::is_same_v<QueryParamType, HNSWQueryParam>) {
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
    } else if constexpr (std::is_same_v<QueryParamType, IVFQueryParam>) {
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
    } else if constexpr (std::is_same_v<QueryParamType, HNSWRabitqQueryParam>) {
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
    } else {
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
    }
    return param;
  }
}

template BaseIndexQueryParam::Pointer
IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(
    const std::string &json_str);
template FlatQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    FlatQueryParam>(const std::string &json_str);
template HNSWQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    HNSWQueryParam>(const std::string &json_str);
template IVFQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    IVFQueryParam>(const std::string &json_str);

}  // namespace zvec::core_interface
