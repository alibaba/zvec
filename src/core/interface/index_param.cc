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

#include <zvec/ailego/logger/logger.h>
#include <zvec/core/interface/index_param.h>
#include "core/interface/utils/utils.h"

namespace zvec {
namespace core_interface {
ailego::JsonObject BaseIndexParam::SerializeToJsonObject(
    bool omit_empty_value) const {
  ailego::JsonObject json_obj;

  if (!omit_empty_value || index_type != IndexType::kNone) {
    json_obj.set("index_type",
                 ailego::JsonValue(magic_enum::enum_name(index_type).data()));
  }
  if (!omit_empty_value || metric_type != MetricType::kNone) {
    json_obj.set("metric_type",
                 ailego::JsonValue(magic_enum::enum_name(metric_type).data()));
  }
  if (!omit_empty_value || dimension != 0) {
    json_obj.set("dimension", ailego::JsonValue(dimension));
  }
  if (!omit_empty_value || version != 0) {
    json_obj.set("version", ailego::JsonValue(version));
  }
  if (!omit_empty_value || is_sparse) {
    json_obj.set("is_sparse", ailego::JsonValue(is_sparse));
  }
  if (!omit_empty_value || data_type != DataType::DT_UNDEFINED) {
    json_obj.set("data_type",
                 ailego::JsonValue(magic_enum::enum_name(data_type).data()));
  }
  if (!omit_empty_value || use_id_map) {
    json_obj.set("use_id_map", ailego::JsonValue(use_id_map));
  }
  if (!omit_empty_value || is_huge_page) {
    json_obj.set("is_huge_page", ailego::JsonValue(is_huge_page));
  }

  // if (preprocess_param) {
  //   json.set("preprocess_param", preprocess_param->SerializeToJson());
  // }
  if (!omit_empty_value || quantizer_param.type != QuantizerType::kNone) {
    json_obj.set("quantizer_param",
                 quantizer_param.SerializeToJsonObject(omit_empty_value));
  }
  // if (refiner_param) {
  //   json.set("refiner_param", refiner_param->SerializeToJson());
  // }
  // if (default_query_param) {
  //   json.set("default_query_param",
  //   default_query_param->SerializeToJson());
  // }
  return json_obj;
}


ailego::JsonObject FlatIndexParam::SerializeToJsonObject(
    bool omit_empty_value) const {
  auto json_obj = BaseIndexParam::SerializeToJsonObject(omit_empty_value);
  if (!omit_empty_value || major_order != IndexMeta::MajorOrder::MO_UNDEFINED) {
    json_obj.set("major_order",
                 ailego::JsonValue(magic_enum::enum_name(major_order).data()));
  }
  return json_obj;
}

ailego::JsonObject HNSWIndexParam::SerializeToJsonObject(
    bool omit_empty_value) const {
  auto json_obj = BaseIndexParam::SerializeToJsonObject(omit_empty_value);
  json_obj.set("m", ailego::JsonValue(m));
  json_obj.set("ef_construction", ailego::JsonValue(ef_construction));
  return json_obj;
}

bool BaseIndexParam::DeserializeFromJsonObject(
    const ailego::JsonObject &json_obj) {
  DESERIALIZE_ENUM_FIELD(json_obj, index_type, IndexType);
  DESERIALIZE_ENUM_FIELD(json_obj, metric_type, MetricType);
  DESERIALIZE_ENUM_FIELD(json_obj, data_type, DataType);

  DESERIALIZE_VALUE_FIELD(json_obj, dimension);
  DESERIALIZE_VALUE_FIELD(json_obj, version);
  DESERIALIZE_VALUE_FIELD(json_obj, is_sparse);
  DESERIALIZE_VALUE_FIELD(json_obj, use_id_map);
  DESERIALIZE_VALUE_FIELD(json_obj, is_huge_page);

  ailego::JsonValue tmp_json_value;
  if (json_obj.has("quantizer_param")) {
    if (json_obj.get("quantizer_param", &tmp_json_value);
        tmp_json_value.is_object()) {
      quantizer_param.DeserializeFromJsonObject(tmp_json_value.as_object());
    }
  }

  return true;
}

bool FlatIndexParam::DeserializeFromJsonObject(
    const ailego::JsonObject &json_obj) {
  if (!BaseIndexParam::DeserializeFromJsonObject(json_obj)) {
    return false;
  }

  if (index_type != IndexType::kFlat) {
    LOG_ERROR("index_type is not kFlat");
    return false;
  }

  DESERIALIZE_ENUM_FIELD(json_obj, major_order, IndexMeta::MajorOrder);
  return true;
}

bool HNSWIndexParam::DeserializeFromJsonObject(
    const ailego::JsonObject &json_obj) {
  if (!BaseIndexParam::DeserializeFromJsonObject(json_obj)) {
    return false;
  }

  if (index_type != IndexType::kHNSW) {
    LOG_ERROR("index_type is not kHNSW");
    return false;
  }

  DESERIALIZE_VALUE_FIELD(json_obj, m);
  DESERIALIZE_VALUE_FIELD(json_obj, ef_construction);

  return true;
}

bool HNSWRabitqIndexParam::DeserializeFromJsonObject(
    const ailego::JsonObject &json_obj) {
  if (!BaseIndexParam::DeserializeFromJsonObject(json_obj)) {
    return false;
  }

  if (index_type != IndexType::kHNSWRabitq) {
    LOG_ERROR("index_type is not kHNSWRabitq");
    return false;
  }

  DESERIALIZE_VALUE_FIELD(json_obj, m);
  DESERIALIZE_VALUE_FIELD(json_obj, ef_construction);
  DESERIALIZE_VALUE_FIELD(json_obj, total_bits);
  DESERIALIZE_VALUE_FIELD(json_obj, num_clusters);
  DESERIALIZE_VALUE_FIELD(json_obj, sample_count);

  return true;
}

ailego::JsonObject HNSWRabitqIndexParam::SerializeToJsonObject(
    bool omit_empty_value) const {
  auto json_obj = BaseIndexParam::SerializeToJsonObject(omit_empty_value);
  json_obj.set("m", ailego::JsonValue(m));
  json_obj.set("ef_construction", ailego::JsonValue(ef_construction));
  json_obj.set("total_bits", ailego::JsonValue(total_bits));
  json_obj.set("num_clusters", ailego::JsonValue(num_clusters));
  if (!omit_empty_value || sample_count != 0) {
    json_obj.set("sample_count", ailego::JsonValue(sample_count));
  }
  return json_obj;
}

ailego::JsonObject QuantizerParam::SerializeToJsonObject(
    bool omit_empty_value) const {
  ailego::JsonObject json_obj;
  if (!omit_empty_value || type != QuantizerType::kNone) {
    json_obj.set("type",
                 zvec::ailego::JsonValue(magic_enum::enum_name(type).data()));
  }
  return json_obj;
}

bool QuantizerParam::DeserializeFromJsonObject(
    const ailego::JsonObject &json_obj) {
  DESERIALIZE_ENUM_FIELD(json_obj, type, QuantizerType);
  return true;
}

// bool BaseIndexQueryParam::DeserializeFromJsonObject(
//     const ailego::JsonObject &json_obj) {
//   DESERIALIZE_ENUM_FIELD(json_obj, index_type, IndexType);
//   DESERIALIZE_VALUE_FIELD(json_obj, topk);
//   DESERIALIZE_VALUE_FIELD(json_obj, fetch_vector);
//   DESERIALIZE_VALUE_FIELD(json_obj, radius);
//   DESERIALIZE_VALUE_FIELD(json_obj, is_linear);
//   return true;
// }

// ailego::JsonObject BaseIndexQueryParam::SerializeToJsonObject(
//     bool omit_empty_value) const {
//   ailego::JsonObject json_obj;
//   if (!omit_empty_value || index_type != IndexType::kNone) {
//     json_obj.set("index_type",
//                  ailego::JsonValue(magic_enum::enum_name(index_type).data()));
//   }
//   if (!omit_empty_value || topk != 0) {
//     json_obj.set("topk", ailego::JsonValue(topk));
//   }
//   if (!omit_empty_value || fetch_vector) {
//     json_obj.set("fetch_vector", ailego::JsonValue(fetch_vector));
//   }
//   if (!omit_empty_value || radius != 0.0f) {
//     json_obj.set("radius", ailego::JsonValue(radius));
//   }
//   if (!omit_empty_value || is_linear) {
//     json_obj.set("is_linear", ailego::JsonValue(is_linear));
//   }
//   return json_obj;
// }

// bool FlatQueryParam::DeserializeFromJsonObject(
//     const ailego::JsonObject &json_obj) {
//   if (!BaseIndexQueryParam::DeserializeFromJsonObject(json_obj)) {
//     return false;
//   }
//   if (index_type != IndexType::kFlat) {
//     LOG_ERROR("index_type is not kFlat");
//     return false;
//   }
//   return true;
// }

// ailego::JsonObject FlatQueryParam::SerializeToJsonObject(
//     bool omit_empty_value) const {
//   auto json_obj =
//       BaseIndexQueryParam::SerializeToJsonObject(omit_empty_value);
//   return json_obj;
// }

// bool HNSWQueryParam::DeserializeFromJsonObject(
//     const ailego::JsonObject &json_obj) {
//   if (!BaseIndexQueryParam::DeserializeFromJsonObject(json_obj)) {
//     return false;
//   }
//   if (index_type != IndexType::kHNSW) {
//     LOG_ERROR("index_type is not kHNSW");
//     return false;
//   }
//   DESERIALIZE_VALUE_FIELD(json_obj, ef_search);
//   return true;
// }

// ailego::JsonObject HNSWQueryParam::SerializeToJsonObject(
//     bool omit_empty_value) const {
//   auto json_obj =
//       BaseIndexQueryParam::SerializeToJsonObject(omit_empty_value);
//   if (!omit_empty_value || ef_search != 0) {
//     json_obj.set("ef_search", ailego::JsonValue(ef_search));
//   }
//   return json_obj;
// }


}  // namespace core_interface
}  // namespace zvec