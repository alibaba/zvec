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

#include <iostream>
#include <string>
#include "zvec/core/framework/index_meta.h"

namespace zvec {
namespace core {

class IndexMetaHelper {
 public:
  static std::string to_string(IndexMeta::DataType type) {
    switch (type) {
      case IndexMeta::DataType::DT_FP32:
        return std::string("FP32");
      case IndexMeta::DataType::DT_FP64:
        return std::string("FP64");
      case IndexMeta::DataType::DT_INT16:
        return std::string("INT16");
      case IndexMeta::DataType::DT_INT8:
        return std::string("INT8");
      case IndexMeta::DataType::DT_BINARY32:
        return std::string("Binary");
      case IndexMeta::DataType::DT_BINARY64:
        return std::string("Binary64");
      case IndexMeta::DataType::DT_FP16:
        return std::string("FP16");
      default:
        return std::string("NotSupportedType");
    }
  }

  static std::string to_string(IndexMeta meta) {
    char buffer[1024];
    snprintf(buffer, 1024,
             "IndexMeta: type[%s] method[%s] dimension[%u] element_size[%u]",
             to_string(meta.data_type()).c_str(), meta.metric_name().c_str(),
             meta.dimension(), meta.element_size());
    return std::string(buffer);
  }

  static bool parse_from(const std::string &type, const std::string &method,
                         const std::string &vector_type, IndexMeta &meta) {
    return parse_from(type, method, 0, vector_type, meta);
  }

  static bool parse_from(const std::string &type, const std::string &method,
                         const size_t dimension, const std::string &vector_type,
                         IndexMeta &meta) {
    if (vector_type != "dense" && vector_type != "sparse") {
      std::cerr << "vector type should be dense or sparse!!!" << std::endl;
      return false;
    }

    auto feature_type = IndexMeta::DataType::DT_UNDEFINED;
    if (type == std::string("float")) {
      feature_type = IndexMeta::DataType::DT_FP32;
    } else if (type == std::string("double")) {
      feature_type = IndexMeta::DataType::DT_FP64;
    } else if (type == std::string("int16")) {
      feature_type = IndexMeta::DataType::DT_INT16;
    } else if (type == std::string("int8")) {
      feature_type = IndexMeta::DataType::DT_INT8;
    } else if (type == std::string("binary")) {
      feature_type = IndexMeta::DataType::DT_BINARY32;
    } else if (type == std::string("binary64")) {
      feature_type = IndexMeta::DataType::DT_BINARY64;
    } else {
      std::cerr << "Not supported type: " << type << std::endl;
      return false;
    }

    meta.set_meta(feature_type, dimension);
    ailego::Params params;
    if (method == std::string("L2")) {
      if (feature_type == IndexMeta::DataType::DT_FP32) {
        meta.set_metric("SquaredEuclidean", 0, std::move(params));
      } else if (feature_type == IndexMeta::DataType::DT_INT8) {
        meta.set_metric("SquaredEuclidean", 0, std::move(params));
      } else if (feature_type == IndexMeta::DataType::DT_FP16) {
        meta.set_metric("SquaredEuclidean", 0, std::move(params));
      } else {
        std::cerr << "Not supported type(" << type << ") for L2" << std::endl;
        return false;
      }
    } else if (method == std::string("IP")) {
      if (feature_type == IndexMeta::DataType::DT_FP32) {
        meta.set_metric("InnerProduct", 0, std::move(params));
      } else if (feature_type == IndexMeta::DataType::DT_INT8) {
        meta.set_metric("InnerProduct", 0, std::move(params));
      } else if (feature_type == IndexMeta::DataType::DT_FP16) {
        meta.set_metric("InnerProduct", 0, std::move(params));
      } else {
        std::cerr << "Not supported type(" << type << ") for IP" << std::endl;
        return false;
      }
    } else if (method == std::string("Cosine")) {
      if (feature_type == IndexMeta::DataType::DT_FP32) {
        meta.set_metric("Cosine", 0, std::move(params));
      } else if (feature_type == IndexMeta::DataType::DT_INT8) {
        meta.set_metric("Cosine", 0, std::move(params));
      } else if (feature_type == IndexMeta::DataType::DT_FP16) {
        meta.set_metric("Cosine", 0, std::move(params));
      } else {
        std::cerr << "Not supported type(" << type << ") for Cosine"
                  << std::endl;
        return false;
      }
    } else if (method == std::string("HAMMING")) {
      if (feature_type == IndexMeta::DataType::DT_BINARY32) {
        meta.set_metric("Hamming", 0, std::move(params));
      } else if (feature_type == IndexMeta::DataType::DT_BINARY64) {
        meta.set_metric("Hamming", 0, std::move(params));
      } else {
        std::cerr << "Not supported type(" << type << ") for hamming"
                  << std::endl;
        return false;
      }
    } else {
      std::cerr << "Not supported method: " << method << std::endl;
      return false;
    }

    return true;
  }
};

}  // namespace core
}  // namespace zvec