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

#include <sys/stat.h>
#include <signal.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ailego/container/bitmap.h>
#include <ailego/hash/crc32c.h>
#include <ailego/io/file.h>
#include <ailego/logger/logger.h>
#include <ailego/parallel/lock.h>
#include <ailego/parallel/thread_pool.h>
#include <ailego/utility/time_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include "framework/index_error.h"
#include "framework/index_factory.h"
#include "framework/index_plugin.h"
#include "framework/index_storage.h"
#include "interface/index.h"
#include "interface/index_factory.h"
#include "interface/index_param.h"
#include "filter_result_cache.h"
#include "meta_segment_common.h"
#include "txt_input_reader.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <yaml-cpp/yaml.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

using namespace std;
using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;


int parse_and_load_index_param(
    YAML::Node &config_node, string &index_dir,
    core_interface::Index::Pointer &index,
    core_interface::BaseIndexQueryParam::Pointer &query_param) {
  // Create Index from config

  if (auto index_config = config_node["IndexCommon"]["IndexConfig"]) {
    std::cout << "IndexConfig: " << index_config.as<string>() << std::endl;
    auto params = core_interface::IndexFactory::DeserializeIndexParamFromJson(
        index_config.as<string>());
    index = core_interface::IndexFactory::CreateAndInitIndex(*params);
    if (!index) {
      cerr << "Failed to create index" << endl;
      return -1;
    }
    core_interface::StorageOptions storage_options;
    storage_options.type = core_interface::StorageOptions::StorageType::kMMAP;
    storage_options.create_new = false;
    storage_options.read_only = true;

    int ret = index->Open(index_dir, storage_options);
    if (0 != ret) {
      cerr << "Index open failed with ret " << ret << endl;
      return false;
    }

    cout << "Load index done!" << endl;
  } else {
    cerr << "IndexCommon.IndexConfig is required" << endl;
    return -1;
  }

  /*
      QueryConfig:
      QueryParam: '{"ef_search":100,"index_type":"kHNSW"}'
      RefinerConfig:
        ScaleFactor: !!int 2
        ReferenceIndex:
          Config:
     '{"use_id_map":false,"data_type":"DT_FP32","dimension":768,"index_type":"kHNSW","metric_type":"kCosine"}'
          Path: ./cohere_train_vector_1m.2.index
  */

  // QUERY PARAM
  if (auto query_config = config_node["QueryConfig"]; query_config) {
    // QueryConfig.QueryParam
    if (auto query_param_config = query_config["QueryParam"];
        query_param_config) {
      std::cout << "QueryParam: " << query_param_config.as<string>()
                << std::endl;
      query_param = core_interface::IndexFactory::QueryParamDeserializeFromJson<
          core_interface::BaseIndexQueryParam>(
          query_param_config.as<std::string>());
      if (!query_param) {
        cerr << "Failed to deserialize query params" << endl;
        return -1;
      }
    }

    // QueryConfig.RefinerConfig
    if (auto refiner_config = query_config["RefinerConfig"]; refiner_config) {
      core_interface::Index::Pointer reference_index = nullptr;
      auto refiner_param = std::make_shared<core_interface::RefinerParam>();

      // QueryConfig.RefinerConfig.ScaleFactor
      if (auto scale_factor_config = refiner_config["ScaleFactor"];
          scale_factor_config) {
        auto scale_factor = scale_factor_config.as<float>();
        refiner_param->scale_factor_ = scale_factor;
      } else {
        cerr << "QueryConfig.RefinerConfig.ScaleFactor config is required"
             << endl;
        return -1;
      }

      // QueryConfig.RefinerConfig.ReferenceIndex
      if (auto reference_index_config = refiner_config["ReferenceIndex"];
          reference_index_config) {
        // QueryConfig.RefinerConfig.ReferenceIndex.Config
        if (auto reference_index_config_config =
                reference_index_config["Config"];
            reference_index_config_config) {
          auto params =
              core_interface::IndexFactory::DeserializeIndexParamFromJson(
                  reference_index_config_config.as<std::string>());

          reference_index =
              core_interface::IndexFactory::CreateAndInitIndex(*params);
        } else {
          cerr << "QueryConfig.RefinerConfig.ReferenceIndex.Config config is "
                  "required"
               << endl;
          return -1;
        }

        // QueryConfig.RefinerConfig.ReferenceIndex.Path
        if (auto reference_index_path_config = reference_index_config["Path"];
            reference_index_path_config) {
          auto reference_index_path =
              reference_index_path_config.as<std::string>();
          core_interface::StorageOptions storage_options;
          storage_options.type =
              core_interface::StorageOptions::StorageType::kMMAP;
          storage_options.create_new = false;
          storage_options.read_only = true;

          int ret =
              reference_index->Open(reference_index_path, storage_options);
          if (0 != ret) {
            cerr << "Index open failed with ret " << ret << endl;
            return -1;
          }

          cout << "Load reference index done!" << endl;
        } else {
          cerr << "QueryConfig.RefinerConfig.ReferenceIndex.Path is required"
               << endl;
          return -1;
        }
        refiner_param->reference_index = reference_index;
      } else {
        cerr << "QueryConfig.RefinerConfig.ReferenceIndex section is required"
             << endl;
        return -1;
      }  // QueryConfig.RefinerConfig.ReferenceIndex

      query_param->refiner_param = refiner_param;
    }  // QueryConfig.RefinerConfig
  }  // QUERY PARAM
  return 0;
}

//--------------------------------------------------
// Helper functions for loading tag lists
//--------------------------------------------------
int load_taglists(const std::string &path,
                  std::vector<std::vector<uint64_t>> &id_to_tags_list,
                  std::vector<uint64_t> &tag_key_list) {
  // Load tag lists
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");

  int ret = storage->open(path, false);
  if (ret != 0) {
    LOG_ERROR("Failed to load index with storage %s", storage->name().c_str());
    return ret;
  }

  auto segment_taglist_header = storage->get(TAGLIST_HEADER_SEGMENT_NAME);
  if (!segment_taglist_header) {
    LOG_INFO("No Tag Lists Found!");
    return 0;
  }

  TagListHeader taglist_header;
  void *data_ptr;
  if (segment_taglist_header->read(0, (const void **)(&data_ptr),
                                   sizeof(TagListHeader)) !=
      sizeof(TagListHeader)) {
    LOG_ERROR("Read tag list meta failed");
    return IndexError_ReadData;
  }

  memcpy(&taglist_header, data_ptr, sizeof(TagListHeader));

  auto segment_taglist_key = storage->get(TAGLIST_KEY_SEGMENT_NAME);
  if (!segment_taglist_key) {
    LOG_ERROR("IndexStorage get segment %s failed",
              TAGLIST_KEY_SEGMENT_NAME.c_str());
    return IndexError_InvalidValue;
  }

  size_t offset = 0;
  for (size_t i = 0; i < taglist_header.num_vecs; ++i) {
    if (segment_taglist_key->read(offset, (const void **)(&data_ptr),
                                  sizeof(uint64_t)) != sizeof(uint64_t)) {
      LOG_ERROR("Read tag list key failed");
      return IndexError_ReadData;
    }

    uint64_t key = *reinterpret_cast<const uint64_t *>(data_ptr);
    tag_key_list.push_back(key);

    offset += sizeof(uint64_t);
  }

  auto segment_taglist_data = storage->get(TAGLIST_DATA_SEGMENT_NAME);
  if (!segment_taglist_data) {
    LOG_ERROR("IndexStorage get segment %s failed",
              TAGLIST_DATA_SEGMENT_NAME.c_str());
    return IndexError_InvalidValue;
  }

  std::vector<uint64_t> taglist_offsets;
  offset = 0;
  for (size_t i = 0; i < taglist_header.num_vecs; ++i) {
    if (segment_taglist_data->read(offset, (const void **)(&data_ptr),
                                   sizeof(uint64_t)) != sizeof(uint64_t)) {
      LOG_ERROR("Read tag list data failed");
      return IndexError_ReadData;
    }

    uint64_t tag_offset = *reinterpret_cast<const uint64_t *>(data_ptr);
    taglist_offsets.push_back(tag_offset);

    offset += sizeof(uint64_t);
  }

  offset = taglist_header.num_vecs * sizeof(uint64_t);
  for (size_t i = 0; i < taglist_header.num_vecs; ++i) {
    if (segment_taglist_data->read(offset, (const void **)(&data_ptr),
                                   sizeof(uint64_t)) != sizeof(uint64_t)) {
      LOG_ERROR("Read tag list data failed");
      return IndexError_ReadData;
    }
    offset += sizeof(uint64_t);

    uint64_t tag_count = *reinterpret_cast<const uint64_t *>(data_ptr);

    if (segment_taglist_data->read(offset, (const void **)(&data_ptr),
                                   tag_count * sizeof(uint64_t)) !=
        tag_count * sizeof(uint64_t)) {
      LOG_ERROR("Read tag list data failed");
      return IndexError_ReadData;
    }
    offset += tag_count * sizeof(uint64_t);

    std::vector<uint64_t> tag_list;
    tag_list.reserve(tag_count);
    for (size_t j = 0; j < tag_count; ++j) {
      tag_list.push_back(reinterpret_cast<const uint64_t *>(data_ptr)[j]);
    }
    id_to_tags_list.push_back(std::move(tag_list));
  }

  return 0;
}
