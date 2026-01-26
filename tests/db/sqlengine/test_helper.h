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
// limitations under the License

#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "db/sqlengine/sqlengine.h"
#include "zvec/db/index_params.h"
#include "zvec/db/schema.h"
#include "zvec/db/type.h"

namespace zvec::sqlengine {

using CreateDocFun = Doc (*)(const uint64_t doc_id);

inline Status InsertDoc(const Segment::Ptr &segment,
                        const uint64_t start_doc_id, const uint64_t end_doc_id,
                        CreateDocFun create_doc) {
  srand(time(NULL));
  long long create_total = 0;
  long long insert_total = 0;
  for (auto doc_id = start_doc_id; doc_id < end_doc_id; doc_id++) {
    if (segment) {
      auto start = std::chrono::system_clock::now();
      Doc new_doc = create_doc(doc_id);
      auto end = std::chrono::system_clock::now();
      auto create_cost =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      create_total += create_cost;

      start = std::chrono::system_clock::now();
      auto status = segment->Insert(new_doc);
      if (!status.ok()) {
        return status;
      }
      end = std::chrono::system_clock::now();
      auto insert_cost =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      insert_total += insert_cost;
    }
  }
  std::cout << "pure create cost " << create_total << "us" << std::endl;
  std::cout << "pure insert cost " << insert_total << "us" << std::endl;
  return Status::OK();
}

inline Segment::Ptr create_segment(const std::string &seg_path,
                                   const CollectionSchema &schema) {
  auto segment_meta = std::make_shared<SegmentMeta>();
  segment_meta->set_id(0);

  auto id_map = IDMap::CreateAndOpen("test_collection", seg_path + "/id_map",
                                     true, false);
  auto delete_store = std::make_shared<DeleteStore>("test_collection");

  Version v1;
  v1.set_schema(schema);
  std::string v_path = seg_path + "/test_manifest";
  FileHelper::CreateDirectory(v_path);
  auto vm = VersionManager::Create(v_path, v1);
  if (!vm.has_value()) {
    LOG_ERROR("create version manager failed: %s", vm.error().c_str());
    return nullptr;
  }

  BlockMeta mem_block;
  mem_block.id_ = 0;
  mem_block.type_ = BlockType::SCALAR;
  mem_block.min_doc_id_ = 0;
  mem_block.max_doc_id_ = 0;
  mem_block.doc_count_ = 0;
  segment_meta->set_writing_forward_block(mem_block);

  SegmentOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  auto result = Segment::CreateAndOpen(seg_path, schema, 0, 0, id_map,
                                       delete_store, vm.value(), options);

  if (!result) {
    LOG_ERROR("create segment failed: %s", result.error().c_str());
    return nullptr;
  }
  auto segment = result.value();
  return segment;
}

}  // namespace zvec::sqlengine