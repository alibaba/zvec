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

#include "diskann_entity.h"

namespace zvec {
namespace core {

const std::string DiskAnnEntity::kDiskAnnVectorSegmentId = "diskann.vector";
const std::string DiskAnnEntity::kDiskAnnMetaSegmentId = "diskann.meta";
const std::string DiskAnnEntity::kDiskAnnPqMetaSegmentId = "diskann.pq_meta";
const std::string DiskAnnEntity::kDiskAnnPqDataSegmentId = "diskann.pq_data";
const std::string DiskAnnEntity::kDiskAnnDummpySegmentId = "diskann.dummy";
const std::string DiskAnnEntity::kDiskAnnKeyMappingSegmentId =
    "diskann.key_mapping";
const std::string DiskAnnEntity::kDiskAnnEntryPointSegmentId =
    "diskann.entrypoint";
const std::string DiskAnnEntity::kDiskAnnKeySegmentId = "diskann.key";

}  // namespace core
}  // namespace zvec
