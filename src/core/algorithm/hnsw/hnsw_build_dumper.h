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

#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_mapping.h>

namespace zvec::core {

// Write mutable mmap chunks without mapping their vector payloads into the
// process. Only the segment table is mapped while packing the completed graph.
class HnswBuildDumper final : public IndexDumper {
 public:
  // Includes IndexVersion, IndexMeta and every packed HNSW chunk. Reserve the
  // table before writing payloads so append() never inserts metadata over them.
  explicit HnswBuildDumper(size_t segment_count)
      : segment_count_(segment_count) {}
  ~HnswBuildDumper() override;
  int init(const ailego::Params &) override;
  int cleanup() override;
  int create(const std::string &path) override;
  int close() override;
  size_t write(const void *data, size_t bytes) override;
  int append(const std::string &id, size_t data_size, size_t padding_size,
             uint32_t crc) override;
  uint32_t magic() const override;

 private:
  IndexMapping mapping_;
  ailego::File file_;
  const size_t segment_count_;
  size_t metadata_remaining_{0};
  size_t pending_{0};
  bool opened_{false};
};

}  // namespace zvec::core
