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

#include <memory>
#include <string>
#include <zvec/core/framework/index_builder.h>

namespace zvec::core {
class DiskAnnBuilder;

// Owns anonymous temporary files until cleanup/destruction. No full-dataset
// vector, adjacency, key mapping, or PQ-code array is retained in memory.
class DiskAnnPartitionBuild {
 public:
  explicit DiskAnnPartitionBuild(DiskAnnBuilder &owner);
  ~DiskAnnPartitionBuild();
  static int check_temp_directory(const std::string &directory);
  static int train_quantizer(DiskAnnBuilder &owner,
                             IndexThreads::Pointer threads);
  int build(IndexThreads::Pointer threads, IndexHolder::Pointer holder);
  int dump(const IndexDumper::Pointer &dumper);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace zvec::core
