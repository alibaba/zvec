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
#include <mutex>
#include <string>
#include <unordered_map>
#include <zvec/db/collection.h>

namespace duckdb {

class CollectionRegistry {
 public:
  static CollectionRegistry &Instance();

  zvec::Collection::Ptr GetOrOpen(const std::string &path);
  zvec::Collection::Ptr Create(const std::string &path,
                               const zvec::CollectionSchema &schema);
  void Close(const std::string &path);
  void CloseAll();

 private:
  CollectionRegistry() = default;
  ~CollectionRegistry() = default;
  CollectionRegistry(const CollectionRegistry &) = delete;
  CollectionRegistry &operator=(const CollectionRegistry &) = delete;

  std::mutex mutex_;
  std::unordered_map<std::string, zvec::Collection::Ptr> collections_;
};

}  // namespace duckdb
