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

#include "collection_registry.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {

CollectionRegistry &CollectionRegistry::Instance() {
  static CollectionRegistry instance;
  return instance;
}

zvec::Collection::Ptr CollectionRegistry::GetOrOpen(const std::string &path) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = collections_.find(path);
  if (it != collections_.end()) {
    return it->second;
  }
  zvec::CollectionOptions options{false, true};
  auto result = zvec::Collection::Open(path, options);
  if (!result.has_value()) {
    throw IOException("zvec: failed to open collection at '%s': %s", path,
                      result.error().message());
  }
  collections_[path] = std::move(result).value();
  return collections_[path];
}

zvec::Collection::Ptr CollectionRegistry::Create(
    const std::string &path, const zvec::CollectionSchema &schema) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = collections_.find(path);
  if (it != collections_.end()) {
    throw IOException("zvec: collection already open at '%s'", path);
  }
  zvec::CollectionOptions options{false, true};
  auto result = zvec::Collection::CreateAndOpen(path, schema, options);
  if (!result.has_value()) {
    throw IOException("zvec: failed to create collection at '%s': %s", path,
                      result.error().message());
  }
  collections_[path] = std::move(result).value();
  return collections_[path];
}

void CollectionRegistry::Close(const std::string &path) {
  std::lock_guard<std::mutex> lock(mutex_);
  collections_.erase(path);
}

void CollectionRegistry::CloseAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  collections_.clear();
}

}  // namespace duckdb
