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

#include <fstream>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <zvec/ailego/hash/crc32c.h>
#include "db/common/file_helper.h"
#include "vector_column_indexer.h"

namespace zvec {
namespace {
void Put32(std::string *out, uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) out->push_back(value >> (i * 8));
}
uint32_t Get32(const char *in) {
  uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i)
    value |= uint32_t(static_cast<unsigned char>(in[i])) << (i * 8);
  return value;
}
}  // namespace

bool VectorColumnIndexer::has_vector(uint32_t row) const {
  return !has_validity() || (row < row_count_ && valid_rows_.contains(row));
}

uint32_t VectorColumnIndexer::physical_id(uint32_t row) const {
  return compact_ids_ ? static_cast<uint32_t>(valid_rows_.rank(row) - 1) : row;
}

bool VectorColumnIndexer::logical_id(uint64_t physical, uint32_t *row) const {
  if (physical > UINT32_MAX) return false;
  if (compact_ids_) return valid_rows_.select(physical, row);
  *row = physical;
  return has_vector(*row);
}

Status VectorColumnIndexer::insert_null(uint32_t doc_id) {
  std::unique_lock lock(validity_mutex_);
  if (!index || read_only_)
    return Status::InvalidArgument("Index is not writable");
  if (!has_validity())
    return Status::InvalidArgument("Vector field is not nullable");
  if (doc_id == UINT32_MAX)
    return Status::InvalidArgument("Vector row ID is out of range");
  if (doc_id < row_count_)
    return Status::InvalidArgument("Vector rows must be appended");
  row_count_ = doc_id + 1;
  validity_dirty_ = true;
  return Status::OK();
}

Status VectorColumnIndexer::load_validity(
    const vector_column_params::ReadOptions &options) {
  read_only_ = options.read_only;
  if (!has_validity()) return Status::OK();
  valid_rows_ = roaring::Roaring{};
  row_count_ = 0;
  compact_ids_ = options.create_new;
  validity_dirty_ = options.create_new;
  const auto path = index_file_path_ + ".validity";
  if (options.create_new) {
    // A new block may replace an abandoned, unpublished block after recovery.
    if (FileHelper::FileExists(path) && !FileHelper::RemoveFile(path))
      return Status::InternalError("Failed to remove stale vector validity: ",
                                   path);
    return Status::OK();
  }
  if (FileHelper::FileExists(path)) {
    std::ifstream in(ailego::FileHelper::PathFromUtf8(path), std::ios::binary);
    if (!in)
      return Status::InternalError("Failed to read vector validity: ", path);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    if (in.bad() || bytes.size() < 20 || bytes.compare(0, 8, "ZVECNV01") != 0 ||
        Get32(bytes.data() + 12) > 1 ||
        Get32(bytes.data() + bytes.size() - 4) !=
            ailego::Crc32c::Hash(bytes.data(), bytes.size() - 4)) {
      return Status::InternalError("Invalid vector validity: ", path);
    }
    row_count_ = Get32(bytes.data() + 8);
    compact_ids_ = Get32(bytes.data() + 12) != 0;
    try {
      valid_rows_ =
          roaring::Roaring::readSafe(bytes.data() + 16, bytes.size() - 20);
    } catch (const std::exception &) {
      return Status::InternalError("Corrupt vector validity bitmap: ", path);
    }
    if ((!valid_rows_.isEmpty() && valid_rows_.maximum() >= row_count_) ||
        (options.row_count && *options.row_count != row_count_) ||
        (index && compact_ids_ && valid_rows_.cardinality() != doc_count())) {
      return Status::InternalError(
          "Vector validity disagrees with block/index: ", path);
    }
    return Status::OK();
  }
  if (options.require_validity)
    return Status::InternalError("Missing vector validity: ", path);

  // Legacy blocks have identity IDs. Recover missingness from their stored
  // keys, never from the vector's numeric value (a real zero vector is valid).
  row_count_ = options.row_count.value_or(doc_count());
  auto recover = [&](const auto &provider) -> Status {
    if (!provider)
      return Status::InternalError("Missing legacy vector provider");
    auto iter = provider->create_iterator();
    if (!iter) return Status::InternalError("Missing legacy vector iterator");
    auto failed = [&] {
      if constexpr (std::is_same_v<std::decay_t<decltype(provider)>,
                                   core::IndexProvider::Pointer>)
        return iter->status() != 0;
      return false;
    };
    size_t visited = 0;
    for (; iter->is_valid(); iter->next()) {
      ++visited;
      const auto key = iter->key();
      if (failed())
        return Status::InternalError("Failed to read legacy vector keys");
      if (key == UINT64_MAX) continue;
      if (key >= row_count_)
        return Status::InternalError("Legacy vector key outside block");
      valid_rows_.add(key);
    }
    if (failed())
      return Status::InternalError("Failed to read legacy vector keys");
    if (visited != provider->count())
      return Status::InternalError("Incomplete legacy vector keys");
    return Status::OK();
  };
  return is_sparse_ ? recover(index->create_sparse_index_provider())
                    : recover(index->create_index_provider());
}

Status VectorColumnIndexer::flush_validity() {
  if (!has_validity() || read_only_ || !validity_dirty_) return Status::OK();
  std::string bytes("ZVECNV01", 8);
  Put32(&bytes, row_count_);
  Put32(&bytes, compact_ids_ ? 1 : 0);
  const auto offset = bytes.size();
  bytes.resize(offset + valid_rows_.getSizeInBytes());
  valid_rows_.write(bytes.data() + offset);
  Put32(&bytes, ailego::Crc32c::Hash(bytes.data(), bytes.size()));
  const auto path = index_file_path_ + ".validity";
  const auto tmp = path + ".tmp";
  ailego::File file;
  if (!file.create(tmp, 0) ||
      file.write(bytes.data(), bytes.size()) != bytes.size() || !file.flush()) {
    return Status::InternalError("Failed to write vector validity: ", path);
  }
  file.close();
  if (!FileHelper::MoveFile(tmp, path))
    return Status::InternalError("Failed to publish vector validity: ", path);
  validity_dirty_ = false;
  return Status::OK();
}
}  // namespace zvec
