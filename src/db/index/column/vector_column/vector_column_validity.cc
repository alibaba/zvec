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

core::IndexProvider::Pointer VectorColumnIndexer::create_index_provider()
    const {
  std::shared_lock lock(nulls_mutex_);
  return index ? index->create_index_provider() : nullptr;
}

bool VectorColumnIndexer::has_vector(uint32_t row) const {
  return !has_null_bitmap() || (row < row_count_ && !null_rows_.contains(row));
}

Status VectorColumnIndexer::insert_null(uint32_t doc_id) {
  std::unique_lock lock(nulls_mutex_);
  if (!index || read_only_)
    return Status::InvalidArgument("Index is not writable");
  if (!has_null_bitmap())
    return Status::InvalidArgument("Vector field is not nullable");
  if (doc_id == UINT32_MAX)
    return Status::InvalidArgument("Vector row ID is out of range");
  if (doc_id < row_count_)
    return Status::InvalidArgument("Vector rows must be appended");
  if (all_null() && !empty_flat_index_) {
    auto status = recreate_empty_index(true);
    if (!status.ok()) return status;
  }
  if (index->add({core_interface::NullVector{}}, doc_id) != 0)
    return Status::InternalError("Failed to allocate NULL vector document");
  null_rows_.addRange(row_count_, uint64_t{doc_id} + 1);
  row_count_ = doc_id + 1;
  nulls_dirty_ = true;
  return Status::OK();
}

Status VectorColumnIndexer::load_nulls(
    const vector_column_params::ReadOptions &options) {
  read_only_ = options.read_only;
  if (!has_null_bitmap()) return Status::OK();
  null_rows_ = roaring::Roaring{};
  row_count_ = 0;
  nulls_dirty_ = options.create_new;
  legacy_without_validity_ = false;
  const auto path = index_file_path_ + ".nulls";
  if (options.create_new) {
    // A new block may replace an abandoned, unpublished block after recovery.
    if (FileHelper::FileExists(path) && !FileHelper::RemoveFile(path))
      return Status::InternalError(
          "Failed to remove stale vector NULL bitmap: ", path);
    return Status::OK();
  }
  // A legacy block (manifest without validity metadata) never marked NULL
  // rows, so the original data may already be lost. Open it best-effort with
  // a warning rather than refusing: every stored slot is treated as present.
  auto warn_legacy = [&] {
    LOG_WARN(
        "Legacy nullable vector field '%s' at %s has no NULL bitmap. NULL "
        "vectors inserted by an older version were not marked, so "
        "vector-to-document mappings may be incorrect and some data may be "
        "lost. Opening anyway; reinsert the original data into a new "
        "collection if exact NULL semantics are required. For help or "
        "feedback, please open an issue at "
        "https://github.com/alibaba/zvec/issues.",
        field_schema_.name().c_str(), index_file_path_.c_str());
    legacy_without_validity_ = true;
    null_rows_ = roaring::Roaring{};
    row_count_ = index ? index->get_doc_count() : 0;
    nulls_dirty_ = false;
    return Status::OK();
  };
  if (FileHelper::FileExists(path)) {
    std::ifstream in(ailego::FileHelper::PathFromUtf8(path), std::ios::binary);
    if (!in)
      return Status::InternalError("Failed to read vector NULL bitmap: ", path);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    if (in.bad() || bytes.size() < 16 || bytes.compare(0, 8, "ZVECNL01") != 0 ||
        Get32(bytes.data() + bytes.size() - 4) !=
            ailego::Crc32c::Hash(bytes.data(), bytes.size() - 4)) {
      return Status::InternalError("Invalid vector NULL bitmap: ", path);
    }
    row_count_ = Get32(bytes.data() + 8);
    try {
      null_rows_ =
          roaring::Roaring::readSafe(bytes.data() + 12, bytes.size() - 16);
    } catch (const std::exception &) {
      return Status::InternalError("Corrupt vector NULL bitmap: ", path);
    }
    if ((!null_rows_.isEmpty() && null_rows_.maximum() >= row_count_) ||
        (options.row_count && *options.row_count != row_count_) ||
        (index && index->get_doc_count() != row_count_)) {
      return Status::InternalError(
          "Vector NULL bitmap disagrees with block/index: ", path,
          " rows=", row_count_,
          " index_rows=", index ? index->get_doc_count() : 0);
    }
    return Status::OK();
  }
  if (options.require_validity)
    return Status::InternalError("Missing vector NULL bitmap: ", path);

  return warn_legacy();
}

Status VectorColumnIndexer::flush_nulls() {
  if (!has_null_bitmap() || read_only_ || !nulls_dirty_) return Status::OK();
  std::string bytes("ZVECNL01", 8);
  Put32(&bytes, row_count_);
  const auto offset = bytes.size();
  bytes.resize(offset + null_rows_.getSizeInBytes());
  null_rows_.write(bytes.data() + offset);
  Put32(&bytes, ailego::Crc32c::Hash(bytes.data(), bytes.size()));
  const auto path = index_file_path_ + ".nulls";
  const auto tmp = path + ".tmp";
  ailego::File file;
  if (!file.create(tmp, 0) ||
      file.write(bytes.data(), bytes.size()) != bytes.size() || !file.flush()) {
    return Status::InternalError("Failed to write vector NULL bitmap: ", path);
  }
  file.close();
  if (!FileHelper::MoveFile(tmp, path))
    return Status::InternalError("Failed to publish vector NULL bitmap: ",
                                 path);
  nulls_dirty_ = false;
  return Status::OK();
}
}  // namespace zvec
