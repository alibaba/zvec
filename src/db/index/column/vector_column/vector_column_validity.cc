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
// Expose only real vectors to quantizer training and external-vector builds.
// Keys stay equal to document IDs; this view never rewrites them.
class NonNullVectorProvider final : public core::IndexProvider {
 public:
  NonNullVectorProvider(core::IndexProvider::Pointer provider,
                        const roaring::Roaring &nulls, uint32_t rows)
      : provider_(std::move(provider)),
        nulls_(std::make_shared<const roaring::Roaring>(nulls)),
        rows_(rows) {}

  size_t count() const override {
    return rows_ - nulls_->cardinality();
  }
  size_t slot_count() const override {
    return provider_->slot_count();
  }
  size_t dimension() const override {
    return provider_->dimension();
  }
  core::IndexMeta::DataType data_type() const override {
    return provider_->data_type();
  }
  size_t element_size() const override {
    return provider_->element_size();
  }
  const std::string &owner_class() const override {
    return provider_->owner_class();
  }
  const void *get_vector(uint64_t key) const override {
    return key < rows_ && !nulls_->contains(key) ? provider_->get_vector(key)
                                                 : nullptr;
  }
  int get_vector(uint64_t key,
                 core::IndexStorage::MemoryBlock &block) const override {
    if (key >= rows_ || nulls_->contains(key)) return core::IndexError_NoExist;
    return provider_->get_vector(key, block);
  }

  class Iterator final : public core::IndexHolder::Iterator {
   public:
    Iterator(core::IndexProvider::Pointer provider,
             core::IndexHolder::Iterator::Pointer iter,
             std::shared_ptr<const roaring::Roaring> nulls, uint32_t rows)
        : provider_(std::move(provider)),
          iter_(std::move(iter)),
          nulls_(std::move(nulls)),
          rows_(rows) {
      seek();
    }
    const void *data() const override {
      return iter_->data();
    }
    bool is_valid() const override {
      return status() == 0 && iter_->is_valid();
    }
    int status() const override {
      return status_ ? status_ : iter_->status();
    }
    uint64_t key() const override {
      return iter_->key();
    }
    void next() override {
      iter_->next();
      seek();
    }

   private:
    void seek() {
      while (iter_->is_valid() && iter_->status() == 0) {
        const auto key = iter_->key();
        if (iter_->status() != 0) return;
        if (key != UINT64_MAX) {
          if (key >= rows_) {
            status_ = core::IndexError_InvalidFormat;
            return;
          }
          if (!nulls_->contains(key)) return;
        }
        iter_->next();
      }
    }
    core::IndexProvider::Pointer provider_;
    core::IndexHolder::Iterator::Pointer iter_;
    std::shared_ptr<const roaring::Roaring> nulls_;
    uint32_t rows_;
    int status_{0};
  };

  core::IndexHolder::Iterator::Pointer create_iterator() override {
    auto iter = provider_->create_iterator();
    if (!iter) return nullptr;
    return std::make_unique<Iterator>(provider_, std::move(iter), nulls_,
                                      rows_);
  }

 private:
  core::IndexProvider::Pointer provider_;
  std::shared_ptr<const roaring::Roaring> nulls_;
  uint32_t rows_;
};

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
  if (!index) return nullptr;
  auto provider = index->create_index_provider();
  if (!provider || !has_null_bitmap()) return provider;
  return std::make_shared<NonNullVectorProvider>(std::move(provider),
                                                 null_rows_, row_count_);
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
  const auto path = index_file_path_ + ".nulls";
  if (options.create_new) {
    // A new block may replace an abandoned, unpublished block after recovery.
    for (const auto &stale : {path, index_file_path_ + ".validity"}) {
      if (FileHelper::FileExists(stale) && !FileHelper::RemoveFile(stale))
        return Status::InternalError(
            "Failed to remove stale vector NULL bitmap: ", stale);
    }
    return Status::OK();
  }
  auto unsupported_legacy = [&] {
    auto status = Status::NotSupported(
        "Legacy index detected for nullable vector field '",
        field_schema_.name(), "' at ", index_file_path_,
        ". If NULL vectors were inserted in an older version, "
        "vector-to-document "
        "mappings may be incorrect. You may need to reinsert the original data "
        "into a new collection. For help or feedback, please open an issue at "
        "https://github.com/alibaba/zvec/issues.");
    LOG_ERROR("%s", status.message().c_str());
    return status;
  };
  if (FileHelper::FileExists(index_file_path_ + ".validity"))
    return unsupported_legacy();
  if (FileHelper::FileExists(path)) {
    std::ifstream in(ailego::FileHelper::PathFromUtf8(path), std::ios::binary);
    if (!in)
      return Status::InternalError("Failed to read vector NULL bitmap: ", path);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    if (in.bad() || bytes.size() < 20 || bytes.compare(0, 8, "ZVECNL01") != 0 ||
        Get32(bytes.data() + bytes.size() - 4) !=
            ailego::Crc32c::Hash(bytes.data(), bytes.size() - 4)) {
      return Status::InternalError("Invalid vector NULL bitmap: ", path);
    }
    if (Get32(bytes.data() + 12) != 3) return unsupported_legacy();
    row_count_ = Get32(bytes.data() + 8);
    try {
      null_rows_ =
          roaring::Roaring::readSafe(bytes.data() + 16, bytes.size() - 20);
    } catch (const std::exception &) {
      return Status::InternalError("Corrupt vector NULL bitmap: ", path);
    }
    if ((!null_rows_.isEmpty() && null_rows_.maximum() >= row_count_) ||
        (options.row_count && *options.row_count != row_count_) ||
        (index && index->get_doc_count() > row_count_)) {
      return Status::InternalError(
          "Vector NULL bitmap disagrees with block/index: ", path);
    }
    if (index) {
      const uint32_t expected = row_count_;
      if (index->get_doc_count() != expected)
        return Status::InternalError(
            "Vector NULL bitmap disagrees with index: ", path,
            " expected rows=", expected,
            " actual rows=", index->get_doc_count());
    }
    return Status::OK();
  }
  if (options.require_validity)
    return Status::InternalError("Missing vector NULL bitmap: ", path);

  return unsupported_legacy();
}

Status VectorColumnIndexer::flush_nulls() {
  if (!has_null_bitmap() || read_only_ || !nulls_dirty_) return Status::OK();
  std::string bytes("ZVECNL01", 8);
  Put32(&bytes, row_count_);
  Put32(&bytes,
        3);  // Core persists the full document range, including NULL tails.
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
