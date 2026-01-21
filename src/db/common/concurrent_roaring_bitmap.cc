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

#include "concurrent_roaring_bitmap.h"
#include <zvec/ailego/hash/crc32c.h>


namespace zvec {


Status ConcurrentRoaringBitmap32::serialize(std::string *out) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  size_t bitmap_size = roaring_bitmap_portable_size_in_bytes(bitmap_);
  out->resize(bitmap_size);
  size_t written_size = roaring_bitmap_portable_serialize(bitmap_, out->data());
  if (written_size == bitmap_size) {
    return Status::OK();
  } else {
    LOG_ERROR("Failed to serialize bitmap");
    return Status::InternalError();
  }
}


Status ConcurrentRoaringBitmap32::deserialize(const std::string &in) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  roaring_bitmap_free(bitmap_);
  bitmap_ = nullptr;
  bitmap_ = roaring_bitmap_portable_deserialize_safe(in.data(), in.size());
  if (bitmap_) {
    return Status::OK();
  } else {
    LOG_ERROR("Failed to deserialize bitmap");
    return Status::InternalError();
  }
}


Status ConcurrentRoaringBitmap64::serialize(const std::string &file_path,
                                            bool overwrite) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  FILE file;
  const std::string file_str = "[" + file_path + "]";

  if (FILE::IsExist(file_path)) {
    if (!FILE::IsRegular(file_path)) {
      auto msg = debug_str(file_str, " is not a regular file");
      LOG_ERROR("%s", msg.c_str());
      return Status::InvalidArgument(msg);
    }
    if (!overwrite) {
      auto msg = debug_str(file_str, " already exists");
      LOG_ERROR("%s", msg.c_str());
      return Status::AlreadyExists(msg);
    }
    if (!FILE::RemovePath(file_path)) {
      auto msg = debug_str("failed to remove ", file_str);
      LOG_ERROR("%s", msg.c_str());
      return Status::InternalError(msg);
    }
  }
  if (!file.create(file_path.c_str(), 0)) {
    auto msg = debug_str("failed to create ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Serialize bitmap to buffer
  BitmapMetaHeader header;
  size_t bitmap_size;
  std::vector<char> bitmap_buffer;
  if (is_32bit_) {
    bitmap_size = bitmap32_->getSizeInBytes();
    bitmap_buffer.resize(bitmap_size);
    if (bitmap32_->write(bitmap_buffer.data()) != bitmap_size) {
      auto msg = debug_str("failed to serialize bitmap to buffer");
      LOG_ERROR("%s", msg.c_str());
      return Status::InternalError(msg);
    }
    header.is_32bit = 1;
  } else {
    bitmap_size = bitmap64_->getSizeInBytes();
    bitmap_buffer.resize(bitmap_size);
    if (bitmap64_->write(bitmap_buffer.data()) != bitmap_size) {
      auto msg = debug_str("failed to serialize bitmap to buffer");
      LOG_ERROR("%s", msg.c_str());
      return Status::InternalError(msg);
    }
    header.is_32bit = 0;
  }
  header.magic = roaring_magic_number;
  header.checksum = ailego::Crc32c::Hash(bitmap_buffer.data(), bitmap_size);
  header.timestamp = time(nullptr);

  // Write meta header to file
  if (file.write(&header, sizeof(header)) != sizeof(header)) {
    auto msg = debug_str("failed to serialize header to ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Write serialized bitmap to file
  if (file.write(bitmap_buffer.data(), bitmap_size) != bitmap_size) {
    auto msg = debug_str("failed to write bitmap data to ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  };

  LOG_DEBUG("%s: serialized bitmap to file[%s], checksum[%u], timestamp[%zu]",
            identifier_.c_str(), file_path.c_str(), header.checksum,
            (size_t)header.timestamp);
  return Status::OK();
}


Status ConcurrentRoaringBitmap64::deserialize(const std::string &file_path) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  FILE file;
  const std::string file_str = "[" + file_path + "]";

  if (!FILE::IsExist(file_path)) {
    auto msg = debug_str(file_str, " does not exist");
    LOG_ERROR("%s", msg.c_str());
    return Status::NotFound(msg);
  }
  if (!FILE::IsRegular(file_path)) {
    auto msg = debug_str(file_str, " is not a regular file");
    LOG_ERROR("%s", msg.c_str());
    return Status::InvalidArgument(msg);
  }
  if (!file.open(file_path.c_str(), true, false)) {
    auto msg = debug_str("failed to open ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Deserialize and verify the meta header
  BitmapMetaHeader header;
  if (file.size() < sizeof(BitmapMetaHeader)) {
    auto msg =
        debug_str(file_str, " is too small to to contain a valid bitmap");
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (file.read(&header, sizeof(header)) != sizeof(header)) {
    auto msg = debug_str("failed to read meta header from ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (header.magic != roaring_magic_number) {
    auto msg = debug_str("magic number mismatch, ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (header.is_32bit != 0 && header.is_32bit != 1) {
    auto msg = debug_str("bitmap type mismatch, ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  is_32bit_ = header.is_32bit == 1 ? true : false;

  // Read from file to buffer
  size_t bitmap_size = file.size() - sizeof(BitmapMetaHeader);
  std::vector<char> bitmap_buffer(bitmap_size);
  if (file.read(bitmap_buffer.data(), bitmap_size) != bitmap_size) {
    auto msg = debug_str("failed to read bitmap data from ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (header.checksum !=
      ailego::Crc32c::Hash(bitmap_buffer.data(), bitmap_size)) {
    auto msg = debug_str("checksum mismatch, ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Deserialize from buffer
  try {
    if (is_32bit_) {
      bitmap32_ = std::make_unique<roaring::Roaring>(
          roaring::Roaring::readSafe(bitmap_buffer.data(), bitmap_size));
    } else {
      bitmap64_ = std::make_unique<roaring::Roaring64Map>(
          roaring::Roaring64Map::readSafe(bitmap_buffer.data(), bitmap_size));
    }
  } catch (...) {
    auto msg = debug_str("failed to deserialize bitmap from ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  LOG_DEBUG(
      "%s: deserialized bitmap from file[%s], checksum[%u], timestamp[%zu]",
      identifier_.c_str(), file_path.c_str(), header.checksum,
      (size_t)header.timestamp);
  return Status::OK();
}


}  // namespace zvec
