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

#include <stdint.h>
#include <cstdint>
#include <string>
#include <ailego/io/file.h>
#include <ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec {

/*
 * File type and id
 */
enum class FileID : uint32_t {
  UNDEFINED = 0,
  ID_FILE,
  DELETE_FILE,
  FORWARD_FILE,
  PROXIMA_FILE,
  SEGMENT_FILE,
  LSN_FILE,
  MANIFEST_FILE,
  WAL_FILE,
  RESHARD_STATE,
};

/*
 * File name coresponding to file id
 */
static const char *GetFileName(FileID t) {
  switch (t) {
    case FileID::ID_FILE:
      return "idmap";
    case FileID::DELETE_FILE:
      return "del";
    case FileID::FORWARD_FILE:
      return "data.fwd";
    case FileID::PROXIMA_FILE:
      return "data.pxa";
    case FileID::SEGMENT_FILE:
      return "data.seg";
    case FileID::LSN_FILE:
      return "data.lsn";
    case FileID::MANIFEST_FILE:
      return "manifest";
    case FileID::WAL_FILE:
      return "data.wal";
    case FileID::RESHARD_STATE:
      return "reshard.state";
    default:
      return "UnknownFile";
  };
}

/*
 * This helper class is mainly to wrapper filesystem operations.
 */
class FileHelper {
 public:
  static const std::string MakeWalPath(const std::string &path, uint32_t seg_id,
                                       uint32_t block_id) {
    return ailego::StringHelper::Concat(path, "/", seg_id, "/", block_id,
                                        ".wal");
  }

  static std::string MakeSegmentPath(const std::string &path, uint32_t id,
                                     const std::string &suffix = "") {
    if (suffix.empty()) {
      return ailego::StringHelper::Concat(path, "/", id);
    }
    return ailego::StringHelper::Concat(path, "/", id, ".", suffix);
  }

  static std::string MakeTempSegmentPath(const std::string &path, uint32_t id) {
    return MakeSegmentPath(path, id, "tmp");
  }

  // e.g.: **/seg1/scalar.block.1.ipc, **/seg1/scalar.block.1.parquet
  static const std::string MakeForwardBlockPath(const std::string &path,
                                                uint32_t seg_id,
                                                uint32_t block_id,
                                                bool use_parquet = false) {
    return use_parquet ? MakeForwardBlockPath(path, seg_id, block_id,
                                              std::string("parquet"))
                       : MakeForwardBlockPath(path, seg_id, block_id,
                                              std::string("ipc"));
  }

  static const std::string MakeForwardBlockPath(const std::string &path,
                                                uint32_t seg_id,
                                                uint32_t block_id,
                                                const std::string &suffix) {
    return ailego::StringHelper::Concat(path, "/", seg_id, "/scalar.", block_id,
                                        ".", suffix);
  }

  static const std::string MakeForwardBlockPath(const std::string &seg_path,
                                                uint32_t block_id,
                                                bool use_parquet = false) {
    return use_parquet ? ailego::StringHelper::Concat(seg_path, "/scalar.",
                                                      block_id, ".parquet")
                       : ailego::StringHelper::Concat(seg_path, "/scalar.",
                                                      block_id, ".ipc");
  }

  static const std::string MakeForwardBlockPath(const std::string &seg_path,
                                                uint32_t block_id,
                                                const std::string &suffix) {
    return ailego::StringHelper::Concat(seg_path, "/scalar.", block_id, ".",
                                        suffix);
  }

  // e.g.: **/seg1/scalar.index.block.1.rocksdb
  static const std::string MakeInvertIndexPath(const std::string &path,
                                               uint32_t seg_id,
                                               uint32_t block_id) {
    return ailego::StringHelper::Concat(path, "/", seg_id, "/scalar.index.",
                                        block_id, ".rocksdb");
  }

  static const std::string MakeInvertIndexPath(const std::string &seg_path,
                                               uint32_t block_id) {
    return ailego::StringHelper::Concat(seg_path, "/scalar.index.", block_id,
                                        ".rocksdb");
  }

  static const std::string MakeVectorIndexPath(const std::string &path,
                                               const std::string &column,
                                               uint32_t seg_id,
                                               uint32_t block_id) {
    return ailego::StringHelper::Concat(path, "/", seg_id, "/", column,
                                        ".index.", block_id, ".proxima");
  }

  static const std::string MakeVectorIndexPath(const std::string &seg_path,
                                               const std::string &column,
                                               uint32_t block_id) {
    return ailego::StringHelper::Concat(seg_path, "/", column, ".index.",
                                        block_id, ".proxima");
  }

  // e.g.: **/{seg_id}/{column}.index.block.{block_id}.proxima
  static const std::string MakeQuantizeVectorIndexPath(
      const std::string &path, const std::string &column, uint32_t seg_id,
      uint32_t block_id) {
    return ailego::StringHelper::Concat(path, "/", seg_id, "/", column,
                                        ".qindex.", block_id, ".proxima");
  }

  static const std::string MakeQuantizeVectorIndexPath(
      const std::string &seg_path, const std::string &column,
      uint32_t block_id) {
    return ailego::StringHelper::Concat(seg_path, "/", column, ".qindex.",
                                        block_id, ".proxima");
  }

  //! Make file path with ${prefix_path}/${file_name}
  static std::string MakeFilePath(const std::string &prefix_path,
                                  FileID file_id) {
    return ailego::StringHelper::Concat(prefix_path, "/", GetFileName(file_id));
  }

  //! Make file path with ${prefix_path}/${file_name}.${number}
  static std::string MakeFilePath(const std::string &prefix_path,
                                  FileID file_id, uint32_t number) {
    return ailego::StringHelper::Concat(prefix_path, "/", GetFileName(file_id),
                                        ".", number);
  }

  //! Make file path with ${prefix_path}/${file_name}.${suffix_name}.${number}
  static std::string MakeFilePath(const std::string &prefix_path,
                                  FileID file_id, uint32_t number,
                                  const std::string &suffix_name) {
    return ailego::StringHelper::Concat(prefix_path, "/", GetFileName(file_id),
                                        ".", suffix_name, ".", number);
  }

  //! Create directory
  static bool CreateDirectory(const std::string &dir_path) {
    return ailego::File::MakePath(dir_path);
  }

  //! Remove directory
  static bool RemoveDirectory(const std::string &dir_path) {
    return ailego::File::RemoveDirectory(dir_path);
  }

  //! Remove file
  static bool RemoveFile(const std::string &file_path) {
    return ailego::File::Delete(file_path);
  }

  //! Move file
  static bool MoveFile(const std::string &src_path,
                       const std::string &dest_path) {
    return ailego::File::Rename(src_path, dest_path);
  }

  //! Move directory
  static bool MoveDirectory(const std::string &src_path,
                            const std::string &dest_path) {
    return ailego::File::Rename(src_path, dest_path);
  }

  //! Check if file exists
  static bool FileExists(const std::string &file_path) {
    return ailego::File::IsExist(file_path);
  }

  //! Check if directory exists
  static bool DirectoryExists(const std::string &dir_path) {
    return ailego::File::IsExist(dir_path);
  }

  //! Return file size
  static size_t FileSize(const std::string &file_path) {
    return ailego::FileHelper::FileSize(file_path.c_str());
  }

  //! Copy file
  //! src_file_path and dst_file_path must be the full path
  //! dst_file_path/.. must exist
  static bool CopyFile(const std::string &src_file_path,
                       const std::string &dst_file_path);

  //! Copy directory recursively
  //! src_dir_path and dst_dir_path must be the full path
  //! dst_dir_path will be created if not exist
  static bool CopyDirectory(const std::string &src_dir_path,
                            const std::string &dst_dir_path);

  //! Clean up file or directory with the prefix `prefix_name` under
  //! `backup_dir`, keep at most `max_backup_count` file or directory.
  //! If `max_backup_count` is 0, nothing is performed.
  //!
  //! The name pattern must be `prefix_name`_`number`, comparable by name.
  static void CleanupDirectory(const std::string &backup_dir,
                               size_t max_backup_count,
                               const char *prefix_name);

  static const std::string BACKUP_SUFFIX;
  static const std::string RECOVER_SUFFIX;
};


}  // namespace zvec