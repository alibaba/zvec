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

#include <string>

namespace zvec {
namespace core {

inline const std::string INDEX_META_SEGMENT_NAME = "IndexMeta";
inline const std::string INDEX_VERSION_SEGMENT_NAME = "IndexVersion";

//! FileLogger
inline const std::string FILE_LOGGER_PATH = "proxima.file.logger.path";

//! FileContainer
inline const std::string FILE_READ_STORAGE_CHECKSUM_VALIDATION =
    "proxima.file.read_storage.checksum_validation";
inline const std::string FILE_READ_STORAGE_ENABLE_DIRECT_IO =
    "proxima.file.read_storage.enable_direct_io";
inline const std::string FILE_READ_STORAGE_ALONE_FILE_HANDLE =
    "proxima.file.read_storage.alone_file_handle";
inline const std::string FILE_READ_STORAGE_MEMORY_LOCKED =
    "proxima.file.read_storage.memory_locked";
inline const std::string FILE_READ_STORAGE_MEMORY_WARMUP =
    "proxima.file.read_storage.memory_warmup";
inline const std::string FILE_READ_STORAGE_MEMORY_SHARED =
    "proxima.file.read_storage.memory_shared";
inline const std::string FILE_READ_STORAGE_HEADER_OFFSET =
    "proxima.file.read_storage.header_offset";
inline const std::string FILE_READ_STORAGE_FOOTER_OFFSET =
    "proxima.file.read_storage.footer_offset";

//! MemoryContainer
inline const std::string MEMORY_CONTAINER_CHECKSUM_VALIDATION =
    "proxima.memory.container.checksum_validation";

//! MMapFileContainer
inline const std::string MMAPFILE_READ_STORAGE_MEMORY_LOCKED =
    "proxima.mmap_file.container.memory_locked";
inline const std::string MMAPFILE_READ_STORAGE_MEMORY_WARMUP =
    "proxima.mmap_file.container.memory_warmup";
inline const std::string MMAPFILE_READ_STORAGE_MEMORY_SHARED =
    "proxima.mmap_file.container.memory_shared";
inline const std::string MMAPFILE_READ_STORAGE_CHECKSUM_VALIDATION =
    "proxima.mmap_file.container.checksum_validation";
inline const std::string MMAPFILE_READ_STORAGE_HEADER_OFFSET =
    "proxima.mmap_file.container.header_offset";
inline const std::string MMAPFILE_READ_STORAGE_FOOTER_OFFSET =
    "proxima.mmap_file.container.footer_offset";

//! BufferReadStorage (read-only storage backed by VecBufferPool)
inline const std::string BUFFER_READ_STORAGE_CHECKSUM_VALIDATION =
    "proxima.buffer.read_storage.checksum_validation";
inline const std::string BUFFER_READ_STORAGE_HEADER_OFFSET =
    "proxima.buffer.read_storage.header_offset";
inline const std::string BUFFER_READ_STORAGE_FOOTER_OFFSET =
    "proxima.buffer.read_storage.footer_offset";
inline const std::string BUFFER_READ_STORAGE_WARMUP_MODE =
    "proxima.buffer.read_storage.warmup_mode";
inline const std::string BUFFER_READ_STORAGE_WARMUP_NONE = "none";
inline const std::string BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL = "sequential";

//! MMapFileStorage
inline const std::string MMAPFILE_STORAGE_MEMORY_LOCKED =
    "proxima.mmap_file.storage.memory_locked";
inline const std::string MMAPFILE_STORAGE_MEMORY_WARMUP =
    "proxima.mmap_file.storage.memory_warmup";
inline const std::string MMAPFILE_STORAGE_COPY_ON_WRITE =
    "proxima.mmap_file.storage.copy_on_write";
inline const std::string MMAPFILE_STORAGE_FORCE_FLUSH =
    "proxima.mmap_file.storage.force_flush";
inline const std::string MMAPFILE_STORAGE_SEGMENT_META_CAPACITY =
    "proxima.mmap_file.storage.segment_meta_capacity";

//! SimpleForward
inline const std::string SIMPLE_FORWARD_DATA_BLOCK_SIZE =
    "proxima.simple.forward.data_block_size";
inline const std::string SIMPLE_FORWARD_INDEX_BLOCK_SIZE =
    "proxima.simple.forward.index_block_size";

//! SimpleForward
inline const std::string SIMPLE_CLOSET_DATA_BLOCK_SIZE =
    "proxima.simple.closet.data_block_size";
inline const std::string SIMPLE_CLOSET_INDEX_BLOCK_SIZE =
    "proxima.simple.closet.index_block_size";

//! ChainCloset
inline const std::string CHAIN_CLOSET_SLOT_SIZE =
    "proxima.chain.closet.slot_size";
inline const std::string CHAIN_CLOSET_INDEX_BLOCK_SIZE =
    "proxima.chain.closet.index_block_size";
inline const std::string CHAIN_CLOSET_DATA_BLOCK_SIZE =
    "proxima.chain.closet.data_block_size";

//! IndexForward
inline const std::string PARAM_FORWARD_MULTI_VALUE =
    "proxima.param.forward.multi_value";
inline const std::string PARAM_FORWARD_MULTI_COUNT =
    "proxima.param.forward.multi_count";

}  // namespace core
}  // namespace zvec
