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

#include <zvec/ailego/io/mmap_file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/framework/index_mapping.h>
#include "ailego/utility/memory_helper.h"

#ifdef __linux__
#include <sys/statfs.h>
#include <sys/vfs.h>
#ifndef HUGETLBFS_MAGIC
#define HUGETLBFS_MAGIC 0x958458f6
#endif
#endif

namespace zvec {
namespace core {

static inline size_t CalcPageAlignedSize(size_t size, bool huge_size) {
  size_t page_size = ailego::MemoryHelper::PageSize();
  if (huge_size) {
    page_size = ailego::MemoryHelper::HugePageSize();
  }
  return (size + page_size - 1) / page_size * page_size;
}

static inline bool WritePadding(ailego::File &file, size_t size) {
  std::string padding(ailego::MemoryHelper::PageSize(), 0);
  for (size_t i = 0, count = size / padding.size(); i < count; ++i) {
    if (file.write(padding.data(), padding.size()) != padding.size()) {
      return false;
    }
  }
  padding.resize(size % padding.size());
  if (padding.size()) {
    if (file.write(padding.data(), padding.size()) != padding.size()) {
      return false;
    }
  }
  return true;
}

static inline int UnpackMappingSize(ailego::File &file, size_t *len) {
  IndexFormat::MetaHeader header;
  if (file.read(&header, sizeof(header)) != sizeof(header)) {
    LOG_ERROR("Failed to read file, errno %d, %s", errno, std::strerror(errno));
    return IndexError_ReadData;
  }

  if (header.meta_header_size != sizeof(IndexFormat::MetaHeader) ||
      header.meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
    return IndexError_InvalidValue;
  }

  if (ailego::Crc32c::Hash(&header, sizeof(header), header.header_crc) !=
      header.header_crc) {
    return IndexError_InvalidChecksum;
  }

  if ((int32_t)header.meta_footer_offset < 0) {
    return IndexError_Unsupported;
  }

  *len = header.meta_footer_offset + header.meta_footer_size;
  if (*len > file.size()) {
    return IndexError_InvalidLength;
  }
  return 0;
}

int IndexMapping::open(const std::string &path, bool cow, bool full_mode) {
  path_ = path;
  full_mode_ = full_mode;
  copy_on_write_ = cow;
  huge_page_ = Ishugetlbfs(path);

  bool read_only = copy_on_write_ && !full_mode_;
  if (!file_.open(path.c_str(), read_only, false)) {
    LOG_ERROR("Failed to open file %s, errno %d, %s", path.c_str(), errno,
              std::strerror(errno));
    return IndexError_OpenFile;
  }

  size_t mapping_size = 0u;
  int error_code = UnpackMappingSize(file_, &mapping_size);
  if (error_code != 0) {
    file_.close();
    return error_code;
  }

  if (!file_.seek(0, ailego::File::Origin::End)) {
    LOG_ERROR("Failed to seek file %s, errno %d, %s", path.c_str(), errno,
              std::strerror(errno));
    return IndexError_SeekFile;
  }
  return this->init_index_mapping(mapping_size);
}

int IndexMapping::create(const std::string &path, size_t seg_meta_capacity) {
  path_ = path;
  seg_meta_capacity_ = seg_meta_capacity;
  current_header_start_offset_ = 0;

  // write() & copying to mmap() will auto extend the file size
  if (!file_.create(path.c_str(), 0)) {
    LOG_ERROR("Failed to create file %s, errno %d, %s", path.c_str(), errno,
              std::strerror(errno));
    return IndexError_CreateFile;
  }
  huge_page_ = Ishugetlbfs(path);
  if (huge_page_) {
    return init_hugepage_meta_section();
  }
  return init_meta_section();
}

int IndexMapping::init_meta_section() {
  if (current_header_start_offset_ % ailego::MemoryHelper::PageSize() != 0) {
    LOG_ERROR("File offset %zu is not a multiple of the page size: %zu",
              current_header_start_offset_, ailego::MemoryHelper::PageSize());
    return IndexError_InvalidValue;
  }

  auto &path = path_;
  size_t len =
      CalcPageAlignedSize(seg_meta_capacity_ + sizeof(IndexFormat::MetaHeader) +
                              sizeof(IndexFormat::MetaFooter),
                          false);

  IndexFormat::MetaHeader meta_header;
  IndexFormat::MetaFooter meta_footer;

  // Write index header
  IndexFormat::SetupMetaHeader(&meta_header, len - sizeof(meta_footer), len);
  if (!file_.seek(current_header_start_offset_, ailego::File::Origin::Begin)) {
    LOG_ERROR("Failed to seek file %s, errno %d, %s", path.c_str(), errno,
              std::strerror(errno));
    return IndexError_SeekFile;
  }
  if (file_.write(&meta_header, sizeof(meta_header)) != sizeof(meta_header)) {
    LOG_ERROR("Failed to write file: %s, errno %d, %s", path.c_str(), errno,
              std::strerror(errno));
    return IndexError_WriteData;
  }

  // Write padding data
  uint32_t segments_meta_size =
      static_cast<uint32_t>(len - (sizeof(meta_header) + sizeof(meta_footer)));
  if (!WritePadding(file_, segments_meta_size)) {
    LOG_ERROR("Failed to write file: %s, errno %d, %s", path.c_str(), errno,
              std::strerror(errno));
    return IndexError_WriteData;
  }

  // Write index footer
  IndexFormat::SetupMetaFooter(&meta_footer);
  meta_footer.segments_meta_size = segments_meta_size;
  meta_footer.total_size = len;
  IndexFormat::UpdateMetaFooter(&meta_footer, 0);
  if (file_.write(&meta_footer, sizeof(meta_footer)) != sizeof(meta_footer)) {
    LOG_ERROR("Failed to write file: %s, errno %d, %s", path.c_str(), errno,
              std::strerror(errno));
    return IndexError_WriteData;
  }
  return this->init_index_mapping(len);
}

int IndexMapping::init_hugepage_meta_section() {
  ssize_t file_offset = (ssize_t)current_header_start_offset_;
  if (file_offset % ailego::MemoryHelper::HugePageSize() != 0) {
    LOG_ERROR("File offset %zu is not a multiple of the page size: %zu",
              file_offset, ailego::MemoryHelper::HugePageSize());
    return IndexError_InvalidValue;
  }

  size_t len =
      CalcPageAlignedSize(seg_meta_capacity_ + sizeof(IndexFormat::MetaHeader) +
                              sizeof(IndexFormat::MetaFooter),
                          true);
  int opts = ailego::File::MMAP_SHARED | ailego::File::MMAP_HUGE_PAGE;
  void *addr =
      ailego::File::MemoryMap(file_.native_handle(), file_offset, len, opts);

  IndexFormat::MetaHeader meta_header;
  IndexFormat::MetaFooter meta_footer;

  // Write index header
  IndexFormat::SetupMetaHeader(&meta_header, len - sizeof(meta_footer), len);
  memcpy((char *)addr + file_offset, &meta_header, sizeof(meta_header));
  file_offset += sizeof(meta_header);

  // Write padding data
  uint32_t segments_meta_size =
      static_cast<uint32_t>(len - (sizeof(meta_header) + sizeof(meta_footer)));
  std::string padding(ailego::MemoryHelper::HugePageSize(), 0);
  for (size_t i = 0, count = segments_meta_size / padding.size(); i < count;
       ++i) {
    memcpy((char *)addr + file_offset, padding.data(), padding.size());
    file_offset += padding.size();
  }
  padding.resize(segments_meta_size % padding.size());
  if (padding.size()) {
    memcpy((char *)addr + file_offset, padding.data(), padding.size());
    file_offset += padding.size();
  }

  // Write index footer
  IndexFormat::SetupMetaFooter(&meta_footer);
  meta_footer.segments_meta_size = segments_meta_size;
  meta_footer.total_size = len;
  IndexFormat::UpdateMetaFooter(&meta_footer, 0);
  memcpy((char *)addr + file_offset, &meta_footer, sizeof(meta_footer));
  file_offset += sizeof(meta_footer);

  return this->init_index_mapping(len);
}

void IndexMapping::close(void) {
  // Unmap all memory
  this->unmap_all();
  if (header_) {
    for (auto item : header_addr_map_) {
      auto header = item.second;
      ailego::File::MemoryUnmap(header, header->content_offset);
    }
  }
  // Reset members
  segment_ids_offset_ = 0;
  segment_start_ = nullptr;
  header_ = nullptr;
  header_addr_map_.clear();
  footer_ = nullptr;
  index_size_ = 0u;
  segments_.clear();
  file_.close();
  copy_on_write_ = false;
  full_mode_ = false;
  header_dirty_ = false;
  huge_page_ = false;
}

void IndexMapping::refresh(uint64_t check_point) {
  // support add_with_id
  for (auto item : header_addr_map_) {
    auto header_start_offset = item.first;
    auto header = item.second;
    auto footer = reinterpret_cast<IndexFormat::MetaFooter *>(
        reinterpret_cast<uint8_t *>(header) + header->meta_footer_offset);
    auto segment_start = reinterpret_cast<IndexFormat::SegmentMeta *>(
        reinterpret_cast<uint8_t *>(header) +
        (header->meta_footer_offset - footer->segments_meta_size));
    footer->segments_meta_crc =
        ailego::Crc32c::Hash(segment_start, footer->segments_meta_size, 0);
    IndexFormat::UpdateMetaFooter(footer, check_point);
  }
  header_dirty_ = true;
}

int IndexMapping::append(const std::string &id, size_t size) {
  size = CalcPageAlignedSize(size, huge_page_);
  if (size == 0) {
    return IndexError_InvalidArgument;
  }

  if (segments_.find(id) != segments_.end()) {
    return IndexError_Duplicate;
  }

  size_t id_size = std::strlen(id.c_str()) + 1;
  size_t need_size = sizeof(IndexFormat::SegmentMeta) + id_size;
  if (sizeof(IndexFormat::SegmentMeta) * footer_->segment_count + need_size >
      segment_ids_offset_) {
    LOG_DEBUG("segment meta section expanded: %s", path_.c_str());
    footer_->next_meta_header_offset = index_size_;
    refresh(0);
    flush();
    // mmap file storage write() will update segment's meta
    // ailego::File::MemoryUnmap(header_, header_->content_offset);
    header_ = nullptr;
    footer_ = nullptr;

    current_header_start_offset_ = index_size_;
    const int ret =
        huge_page_ ? init_hugepage_meta_section() : init_meta_section();
    if (ret != 0) {
      return ret;
    }
  }

  if (!copy_on_write_ && !file_.truncate(index_size_ + size)) {
    LOG_ERROR("Failed to truncate file, errno %d, %s", errno,
              std::strerror(errno));
    return IndexError_TruncateFile;
  }

  // Update segment table
  segment_ids_offset_ -= static_cast<uint32_t>(id_size);
  IndexFormat::SegmentMeta *segment = segment_start_ + footer_->segment_count;
  segment->segment_id_offset = segment_ids_offset_;
  segment->data_index =
      index_size_ - header_->content_offset - current_header_start_offset_;
  segment->data_size = 0;
  segment->data_crc = 0;
  segment->padding_size = size;
  memcpy((uint8_t *)segment_start_ + segment_ids_offset_, id.c_str(), id_size);
  index_size_ += size;

  // Update index footer
  footer_->segments_meta_crc =
      ailego::Crc32c::Hash(segment_start_, footer_->segments_meta_size, 0);
  footer_->segment_count += 1;
  footer_->content_size += size;
  footer_->total_size += size;
  IndexFormat::UpdateMetaFooter(footer_, 0);
  segments_.emplace(
      id, SegmentInfo{Segment{segment}, current_header_start_offset_, header_});
  header_dirty_ = true;
  return 0;
}

IndexMapping::Segment *IndexMapping::map(const std::string &id, bool warmup,
                                         bool locked) {
  auto iter = segments_.find(id);
  if (iter == segments_.end()) {
    return nullptr;
  }
  SegmentInfo &segment_info = iter->second;
  Segment *item = &segment_info.segment;
  if (!item->data()) {
    auto meta = item->meta();
    size_t mapping_size = meta->data_size + meta->padding_size;
    size_t offset = segment_info.segment_header_start_offset +
                    segment_info.segment_header->content_offset +
                    meta->data_index;

    void *addr = nullptr;
    if (!copy_on_write_) {
      int opts = ailego::File::MMAP_SHARED;
      if (huge_page_) {
        opts |= ailego::File::MMAP_HUGE_PAGE;
      }
      addr = ailego::File::MemoryMap(file_.native_handle(), offset,
                                     mapping_size, opts);
    } else {
      size_t file_size = file_.size();
      int opts = ailego::File::MMAP_POPULATE;
      if (huge_page_) {
        opts |= ailego::File::MMAP_HUGE_PAGE;
      }
      if (offset < file_size) {
        ailego_assert(offset + mapping_size <= file_size);
        addr = ailego::File::MemoryMap(file_.native_handle(), offset,
                                       mapping_size, opts);
      } else {
        addr = ailego::File::MemoryMap(mapping_size, opts);
      }
    }

    if (!addr) {
      LOG_ERROR("Map segment failed, segment id %s", id.c_str());
      return nullptr;
    }
    item->set_data(addr);

    // Lock memory
    if (locked) {
      ailego::File::MemoryLock(item->data(), mapping_size);
    }
    // Warmup memory
    if (warmup && meta->data_size) {
      ailego::File::MemoryWarmup(item->data(), meta->data_size);
    }
  }
  return item;
}

void IndexMapping::unmap(const std::string &id) {
  auto iter = segments_.find(id);
  if (iter != segments_.end()) {
    SegmentInfo &segment_info = iter->second;
    Segment *item = &segment_info.segment;

    if (item->data()) {
      ailego::File::MemoryUnmap(
          item->data(), item->meta()->data_size + item->meta()->padding_size);
      item->set_data(nullptr);
    }
  }
}

void IndexMapping::unmap_all(void) {
  for (auto iter = segments_.begin(); iter != segments_.end(); ++iter) {
    SegmentInfo &segment_info = iter->second;
    Segment *item = &segment_info.segment;

    if (item->data()) {
      ailego::File::MemoryUnmap(
          item->data(), item->meta()->data_size + item->meta()->padding_size);
      item->set_data(nullptr);
    }
  }
}

int IndexMapping::flush(void) {
  if ((file_.size() < index_size_) && !file_.truncate(index_size_)) {
    LOG_ERROR("Failed to truncate file size %zu, errno %d, %s", index_size_,
              errno, std::strerror(errno));
    return IndexError_TruncateFile;
  }

  for (auto iter = segments_.begin(); iter != segments_.end(); ++iter) {
    SegmentInfo &segment_info = iter->second;
    Segment *item = &segment_info.segment;
    if (!item->data() || !item->dirty()) {
      continue;
    }

    size_t segment_size = item->meta()->data_size + item->meta()->padding_size;
    if (full_mode_ && copy_on_write_) {
      size_t off = segment_info.segment_header_start_offset +
                   segment_info.segment_header->content_offset +
                   item->meta()->data_index;
      if (file_.write(off, item->data(), segment_size) != segment_size) {
        LOG_ERROR("Failed to write segment, size %zu, errno %d, %s",
                  segment_size, errno, std::strerror(errno));
        return IndexError_WriteData;
      }
    } else {
      ailego::File::MemoryFlush(item->data(), segment_size);
    }
    item->reset_dirty();
  }

  if (!header_dirty_) {
    return 0;
  }

  header_dirty_ = false;
  if (full_mode_ && copy_on_write_) {
    for (auto item : header_addr_map_) {
      auto header_start_offset = item.first;
      auto header = item.second;
      if (file_.write(header_start_offset, header, header->content_offset) !=
          header->content_offset) {
        LOG_ERROR("Failed to write segment, size %lu, errno %d, %s",
                  header->content_offset, errno, std::strerror(errno));
        return IndexError_WriteData;
      }
    }
  } else {
    for (auto item : header_addr_map_) {
      auto header = item.second;
      ailego::File::MemoryFlush(header, header->content_offset);
    }
  }
  return 0;
}

int IndexMapping::init_index_mapping(size_t len) {
  int opts =
      copy_on_write_ ? ailego::File::MMAP_POPULATE : ailego::File::MMAP_SHARED;
  if (huge_page_) {
    opts |= ailego::File::MMAP_HUGE_PAGE;
  }
  uint8_t *start = reinterpret_cast<uint8_t *>(ailego::File::MemoryMap(
      file_.native_handle(), current_header_start_offset_, len, opts));
  if (!start) {
    LOG_ERROR("Failed to map file, errno %d, %s", errno, std::strerror(errno));
    return IndexError_MMapFile;
  }

  // Unpack header
  header_ = reinterpret_cast<IndexFormat::MetaHeader *>(start);
  header_addr_map_.insert({current_header_start_offset_, header_});
  if (header_->meta_header_size != sizeof(IndexFormat::MetaHeader)) {
    return IndexError_InvalidLength;
  }
  if (ailego::Crc32c::Hash(header_, sizeof(*header_), header_->header_crc) !=
      header_->header_crc) {
    return IndexError_InvalidChecksum;
  }

  switch (header_->version) {
    case IndexFormat::FORMAT_VERSION:
      break;
    default:
      LOG_ERROR("Unsupported index version: %u", header_->version);
      return IndexError_Unsupported;
  }

  // Unpack footer
  if (header_->meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
    return IndexError_InvalidLength;
  }
  if ((int32_t)header_->meta_footer_offset < 0) {
    return IndexError_Unsupported;
  }
  size_t footer_offset = header_->meta_footer_offset;
  if (footer_offset + header_->meta_footer_size > len) {
    return IndexError_InvalidLength;
  }

  footer_ = reinterpret_cast<IndexFormat::MetaFooter *>(start + footer_offset);
  if (footer_offset < footer_->segments_meta_size) {
    return IndexError_InvalidLength;
  }

  index_size_ = file_.size();
  if ((footer_->total_size > index_size_) ||
      (footer_->content_size + footer_->content_padding_size +
           header_->content_offset >
       index_size_)) {
    return IndexError_InvalidLength;
  }
  if (ailego::Crc32c::Hash(footer_, sizeof(*footer_), footer_->footer_crc) !=
      footer_->footer_crc) {
    return IndexError_InvalidChecksum;
  }

  // Unpack segment table
  if (sizeof(IndexFormat::SegmentMeta) * footer_->segment_count >
      footer_->segments_meta_size) {
    return IndexError_InvalidLength;
  }

  segment_start_ = reinterpret_cast<IndexFormat::SegmentMeta *>(
      start + (footer_offset - footer_->segments_meta_size));
  if (ailego::Crc32c::Hash(segment_start_, footer_->segments_meta_size, 0u) !=
      footer_->segments_meta_crc) {
    LOG_ERROR("Index segments meta checksum is invalid.");
    return IndexError_InvalidChecksum;
  }

  segment_ids_offset_ = footer_->segments_meta_size;
  for (IndexFormat::SegmentMeta *iter = segment_start_,
                                *end = segment_start_ + footer_->segment_count;
       iter != end; ++iter) {
    if (iter->segment_id_offset > footer_->segments_meta_size) {
      return IndexError_InvalidValue;
    }
    if (iter->data_index > footer_->content_size) {
      return IndexError_InvalidValue;
    }
    if (iter->data_index + iter->data_size > footer_->content_size) {
      return IndexError_InvalidLength;
    }

    if (iter->segment_id_offset < segment_ids_offset_) {
      segment_ids_offset_ = iter->segment_id_offset;
    }
    segments_.emplace(
        std::string(reinterpret_cast<const char *>(segment_start_) +
                    iter->segment_id_offset),
        SegmentInfo{Segment{iter}, current_header_start_offset_, header_});
  }
  if (sizeof(IndexFormat::SegmentMeta) * footer_->segment_count >
      segment_ids_offset_) {
    return IndexError_InvalidLength;
  }

  // if (header_->version == IndexFormat::COMPATIBLE_FORMAT_VERSION_0X0002) {
  //   header_->version = IndexFormat::CURRENT_FORMAT_VERSION;
  //   LOG_INFO("Index file format upgraded");
  //   IndexFormat::UpdateMetaHeader(header_);
  //   footer_->segments_meta_crc =
  //       ailego::Crc32c::Hash(segment_start_, footer_->segments_meta_size, 0);
  //   IndexFormat::UpdateMetaFooter(footer_, 0);
  //   header_dirty_ = true;
  // }

  if (footer_->next_meta_header_offset > 0) {
    current_header_start_offset_ = footer_->next_meta_header_offset;
    // Meta sections have all the same size, so we can use the same size to map
    // the next meta section
    return this->init_index_mapping(len);
  }

  return 0;
}

bool IndexMapping::Ishugetlbfs(const std::string &path) const {
#ifdef __linux__
  struct statfs buf;
  if (statfs(path.c_str(), &buf) != 0) {
    perror("statfs");
    return false;
  }
  return static_cast<unsigned long>(buf.f_type) == HUGETLBFS_MAGIC;
#else
  static_cast<void>(path);
  return false;
#endif
}

}  // namespace core
}  // namespace zvec
