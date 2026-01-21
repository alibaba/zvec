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

#include <map>
#include <zvec/ailego/utility/type_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_format.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace core {

/*! Index Unpacker
 */
class IndexUnpacker {
 public:
  /*! Index Unpacker Segment Meta
   */
  class SegmentMeta {
   public:
    //! Constructor
    SegmentMeta(size_t offset, size_t dsz, size_t psz, uint32_t crc)
        : data_offset_(offset),
          data_size_(dsz),
          padding_size_(psz),
          data_crc_(crc) {}

    //! Retrieve offset of data
    size_t data_offset(void) const {
      return data_offset_;
    }

    //! Retrieve size of data
    size_t data_size(void) const {
      return data_size_;
    }

    //! Retrieve crc of data
    uint32_t data_crc(void) const {
      return data_crc_;
    }

    //! Retrieve size of padding
    size_t padding_size(void) const {
      return padding_size_;
    }

   private:
    size_t data_offset_{0};
    size_t data_size_{0};
    size_t padding_size_{0};
    uint32_t data_crc_{0};
  };

  //! Reset the unpacker
  void reset(void) {
    segments_.clear();
  }

  //! Retrieve segments of index package
  const std::map<std::string, SegmentMeta> &segments(void) const {
    return segments_;
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const {
    return header_.magic;
  }

  //! Retrieve header of index package
  const IndexFormat::MetaHeader &header(void) const {
    return header_;
  }

  //! Retrieve footer of index package
  const IndexFormat::MetaFooter &footer(void) const {
    return footer_;
  }

  //! Retrieve version information
  const std::string &version(void) const {
    return version_;
  }

  //! Retrieve mutable segments of index package
  std::map<std::string, SegmentMeta> *mutable_segments(void) {
    return &segments_;
  }

  //! Unpack index data
  template <typename TFunc>
  bool unpack(TFunc read_data, size_t total, bool checksum) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");

    if (!this->unpack_header(read_data)) {
      LOG_ERROR("Failed to unpack index header");
      return false;
    }
    if (!this->unpack_footer(read_data, total)) {
      LOG_ERROR("Failed to unpack index footer");
      return false;
    }
    if (!this->unpack_segments(read_data, total)) {
      LOG_ERROR("Failed to unpack index segments' meta");
      return false;
    }
    if (checksum && !this->validate_checksum(read_data)) {
      LOG_ERROR("Failed to validate checksum of index content");
      return false;
    }
    if (!this->unpack_version(read_data)) {
      LOG_ERROR("Failed to unpack index version");
      return false;
    }
    return true;
  }

  //! Unpack index header
  template <typename TFunc>
  bool unpack_header(TFunc read_data) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    const void *data = nullptr;
    if (read_data(0u, &data, sizeof(header_)) != sizeof(header_)) {
      return false;
    }

    memcpy(&header_, data, sizeof(header_));
    if (header_.meta_header_size != sizeof(header_)) {
      return false;
    }
    if (ailego::Crc32c::Hash(&header_, sizeof(header_), header_.header_crc) !=
        header_.header_crc) {
      return false;
    }
    return true;
  }

  //! Unpack index footer
  template <typename TFunc>
  bool unpack_footer(TFunc read_data, size_t total) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    if (header_.meta_footer_size != sizeof(footer_)) {
      return false;
    }

    size_t footer_offset = ((int32_t)header_.meta_footer_offset < 0)
                               ? total + (int32_t)header_.meta_footer_offset
                               : header_.meta_footer_offset;
    if (footer_offset + sizeof(footer_) > total) {
      return false;
    }

    const void *data = nullptr;
    if (read_data(footer_offset, &data, sizeof(footer_)) != sizeof(footer_)) {
      return false;
    }

    memcpy(&footer_, data, sizeof(footer_));
    if ((footer_.total_size != total) ||
        (footer_.content_size + footer_.content_padding_size +
             header_.content_offset >
         total)) {
      return false;
    }
    if (ailego::Crc32c::Hash(&footer_, sizeof(footer_), footer_.footer_crc) !=
        footer_.footer_crc) {
      return false;
    }
    return true;
  }

  //! Unpack segments' meta
  template <typename TFunc>
  bool unpack_segments(TFunc read_data, size_t total) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
        footer_.segments_meta_size) {
      return false;
    }
    size_t offset = ((int32_t)header_.meta_footer_offset < 0)
                        ? total + (int32_t)header_.meta_footer_offset
                        : header_.meta_footer_offset;
    if (offset < footer_.segments_meta_size || offset > total) {
      return false;
    }
    offset -= footer_.segments_meta_size;

    const void *data = nullptr;
    if (read_data(offset, &data, footer_.segments_meta_size) !=
        footer_.segments_meta_size) {
      return false;
    }
    if (ailego::Crc32c::Hash(data, footer_.segments_meta_size, 0u) !=
        footer_.segments_meta_crc) {
      return false;
    }

    IndexFormat::SegmentMeta *seg = (IndexFormat::SegmentMeta *)data;
    for (size_t i = 0; i < footer_.segment_count; ++i, ++seg) {
      if (seg->segment_id_offset > footer_.segments_meta_size) {
        return false;
      }
      if (seg->data_index > footer_.content_size) {
        return false;
      }
      if (seg->data_index + seg->data_size > footer_.content_size) {
        return false;
      }
      segments_.emplace(
          std::string(reinterpret_cast<const char *>(data) +
                      seg->segment_id_offset),
          SegmentMeta(seg->data_index + header_.content_offset, seg->data_size,
                      seg->padding_size, seg->data_crc));
    }
    return true;
  }

  //! Unpack index version
  template <typename TFunc>
  bool unpack_version(TFunc read_data) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");

    auto it = segments_.find("IndexVersion");
    if (it == segments_.end()) {
      return false;
    }

    const SegmentMeta &segment = it->second;
    const void *data = nullptr;

    if (read_data(segment.data_offset(), &data, segment.data_size()) !=
        segment.data_size()) {
      return false;
    }
    if (segment.data_crc() != 0u &&
        ailego::Crc32c::Hash(data, segment.data_size(), 0u) !=
            segment.data_crc()) {
      return false;
    }
    version_.assign(reinterpret_cast<const char *>(data), segment.data_size());
    return true;
  }

  //! Validate checksum of content
  template <typename TFunc>
  bool validate_checksum(TFunc read_data) const {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    if (footer_.content_crc == 0) {
      return true;
    }
    const size_t block_size = 4096u;
    const void *data = nullptr;
    uint32_t checksum = 0u;
    size_t total = footer_.content_size;
    size_t offset = sizeof(header_);

    while (total >= block_size) {
      if (read_data(offset, &data, block_size) != block_size) {
        return false;
      }
      checksum = ailego::Crc32c::Hash(data, block_size, checksum);
      total -= block_size;
      offset += block_size;
    }
    if (read_data(offset, &data, total) != total) {
      return false;
    }
    checksum = ailego::Crc32c::Hash(data, total, checksum);
    return (checksum == footer_.content_crc);
  }

 private:
  IndexFormat::MetaHeader header_;
  IndexFormat::MetaFooter footer_;
  std::string version_{};
  std::map<std::string, SegmentMeta> segments_{};
};

}  // namespace core
}  // namespace zvec
