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

#include <cstring>
#include <random>
#include <string>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/utility/time_helper.h>

namespace zvec {
namespace core {

/*! Index Format
 */
struct IndexFormat {
  /*! Version Number
   */
  enum { FORMAT_VERSION = 0x0002 };

  /*! Index Format Meta Header
   */
  struct MetaHeader {
    uint32_t header_crc;
    uint16_t reserved1_;
    uint16_t version;
    uint32_t revision;
    uint32_t magic;
    uint16_t meta_header_size;
    uint16_t meta_footer_size;
    uint32_t meta_footer_offset;
    uint32_t content_offset;
    uint32_t reserved2_;
    uint64_t setup_time;
    uint64_t reserved3_[3];
  };

  static_assert(sizeof(MetaHeader) % 32 == 0,
                "MetaHeader must be aligned with 32 bytes");

  /*! Index Format Meta Footer
   */
  struct MetaFooter {
    uint32_t footer_crc;
    uint32_t segments_meta_crc;
    uint32_t content_crc;
    uint32_t segment_count;
    uint32_t segments_meta_size;
    uint32_t reserved1_;
    uint64_t content_size;
    uint64_t content_padding_size;
    uint64_t check_point;
    uint64_t update_time;
    uint64_t reserved2_[8];
    uint64_t total_size;
  };

  static_assert(sizeof(MetaFooter) % 32 == 0,
                "MetaFooter must be aligned with 32 bytes");

  /*! Index Format Segment Meta
   */
  struct SegmentMeta {
    uint32_t segment_id_offset;
    uint32_t data_crc;
    uint64_t data_index;
    uint64_t data_size;
    uint64_t padding_size;
  };

  static_assert(sizeof(SegmentMeta) % 32 == 0,
                "SegmentMeta must be aligned with 32 bytes");

  /*! Index Format Segment Meta Buffer
   */
  class SegmentMetaBuffer {
   public:
    //! Constructor
    SegmentMetaBuffer(uint32_t count) : capacity_(count) {
      buffer_.clear();
      buffer_.resize(sizeof(SegmentMeta) * capacity_);
    }

    //! Append a segment meta into buffer
    bool append(const std::string &id, size_t data_size, size_t padding_size,
                uint32_t data_crc) {
      if (count_ >= capacity_) {
        return false;
      }
      SegmentMeta *meta = (SegmentMeta *)buffer_.data() + count_;
      meta->segment_id_offset = static_cast<uint32_t>(buffer_.size());
      meta->data_index = offset_;
      meta->data_size = data_size;
      meta->data_crc = data_crc;
      meta->padding_size = padding_size;
      buffer_.append(id.c_str(), std::strlen(id.c_str()) + 1);
      count_ += 1;
      offset_ += data_size + padding_size;
      return true;
    }

    //! Resize the buffer
    void resize(size_t val) {
      buffer_.resize(val);
    }

    //! Retrieve pointer of data
    const void *data(void) const {
      return buffer_.data();
    }

    //! Retrieve size of data
    size_t size(void) const {
      return buffer_.size();
    }

    //! Retrieve crc of buffer
    uint32_t crc(void) const {
      return ailego::Crc32c::Hash(buffer_.data(), buffer_.size(), 0);
    }

   private:
    //! Disable them
    SegmentMetaBuffer(void) = delete;

    //! Members
    std::string buffer_{};
    size_t offset_{0u};
    uint32_t capacity_{0u};
    uint32_t count_{0u};
  };

  //! Setup meta header structure
  static void SetupMetaHeader(MetaHeader *header, uint32_t footer_offset,
                              uint32_t content_offset) {
    memset(header, 0, sizeof(MetaHeader));
    header->version = IndexFormat::FORMAT_VERSION;
    header->revision = 0;
    header->magic = std::random_device()();
    header->meta_header_size = sizeof(MetaHeader);
    header->meta_footer_size = sizeof(MetaFooter);
    header->meta_footer_offset = footer_offset;
    header->content_offset = content_offset;
    header->setup_time = ailego::Realtime::Seconds();
    header->header_crc = ailego::Crc32c::Hash(header, sizeof(MetaHeader), 0);
  }

  //! Setup meta footer structure
  static void SetupMetaFooter(MetaFooter *footer) {
    memset(footer, 0, sizeof(MetaFooter));
  }

  //! Update meta footer structure
  static void UpdateMetaFooter(MetaFooter *footer, uint64_t check_point) {
    if (check_point != 0) {
      footer->check_point = check_point;
    }
    footer->update_time = ailego::Realtime::Seconds();
    footer->footer_crc = 0;
    footer->footer_crc = ailego::Crc32c::Hash(footer, sizeof(MetaFooter), 0);
  }
};

}  // namespace core
}  // namespace zvec
