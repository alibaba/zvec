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

#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_format.h>
#include <zvec/core/framework/index_version.h>

namespace zvec {
namespace core {

/*! Index Packer
 */
class IndexPacker {
 public:
  /*! Index Packer Segment Meta
   */
  class SegmentMeta {
   public:
    //! Constructor
    SegmentMeta(const std::string &str, size_t dsz, size_t psz, uint32_t crc)
        : data_size_(dsz), padding_size_(psz), data_crc_(crc), id_(str) {}

    //! Constructor
    SegmentMeta(std::string &&str, size_t dsz, size_t psz, uint32_t crc)
        : data_size_(dsz),
          padding_size_(psz),
          data_crc_(crc),
          id_(std::forward<std::string>(str)) {}

    //! Constructor
    SegmentMeta(const SegmentMeta &rhs)
        : data_size_(rhs.data_size_),
          padding_size_(rhs.padding_size_),
          data_crc_(rhs.data_crc_),
          id_(rhs.id_) {}

    //! Constructor
    SegmentMeta(SegmentMeta &&rhs)
        : data_size_(rhs.data_size_),
          padding_size_(rhs.padding_size_),
          data_crc_(rhs.data_crc_),
          id_(std::move(rhs.id_)) {}

    //! Retrieve id of segment
    const std::string &id(void) const {
      return id_;
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
    size_t data_size_{0};
    size_t padding_size_{0};
    uint32_t data_crc_{0};
    std::string id_{};
  };

  //! Retrieve magic number of index
  uint32_t magic(void) const {
    return magic_;
  }

  //! Reset the packer
  void reset(void) {
    magic_ = 0;
    data_crc_ = 0u;
    data_size_ = 0u;
  }

  //! Setup header of index pacakge
  template <typename TFunc>
  bool setup(TFunc write_data) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, const void *,
                                                size_t>::value,
                  "Invocable function type");
    IndexFormat::MetaHeader header;
    IndexFormat::SetupMetaHeader(
        &header, (uint32_t)(0 - sizeof(IndexFormat::MetaFooter)),
        sizeof(IndexFormat::MetaHeader));

    if (write_data(&header, sizeof(header)) != sizeof(header)) {
      return false;
    }
    magic_ = header.magic;
    return true;
  }

  //! Pack index data
  template <typename TFunc>
  size_t pack(TFunc write_data, const void *data, size_t len) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, const void *,
                                                size_t>::value,
                  "Invocable function type");
    size_t wrlen = write_data(data, len);
    if (wrlen > 0u) {
      data_crc_ = ailego::Crc32c::Hash(data, wrlen, data_crc_);
      data_size_ += wrlen;
    }
    return wrlen;
  }

  //! Finish packing data
  template <typename TFunc>
  bool finish(TFunc write_data, std::vector<SegmentMeta> &stab) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, const void *,
                                                size_t>::value,
                  "Invocable function type");

    size_t content_size = 0u;
    for (const auto &it : stab) {
      content_size += it.data_size() + it.padding_size();
    }

    if (content_size != data_size_) {
      return false;
    }

    if (!this->pack_version(write_data, stab)) {
      return false;
    }

    // Write the padding if need
    size_t content_padding_size = ailego_align(data_size_, 32) - data_size_;
    if (content_padding_size) {
      std::string padding(content_padding_size, '\0');

      if (write_data(padding.data(), padding.size()) != padding.size()) {
        return false;
      }
    }

    // Prepare segment meta buffer
    IndexFormat::SegmentMetaBuffer buffer(stab.size());
    for (const auto &it : stab) {
      buffer.append(it.id(), it.data_size(), it.padding_size(), it.data_crc());
    }
    buffer.resize(ailego_align(buffer.size(), 32));

    // Write segment table into file
    if (write_data(buffer.data(), buffer.size()) != buffer.size()) {
      return false;
    }

    // Update footer
    IndexFormat::MetaFooter footer;
    IndexFormat::SetupMetaFooter(&footer);
    footer.segments_meta_crc = buffer.crc();
    footer.content_crc = data_crc_;
    footer.segment_count = stab.size();
    footer.segments_meta_size = buffer.size();
    footer.content_size = data_size_;
    footer.content_padding_size = content_padding_size;
    footer.total_size = footer.content_size + footer.content_padding_size +
                        footer.segments_meta_size +
                        sizeof(IndexFormat::MetaHeader) +
                        sizeof(IndexFormat::MetaFooter);
    IndexFormat::UpdateMetaFooter(&footer, 0);

    // Write footer into file
    if (write_data(&footer, sizeof(footer)) != sizeof(footer)) {
      return false;
    }
    return true;
  }

  //! Pack index version
  template <typename TFunc>
  bool pack_version(TFunc write_data, std::vector<SegmentMeta> &stab) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, const void *,
                                                size_t>::value,
                  "Invocable function type");
    std::string buffer(IndexVersion::Details());

    size_t data_size = buffer.size();
    uint32_t data_crc = ailego::Crc32c::Hash(buffer.data(), buffer.size(), 0);
    buffer.resize((data_size + 31u) & ~31u);

    if (write_data(buffer.data(), buffer.size()) != buffer.size()) {
      return false;
    }
    data_crc_ = ailego::Crc32c::Hash(buffer.data(), buffer.size(), data_crc_);
    data_size_ += buffer.size();
    stab.emplace_back(std::string("IndexVersion"), data_size,
                      buffer.size() - data_size, data_crc);
    return true;
  }

 private:
  uint32_t magic_{0u};
  uint32_t data_crc_{0u};
  size_t data_size_{0u};
};

}  // namespace core
}  // namespace zvec
