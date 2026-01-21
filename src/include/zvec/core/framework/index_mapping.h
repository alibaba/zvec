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
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_format.h>

namespace zvec {
namespace core {

/*! Index Mapping
 */
class IndexMapping {
 public:
  /*! Index Mapping Segment
   */
  class Segment {
   public:
    //! Constructor
    Segment(void) {}

    //! Constructor
    Segment(IndexFormat::SegmentMeta *segmeta) : meta_(segmeta) {}

    //! Flush the segment
    bool flush(void) const {
      ailego_false_if_false(this->meta_ && this->data_);
      return ailego::File::MemoryFlush(
          this->data_, this->meta_->data_size + this->meta_->padding_size);
    }

    //! Retrieve mapping address of the segment
    void *data(void) const {
      return data_;
    }

    //! Retrieve pointer of SegmentMeta
    IndexFormat::SegmentMeta *meta(void) const {
      return meta_;
    }

    //! Retrieve dirty flag of the segment
    bool dirty(void) const {
      return dirty_;
    }

    //! Set the segment as dirty
    void set_dirty(void) const {
      dirty_ = true;
    }

   private:
    friend class IndexMapping;

    //! Set the mapping address of the segment
    void set_data(void *addr) {
      data_ = addr;
    }

    //! Clear the dirty flag
    void reset_dirty(void) const {
      dirty_ = false;
    }

   private:
    //! Members
    IndexFormat::SegmentMeta *meta_{nullptr};
    void *data_{nullptr};
    mutable bool dirty_{false};
  };

  //! Constructor
  IndexMapping(void) {}

  //! Constructor
  IndexMapping(IndexMapping &&rhs)
      : segment_ids_offset_(rhs.segment_ids_offset_),
        segment_start_(rhs.segment_start_),
        header_(rhs.header_),
        footer_(rhs.footer_),
        segments_(std::move(rhs.segments_)),
        file_(std::move(rhs.file_)) {
    rhs.segment_ids_offset_ = 0;
    rhs.segment_start_ = nullptr;
    rhs.header_ = nullptr;
    rhs.footer_ = nullptr;
  }

  //! Assignment
  IndexMapping &operator=(IndexMapping &&rhs) {
    segment_ids_offset_ = rhs.segment_ids_offset_;
    segment_start_ = rhs.segment_start_;
    header_ = rhs.header_;
    footer_ = rhs.footer_;
    segments_ = std::move(rhs.segments_);
    file_ = std::move(rhs.file_);
    rhs.segment_ids_offset_ = 0;
    rhs.segment_start_ = nullptr;
    rhs.header_ = nullptr;
    rhs.footer_ = nullptr;
    return *this;
  }

  //! Open a index file
  int open(const std::string &path, bool cow, bool full_mode);

  //! Create a index file
  int create(const std::string &path, size_t segs_size);

  int create_hugepage(size_t segs_size);

  //! Close the index
  void close(void);

  //! Refresh meta information (checksum, update time, etc.)
  void refresh(uint64_t check_point);

  //! Append a segment into index
  int append(const std::string &id, size_t size);

  //! Map a segment by id
  Segment *map(const std::string &id, bool warmup, bool lock);

  //! Unmap a segment by id
  void unmap(const std::string &id);

  //! Unmap all segments
  void unmap_all(void);

  //! Flush the index mapping
  int flush(void);

  //! Test if the segment is exist
  bool has(const std::string &id) const {
    return (segments_.find(id) != segments_.end());
  }

  //! Retrieve count of segments
  size_t segment_count(void) const {
    return segments_.size();
  }

  //! Retrieve size of index mapping
  size_t index_size(void) const {
    return index_size_;
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const {
    return (header_ ? header_->magic : 0);
  }

  //! Retrieve header information
  const IndexFormat::MetaHeader &header(void) const {
    return *header_;
  }

  //! Retrieve footer information
  const IndexFormat::MetaFooter &footer(void) const {
    return *footer_;
  }

  bool huge_page() const {
    return huge_page_;
  }

 protected:
  //! Initialize index file mapping
  int init_index_mapping(size_t len);

  bool Ishugetlbfs(const std::string &path) const;

 private:
  //! Disable them
  IndexMapping(const IndexMapping &) = delete;
  IndexMapping &operator=(const IndexMapping &) = delete;

  //! Members
  uint32_t segment_ids_offset_{0};
  IndexFormat::SegmentMeta *segment_start_{nullptr};
  IndexFormat::MetaHeader *header_{nullptr};
  IndexFormat::MetaFooter *footer_{nullptr};
  std::map<std::string, Segment> segments_{};
  size_t index_size_{0u};
  ailego::File file_{};
  bool copy_on_write_{false};
  bool full_mode_{false};
  bool header_dirty_{false};
  bool huge_page_{false};
};

}  // namespace core
}  // namespace zvec
