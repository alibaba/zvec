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

#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include "diskann_entity.h"
#include "diskann_file_reader.h"

namespace zvec {
namespace core {

class DiskAnnSearcherEntity : public DiskAnnEntity {
 public:
  using Pointer = std::shared_ptr<DiskAnnSearcherEntity>;
  using SegmentPointer = IndexStorage::Segment::Pointer;

 public:
  DiskAnnSearcherEntity() = default;
  virtual ~DiskAnnSearcherEntity() = default;

 public:
  const DiskAnnEntity::Pointer clone() const override;

  void clear();
  void release_storage();
  int load(const IndexMeta &meta, IndexStorage::Pointer storage);
  int load_pq_segment();
  int load_header_segment();
  int load_vector_segment();
  int load_key_segment();
  int load_key_mapping_segment();
  int load_entrypoint_segment();

  //! Read the serialized PQ quantizer meta buffer from the PQ meta segment.
  //! The quantizer itself is constructed by the searcher/streamer and handed
  //! to the indexer; the entity only owns the persisted bytes.  For a legacy
  //! layout the raw codebook is returned instead (see DiskAnnUtil).
  int read_pq_quantizer_meta_buffer(std::string *meta_buffer) const;

  const uint8_t *pq_codes() const {
    return pq_codes_ ? reinterpret_cast<const uint8_t *>(pq_codes_->data())
                     : nullptr;
  }

  IndexStorage::Pointer get_storage() {
    return storage_;
  }

  SegmentPointer get_vector_segment() {
    return vector_segment_;
  }

  std::vector<diskann_id_t> &entrypoints() {
    return entrypoints_;
  }

  diskann_id_t get_id(diskann_key_t key) const override;
  diskann_key_t get_key(diskann_id_t id) const override;

 private:
  //! Interpret the PQ meta header just read into pq_meta_.  Legacy indexes
  //! carry DiskAnnLegacyPqMeta in the same bytes; on success pq_meta_ holds the
  //! normalized chunk count and payload size for both layouts.
  int normalize_pq_meta();

 private:
  IndexStorage::Pointer storage_{};

  SegmentPointer meta_segment_{nullptr};
  SegmentPointer pq_meta_segment_{nullptr};
  SegmentPointer pq_data_segment_{nullptr};
  SegmentPointer vector_segment_{nullptr};
  SegmentPointer key_segment_{nullptr};
  SegmentPointer key_mapping_segment_{nullptr};
  SegmentPointer entrypoint_segment_{nullptr};

  IndexMeta meta_;

  //! Shared so that clone() stays cheap for every search context.
  std::shared_ptr<const std::string> pq_codes_;
  std::shared_ptr<const std::string> key_buffer_;
  std::shared_ptr<const std::string> key_mapping_buffer_;
  std::vector<diskann_id_t> entrypoints_;
};

}  // namespace core
}  // namespace zvec
