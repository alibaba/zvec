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

#include "diskann_context.h"
#include <chrono>
#include <new>
#include "diskann_params.h"
#include "diskann_pq_table.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

DiskAnnContext::DiskAnnContext(const IndexMeta &meta,
                               const IndexMetric::Pointer &measure,
                               const DiskAnnEntity::Pointer &entity)
    : IndexContext(measure),
      dc_(entity.get(), measure, meta.dimension()),
      entity_{entity} {}

DiskAnnContext::Pointer DiskAnnContext::create_fetch_context(
    const IndexMeta &meta, const IndexMetric::Pointer &measure,
    const DiskAnnEntity::Pointer &entity) {
  if (!measure || !entity) {
    return nullptr;
  }

  Pointer context(new (std::nothrow) DiskAnnContext(meta, measure, entity));
  if (!context ||
      context->init(kFetchContext, entity->max_degree(), entity->pq_chunk_num(),
                    meta.element_size()) != 0) {
    return nullptr;
  }
  return context;
}

int DiskAnnContext::resize_fetch_sector_buffer(
    const DiskAnnEntity::Pointer &entity) {
  if (!entity) {
    LOG_ERROR("Cannot size a DiskAnn fetch buffer without an entity");
    return IndexError_InvalidArgument;
  }

  const uint64_t sector_num_per_node =
      entity->node_per_sector() > 0
          ? 1
          : DiskAnnUtil::div_round_up(entity->max_node_size(),
                                      DiskAnnUtil::kSectorSize);
  if (sector_num_per_node == 0 ||
      sector_num_per_node > DiskAnnUtil::kMaxSectorReadNum) {
    LOG_ERROR("Invalid DiskAnn fetch sector count: %lu",
              static_cast<unsigned long>(sector_num_per_node));
    return IndexError_InvalidArgument;
  }

  const size_t required_size =
      static_cast<size_t>(sector_num_per_node) * DiskAnnUtil::kSectorSize;
  if (sector_buffer_ != nullptr && sector_buffer_size_ == required_size) {
    return 0;
  }

  void *replacement = nullptr;
  DiskAnnUtil::alloc_aligned(&replacement, required_size,
                             DiskAnnUtil::kSectorSize);
  if (!replacement) {
    LOG_ERROR("Failed to allocate DiskAnn fetch buffer");
    return IndexError_NoMemory;
  }

  DiskAnnUtil::free_aligned(sector_buffer_);
  sector_buffer_ = replacement;
  sector_buffer_size_ = required_size;
  return 0;
}

int DiskAnnContext::init(ContextType type, uint32_t graph_degree,
                         uint32_t pq_chunk_num, uint32_t element_size) {
  if (!entity_ || element_size == 0) {
    LOG_ERROR("Invalid DiskAnn context parameters");
    return IndexError_InvalidArgument;
  }
  type_ = type;
  element_size_ = element_size;
  pq_chunk_num_ = pq_chunk_num;

  if (type != kFetchContext) {
    DiskAnnUtil::alloc_aligned((void **)&query_, element_size_, 32);
    DiskAnnUtil::alloc_aligned((void **)&query_rotated_, element_size_, 32);
    if (!query_ || !query_rotated_) {
      LOG_ERROR("Failed to allocate DiskAnn query buffers");
      return IndexError_NoMemory;
    }
  }

  int ret;
  switch (type) {
    case kBuilderContext:
      ret = visit_filter_.init(VisitFilter::ByteMap, entity_->doc_cnt(),
                               entity_->doc_cnt(), negative_probility_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }
      break;

    case kSearcherContext:
      if (graph_degree == 0 || pq_chunk_num_ == 0) {
        LOG_ERROR("Invalid DiskAnn search context dimensions");
        return IndexError_InvalidArgument;
      }

      ret = visit_filter_.init(filter_mode_, entity_->doc_cnt(),
                               entity_->doc_cnt(), negative_probility_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }

      DiskAnnUtil::alloc_aligned((void **)&pq_table_dist_buffer_,
                                 static_cast<size_t>(PQTable::kPQCentroidNum) *
                                     pq_chunk_num_ * sizeof(float),
                                 256);
      DiskAnnUtil::alloc_aligned(
          (void **)&pq_coord_buffer_,
          static_cast<size_t>(graph_degree) * pq_chunk_num_ * sizeof(uint8_t),
          256);
      DiskAnnUtil::alloc_aligned((void **)&coord_buffer_, element_size_, 256);
      sector_buffer_size_ = static_cast<size_t>(DiskAnnUtil::kMaxSectorReadNum *
                                                DiskAnnUtil::kSectorSize);
      DiskAnnUtil::alloc_aligned((void **)&sector_buffer_, sector_buffer_size_,
                                 DiskAnnUtil::kSectorSize);
      if (!pq_table_dist_buffer_ || !pq_coord_buffer_ || !coord_buffer_ ||
          !sector_buffer_) {
        LOG_ERROR("Failed to allocate DiskAnn search buffers");
        return IndexError_NoMemory;
      }

      ret = setup_io_ctx(io_ctx_);
      if (ret != 0) {
        LOG_ERROR("setup io ctx error, ret=%d", ret);
        return ret;
      }
      break;

    case kFetchContext:
      ret = resize_fetch_sector_buffer(entity_);
      if (ret != 0) {
        return ret;
      }

      ret = setup_io_ctx(io_ctx_);
      if (ret != 0) {
        LOG_ERROR("setup fetch io ctx error, ret=%d", ret);
        return ret;
      }
      break;

    default:
      LOG_ERROR("Init context failed");
      return IndexError_Runtime;
  }

  return 0;
}

DiskAnnContext::~DiskAnnContext() {
  // The sector buffer may still be the destination of an overlapped read if a
  // query exits early. Cancel and wait for every request before releasing any
  // memory that the I/O context can reference.
  if (type_ == kSearcherContext || type_ == kFetchContext) {
    destroy_io_ctx(io_ctx_);
  }

  visit_filter_.destroy();
  DiskAnnUtil::free_aligned(query_);
  DiskAnnUtil::free_aligned(query_rotated_);
  DiskAnnUtil::free_aligned(pq_table_dist_buffer_);
  DiskAnnUtil::free_aligned(pq_coord_buffer_);
  DiskAnnUtil::free_aligned(coord_buffer_);
  DiskAnnUtil::free_aligned(sector_buffer_);
}

int DiskAnnContext::update(const ailego::Params &params) {
  uint32_t list_size = list_size_;
  params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size);
  list_size_ = list_size;
  return 0;
}

int DiskAnnContext::update_context(ContextType type, const IndexMeta &meta,
                                   const IndexMetric::Pointer &measure,
                                   const DiskAnnEntity::Pointer &entity,
                                   uint32_t magic_num) {
  if (ailego_unlikely(type != static_cast<ContextType>(type_))) {
    LOG_ERROR(
        "DiskAnnContext does not support shared by different type, "
        "src=%u dst=%u",
        type_, type);
    return IndexError_Unsupported;
  }

  magic_ = kInvalidMgic;

  switch (type) {
    case kBuilderContext:
      LOG_ERROR("BuildContext does not support update");
      return IndexError_NotImplemented;

    case kSearcherContext:
      break;

    case kFetchContext: {
      const int ret = resize_fetch_sector_buffer(entity);
      if (ret != 0) {
        return ret;
      }
      break;
    }

    case kReducerContext:
      break;

    default:
      LOG_ERROR("update context failed");
      return IndexError_Runtime;
  }

  entity_ = entity;
  update_index_metric(measure);
  dc_.update(entity_.get(), measure, meta.dimension());
  magic_ = magic_num;

  return 0;
}

}  // namespace core
}  // namespace zvec
