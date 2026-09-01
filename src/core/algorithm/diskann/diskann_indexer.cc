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

#include "diskann_indexer.h"
#include <algorithm>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <tuple>
#include <unordered_set>
#include <zvec/ailego/io/file.h>

namespace zvec {
namespace core {

namespace {

// DiskAnnContext instances are pooled above the indexer and can therefore
// outlive the reader that prepared their lazy Windows file handle. Keep that
// handle across all I/O batches in one logical operation, then release it on
// every return path (including exceptions). POSIX readers intentionally keep
// their backend queue resources for reuse.
class IOContextReleaseGuard {
 public:
  IOContextReleaseGuard(AlignedFileReader &reader, IOContext &ctx,
                        bool enabled = true)
      : reader_(reader), ctx_(ctx), enabled_(enabled) {}

  ~IOContextReleaseGuard() {
    if (enabled_) {
      reader_.release_io_ctx(ctx_);
    }
  }

  IOContextReleaseGuard(const IOContextReleaseGuard &) = delete;
  IOContextReleaseGuard &operator=(const IOContextReleaseGuard &) = delete;

 private:
  AlignedFileReader &reader_;
  IOContext &ctx_;
  bool enabled_;
};

}  // namespace

DiskAnnIndexer::DiskAnnIndexer(const IndexMeta &meta) {
  meta_ = meta;
}

DiskAnnIndexer::~DiskAnnIndexer() {
  destroy_io_ctx(init_ctx_);
  if (centroid_data_) {
    DiskAnnUtil::free_aligned(centroid_data_);
  }
  reset_cache_storage();
}

int DiskAnnIndexer::init(DiskAnnSearcherEntity &entity) {
  auto storage = entity.get_storage();
  auto vector_segment = entity.get_vector_segment();
  if (!storage || !vector_segment) {
    LOG_ERROR("DiskAnn storage or vector segment is missing");
    return IndexError_InvalidFormat;
  }

  const uint64_t stored_max_node_size = entity.max_node_size();
  if (stored_max_node_size == 0 ||
      stored_max_node_size > (std::numeric_limits<uint32_t>::max)() ||
      stored_max_node_size < sizeof(uint32_t) ||
      meta_.element_size() > stored_max_node_size - sizeof(uint32_t)) {
    LOG_ERROR("Invalid DiskAnn node size: node=%llu vector=%u",
              static_cast<unsigned long long>(stored_max_node_size),
              static_cast<unsigned>(meta_.element_size()));
    return IndexError_InvalidFormat;
  }

  const uint64_t stored_max_degree = entity.max_degree();
  const uint64_t neighbor_bytes =
      stored_max_node_size - sizeof(uint32_t) - meta_.element_size();
  if (stored_max_degree > (std::numeric_limits<uint32_t>::max)() ||
      stored_max_degree > neighbor_bytes / sizeof(diskann_id_t)) {
    LOG_ERROR("Invalid DiskAnn node capacity: node=%llu vector=%u degree=%llu",
              static_cast<unsigned long long>(stored_max_node_size),
              static_cast<unsigned>(meta_.element_size()),
              static_cast<unsigned long long>(stored_max_degree));
    return IndexError_InvalidFormat;
  }

  const uint64_t expected_node_per_sector =
      stored_max_node_size <= DiskAnnUtil::kSectorSize
          ? DiskAnnUtil::kSectorSize / stored_max_node_size
          : 0;
  if (entity.node_per_sector() != expected_node_per_sector) {
    LOG_ERROR(
        "Invalid DiskAnn node layout: node=%llu nodes_per_sector=%llu "
        "expected=%llu",
        static_cast<unsigned long long>(stored_max_node_size),
        static_cast<unsigned long long>(entity.node_per_sector()),
        static_cast<unsigned long long>(expected_node_per_sector));
    return IndexError_InvalidFormat;
  }

  auto cached_file = storage->file();
#if defined(_WIN32) || defined(_WIN64)
  // Windows DiskAnn must be able to close the single buffered handle before
  // opening its unbuffered IOCP handles.  FileReadStorage's
  // alone_file_handle mode gives every Segment an independent handle, which
  // cannot be closed through IndexStorage and may be retained by the caller.
  if (!cached_file) {
    LOG_ERROR(
        "DiskAnn on Windows requires FileReadStorage with "
        "proxima.file.read_storage.alone_file_handle disabled");
    return IndexError_InvalidArgument;
  }
#endif

  max_node_size_ = static_cast<uint32_t>(stored_max_node_size);
  sector_num_per_node_ =
      DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);
  if (sector_num_per_node_ == 0 ||
      beam_width_ > DiskAnnUtil::kMaxSectorReadNum / sector_num_per_node_) {
    LOG_ERROR("DiskAnn node size exceeds the search buffer capacity");
    return IndexError_InvalidArgument;
  }

  pq_table_ = entity.get_pq_table();
  entity_ = entity.clone();
  if (!entity_) {
    LOG_ERROR("Failed to clone in-memory DiskAnn entity");
    return IndexError_NoMemory;
  }

  index_segment_offset_ = vector_segment->data_offset();

  const auto file_path = storage->file_path();
  int ret = 0;
  reader_.reset(new PlatformAlignedFileReader());
#if defined(_WIN32) || defined(_WIN64)
  // Drop every Segment reference created by entity.load() before checking the
  // File control block. Without an external alias, only cached_file and the
  // FileReadStorage itself remain as owners.
  entity.release_storage();
  vector_segment.reset();
  if (cached_file.use_count() != 2) {
    LOG_ERROR(
        "DiskAnn on Windows cannot load while the caller retains the "
        "FileReadStorage file or one of its segments");
    return IndexError_InvalidArgument;
  }

  // Capture the exact file object that supplied the in-memory metadata before
  // releasing FileReadStorage. Reopening file_path after cleanup could bind
  // graph reads to a replacement file while PQ/keys still belong to the old
  // one.
  ret = static_cast<WindowsAlignedFileReader *>(reader_.get())
            ->open_from_handle(file_path, cached_file->native_handle());
#else
  if (cached_file) {
    // POSIX atomic replacement leaves an open descriptor bound to the old
    // inode. Capture an independent descriptor before cleanup so graph reads
    // use the same file object that supplied the in-memory metadata.
    ret = static_cast<LinuxAlignedFileReader *>(reader_.get())
              ->open_from_handle(file_path, cached_file->native_handle());
  } else {
    // Preserve support for FileReadStorage's alone_file_handle mode. Its
    // Segment abstraction does not expose a descriptor, so retain the
    // origin/main ordering and bind the path before releasing the storage.
    reader_->open(file_path);
  }
#endif
  if (ret != 0) {
    LOG_ERROR("Failed to capture DiskAnn index file, ret=%d", ret);
    return ret;
  }

  ret = storage->cleanup();
#if !defined(_WIN32) && !defined(_WIN64)
  entity.release_storage();
  vector_segment.reset();
#endif
  storage.reset();
  if (ret != 0) {
    reader_->close();
    LOG_ERROR("Failed to release DiskAnn index storage, ret=%d", ret);
    return ret;
  }

#if defined(_WIN32) || defined(_WIN64)
  // Windows cannot keep an ordinary buffered alias to this file object beside
  // DiskAnn's unbuffered handles without a severe random-read regression. The
  // preflight check above avoids consuming the storage on an ordinary
  // ownership error. Check again after cleanup so an unexpected remaining
  // owner cannot make the successful load retain a buffered handle.
  if (cached_file.use_count() != 1) {
    reader_->close();
    LOG_ERROR(
        "DiskAnn on Windows cannot load while the caller retains the "
        "FileReadStorage file or one of its segments");
    return IndexError_InvalidArgument;
  }
#endif
  // Releasing the last internal reference closes the buffered source handle.
  // POSIX caller-owned aliases remain valid; Windows has rejected them above.
  cached_file.reset();

  ret = setup_io_ctx(init_ctx_);
  if (ret != 0) {
    LOG_ERROR("setup io ctx error");
    return ret;
  }

  disk_bytes_per_point_ = meta_.element_size();

  node_per_sector_ = entity.node_per_sector();
  pq_chunk_num_ = entity.pq_chunk_num();

  medoid_ = entity.medoid();

  entrypoints_.push_back(medoid_);
  auto &entrypoints = entity.entrypoints();
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    entrypoints_.push_back(entrypoints[i]);
  }

  doc_cnt_ = entity.doc_cnt();

  max_degree_ = static_cast<uint32_t>(stored_max_degree);

  centroid_stride_ = DiskAnnUtil::round_up(meta_.element_size(), 32);
  DiskAnnUtil::alloc_aligned(&centroid_data_,
                             entrypoints_.size() * centroid_stride_, 32);
  if (centroid_data_ == nullptr) {
    LOG_ERROR("Failed to allocate entrypoint vector buffer");
    return IndexError_NoMemory;
  }

  ret = use_medroids_data_as_centroids();
  if (ret != 0) {
    return ret;
  }

  return 0;
}

int DiskAnnIndexer::use_medroids_data_as_centroids() {
  LOG_INFO("Loading centroid data from medoid vector data");

  std::vector<void *> entrypoint_buffers(entrypoints_.size());
  std::vector<std::pair<uint32_t, diskann_id_t *>> neighbor_buffers(
      entrypoints_.size(), std::make_pair(0U, nullptr));
  auto *entrypoint_data = static_cast<uint8_t *>(centroid_data_);
  for (size_t i = 0; i < entrypoints_.size(); ++i) {
    entrypoint_buffers[i] = entrypoint_data + i * centroid_stride_;
  }

  auto read_status =
      read_nodes(entrypoints_, entrypoint_buffers, neighbor_buffers);
  if (std::find(read_status.begin(), read_status.end(), false) !=
      read_status.end()) {
    LOG_ERROR("Failed to read one or more entrypoint vectors");
    return IndexError_ReadData;
  }

  return 0;
}

diskann_key_t DiskAnnIndexer::get_key(diskann_id_t id) const {
  return entity_->get_key(id);
}

diskann_id_t DiskAnnIndexer::get_id(diskann_key_t key) const {
  return entity_->get_id(key);
}

std::vector<bool> DiskAnnIndexer::read_nodes(
    const std::vector<diskann_id_t> &node_ids,
    std::vector<void *> &coord_buffers,
    std::vector<std::pair<uint32_t, diskann_id_t *>> &neighbor_buffers) {
  std::vector<bool> retval(node_ids.size(), true);
  if (coord_buffers.size() != node_ids.size() ||
      neighbor_buffers.size() != node_ids.size()) {
    LOG_ERROR(
        "read_nodes: node, coordinate, and neighbor buffer counts must "
        "match");
    std::fill(retval.begin(), retval.end(), false);
    return retval;
  }
  if (node_ids.empty()) {
    return retval;
  }

  std::vector<AlignedRead> read_reqs;
  read_reqs.reserve(node_ids.size());

  uint8_t *buf = nullptr;
  auto sector_num =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);
  DiskAnnUtil::alloc_aligned(
      (void **)&buf, node_ids.size() * sector_num * DiskAnnUtil::kSectorSize,
      DiskAnnUtil::kSectorSize);
  if (buf == nullptr) {
    LOG_ERROR("read_nodes: failed to allocate aligned read buffer");
    std::fill(retval.begin(), retval.end(), false);
    return retval;
  }

  for (size_t i = 0; i < node_ids.size(); ++i) {
    auto node_id = node_ids[i];

    AlignedRead read;
    read.len = sector_num * DiskAnnUtil::kSectorSize;
    read.buf = buf + i * sector_num * DiskAnnUtil::kSectorSize;
    read.offset =
        index_segment_offset_ +
        DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                     DiskAnnUtil::kSectorSize, node_id) *
            DiskAnnUtil::kSectorSize;
    read_reqs.push_back(read);
  }

  int read_ret = reader_->read(read_reqs, init_ctx_);
  if (read_ret != 0) {
    LOG_ERROR("read_nodes: reader_->read failed, ret=%d", read_ret);
    for (size_t i = 0; i < retval.size(); i++) {
      retval[i] = false;
    }
    DiskAnnUtil::free_aligned(buf);
    return retval;
  }

  for (uint32_t i = 0; i < read_reqs.size(); i++) {
    uint8_t *node_buf =
        DiskAnnUtil::offset_to_node(node_per_sector_, max_node_size_,
                                    (uint8_t *)read_reqs[i].buf, node_ids[i]);

    if (coord_buffers[i] != nullptr) {
      void *node_coords = node_buf;
      memcpy(coord_buffers[i], node_coords, disk_bytes_per_point_);
    }

    if (neighbor_buffers[i].second != nullptr) {
      uint32_t *node_neighbor =
          DiskAnnUtil::offset_to_node_neighbor(node_buf, meta_.element_size());
      uint32_t neighbor_num = *node_neighbor;

      if (neighbor_num > max_degree_) {
        LOG_ERROR(
            "read_nodes: node %u has %u neighbors, exceeding max degree %u",
            node_ids[i], neighbor_num, max_degree_);
        retval[i] = false;
        continue;
      }

      neighbor_buffers[i].first = neighbor_num;
      memcpy(neighbor_buffers[i].second, node_neighbor + 1,
             neighbor_num * sizeof(diskann_id_t));
    }
  }

  DiskAnnUtil::free_aligned(buf);

  return retval;
}

void DiskAnnIndexer::reset_cache_storage() {
  // The maps contain pointers into the two backing buffers. Drop the maps
  // first so no stale pointer remains observable while storage is replaced.
  coord_cache_.clear();
  neighbor_cache_.clear();
  DiskAnnUtil::free_aligned(coord_cache_buf_);
  coord_cache_buf_ = nullptr;
  std::vector<diskann_id_t>().swap(neighbor_cache_buffer_);
}

uint32_t DiskAnnIndexer::effective_cache_node_count(
    uint32_t requested_nodes) const {
  uint64_t max_nodes = 0;
  if (doc_cnt_ != 0) {
    max_nodes =
        doc_cnt_ / 10 + (doc_cnt_ % 10 >= 5 ? static_cast<uint64_t>(1) : 0);
    max_nodes = std::max<uint64_t>(1, max_nodes);
  }
  const uint32_t effective_nodes =
      static_cast<uint32_t>(std::min<uint64_t>(requested_nodes, max_nodes));
  if (effective_nodes != requested_nodes) {
    LOG_WARN(
        "Reducing nodes to cache from: %u, to: (10 percent of total nodes: "
        "%u)",
        requested_nodes, effective_nodes);
  }
  return effective_nodes;
}

int DiskAnnIndexer::prepare_cache_storage(size_t capacity,
                                          CacheLoadState &state) {
  reset_cache_storage();
  state = {};
  state.capacity = capacity;

  if (capacity == 0) {
    return 0;
  }

  const uint64_t neighbor_entries_per_node_u64 =
      static_cast<uint64_t>(max_degree_) + 1;
  if (neighbor_entries_per_node_u64 > std::numeric_limits<size_t>::max()) {
    LOG_ERROR("DiskANN node cache neighbor stride overflow");
    return IndexError_InvalidArgument;
  }
  const size_t neighbor_entries_per_node =
      static_cast<size_t>(neighbor_entries_per_node_u64);
  const size_t max_neighbor_entries =
      std::numeric_limits<size_t>::max() / sizeof(diskann_id_t);
  if (capacity > max_neighbor_entries / neighbor_entries_per_node) {
    LOG_ERROR("DiskANN node cache neighbor allocation size overflow");
    return IndexError_InvalidArgument;
  }

  const size_t element_size = meta_.element_size();
  if (element_size == 0 || meta_.unit_size() == 0 ||
      capacity > std::numeric_limits<size_t>::max() / element_size) {
    LOG_ERROR("DiskANN node cache coordinate byte size overflow");
    reset_cache_storage();
    return IndexError_InvalidArgument;
  }
  const size_t coord_cache_bytes = capacity * element_size;

  try {
    state.slots.reserve(capacity);
    neighbor_cache_buffer_.resize(capacity * neighbor_entries_per_node, 0);
  } catch (const std::exception &e) {
    LOG_ERROR("Failed to allocate DiskANN node cache storage: %s", e.what());
    reset_cache_storage();
    state = {};
    return IndexError_NoMemory;
  }

  DiskAnnUtil::alloc_aligned(&coord_cache_buf_, coord_cache_bytes,
                             8 * meta_.unit_size());
  if (coord_cache_buf_ == nullptr) {
    LOG_ERROR("Failed to allocate coordinate cache buffer");
    reset_cache_storage();
    return IndexError_NoMemory;
  }
  return 0;
}

int DiskAnnIndexer::load_cache_list(CacheLoadState &state) {
  LOG_INFO("Loading the remaining cache nodes into memory");

  std::vector<size_t> pending_slots;
  pending_slots.reserve(state.slots.size());
  for (size_t i = 0; i < state.slots.size(); ++i) {
    if (!state.slots[i].loaded) {
      pending_slots.push_back(i);
    }
  }
  std::sort(pending_slots.begin(), pending_slots.end(),
            [&state](size_t lhs, size_t rhs) {
              return state.slots[lhs].id < state.slots[rhs].id;
            });

  const size_t neighbor_entries_per_node = static_cast<size_t>(max_degree_) + 1;
  const size_t batch_size = static_cast<size_t>(
      DiskAnnUtil::cache_load_batch_size(sector_num_per_node_));
  const size_t num_blocks =
      DiskAnnUtil::div_round_up(pending_slots.size(), batch_size);

  std::vector<diskann_id_t> nodes_to_read;
  std::vector<void *> coord_buffers;
  std::vector<std::pair<uint32_t, diskann_id_t *>> neighbor_buffers;
  nodes_to_read.reserve(batch_size);
  coord_buffers.reserve(batch_size);
  neighbor_buffers.reserve(batch_size);

  for (size_t block = 0; block < num_blocks; ++block) {
    const size_t start_idx = block * batch_size;
    const size_t end_idx =
        std::min(pending_slots.size(), (block + 1) * batch_size);

    nodes_to_read.clear();
    coord_buffers.clear();
    neighbor_buffers.clear();
    for (size_t i = start_idx; i < end_idx; ++i) {
      const size_t slot_idx = pending_slots[i];
      nodes_to_read.push_back(state.slots[slot_idx].id);
      coord_buffers.push_back(reinterpret_cast<uint8_t *>(coord_cache_buf_) +
                              slot_idx * meta_.element_size());
      neighbor_buffers.emplace_back(
          0,
          neighbor_cache_buffer_.data() + slot_idx * neighbor_entries_per_node);
    }

    const auto read_status =
        read_nodes(nodes_to_read, coord_buffers, neighbor_buffers);
    for (size_t i = 0; i < read_status.size(); ++i) {
      if (read_status[i]) {
        const size_t slot_idx = pending_slots[start_idx + i];
        state.slots[slot_idx].loaded = true;
        state.slots[slot_idx].neighbor_count = neighbor_buffers[i].first;
      }
    }
  }

  // Publish both maps together only after all optional I/O has completed.
  // Their values point into fixed-capacity buffers that will not move.
  std::vector<size_t> loaded_slots;
  loaded_slots.reserve(state.slots.size());
  for (size_t i = 0; i < state.slots.size(); ++i) {
    if (state.slots[i].loaded) {
      loaded_slots.push_back(i);
    }
  }
  std::sort(loaded_slots.begin(), loaded_slots.end(),
            [&state](size_t lhs, size_t rhs) {
              return state.slots[lhs].id < state.slots[rhs].id;
            });

  try {
    for (size_t slot_idx : loaded_slots) {
      const CacheSlot &slot = state.slots[slot_idx];
      void *coord = reinterpret_cast<uint8_t *>(coord_cache_buf_) +
                    slot_idx * meta_.element_size();
      diskann_id_t *neighbors =
          neighbor_cache_buffer_.data() + slot_idx * neighbor_entries_per_node;
      coord_cache_.emplace_hint(coord_cache_.end(), slot.id, coord);
      neighbor_cache_.emplace_hint(
          neighbor_cache_.end(), slot.id,
          std::make_pair(slot.neighbor_count, neighbors));
    }
  } catch (const std::exception &e) {
    LOG_ERROR("Failed to publish DiskANN node cache: %s", e.what());
    reset_cache_storage();
    return IndexError_NoMemory;
  }

  const size_t failed_nodes = state.slots.size() - loaded_slots.size();
  if (failed_nodes != 0) {
    LOG_WARN(
        "DiskANN node cache preload completed with read failures: "
        "selected_nodes=%zu loaded_nodes=%zu failed_nodes=%zu",
        state.slots.size(), loaded_slots.size(), failed_nodes);
  }

  return 0;
}

int DiskAnnIndexer::configure_cache(uint32_t cache_node_num) {
  cache_node_num = effective_cache_node_count(cache_node_num);
  if (cache_node_num == 0) {
    reset_cache_storage();
    return 0;
  }

  CacheLoadState state;
  int ret = prepare_cache_storage(cache_node_num, state);
  if (ret != 0) {
    return ret;
  }

  ailego::ElapsedTime cache_timer;
  LOG_INFO("Caching %u nodes around medoid(s)", cache_node_num);
  ret = cache_bfs_levels(cache_node_num, state);
  if (ret != 0) {
    reset_cache_storage();
    return ret;
  }
  ret = load_cache_list(state);
  if (ret != 0) {
    return ret;
  }

  const size_t selected_nodes = state.slots.size();
  const size_t loaded_nodes = coord_cache_.size();
  LOG_INFO(
      "Load Cache List Done: requested_nodes=%u selected_nodes=%zu "
      "loaded_nodes=%zu failed_nodes=%zu elapsed_ms=%llu",
      cache_node_num, selected_nodes, loaded_nodes,
      selected_nodes - loaded_nodes,
      static_cast<unsigned long long>(cache_timer.milli_seconds()));
  return 0;
}

int DiskAnnIndexer::cache_bfs_levels(uint64_t num_nodes_to_cache,
                                     CacheLoadState &state) {
  std::set<diskann_id_t> node_set;

  LOG_INFO("Begin to cache %zu Nodes", (size_t)num_nodes_to_cache);

  std::unordered_set<diskann_id_t> cur_level;
  std::unordered_set<diskann_id_t> prev_level;

  for (uint64_t iter = 0;
       iter < entrypoints_.size() && cur_level.size() < num_nodes_to_cache;
       iter++) {
    cur_level.insert(entrypoints_[iter]);
  }

  uint64_t level = 1;
  uint64_t prev_node_set_size = 0;
  while ((node_set.size() + cur_level.size() < num_nodes_to_cache) &&
         cur_level.size() != 0) {
    prev_level.swap(cur_level);

    cur_level.clear();

    std::vector<diskann_id_t> nodes_to_expand;
    nodes_to_expand.reserve(prev_level.size());

    for (const diskann_id_t &id : prev_level) {
      if (node_set.find(id) != node_set.end()) {
        continue;
      }

      node_set.insert(id);
      nodes_to_expand.push_back(id);
    }

    std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

    if (nodes_to_expand.size() > state.capacity - state.slots.size()) {
      LOG_ERROR("DiskANN node cache BFS exceeded its allocated capacity");
      return IndexError_Runtime;
    }
    const size_t first_slot = state.slots.size();
    for (diskann_id_t id : nodes_to_expand) {
      state.slots.push_back(CacheSlot{id, 0, false});
    }

    bool finish_flag = false;

    constexpr uint64_t BLOCK_SIZE = 1024;
    uint64_t nblocks =
        DiskAnnUtil::div_round_up(nodes_to_expand.size(), BLOCK_SIZE);
    for (size_t block = 0; block < nblocks && !finish_flag; block++) {
      size_t start = block * BLOCK_SIZE;
      size_t end = std::min((uint64_t)((block + 1) * BLOCK_SIZE),
                            (uint64_t)(nodes_to_expand.size()));
      const size_t block_size = end - start;

      std::vector<diskann_id_t> nodes_to_read(nodes_to_expand.begin() + start,
                                              nodes_to_expand.begin() + end);
      std::vector<void *> coord_buffers;
      coord_buffers.reserve(block_size);

      std::vector<std::pair<uint32_t, diskann_id_t *>> neighbor_buffers;
      neighbor_buffers.reserve(block_size);
      const size_t neighbor_entries_per_node =
          static_cast<size_t>(max_degree_) + 1;
      for (size_t i = 0; i < block_size; i++) {
        const size_t slot_idx = first_slot + start + i;
        coord_buffers.push_back(reinterpret_cast<uint8_t *>(coord_cache_buf_) +
                                slot_idx * meta_.element_size());
        neighbor_buffers.emplace_back(0,
                                      neighbor_cache_buffer_.data() +
                                          slot_idx * neighbor_entries_per_node);
      }

      const auto read_status =
          read_nodes(nodes_to_read, coord_buffers, neighbor_buffers);

      for (size_t i = 0; i < read_status.size(); i++) {
        if (!read_status[i]) {
          continue;
        }

        const size_t slot_idx = first_slot + start + i;
        state.slots[slot_idx].loaded = true;
        state.slots[slot_idx].neighbor_count = neighbor_buffers[i].first;

        const uint32_t neighbor_num = neighbor_buffers[i].first;
        diskann_id_t *neighbors = neighbor_buffers[i].second;
        for (uint32_t j = 0; j < neighbor_num && !finish_flag; j++) {
          if (node_set.find(neighbors[j]) == node_set.end()) {
            cur_level.insert(neighbors[j]);
          }
          if (cur_level.size() + node_set.size() >= num_nodes_to_cache) {
            finish_flag = true;
          }
        }
      }
    }

    size_t total_size = node_set.size();

    LOG_INFO("Level: %zu, Cached Size: %zu, Total Cached Size: %zu",
             (size_t)level, (size_t)(total_size - prev_node_set_size),
             total_size);

    prev_node_set_size = total_size;
    level++;
  }

  ailego_assert(node_set.size() + cur_level.size() == num_nodes_to_cache ||
                cur_level.size() == 0);

  std::vector<diskann_id_t> final_level(cur_level.begin(), cur_level.end());
  std::sort(final_level.begin(), final_level.end());
  if (final_level.size() > state.capacity - state.slots.size()) {
    LOG_ERROR("DiskANN node cache frontier exceeded its allocated capacity");
    return IndexError_Runtime;
  }
  for (diskann_id_t id : final_level) {
    state.slots.push_back(CacheSlot{id, 0, false});
  }

  const size_t total_size = state.slots.size();
  LOG_INFO("Level: %zu, Cached Size: %zu, Total Cached Size: %zu",
           (size_t)level, (size_t)(total_size - prev_node_set_size),
           (size_t)total_size);

  return 0;
}

int DiskAnnIndexer::linear_search(DiskAnnContext *ctx) {
  auto &stats = ctx->query_stats();
  auto &dc = ctx->dist_calculator();
  auto &topk_heap = ctx->topk_heap();

  topk_heap.clear();
  auto &group_topk_heaps = ctx->group_topk_heaps();
  group_topk_heaps.clear();
  auto emplace_candidate = [&](diskann_id_t id, VectorInfo info) {
    if (ctx->group_by_search() && ctx->group_by().is_valid()) {
      topk_heap.emplace(id, info);
      std::string group_id = ctx->group_by()(get_key(id));
      auto &group_topk_heap = group_topk_heaps[group_id];
      if (group_topk_heap.empty()) {
        group_topk_heap.limit(ctx->group_topk());
      }
      group_topk_heap.emplace(id, std::move(info));
    } else {
      topk_heap.emplace(id, std::move(info));
    }
  };

  IOContext &io_ctx = ctx->io_ctx();
  void *aligned_query_raw = ctx->query();

  void *data_buf = reinterpret_cast<void *>(ctx->coord_buffer());

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);

  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime cpu_timer;

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * beam_width_);

  std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * beam_width_);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * beam_width_);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * beam_width_);

  uint64_t sector_buffer_idx = 0;

  diskann_id_t id = 0;
  while (id < doc_cnt_) {
    while (frontier.size() < beam_width_) {
      if (!ctx->filter().is_valid() || !ctx->filter()(get_key(id))) {
        auto iter = neighbor_cache_.find(id);
        if (iter != neighbor_cache_.end()) {
          cached_neighbors.push_back(
              std::make_tuple(id, iter->second.first, iter->second.second));
          stats.cache_hits++;
        } else {
          frontier.push_back(id);
        }
      }

      id++;
      if (id >= doc_cnt_) {
        break;
      }
    }

    if (!frontier.empty()) {
      for (uint64_t i = 0; i < frontier.size(); i++) {
        diskann_id_t cur_id = frontier[i];

        std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
        frontier_neighbor.first = cur_id;
        frontier_neighbor.second = sector_buffer + sector_num_per_node *
                                                       sector_buffer_idx *
                                                       DiskAnnUtil::kSectorSize;
        frontier_neighbors.push_back(frontier_neighbor);

        sector_buffer_idx++;

        frontier_read_reqs.emplace_back(
            index_segment_offset_ +
                DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                             DiskAnnUtil::kSectorSize, cur_id) *
                    DiskAnnUtil::kSectorSize,
            sector_num_per_node * DiskAnnUtil::kSectorSize,
            frontier_neighbor.second);

        stats.disk_page_reads++;
        stats.io_num++;
      }

      io_timer.reset();

      int read_ret = reader_->read(frontier_read_reqs, io_ctx);
      stats.io_us += io_timer.micro_seconds();
      if (read_ret != 0) {
        LOG_ERROR("linear_search: reader_->read failed, ret=%d", read_ret);
        ctx->set_error(true);
        return IndexError_Runtime;
      }
    }

    for (auto &cached_neighbor : cached_neighbors) {
      auto global_cache_iter = coord_cache_.find(std::get<0>(cached_neighbor));
      void *node_fp_coords_copy = global_cache_iter->second;

      float cur_expanded_dist = dc.dist(aligned_query_raw, node_fp_coords_copy);

      emplace_candidate(
          std::get<0>(cached_neighbor),
          VectorInfo(cur_expanded_dist, make_vector_copy(node_fp_coords_copy)));
    }

    for (auto &frontier_neighbor : frontier_neighbors) {
      uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
          node_per_sector_, max_node_size_, frontier_neighbor.second,
          frontier_neighbor.first);

      void *node_fp_coords = node_disk_buf;
      memcpy(data_buf, node_fp_coords, disk_bytes_per_point_);

      float cur_expanded_dist = dc.dist(aligned_query_raw, data_buf);

      emplace_candidate(
          frontier_neighbor.first,
          VectorInfo(cur_expanded_dist, make_vector_copy(data_buf)));

      stats.cpu_us += cpu_timer.micro_seconds();
    }

    frontier.clear();
    frontier_neighbors.clear();
    frontier_read_reqs.clear();
    cached_neighbors.clear();
    sector_buffer_idx = 0;
  }

  stats.total_us += query_timer.micro_seconds();

  return 0;
}

int DiskAnnIndexer::keys_search(const std::vector<uint64_t> &keys,
                                DiskAnnContext *ctx) {
  auto &stats = ctx->query_stats();
  auto &dc = ctx->dist_calculator();
  auto &topk_heap = ctx->topk_heap();

  topk_heap.clear();
  auto &group_topk_heaps = ctx->group_topk_heaps();
  group_topk_heaps.clear();
  auto emplace_candidate = [&](diskann_id_t id, VectorInfo info) {
    if (ctx->group_by_search() && ctx->group_by().is_valid()) {
      topk_heap.emplace(id, info);
      std::string group_id = ctx->group_by()(get_key(id));
      auto &group_topk_heap = group_topk_heaps[group_id];
      if (group_topk_heap.empty()) {
        group_topk_heap.limit(ctx->group_topk());
      }
      group_topk_heap.emplace(id, std::move(info));
    } else {
      topk_heap.emplace(id, std::move(info));
    }
  };

  IOContext &io_ctx = ctx->io_ctx();
  void *aligned_query_raw = ctx->query();

  void *data_buf = reinterpret_cast<void *>(ctx->coord_buffer());

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);

  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime cpu_timer;

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * beam_width_);

  std::vector<std::pair<uint32_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * beam_width_);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * beam_width_);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * beam_width_);

  uint64_t sector_buffer_idx = 0;

  size_t idx = 0;
  while (idx < keys.size()) {
    while (frontier.size() < beam_width_) {
      if (!ctx->filter().is_valid() || !ctx->filter()(keys[idx])) {
        diskann_id_t id = get_id(keys[idx]);
        if (id == kInvalidId) {
          ++idx;
          if (idx >= keys.size()) {
            break;
          }
          continue;
        }

        auto iter = neighbor_cache_.find(id);
        if (iter != neighbor_cache_.end()) {
          cached_neighbors.push_back(
              std::make_tuple(id, iter->second.first, iter->second.second));
          stats.cache_hits++;
        } else {
          frontier.push_back(id);
        }
      }

      idx++;
      if (idx >= keys.size()) {
        break;
      }
    }

    if (!frontier.empty()) {
      for (uint64_t i = 0; i < frontier.size(); i++) {
        diskann_id_t cur_id = frontier[i];

        std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
        frontier_neighbor.first = cur_id;
        frontier_neighbor.second = sector_buffer + sector_num_per_node *
                                                       sector_buffer_idx *
                                                       DiskAnnUtil::kSectorSize;
        frontier_neighbors.push_back(frontier_neighbor);

        sector_buffer_idx++;

        frontier_read_reqs.emplace_back(
            index_segment_offset_ +
                DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                             DiskAnnUtil::kSectorSize, cur_id) *
                    DiskAnnUtil::kSectorSize,
            sector_num_per_node * DiskAnnUtil::kSectorSize,
            frontier_neighbor.second);

        stats.disk_page_reads++;
        stats.io_num++;
      }

      io_timer.reset();

      int read_ret = reader_->read(frontier_read_reqs, io_ctx);
      stats.io_us += io_timer.micro_seconds();
      if (read_ret != 0) {
        LOG_ERROR("keys_search: reader_->read failed, ret=%d", read_ret);
        ctx->set_error(true);
        return IndexError_Runtime;
      }
    }

    for (auto &cached_neighbor : cached_neighbors) {
      auto global_cache_iter = coord_cache_.find(std::get<0>(cached_neighbor));
      void *node_fp_coords_copy = global_cache_iter->second;

      float cur_expanded_dist = dc.dist(aligned_query_raw, node_fp_coords_copy);

      emplace_candidate(
          std::get<0>(cached_neighbor),
          VectorInfo(cur_expanded_dist, make_vector_copy(node_fp_coords_copy)));
    }

    for (auto &frontier_neighbor : frontier_neighbors) {
      uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
          node_per_sector_, max_node_size_, frontier_neighbor.second,
          frontier_neighbor.first);

      void *node_fp_coords = node_disk_buf;
      memcpy(data_buf, node_fp_coords, disk_bytes_per_point_);

      float cur_expanded_dist = dc.dist(aligned_query_raw, data_buf);

      emplace_candidate(
          frontier_neighbor.first,
          VectorInfo(cur_expanded_dist, make_vector_copy(data_buf)));

      stats.cpu_us += cpu_timer.micro_seconds();
    }

    frontier.clear();
    frontier_neighbors.clear();
    frontier_read_reqs.clear();
    cached_neighbors.clear();
    sector_buffer_idx = 0;
  }

  stats.total_us += query_timer.micro_seconds();

  return 0;
}

int DiskAnnIndexer::get_vector(diskann_id_t id, IndexContext::Pointer &context,
                               std::string &vector) {
  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  if (ctx == nullptr) {
    LOG_ERROR("get_vector: invalid DiskAnn context");
    return IndexError_InvalidArgument;
  }

  auto &stats = ctx->query_stats();

  IOContext &io_ctx = ctx->io_ctx();
  // Search contexts are owned by an external pool and may outlive this index.
  // Fetch contexts, however, are owned by the provider/iterator that also owns
  // the reader; retaining their private handle avoids reopening it per vector.
  IOContextReleaseGuard release_guard(
      *reader_, io_ctx,
      ctx->context_type() == DiskAnnContext::kSearcherContext);

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);
  const size_t sector_read_size =
      static_cast<size_t>(sector_num_per_node) * DiskAnnUtil::kSectorSize;
  const size_t node_offset =
      node_per_sector_ == 0
          ? 0
          : static_cast<size_t>(id % node_per_sector_) * max_node_size_;
  if (sector_num_per_node == 0 || sector_buffer == nullptr ||
      sector_read_size > ctx->sector_buffer_size() ||
      node_offset > sector_read_size ||
      meta_.element_size() > sector_read_size - node_offset) {
    LOG_ERROR(
        "get_vector: invalid sector buffer range, read=%zu offset=%zu "
        "vector=%u available=%zu",
        sector_read_size, node_offset,
        static_cast<unsigned>(meta_.element_size()), ctx->sector_buffer_size());
    return IndexError_InvalidArgument;
  }

  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime cpu_timer;

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * beam_width_);

  std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * beam_width_);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * beam_width_);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * beam_width_);

  auto iter = coord_cache_.find(id);
  if (iter != coord_cache_.end()) {
    void *node_fp_coords_copy = iter->second;

    vector.resize(meta_.element_size());
    ::memcpy(&(vector[0]), node_fp_coords_copy, meta_.element_size());

    return 0;
  } else {
    std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
    frontier_neighbor.first = id;
    frontier_neighbor.second = sector_buffer;
    frontier_neighbors.push_back(frontier_neighbor);

    frontier_read_reqs.emplace_back(
        index_segment_offset_ +
            DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                         DiskAnnUtil::kSectorSize, id) *
                DiskAnnUtil::kSectorSize,
        sector_read_size, frontier_neighbor.second);

    stats.disk_page_reads++;
    stats.io_num++;

    io_timer.reset();

    int read_ret = reader_->read(frontier_read_reqs, io_ctx);
    stats.io_us += io_timer.micro_seconds();
    if (read_ret != 0) {
      LOG_ERROR("get_vector: reader_->read failed, ret=%d", read_ret);
      ctx->set_error(true);
      return IndexError_Runtime;
    }

    uint8_t *node_disk_buf = frontier_neighbor.second + node_offset;

    void *node_fp_coords = node_disk_buf;

    vector.resize(meta_.element_size());
    ::memcpy(&(vector[0]), node_fp_coords, meta_.element_size());

    stats.cpu_us += cpu_timer.micro_seconds();
  }

  return 0;
}

int DiskAnnIndexer::knn_search(DiskAnnContext *ctx) {
  int ret = cached_beam_search(ctx);
  if (ret != 0) {
    return ret;
  }

  if (ctx->group_by_search()) {
    ret = cached_beam_search_by_group(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

void DiskAnnIndexer::release_io_ctx(DiskAnnContext *ctx) {
  if (reader_ && ctx) {
    reader_->release_io_ctx(ctx->io_ctx());
  }
}

int DiskAnnIndexer::cached_beam_search(DiskAnnContext *ctx) {
  int error_code = IndexError_Runtime;
  try {
    return cached_beam_search_impl(ctx);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("cached_beam_search: memory allocation failed");
    error_code = IndexError_NoMemory;
  } catch (const std::exception &e) {
    LOG_ERROR("cached_beam_search: unexpected exception: %s", e.what());
  } catch (...) {
    LOG_ERROR("cached_beam_search: unknown exception");
  }

  // An exception may occur after an asynchronous batch has been submitted.
  // Recreate the context only after destroy_io_ctx has cancelled and waited
  // for all requests, so the caller can safely reuse or destroy this context.
  IOContext &io_ctx = ctx->io_ctx();
  destroy_io_ctx(io_ctx);
  if (setup_io_ctx(io_ctx) != 0) {
    LOG_ERROR("cached_beam_search: failed to recreate I/O context");
  }
  ctx->set_error(true);
  return error_code;
}

int DiskAnnIndexer::cached_beam_search_impl(DiskAnnContext *ctx) {
  auto &stats = ctx->query_stats();
  auto &dc = ctx->dist_calculator();
  auto &topk_heap = ctx->topk_heap();
  auto &visit_filter = ctx->visit_filter();

  topk_heap.clear();

  IOContext &io_ctx = ctx->io_ctx();

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);

  pq_table_->preprocess_pq_dist_table(ctx->query_rotated(),
                                      ctx->pq_table_dist_buffer());

  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime cpu_timer;

  NeighborPriorityQueue candidates;

  candidates.reserve(ctx->list_size());

  diskann_id_t best_medoid = entrypoints_.front();
  float best_dist = (std::numeric_limits<float>::max)();
  for (uint64_t cur_m = 0; cur_m < entrypoints_.size(); cur_m++) {
    const void *entrypoint =
        static_cast<const uint8_t *>(centroid_data_) + centroid_stride_ * cur_m;
    float cur_expanded_dist = dc.dist(ctx->query(), entrypoint);

    if (cur_expanded_dist < best_dist) {
      best_medoid = entrypoints_[cur_m];
      best_dist = cur_expanded_dist;
    }
  }

  float dist;
  pq_table_->compute_dists(1, &best_medoid, pq_chunk_num_,
                           ctx->pq_table_dist_buffer(), ctx->pq_coord_buffer(),
                           &dist);
  candidates.insert(Neighbor(best_medoid, dist));
  visit_filter.set_visited(best_medoid);

  uint32_t num_ios = 0;

  // Cap beam width so one batch of frontier reads never exceeds the sector
  // buffer capacity (kMaxSectorReadNum sectors), as each node occupies
  // sector_num_per_node sectors.
  uint32_t max_beam_width =
      std::max(1u, static_cast<uint32_t>(DiskAnnUtil::kMaxSectorReadNum /
                                         sector_num_per_node));
  uint32_t effective_beam_width = std::min(
      std::max(8u, std::min(ctx->list_size() / 5, 32u)), max_beam_width);

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * effective_beam_width);

  std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * effective_beam_width);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * effective_beam_width);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * effective_beam_width);

  PendingBatch pending;

  while (candidates.has_unexpanded_node() && num_ios < io_limit_) {
    frontier.clear();
    frontier_neighbors.clear();
    frontier_read_reqs.clear();
    cached_neighbors.clear();

    uint64_t sector_buffer_idx = 0;

    uint32_t num_seen = 0;
    while (candidates.has_unexpanded_node() &&
           frontier.size() < effective_beam_width &&
           num_seen < effective_beam_width) {
      auto neighbor = candidates.closest_unexpanded();
      num_seen++;

      auto iter = neighbor_cache_.find(neighbor.id);
      if (iter != neighbor_cache_.end()) {
        cached_neighbors.push_back(std::make_tuple(
            neighbor.id, iter->second.first, iter->second.second));
        stats.cache_hits++;
      } else {
        frontier.push_back(neighbor.id);
      }
    }

    if (!frontier.empty()) {
      stats.hop_num++;

      for (uint64_t i = 0; i < frontier.size(); i++) {
        diskann_id_t cur_id = frontier[i];

        std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
        frontier_neighbor.first = cur_id;
        frontier_neighbor.second = sector_buffer + sector_num_per_node *
                                                       sector_buffer_idx *
                                                       DiskAnnUtil::kSectorSize;
        frontier_neighbors.push_back(frontier_neighbor);

        sector_buffer_idx++;

        frontier_read_reqs.emplace_back(
            index_segment_offset_ +
                DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                             DiskAnnUtil::kSectorSize, cur_id) *
                    DiskAnnUtil::kSectorSize,
            sector_num_per_node * DiskAnnUtil::kSectorSize,
            frontier_neighbor.second);

        stats.disk_page_reads++;
        stats.io_num++;
        num_ios++;
      }

      io_timer.reset();
      int submit_ret = reader_->submit(pending, frontier_read_reqs, io_ctx);
      stats.io_us += io_timer.micro_seconds();
      if (submit_ret != 0) {
        LOG_ERROR("cached_beam_search: submit failed, ret=%d", submit_ret);
        ctx->set_error(true);
        return IndexError_Runtime;
      }
    }

    for (auto &cached_neighbor : cached_neighbors) {
      auto global_cache_iter = coord_cache_.find(std::get<0>(cached_neighbor));
      void *node_fp_coords_copy = global_cache_iter->second;

      float cur_expanded_dist = dc.dist(ctx->query(), node_fp_coords_copy);

      if (!ctx->filter().is_valid() ||
          !ctx->filter()(get_key(std::get<0>(cached_neighbor)))) {
        topk_heap.emplace(std::get<0>(cached_neighbor),
                          VectorInfo(cur_expanded_dist,
                                     make_vector_copy(node_fp_coords_copy)));
      }

      uint32_t neighbor_num = std::get<1>(cached_neighbor);
      diskann_id_t *node_neighbors = std::get<2>(cached_neighbor);

      cpu_timer.reset();

      std::vector<float> distances(neighbor_num);
      pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                               ctx->pq_table_dist_buffer(),
                               ctx->pq_coord_buffer(), distances.data());

      stats.dist_num += neighbor_num;
      stats.cpu_us += cpu_timer.micro_seconds();

      for (uint64_t m = 0; m < neighbor_num; ++m) {
        diskann_id_t id = node_neighbors[m];
        if (!visit_filter.visited(id)) {
          visit_filter.set_visited(id);
          Neighbor nn(id, distances[m]);
          candidates.insert(nn);
        }
      }
    }

    if (!frontier.empty()) {
      std::vector<uint32_t> completed;
      while (pending.n_reaped < pending.n_submitted) {
        completed.clear();
        io_timer.reset();
        int n = reader_->get_completed(pending, io_ctx, 1, completed);
        stats.io_us += io_timer.micro_seconds();
        if (n < 0) {
          LOG_ERROR("cached_beam_search: get_completed failed, ret=%d", n);
          ctx->set_error(true);
          return IndexError_Runtime;
        }

        for (uint32_t idx : completed) {
          auto &frontier_neighbor = frontier_neighbors[idx];
          uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
              node_per_sector_, max_node_size_, frontier_neighbor.second,
              frontier_neighbor.first);
          uint32_t *node_buf = DiskAnnUtil::offset_to_node_neighbor(
              node_disk_buf, meta_.element_size());
          uint32_t neighbor_num = *node_buf;

          void *node_fp_coords = node_disk_buf;

          float cur_expanded_dist = dc.dist(ctx->query(), node_fp_coords);

          if (!ctx->filter().is_valid() ||
              !ctx->filter()(get_key(frontier_neighbor.first))) {
            topk_heap.emplace(frontier_neighbor.first,
                              VectorInfo(cur_expanded_dist,
                                         make_vector_copy(node_fp_coords)));
          }

          diskann_id_t *node_neighbors =
              reinterpret_cast<diskann_id_t *>(node_buf + 1);

          cpu_timer.reset();
          std::vector<float> distances(neighbor_num);
          pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                                   ctx->pq_table_dist_buffer(),
                                   ctx->pq_coord_buffer(), distances.data());

          stats.dist_num += neighbor_num;
          stats.cpu_us += cpu_timer.micro_seconds();

          cpu_timer.reset();
          for (uint64_t m = 0; m < neighbor_num; ++m) {
            diskann_id_t id = node_neighbors[m];
            if (!visit_filter.visited(id)) {
              visit_filter.set_visited(id);
              stats.dist_num++;
              Neighbor nn(id, distances[m]);
              candidates.insert(nn);
            }
          }

          stats.cpu_us += cpu_timer.micro_seconds();
        }
      }
    }
  }

  stats.total_us += query_timer.micro_seconds();

  return 0;
}

int DiskAnnIndexer::cached_beam_search_in_mem(DiskAnnContext * /*ctx*/) {
  return IndexError_NotImplemented;
}

void DiskAnnIndexer::populate_group_topk_heaps(DiskAnnContext *ctx) {
  auto &group_topk_heaps = ctx->group_topk_heaps();
  group_topk_heaps.clear();
  if (!ctx->group_by().is_valid()) {
    return;
  }

  auto &topk_heap = ctx->topk_heap();
  for (uint32_t i = 0; i < topk_heap.size(); ++i) {
    diskann_id_t id = topk_heap[i].first;
    const auto &info = topk_heap[i].second;
    std::string group_id = ctx->group_by()(get_key(id));

    auto &group_topk_heap = group_topk_heaps[group_id];
    if (group_topk_heap.empty()) {
      group_topk_heap.limit(ctx->group_topk());
    }
    group_topk_heap.emplace(id, info);
  }
}

int DiskAnnIndexer::cached_beam_search_by_group(DiskAnnContext *ctx) {
  if (!ctx->group_by().is_valid()) {
    ctx->group_topk_heaps().clear();
    return 0;
  }

  // Divide the initial candidates into groups.
  auto &topk_heap = ctx->topk_heap();
  auto &visit_filter = ctx->visit_filter();
  populate_group_topk_heaps(ctx);
  auto &group_topk_heaps = ctx->group_topk_heaps();

  // stage 2, expand to reach group num as possible
  if (group_topk_heaps.size() < ctx->group_num()) {
    NeighborPriorityQueue candidates;

    candidates.reserve(ctx->list_size());

    for (uint32_t i = 0; i < topk_heap.size(); ++i) {
      diskann_id_t id = topk_heap[i].first;
      float score = topk_heap[i].second.dist_;

      visit_filter.set_visited(id);
      candidates.insert(Neighbor(id, score));
    }

    ailego::ElapsedTime io_timer;
    ailego::ElapsedTime query_timer;
    ailego::ElapsedTime cpu_timer;

    auto &stats = ctx->query_stats();
    auto &dc = ctx->dist_calculator();

    IOContext &io_ctx = ctx->io_ctx();

    void *data_buf = reinterpret_cast<void *>(ctx->coord_buffer());
    uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

    const uint64_t sector_num_per_node =
        node_per_sector_ > 0 ? 1
                             : DiskAnnUtil::div_round_up(
                                   max_node_size_, DiskAnnUtil::kSectorSize);

    pq_table_->preprocess_pq_dist_table(ctx->query_rotated(),
                                        ctx->pq_table_dist_buffer());

    uint32_t num_ios = 0;

    std::vector<diskann_id_t> frontier;
    frontier.reserve(2 * beam_width_);
    std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
    frontier_neighbors.reserve(2 * beam_width_);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width_);
    std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
        cached_neighbors;
    cached_neighbors.reserve(2 * beam_width_);

    uint64_t sector_buffer_idx;

    while (candidates.has_unexpanded_node() && num_ios < io_limit_) {
      frontier.clear();
      frontier_neighbors.clear();
      frontier_read_reqs.clear();
      cached_neighbors.clear();
      sector_buffer_idx = 0;

      uint32_t num_seen = 0;
      while (candidates.has_unexpanded_node() &&
             frontier.size() < beam_width_ && num_seen < beam_width_) {
        auto neighbor = candidates.closest_unexpanded();
        num_seen++;

        auto iter = neighbor_cache_.find(neighbor.id);
        if (iter != neighbor_cache_.end()) {
          cached_neighbors.push_back(std::make_tuple(
              neighbor.id, iter->second.first, iter->second.second));
          stats.cache_hits++;
        } else {
          frontier.push_back(neighbor.id);
        }
      }

      if (!frontier.empty()) {
        stats.hop_num++;

        for (uint64_t i = 0; i < frontier.size(); i++) {
          diskann_id_t cur_id = frontier[i];

          std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
          frontier_neighbor.first = cur_id;
          frontier_neighbor.second =
              sector_buffer + sector_num_per_node * sector_buffer_idx *
                                  DiskAnnUtil::kSectorSize;
          frontier_neighbors.push_back(frontier_neighbor);

          sector_buffer_idx++;

          frontier_read_reqs.emplace_back(
              index_segment_offset_ + DiskAnnUtil::get_node_sector(
                                          node_per_sector_, max_node_size_,
                                          DiskAnnUtil::kSectorSize, cur_id) *
                                          DiskAnnUtil::kSectorSize,
              sector_num_per_node * DiskAnnUtil::kSectorSize,
              frontier_neighbor.second);

          stats.disk_page_reads++;
          stats.io_num++;
          num_ios++;
        }

        io_timer.reset();

        int read_ret = reader_->read(frontier_read_reqs, io_ctx);
        stats.io_us += io_timer.micro_seconds();
        if (read_ret != 0) {
          LOG_ERROR("cached_beam_search_by_group: reader_->read failed, ret=%d",
                    read_ret);
          ctx->set_error(true);
          return IndexError_Runtime;
        }
      }

      for (auto &cached_neighbor : cached_neighbors) {
        auto global_cache_iter =
            coord_cache_.find(std::get<0>(cached_neighbor));
        void *node_fp_coords_copy = global_cache_iter->second;

        float cur_expanded_dist = dc.dist(ctx->query(), node_fp_coords_copy);

        if (!ctx->filter().is_valid() ||
            !ctx->filter()(get_key(std::get<0>(cached_neighbor)))) {
          std::string group_id =
              ctx->group_by()(get_key(std::get<0>(cached_neighbor)));

          auto &group_topk_heap = group_topk_heaps[group_id];
          if (group_topk_heap.empty()) {
            group_topk_heap.limit(ctx->group_topk());
          }

          group_topk_heap.emplace(
              std::get<0>(cached_neighbor),
              VectorInfo(cur_expanded_dist,
                         make_vector_copy(node_fp_coords_copy)));

          if (group_topk_heaps.size() >= ctx->group_num()) {
            break;
          }
        }

        uint64_t neighbor_num = std::get<1>(cached_neighbor);
        diskann_id_t *node_neighbors = std::get<2>(cached_neighbor);

        cpu_timer.reset();

        std::vector<float> distances(neighbor_num);
        pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                                 ctx->pq_table_dist_buffer(),
                                 ctx->pq_coord_buffer(), distances.data());

        stats.dist_num += neighbor_num;
        stats.cpu_us += cpu_timer.micro_seconds();

        for (uint64_t m = 0; m < neighbor_num; ++m) {
          diskann_id_t id = node_neighbors[m];
          visit_filter.set_visited(id);

          Neighbor nn(id, distances[m]);
          candidates.insert(nn);
        }
      }

      for (auto &frontier_neighbor : frontier_neighbors) {
        uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
            node_per_sector_, max_node_size_, frontier_neighbor.second,
            frontier_neighbor.first);
        uint32_t *node_buf = DiskAnnUtil::offset_to_node_neighbor(
            node_disk_buf, meta_.element_size());
        uint32_t neighbor_num = *node_buf;

        void *node_fp_coords = node_disk_buf;
        memcpy(data_buf, node_fp_coords, disk_bytes_per_point_);

        float cur_expanded_dist = dc.dist(ctx->query(), data_buf);

        if (!ctx->filter().is_valid() ||
            !ctx->filter()(get_key(frontier_neighbor.first))) {
          std::string group_id =
              ctx->group_by()(get_key(frontier_neighbor.first));

          auto &group_topk_heap = group_topk_heaps[group_id];
          if (group_topk_heap.empty()) {
            group_topk_heap.limit(ctx->group_topk());
          }

          group_topk_heap.emplace(
              frontier_neighbor.first,
              VectorInfo(cur_expanded_dist, make_vector_copy(data_buf)));

          if (group_topk_heaps.size() >= ctx->group_num()) {
            break;
          }
        }

        cpu_timer.reset();

        std::vector<float> distances(neighbor_num);
        diskann_id_t *node_neighbors =
            reinterpret_cast<diskann_id_t *>(node_buf + 1);
        pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                                 ctx->pq_table_dist_buffer(),
                                 ctx->pq_coord_buffer(), distances.data());

        stats.dist_num += neighbor_num;
        stats.cpu_us += cpu_timer.micro_seconds();

        cpu_timer.reset();
        for (uint64_t m = 0; m < neighbor_num; ++m) {
          diskann_id_t id = node_neighbors[m];
          visit_filter.set_visited(id);
          stats.dist_num++;

          Neighbor nn(id, distances[m]);
          candidates.insert(nn);
        }

        stats.cpu_us += cpu_timer.micro_seconds();
      }
    }

    stats.total_us += query_timer.micro_seconds();
  }

  return 0;
}

}  // namespace core
}  // namespace zvec
