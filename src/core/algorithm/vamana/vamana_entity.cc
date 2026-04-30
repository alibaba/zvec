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
#include "vamana_entity.h"
#include <zvec/ailego/hash/crc32c.h>

namespace zvec {
namespace core {

const std::string VamanaEntity::kGraphHeaderSegmentId = "vamana.graph.header";
const std::string VamanaEntity::kGraphFeaturesSegmentId =
    "vamana.graph.features";
const std::string VamanaEntity::kGraphKeysSegmentId = "vamana.graph.keys";
const std::string VamanaEntity::kGraphNeighborsSegmentId =
    "vamana.graph.neighbors";
const std::string VamanaEntity::kGraphOffsetsSegmentId = "vamana.graph.offsets";
const std::string VamanaEntity::kGraphMappingSegmentId = "vamana.graph.mapping";
const std::string VamanaEntity::kGraphNeighborDistsSegmentId =
    "vamana.graph.neighbor_dists";

int VamanaEntity::CalcAndAddPadding(const IndexDumper::Pointer &dumper,
                                    size_t data_size, size_t *padding_size) {
  *padding_size = AlignSize(data_size) - data_size;
  if (*padding_size == 0) {
    return 0;
  }
  std::string padding(*padding_size, '\0');
  if (dumper->write(padding.data(), *padding_size) != *padding_size) {
    LOG_ERROR("Append padding failed, size %lu", *padding_size);
    return IndexError_WriteData;
  }
  return 0;
}

int64_t VamanaEntity::dump_segment(const IndexDumper::Pointer &dumper,
                                   const std::string &segment_id,
                                   const void *data, size_t size) const {
  size_t len = dumper->write(data, size);
  if (len != size) {
    LOG_ERROR("Dump segment %s data failed, expect: %lu, actual: %lu",
              segment_id.c_str(), size, len);
    return IndexError_WriteData;
  }

  size_t padding_size = AlignSize(size) - size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  uint32_t crc = ailego::Crc32c::Hash(data, size);
  int ret = dumper->append(segment_id, size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d", segment_id.c_str(), ret);
    return ret;
  }

  return len + padding_size;
}

int64_t VamanaEntity::dump_header(const IndexDumper::Pointer &dumper,
                                  const VamanaHeader &hd) const {
  return dump_segment(dumper, kGraphHeaderSegmentId, &hd.graph, hd.graph.size);
}

int64_t VamanaEntity::dump_vectors(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping) const {
  size_t total_size = doc_cnt() * vector_size();
  std::vector<uint8_t> buffer(total_size);

  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    node_id_t old_id = reorder_mapping[i];
    const void *vec = get_vector(old_id);
    if (vec == nullptr) {
      LOG_ERROR("Get vector failed for node %u", old_id);
      return IndexError_ReadData;
    }
    memcpy(buffer.data() + static_cast<size_t>(i) * vector_size(), vec,
           vector_size());
  }

  return dump_segment(dumper, kGraphFeaturesSegmentId, buffer.data(),
                      total_size);
}

int64_t VamanaEntity::dump_neighbors(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping,
    const std::vector<node_id_t> &neighbor_mapping) const {
  size_t nbr_size = neighbors_size();
  size_t total_size = doc_cnt() * nbr_size;
  std::vector<uint8_t> buffer(total_size, 0);

  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    node_id_t old_id = reorder_mapping[i];
    const Neighbors nbrs = get_neighbors(old_id);

    auto *hd = reinterpret_cast<NeighborsHeader *>(
        buffer.data() + static_cast<size_t>(i) * nbr_size);
    hd->neighbor_cnt = nbrs.size();
    for (uint32_t j = 0; j < nbrs.size(); ++j) {
      hd->neighbors[j] = neighbor_mapping[nbrs[j]];
    }
  }

  return dump_segment(dumper, kGraphNeighborsSegmentId, buffer.data(),
                      total_size);
}

int64_t VamanaEntity::dump_neighbor_dists(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping) const {
  if (!dist_storage_loaded()) {
    // No distance data to dump — this is fine for read-only indices
    return 0;
  }

  uint32_t max_deg = static_cast<uint32_t>(max_degree());
  size_t dist_entry = max_deg * sizeof(dist_t);
  size_t total_size = doc_cnt() * dist_entry;
  std::vector<uint8_t> buffer(total_size, 0);

  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    node_id_t old_id = reorder_mapping[i];
    const dist_t *dists = get_neighbor_dists(old_id);
    if (dists != nullptr) {
      memcpy(buffer.data() + static_cast<size_t>(i) * dist_entry, dists,
             dist_entry);
    }
  }

  return dump_segment(dumper, kGraphNeighborDistsSegmentId, buffer.data(),
                      total_size);
}

int64_t VamanaEntity::dump_mapping_segment(const IndexDumper::Pointer &dumper,
                                           const key_t *keys) const {
  size_t total_size = doc_cnt() * sizeof(key_t);
  return dump_segment(dumper, kGraphMappingSegmentId, keys, total_size);
}

void VamanaEntity::reshuffle_vectors(std::vector<node_id_t> *n2o_mapping,
                                     std::vector<node_id_t> *o2n_mapping,
                                     key_t *keys) const {
  uint32_t count = doc_cnt();
  n2o_mapping->resize(count);
  o2n_mapping->resize(count);

  // Simple identity mapping for now (can be optimized with BFS traversal)
  for (uint32_t i = 0; i < count; ++i) {
    (*n2o_mapping)[i] = i;
    (*o2n_mapping)[i] = i;
    keys[i] = get_key(i);
  }
}

int64_t VamanaEntity::dump_segments(const IndexDumper::Pointer &dumper,
                                    key_t *keys) const {
  std::vector<node_id_t> n2o_mapping;
  std::vector<node_id_t> o2n_mapping;
  reshuffle_vectors(&n2o_mapping, &o2n_mapping, keys);

  // Remap entry point
  VamanaHeader dump_header_copy = header_;
  if (dump_header_copy.graph.entry_point != kInvalidNodeId) {
    dump_header_copy.graph.entry_point =
        o2n_mapping[dump_header_copy.graph.entry_point];
  }

  int64_t hd_size = dump_header(dumper, dump_header_copy);
  if (hd_size < 0) return hd_size;

  int64_t vec_size = dump_vectors(dumper, n2o_mapping);
  if (vec_size < 0) return vec_size;

  int64_t nbr_size = dump_neighbors(dumper, n2o_mapping, o2n_mapping);
  if (nbr_size < 0) return nbr_size;

  int64_t map_size = dump_mapping_segment(dumper, keys);
  if (map_size < 0) return map_size;

  int64_t dist_size = dump_neighbor_dists(dumper, n2o_mapping);
  if (dist_size < 0) return dist_size;

  return hd_size + vec_size + nbr_size + map_size + dist_size;
}

}  // namespace core
}  // namespace zvec
