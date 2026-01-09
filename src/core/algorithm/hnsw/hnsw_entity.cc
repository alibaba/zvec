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
#include "hnsw_entity.h"
#include "framework/index_stats.h"
#include "utility/sparse_utility.h"

namespace zvec {
namespace core {

const std::string HnswEntity::kGraphHeaderSegmentId = "graph.header";
const std::string HnswEntity::kGraphFeaturesSegmentId = "graph.features";
const std::string HnswEntity::kGraphKeysSegmentId = "graph.keys";
const std::string HnswEntity::kGraphNeighborsSegmentId = "graph.neighbors";
const std::string HnswEntity::kGraphOffsetsSegmentId = "graph.offsets";
const std::string HnswEntity::kGraphMappingSegmentId = "graph.mapping";
const std::string HnswEntity::kHnswHeaderSegmentId = "hnsw.header";
const std::string HnswEntity::kHnswNeighborsSegmentId = "hnsw.neighbors";
const std::string HnswEntity::kHnswOffsetsSegmentId = "hnsw.offsets";

int HnswEntity::CalcAndAddPadding(const IndexDumper::Pointer &dumper,
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

int64_t HnswEntity::dump_segment(const IndexDumper::Pointer &dumper,
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

int64_t HnswEntity::dump_header(const IndexDumper::Pointer &dumper,
                                const HNSWHeader &hd) const {
  //! dump basic graph header. header is aligned and does not need padding
  int64_t graph_hd_size =
      dump_segment(dumper, kGraphHeaderSegmentId, &hd.graph, hd.graph.size);
  if (graph_hd_size < 0) {
    return graph_hd_size;
  }

  //! dump basic graph header. header is aligned and does not need padding
  int64_t hnsw_hd_size =
      dump_segment(dumper, kHnswHeaderSegmentId, &hd.hnsw, hd.hnsw.size);
  if (hnsw_hd_size < 0) {
    return hnsw_hd_size;
  }

  return graph_hd_size + hnsw_hd_size;
}

void HnswEntity::reshuffle_vectors(
    const std::function<level_t(node_id_t)> & /*get_level*/,
    std::vector<node_id_t> * /*n2o_mapping*/,
    std::vector<node_id_t> * /*o2n_mapping*/, key_t * /*keys*/) const {
  // TODO
  return;
}

int64_t HnswEntity::dump_mapping_segment(const IndexDumper::Pointer &dumper,
                                         const key_t *keys) const {
  std::vector<node_id_t> mapping(doc_cnt());

  std::iota(mapping.begin(), mapping.end(), 0U);
  std::sort(mapping.begin(), mapping.end(),
            [&](node_id_t i, node_id_t j) { return keys[i] < keys[j]; });

  size_t size = mapping.size() * sizeof(node_id_t);

  return dump_segment(dumper, kGraphMappingSegmentId, mapping.data(), size);
}

int64_t HnswEntity::dump_segments(
    const IndexDumper::Pointer &dumper, key_t *keys,
    const std::function<level_t(node_id_t)> &get_level) const {
  HNSWHeader dump_hd(header());

  dump_hd.graph.node_size = AlignSize(vector_size());

  std::vector<node_id_t> n2o_mapping;  // map new id to origin id
  std::vector<node_id_t> o2n_mapping;  // map origin id to new id
  reshuffle_vectors(get_level, &n2o_mapping, &o2n_mapping, keys);
  if (!o2n_mapping.empty()) {
    dump_hd.hnsw.entry_point = o2n_mapping[entry_point()];
  }

  //! Dump header
  int64_t hd_size = dump_header(dumper, dump_hd);
  if (hd_size < 0) {
    return hd_size;
  }

  //! Dump vectors
  int64_t vecs_size = dump_vectors(dumper, n2o_mapping);
  if (vecs_size < 0) {
    return vecs_size;
  }

  //! Dump neighbors
  auto neighbors_size =
      dump_neighbors(dumper, get_level, n2o_mapping, o2n_mapping);
  if (neighbors_size < 0) {
    return neighbors_size;
  }
  //! free memory
  n2o_mapping = std::vector<node_id_t>();
  o2n_mapping = std::vector<node_id_t>();

  //! Dump keys
  size_t key_segment_size = doc_cnt() * sizeof(key_t);
  int64_t keys_size =
      dump_segment(dumper, kGraphKeysSegmentId, keys, key_segment_size);
  if (keys_size < 0) {
    return keys_size;
  }

  //! Dump mapping
  int64_t mapping_size = dump_mapping_segment(dumper, keys);
  if (mapping_size < 0) {
    return mapping_size;
  }

  return hd_size + keys_size + vecs_size + neighbors_size + mapping_size;
}

int64_t HnswEntity::dump_vectors(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping) const {
  size_t vector_dump_size = vector_size();

  size_t padding_size = AlignSize(vector_dump_size) - vector_dump_size;

  char padding[padding_size];
  memset(padding, 0, sizeof(padding));
  const void *data = nullptr;
  uint32_t crc = 0U;
  size_t vecs_size = 0UL;

  //! dump vectors
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    data = get_vector(reorder_mapping.empty() ? id : reorder_mapping[id]);
    if (ailego_unlikely(!data)) {
      return IndexError_ReadData;
    }
    size_t len = dumper->write(data, vector_size());
    if (len != vector_size()) {
      LOG_ERROR("Dump vectors failed, write=%zu expect=%zu", len,
                vector_size());
      return IndexError_WriteData;
    }

    crc = ailego::Crc32c::Hash(data, vector_size(), crc);
    vecs_size += vector_size();

    if (padding_size == 0) {
      continue;
    }

    len = dumper->write(padding, padding_size);
    if (len != padding_size) {
      LOG_ERROR("Dump vectors failed, write=%zu expect=%zu", len, padding_size);
      return IndexError_WriteData;
    }
    crc = ailego::Crc32c::Hash(padding, padding_size, crc);
    vecs_size += padding_size;
  }

  int ret = dumper->append(kGraphFeaturesSegmentId, vecs_size, 0UL, crc);
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment meta failed, ret %d", ret);
    return ret;
  }

  return vecs_size;
}

int64_t HnswEntity::dump_graph_neighbors(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping,
    const std::vector<node_id_t> &neighbor_mapping) const {
  std::vector<GraphNeighborMeta> graph_meta;
  graph_meta.reserve(doc_cnt());
  size_t offset = 0;
  uint32_t crc = 0;
  node_id_t mapping[l0_neighbor_cnt()];

  uint32_t min_neighbor_count = 10000;
  uint32_t max_neighbor_count = 0;
  size_t sum_neighbor_count = 0;

  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    const Neighbors neighbors =
        get_neighbors(0, reorder_mapping.empty() ? id : reorder_mapping[id]);
    ailego_assert_with(!!neighbors.data, "invalid neighbors");
    ailego_assert_with(neighbors.size() <= l0_neighbor_cnt(),
                       "invalid neighbors");

    uint32_t neighbor_count = neighbors.size();
    if (neighbor_count < min_neighbor_count) {
      min_neighbor_count = neighbor_count;
    }
    if (neighbor_count > max_neighbor_count) {
      max_neighbor_count = neighbor_count;
    }
    sum_neighbor_count += neighbor_count;

    graph_meta.emplace_back(offset, neighbor_count);
    size_t size = neighbors.size() * sizeof(node_id_t);
    const node_id_t *data = &neighbors[0];
    if (!neighbor_mapping.empty()) {
      for (node_id_t i = 0; i < neighbors.size(); ++i) {
        mapping[i] = neighbor_mapping[neighbors[i]];
      }
      data = mapping;
    }
    if (dumper->write(data, size) != size) {
      LOG_ERROR("Dump graph neighbor id=%u failed, size %lu", id, size);
      return IndexError_WriteData;
    }
    crc = ailego::Crc32c::Hash(data, size, crc);
    offset += size;
  }

  uint32_t average_neighbor_count = 0;
  if (doc_cnt() > 0) {
    average_neighbor_count = sum_neighbor_count / doc_cnt();
  }
  LOG_INFO(
      "Dump hnsw graph: min_neighbor_count[%u] max_neighbor_count[%u] "
      "average_neighbor_count[%u]",
      min_neighbor_count, max_neighbor_count, average_neighbor_count);

  size_t padding_size = 0;
  int ret = CalcAndAddPadding(dumper, offset, &padding_size);
  if (ret != 0) {
    return ret;
  }
  ret = dumper->append(kGraphNeighborsSegmentId, offset, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s failed, ret %d",
              kGraphNeighborsSegmentId.c_str(), ret);
    return ret;
  }

  //! dump level 0 neighbors meta
  auto len = dump_segment(dumper, kGraphOffsetsSegmentId, graph_meta.data(),
                          graph_meta.size() * sizeof(GraphNeighborMeta));
  if (len < 0) {
    return len;
  }

  return len + offset + padding_size;
}

int64_t HnswEntity::dump_upper_neighbors(
    const IndexDumper::Pointer &dumper,
    const std::function<level_t(node_id_t)> &get_level,
    const std::vector<node_id_t> &reorder_mapping,
    const std::vector<node_id_t> &neighbor_mapping) const {
  std::vector<HnswNeighborMeta> hnsw_meta;
  hnsw_meta.reserve(doc_cnt());
  size_t offset = 0;
  uint32_t crc = 0;
  node_id_t buffer[upper_neighbor_cnt() + 1];
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    node_id_t new_id = reorder_mapping.empty() ? id : reorder_mapping[id];
    auto level = get_level(new_id);
    if (level == 0) {
      hnsw_meta.emplace_back(0U, 0U);
      continue;
    }
    hnsw_meta.emplace_back(offset, level);
    ailego_assert_with((size_t)level < kMaxGraphLayers, "invalid level");
    for (level_t cur_level = 1; cur_level <= level; ++cur_level) {
      const Neighbors neighbors = get_neighbors(cur_level, new_id);
      ailego_assert_with(!!neighbors.data, "invalid neighbors");
      ailego_assert_with(neighbors.size() <= neighbor_cnt(cur_level),
                         "invalid neighbors");
      memset(buffer, 0, sizeof(buffer));
      buffer[0] = neighbors.size();
      if (neighbor_mapping.empty()) {
        memcpy(&buffer[1], &neighbors[0], neighbors.size() * sizeof(node_id_t));
      } else {
        for (node_id_t i = 0; i < neighbors.size(); ++i) {
          buffer[i + 1] = neighbor_mapping[neighbors[i]];
        }
      }
      if (dumper->write(buffer, sizeof(buffer)) != sizeof(buffer)) {
        LOG_ERROR("Dump graph neighbor id=%u failed, size %lu", id,
                  sizeof(buffer));
        return IndexError_WriteData;
      }
      crc = ailego::Crc32c::Hash(buffer, sizeof(buffer), crc);
      offset += sizeof(buffer);
    }
  }
  size_t padding_size = 0;
  int ret = CalcAndAddPadding(dumper, offset, &padding_size);
  if (ret != 0) {
    return ret;
  }

  ret = dumper->append(kHnswNeighborsSegmentId, offset, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s failed, ret %d", kHnswNeighborsSegmentId.c_str(),
              ret);
    return ret;
  }

  //! dump level 0 neighbors meta
  auto len = dump_segment(dumper, kHnswOffsetsSegmentId, hnsw_meta.data(),
                          hnsw_meta.size() * sizeof(HnswNeighborMeta));
  if (len < 0) {
    return len;
  }

  return len + offset + padding_size;
}

}  // namespace core
}  // namespace zvec
