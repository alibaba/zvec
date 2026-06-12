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

#include "diskann_pq_trainer.h"
#include "diskann_entity.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

DiskAnnPqTrainer::DiskAnnPqTrainer(uint32_t max_train_sample_count)
    : max_train_sample_count_{max_train_sample_count} {}

DiskAnnPqTrainer::~DiskAnnPqTrainer() {}

int DiskAnnPqTrainer::gen_random_sample(IndexHolder::Pointer holder,
                                        const IndexMeta &meta,
                                        std::string &sample_data,
                                        size_t &sample_size) {
  double train_sample_ratio =
      max_train_sample_count_ < 1 ? max_train_sample_count_ : 1;

  uint32_t max_train_sample_count = train_sample_ratio * holder->count();
  max_train_sample_count = max_train_sample_count > max_train_sample_count_
                               ? max_train_sample_count_
                               : max_train_sample_count;

  std::vector<std::vector<uint8_t>> sample_vecs;

  // Use a fixed seed for deterministic sampling across runs.
  uint32_t x = 456321;
  std::mt19937 gen(x);
  std::uniform_real_distribution<float> dist(0, 1);

  uint32_t vec_size = meta.element_size();

  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }

  size_t sample_count = 0;
  while (iter->is_valid() && sample_count < max_train_sample_count) {
    float random = dist(gen);

    if (random < train_sample_ratio) {
      const void *vec = iter->data();

      std::vector<uint8_t> temp_vec;
      temp_vec.resize(vec_size);

      std::memcpy(reinterpret_cast<uint8_t *>(&temp_vec[0]), vec, vec_size);

      sample_vecs.push_back(std::move(temp_vec));

      sample_count++;
    }

    iter->next();
  }

  sample_size = sample_vecs.size();
  sample_data.reserve(sample_size * vec_size);

  for (size_t i = 0; i < sample_size; i++) {
    sample_data.append(reinterpret_cast<const char *>(sample_vecs[i].data()),
                       vec_size);
  }

  return 0;
}

template <typename T>
int DiskAnnPqTrainer::prepare_pq_train_data(
    const IndexMeta &meta, size_t num_train, std::string &train_data,
    bool use_zero_mean, std::vector<uint8_t> &centroid,
    std::shared_ptr<CompactIndexFeatures> &train_features) {
  uint32_t dim = meta.dimension();
  uint32_t vec_size = meta.element_size();

  std::string train_data_processed;
  train_data_processed.resize(num_train * vec_size);

  std::memcpy(&(train_data_processed[0]), train_data.data(),
              num_train * vec_size);

  // use fp32 to accumulate to avoid overflow
  std::vector<float> centroid_temp(dim);
  for (uint64_t d = 0; d < dim; d++) {
    centroid_temp[d] = 0;
  }

  T *train_data_processed_ptr = reinterpret_cast<T *>(&train_data_processed[0]);

  if (use_zero_mean) {
    for (uint64_t d = 0; d < dim; d++) {
      for (uint64_t p = 0; p < num_train; p++) {
        centroid_temp[d] += train_data_processed_ptr[p * dim + d];
      }
      centroid_temp[d] /= num_train;
    }

    for (uint64_t d = 0; d < dim; d++) {
      for (uint64_t p = 0; p < num_train; p++) {
        train_data_processed_ptr[p * dim + d] -= centroid_temp[d];
      }
    }
  }

  for (size_t i = 0; i < num_train; ++i) {
    train_features->emplace(train_data_processed_ptr + i * dim);
  }

  // copy the centroid out
  centroid.resize(vec_size);
  T *centroid_ptr = reinterpret_cast<T *>(centroid.data());
  for (uint64_t d = 0; d < dim; d++) {
    centroid_ptr[d] = centroid_temp[d];
  }

  return 0;
}

template <typename T>
int DiskAnnPqTrainer::convert_pivot_data(
    const IndexMeta &meta, uint32_t num_centers, uint32_t pq_chunk_num,
    const std::vector<uint32_t> &chunk_dims,
    const std::vector<uint32_t> &chunk_offsets,
    IndexCluster::CentroidList &centroids,
    std::vector<uint8_t> &full_pivot_data) {
  uint32_t dim = meta.dimension();
  uint32_t element_size = meta.element_size();

  full_pivot_data.resize(num_centers * element_size);

  for (size_t chunk = 0; chunk < pq_chunk_num; ++chunk) {
    for (size_t cluster = 0; cluster < num_centers; ++cluster) {
      size_t idx = chunk * num_centers + cluster;

      T *pivot_data_ptr = reinterpret_cast<T *>(&(full_pivot_data[0])) +
                          cluster * dim + chunk_offsets[chunk];
      const T *feature_ptr =
          reinterpret_cast<const T *>(centroids[idx].feature());
      for (size_t d = 0; d <= chunk_dims[chunk]; ++d) {
        pivot_data_ptr[d] = feature_ptr[d];
      }
    }
  }

  return 0;
}

int DiskAnnPqTrainer::train_pq(IndexThreads::Pointer threads,
                               const IndexMeta &meta, std::string &train_data,
                               size_t num_train, uint32_t num_centers,
                               uint32_t pq_chunk_num, uint32_t max_iterations,
                               bool use_zero_mean,
                               std::vector<uint8_t> &full_pivot_data,
                               std::vector<uint8_t> &centroid,
                               std::vector<uint32_t> &chunk_offsets) {
  uint32_t dim = meta.dimension();
  if (pq_chunk_num > dim) {
    LOG_ERROR("Error: number of chunks more than dimension. chunk: %u, dim: %u",
              pq_chunk_num, dim);
    return IndexError_InvalidArgument;
  }

  std::shared_ptr<CompactIndexFeatures> train_features(
      new CompactIndexFeatures(meta));

  uint32_t type = meta.data_type();

  int ret;
  switch (type) {
    case IndexMeta::DataType::DT_FP32:
      ret = prepare_pq_train_data<float>(
          meta, num_train, train_data, use_zero_mean, centroid, train_features);
      if (ret != 0) {
        LOG_ERROR("Failed to prepare pq train data");
        return ret;
      }
      break;

    case IndexMeta::DataType::DT_FP16:
      ret = prepare_pq_train_data<ailego::Float16>(
          meta, num_train, train_data, use_zero_mean, centroid, train_features);
      if (ret != 0) {
        LOG_ERROR("Failed to prepare pq train data");
        return ret;
      }
      break;
  }

  // Do Train
  ailego::Params params;
  params.set(MULTI_CHUNK_CLUSTER_COUNT, num_centers);
  params.set(MULTI_CHUNK_CLUSTER_CHUNK_COUNT, pq_chunk_num);
  params.set(MULTI_CHUNK_CLUSTER_MAX_ITERATIONS, max_iterations);

  ret = chunk_cluster_.init(meta, params);
  if (ret != 0) {
    LOG_ERROR("Failed to get chunk cluster");
    return IndexError_InvalidArgument;
  }

  ret = chunk_cluster_.mount(train_features);
  if (ret != 0) {
    LOG_ERROR("Cannot mount train features");
    return ret;
  }


  std::vector<uint32_t> labels;

  ret = chunk_cluster_.cluster(threads, cluster_centroids_);
  if (ret != 0) {
    LOG_ERROR("Failed to cluster");
    return ret;
  }

  chunk_offsets = chunk_cluster_.chunk_dim_offsets();
  auto chunk_dims = chunk_cluster_.chunk_dims();

  switch (type) {
    case IndexMeta::DataType::DT_FP32:
      ret = convert_pivot_data<float>(meta, num_centers, pq_chunk_num,
                                      chunk_dims, chunk_offsets,
                                      cluster_centroids_, full_pivot_data);
      if (ret != 0) {
        LOG_ERROR("Failed to convert pivot data");
        return ret;
      }
      break;

    case IndexMeta::DataType::DT_FP16:
      ret = convert_pivot_data<ailego::Float16>(
          meta, num_centers, pq_chunk_num, chunk_dims, chunk_offsets,
          cluster_centroids_, full_pivot_data);
      if (ret != 0) {
        LOG_ERROR("Failed to convert pivot data");
        return ret;
      }
      break;
  }

  return 0;
}

int DiskAnnPqTrainer::train_quantized_data(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    const IndexMeta &meta, std::vector<uint8_t> &pq_full_pivot_data,
    std::vector<uint8_t> &pq_centroid, std::vector<uint32_t> &pq_chunk_offsets,
    size_t pq_chunk_num) {
  size_t train_size;
  std::string train_data;

  int ret = gen_random_sample(holder, meta, train_data, train_size);
  if (ret != 0) {
    LOG_ERROR("Get Random Sample Error, ret: %d", ret);
    return ret;
  }

  LOG_INFO("Training data with %zu samples loaded.", train_size);

  // bool use_zero_mean = (meta.metric_name() != "InnerProduct" ? true :
  // false);
  bool use_zero_mean = false;

  ret = train_pq(threads, meta, train_data, train_size, PQTable::kPQCentroidNum,
                 pq_chunk_num, PQTable::kMeanIterNum, use_zero_mean,
                 pq_full_pivot_data, pq_centroid, pq_chunk_offsets);
  if (ret != 0) {
    LOG_ERROR("Train PQ Error, ret: %d", ret);
    return ret;
  }

  return 0;
}

int DiskAnnPqTrainer::generate_pq(IndexThreads::Pointer threads,
                                  const IndexMeta &meta,
                                  IndexHolder::Pointer holder,
                                  uint32_t pq_chunk_num,
                                  std::vector<uint8_t> &centroid,
                                  std::vector<uint8_t> &block_compressed_data) {
  uint32_t type = meta.data_type();
  uint32_t dim = meta.dimension();

  if (pq_chunk_num > dim) {
    LOG_ERROR("Error: number of chunks more than dimension. chunk: %u, dim: %u",
              pq_chunk_num, dim);
    return IndexError_InvalidArgument;
  }

  // Do Label
  std::vector<uint32_t> labels;
  size_t num_vecs = holder->count();
  size_t batch_size =
      num_vecs <= compress_batch_size_ ? num_vecs : compress_batch_size_;

  std::vector<uint32_t> block_compressed_base(batch_size * pq_chunk_num);

  std::memset(&block_compressed_base[0], 0,
              batch_size * pq_chunk_num * sizeof(uint32_t));

  std::vector<uint8_t> block_data(batch_size * meta.element_size());
  std::vector<uint8_t> block_data_converted(batch_size * meta.element_size());

  size_t block_num = DiskAnnUtil::div_round_up(num_vecs, batch_size);

  block_compressed_data.resize(num_vecs * pq_chunk_num);

  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }

  for (size_t block = 0; block < block_num; block++) {
    size_t start_id = block * batch_size;
    size_t end_id = std::min((block + 1) * batch_size, num_vecs);

    size_t cur_block_size = end_id - start_id;

    for (size_t i = 0; i < cur_block_size && iter->is_valid(); i++) {
      const void *vec = iter->data();
      std::memcpy(
          reinterpret_cast<uint8_t *>(&block_data[0]) + i * meta.element_size(),
          vec, meta.element_size());
      iter->next();
    }

    std::memcpy(block_data_converted.data(), block_data.data(),
                cur_block_size * meta.element_size());

    LOG_INFO("Processing Docs, Range: [%zu, %zu)..", start_id, end_id);

    std::shared_ptr<CompactIndexFeatures> block_features(
        new CompactIndexFeatures(meta));

    switch (type) {
      case IndexMeta::DataType::DT_FP32:
        DiskAnnUtil::convert_vector_to_residual<float>(
            reinterpret_cast<float *>(block_data_converted.data()),
            cur_block_size, dim, centroid.data());
        break;
      case IndexMeta::DataType::DT_FP16:
        DiskAnnUtil::convert_vector_to_residual<ailego::Float16>(
            reinterpret_cast<ailego::Float16 *>(block_data_converted.data()),
            cur_block_size, dim, centroid.data());
        break;
      default:
        return IndexError_InvalidArgument;
    }

    for (size_t i = 0; i < cur_block_size; i++) {
      block_features->emplace(block_data_converted.data() +
                              i * meta.element_size());
    }

    int ret = chunk_cluster_.mount(block_features);
    if (ret != 0) {
      LOG_ERROR("Cannot mount block features");
      return ret;
    }

    ret = chunk_cluster_.label(threads, cluster_centroids_, &labels);
    if (ret != 0) {
      LOG_ERROR("Failed to label");
      return ret;
    }

    std::vector<uint8_t> compressed_data(cur_block_size * pq_chunk_num);

    DiskAnnUtil::convert_types_uint32_to_uint8(
        labels.data(), compressed_data.data(), cur_block_size, pq_chunk_num);

    memcpy(&(block_compressed_data[0]) + start_id * pq_chunk_num,
           compressed_data.data(), cur_block_size * pq_chunk_num);

    LOG_INFO("Generate PQ Data Done.");
  }

  return 0;
}

int DiskAnnPqTrainer::generate_quantized_data(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    const IndexMeta &meta, std::vector<uint8_t> &pq_centroid,
    std::vector<uint8_t> &block_compressed_data, size_t pq_chunk_num) {
  // bool use_zero_mean = (meta.metric_name() != "InnerProduct" ? true :
  // false);

  int ret = generate_pq(threads, meta, holder, pq_chunk_num, pq_centroid,
                        block_compressed_data);
  if (ret != 0) {
    LOG_ERROR("Generate PQ Error, ret: %d", ret);
    return ret;
  }

  return 0;
}

}  // namespace core
}  // namespace zvec
