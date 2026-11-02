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

#include "diskann_partition_build.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#if defined(_WIN32)
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif
#include "algorithm/cluster/vector_mean.h"
#include "diskann_builder.h"
#include "diskann_params.h"
#include "diskann_util.h"

namespace zvec::core {
namespace {
// All seeks are 64-bit, including Windows. Anonymous files are removed by
// fclose, including on failed builds. Files survive dump failures for retry.
class ScratchFile {
 public:
  explicit ScratchFile(const std::string &directory)
      : directory_(directory), file_(open(directory), &std::fclose) {
    if (!file_) error("create", IndexError_OpenFile);
  }
  bool valid() const {
    return file_ != nullptr;
  }
  int seek(uint64_t offset) {
    if (!file_) return IndexError_OpenFile;
    if (offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return IndexError_InvalidLength;
#ifdef _WIN32
    return _fseeki64(file_.get(), static_cast<int64_t>(offset), SEEK_SET) == 0
               ? 0
               : error("seek", IndexError_ReadData);
#else
    return fseeko(file_.get(), static_cast<off_t>(offset), SEEK_SET) == 0
               ? 0
               : error("seek", IndexError_ReadData);
#endif
  }
  int read(void *data, size_t bytes) {
    if (!file_) return IndexError_OpenFile;
    errno = 0;
    return std::fread(data, 1, bytes, file_.get()) == bytes
               ? 0
               : error("read", IndexError_ReadData);
  }
  int write(const void *data, size_t bytes) {
    if (!file_) return IndexError_OpenFile;
    errno = 0;
    return std::fwrite(data, 1, bytes, file_.get()) == bytes
               ? 0
               : error("write", IndexError_WriteData);
  }
  int read_at(uint64_t offset, void *data, size_t bytes) {
    int ret = seek(offset);
    return ret ? ret : read(data, bytes);
  }
  int write_at(uint64_t offset, const void *data, size_t bytes) {
    int ret = seek(offset);
    return ret ? ret : write(data, bytes);
  }
  int flush() {
    errno = 0;
    return file_ && std::fflush(file_.get()) == 0
               ? 0
               : error("flush", IndexError_WriteData);
  }
  int close() {
    if (!file_) return 0;
    errno = 0;
    return std::fclose(file_.release()) == 0
               ? 0
               : error("close", IndexError_WriteData);
  }

 private:
  // Exclusive creation avoids overwriting an existing file. POSIX unlinks
  // immediately; Windows deletes on close. Never fall back from an explicit
  // directory, and never create or remove the caller's directory itself.
  static FILE *open(const std::string &directory) {
    if (directory.empty()) return std::tmpfile();
    static std::atomic<uint64_t> sequence{0};
    for (unsigned attempt = 0; attempt < 128; ++attempt) {
      const auto stamp =
          std::chrono::steady_clock::now().time_since_epoch().count();
      const std::string path = directory + "/zvec-diskann-" +
                               std::to_string(stamp) + "-" +
                               std::to_string(sequence.fetch_add(1)) + ".tmp";
      int fd = -1;
#ifdef _WIN32
      const auto wide = ailego::FileHelper::Utf8ToWide(path);
      if (wide.empty()) {
        errno = EINVAL;
        return nullptr;
      }
      const int ret = _wsopen_s(&fd, wide.c_str(),
                                _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY |
                                    _O_TEMPORARY | _O_NOINHERIT,
                                _SH_DENYRW, _S_IREAD | _S_IWRITE);
      if (ret != 0) {
        if (ret == EEXIST) continue;
        errno = ret;
        return nullptr;
      }
      FILE *file = _fdopen(fd, "w+b");
      if (!file) {
        const int saved = errno;
        _close(fd);
        errno = saved;
      }
#else
      fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
                  S_IRUSR | S_IWUSR);
      if (fd == -1) {
        if (errno == EEXIST) continue;
        return nullptr;
      }
      if (::unlink(path.c_str()) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        return nullptr;
      }
      FILE *file = ::fdopen(fd, "w+b");
      if (!file) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
      }
#endif
      return file;
    }
    errno = EEXIST;
    return nullptr;
  }

  int error(const char *operation, int code) const {
    const int saved = errno ? errno : EIO;
    LOG_ERROR(
        "DiskAnn temporary file %s failed in directory '%s': %s "
        "(errno=%d)",
        operation, directory_.empty() ? "<system default>" : directory_.c_str(),
        std::strerror(saved), saved);
    return code;
  }

  std::string directory_;
  std::unique_ptr<FILE, decltype(&std::fclose)> file_;
};

struct KeyId {
  uint64_t key;
  uint32_t id;
  uint32_t reserved{0};
  bool operator<(const KeyId &rhs) const {
    return key < rhs.key || (key == rhs.key && id < rhs.id);
  }
};

// Two buffered run readers let the key mapping be externally sorted with a
// fixed memory footprint and only two files, regardless of the run count.
struct KeyRun {
  ScratchFile &file;
  uint64_t position, end;
  std::vector<KeyId> buffer = std::vector<KeyId>(4096);
  size_t index{0}, size{0};
  int next(KeyId *item, bool *valid) {
    if (index == size) {
      size = static_cast<size_t>(
          std::min<uint64_t>(buffer.size(), end - position));
      index = 0;
      if (size) {
        int ret = file.read_at(position * sizeof(KeyId), buffer.data(),
                               size * sizeof(KeyId));
        if (ret) return ret;
        position += size;
      }
    }
    *valid = index < size;
    if (*valid) *item = buffer[index++];
    return 0;
  }
};

// Coordinates of one node and its (at most 2R) merge candidates. Global IDs
// never become array offsets into this bounded buffer.
class MergeVectors : public DiskAnnEntity {
 public:
  MergeVectors(size_t capacity, size_t element)
      : element_(element), data_(capacity * element) {
    *mutable_doc_cnt() = capacity;
  }
  const void *get_vector(diskann_id_t id) const override {
    return data_.data() + id * element_;
  }
  void *at(size_t id) {
    return data_.data() + id * element_;
  }
  diskann_id_t get_id(diskann_key_t key) const override {
    return static_cast<diskann_id_t>(key);
  }
  diskann_key_t get_key(diskann_id_t id) const override {
    return id;
  }

 private:
  size_t element_;
  std::vector<uint8_t> data_;
};

// A stable, bounded reservoir shared by the PQ trainer's readers. The raw
// element stride can include a cosine norm tail; PrefixIndexHolder supplies
// the quantizer's vector-only metadata without changing that stride.
class TrainingSample : public IndexHolder {
 public:
  TrainingSample(IndexMeta meta, size_t count)
      : meta_(std::move(meta)), data_(count * meta_.element_size()) {}
  size_t count() const override {
    return data_.size() / element_size();
  }
  size_t dimension() const override {
    return meta_.dimension();
  }
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }
  size_t element_size() const override {
    return meta_.element_size();
  }
  bool multipass() const override {
    return true;
  }
  void *at(size_t id) {
    return data_.data() + id * element_size();
  }
  class Iterator : public IndexHolder::Iterator {
   public:
    explicit Iterator(TrainingSample &sample) : sample_(sample) {}
    const void *data() const override {
      return sample_.at(id_);
    }
    uint64_t key() const override {
      return id_;
    }
    bool is_valid() const override {
      return id_ < sample_.count();
    }
    void next() override {
      ++id_;
    }

   private:
    TrainingSample &sample_;
    size_t id_{0};
  };
  IndexHolder::Iterator::Pointer create_iterator() override {
    return std::make_unique<Iterator>(*this);
  }

 private:
  IndexMeta meta_;
  std::vector<uint8_t> data_;
};

// Every I/O operation is checked, including delayed stdio errors at flush.
#define PARTITION_TRY(expr)         \
  do {                              \
    int result = (expr);            \
    if (result != 0) return result; \
  } while (false)
}  // namespace

struct DiskAnnPartitionBuild::Impl {
  explicit Impl(DiskAnnBuilder &builder)
      : owner(builder),
        vectors(builder.build_temp_dir_),
        keys(builder.build_temp_dir_),
        graph(builder.build_temp_dir_),
        codes(builder.build_temp_dir_),
        mapping(builder.build_temp_dir_) {}
  DiskAnnBuilder &owner;
  ScratchFile vectors, keys, graph, codes, mapping;
  size_t count{0}, element{0}, capacity{0};
  uint32_t degree{0};
  std::vector<uint32_t> entrypoints;
  std::vector<uint8_t> sample;
  std::vector<uint32_t> sample_ids;
  // Global mean in the raw element layout; the medoid is the vector nearest
  // to it, matching DiskAnnBuilder::calculate_entry_point without an
  // in-memory copy of the dataset.
  std::vector<uint8_t> centroid;

  int sort_keys(std::unique_ptr<ScratchFile> runs) {
    for (uint64_t width = capacity; width < count; width *= 2) {
      auto output = std::make_unique<ScratchFile>(owner.build_temp_dir_);
      if (!output->valid()) return IndexError_OpenFile;
      for (uint64_t start = 0; start < count; start += 2 * width) {
        KeyRun left{*runs, start, std::min<uint64_t>(start + width, count)};
        KeyRun right{*runs, std::min<uint64_t>(start + width, count),
                     std::min<uint64_t>(start + 2 * width, count)};
        KeyId a{}, b{};
        bool have_a = false, have_b = false;
        PARTITION_TRY(left.next(&a, &have_a));
        PARTITION_TRY(right.next(&b, &have_b));
        while (have_a || have_b) {
          if (have_a && (!have_b || a < b)) {
            PARTITION_TRY(output->write(&a, sizeof(a)));
            PARTITION_TRY(left.next(&a, &have_a));
          } else {
            PARTITION_TRY(output->write(&b, sizeof(b)));
            PARTITION_TRY(right.next(&b, &have_b));
          }
        }
      }
      PARTITION_TRY(output->flush());
      runs = std::move(output);
    }
    PARTITION_TRY(runs->seek(0));
    for (size_t i = 0; i < count; ++i) {
      KeyId item{};
      PARTITION_TRY(runs->read(&item, sizeof(item)));
      PARTITION_TRY(mapping.write(&item.id, sizeof(item.id)));
    }
    return mapping.flush();
  }

  int snapshot(IndexHolder::Pointer holder) {
    auto runs = std::make_unique<ScratchFile>(owner.build_temp_dir_);
    if (!runs->valid()) return IndexError_OpenFile;
    auto iter = holder->create_iterator();
    if (!iter) return IndexError_Runtime;
    const size_t sample_count = std::min<size_t>(
        count,
        std::min<size_t>(4096, owner.build_memory_budget_ / 16 / element));
    if (sample_count == 0) return IndexError_NoMemory;
    sample.resize(sample_count * element);
    sample_ids.resize(sample_count);
    std::vector<KeyId> key_run;
    key_run.reserve(capacity);
    const size_t dimension = owner.build_meta_.dimension();
    const bool fp16 = owner.build_meta_.data_type() == IndexMeta::DT_FP16;
    const size_t mean_bytes =
        dimension * (fp16 ? sizeof(ailego::Float16) : sizeof(float));
    std::unique_ptr<VectorMean> accumulator;
    if (fp16)
      accumulator.reset(new NumericalVectorMean<ailego::Float16>(dimension));
    else
      accumulator.reset(new NumericalVectorMean<float>(dimension));
    std::mt19937 random(42);
    size_t id = 0;
    for (; iter->is_valid(); iter->next(), ++id) {
      if (id >= count) return IndexError_Mismatch;
      PARTITION_TRY(iter->status());
      const auto key = iter->key();
      const void *data = iter->data();
      PARTITION_TRY(iter->status());
      if (!data || !iter->is_valid()) return IndexError_ReadData;
      if (!accumulator->plus(data, mean_bytes)) return IndexError_Mismatch;
      PARTITION_TRY(vectors.write(data, element));
      PARTITION_TRY(keys.write(&key, sizeof(key)));
      key_run.push_back({key, static_cast<uint32_t>(id), 0});
      if (key_run.size() == capacity || id + 1 == count) {
        std::sort(key_run.begin(), key_run.end());
        PARTITION_TRY(
            runs->write(key_run.data(), key_run.size() * sizeof(KeyId)));
        key_run.clear();
      }
      size_t slot = id;
      if (id >= sample_count)
        slot = std::uniform_int_distribution<size_t>(0, id)(random);
      if (slot < sample_count) {
        memcpy(sample.data() + slot * element, data, element);
        sample_ids[slot] = id;
      }
    }
    PARTITION_TRY(iter->status());
    if (id != count) return IndexError_Mismatch;
    iter.reset();
    centroid.assign(element, 0);
    if (count && !accumulator->mean(centroid.data(), mean_bytes))
      return IndexError_Mismatch;
    PARTITION_TRY(vectors.flush());
    PARTITION_TRY(keys.flush());
    PARTITION_TRY(runs->flush());
    std::vector<KeyId>().swap(key_run);
    return sort_keys(std::move(runs));
  }

  int merge_shard(DiskAnnBuilder &shard, const std::vector<uint32_t> &ids,
                  MergeVectors &merge_vectors, DiskAnnContext &context) {
    const size_t stride = (degree + 1) * sizeof(uint32_t);
    std::vector<uint32_t> row(degree + 1), candidates;
    std::vector<Neighbor> pool;
    std::vector<uint32_t> selected;
    auto &dc = context.dist_calculator();
    for (size_t local = 0; local < ids.size(); ++local) {
      const uint32_t id = ids[local];
      PARTITION_TRY(graph.read_at(static_cast<uint64_t>(id) * stride,
                                  row.data(), stride));
      if (row[0] > degree) return IndexError_InvalidFormat;
      candidates.assign(row.begin() + 1, row.begin() + 1 + row[0]);
      auto neighbors = shard.entity_.get_neighbors(local);
      for (size_t j = 0; j < neighbors.first; ++j) {
        if (neighbors.second[j] >= ids.size()) return IndexError_InvalidFormat;
        candidates.push_back(ids[neighbors.second[j]]);
      }
      std::sort(candidates.begin(), candidates.end());
      candidates.erase(std::unique(candidates.begin(), candidates.end()),
                       candidates.end());
      candidates.erase(std::remove(candidates.begin(), candidates.end(), id),
                       candidates.end());
      if (candidates.size() > degree) {
        memcpy(merge_vectors.at(0), shard.entity_.get_vector(local), element);
        pool.clear();
        selected.clear();
        for (size_t j = 0; j < candidates.size(); ++j) {
          PARTITION_TRY(
              vectors.read_at(static_cast<uint64_t>(candidates[j]) * element,
                              merge_vectors.at(j + 1), element));
          float distance =
              dc.dist(merge_vectors.at(0), merge_vectors.at(j + 1));
          if (!std::isfinite(distance)) return IndexError_InvalidArgument;
          pool.emplace_back(static_cast<uint32_t>(j + 1), distance);
        }
        PARTITION_TRY(
            owner.algo_->prune_neighbors(0, pool, selected, &context));
        row[0] = selected.size();
        for (size_t j = 0; j < selected.size(); ++j)
          row[j + 1] = candidates[selected[j] - 1];
      } else {
        row[0] = candidates.size();
        std::copy(candidates.begin(), candidates.end(), row.begin() + 1);
      }
      PARTITION_TRY(graph.write_at(static_cast<uint64_t>(id) * stride,
                                   row.data(), stride));
    }
    return graph.flush();
  }

  int build_graph(IndexThreads::Pointer threads) {
    const size_t shard_count = std::min<size_t>(
        64, std::min<size_t>(
                sample_ids.size(),
                std::max<size_t>(2, (2 * count + capacity - 1) / capacity)));
    DistCalculator distance(&owner.entity_, owner.build_meta_, owner.metric_,
                            owner.data_quantizer_);
    // Farthest-point centers from a deterministic uniform reservoir. Using
    // actual vectors supports the same FP16/FP32 and metric layouts as build.
    std::vector<size_t> centers;
    std::vector<float> nearest(sample_ids.size(),
                               std::numeric_limits<float>::max());
    std::vector<bool> chosen(sample_ids.size(), false);
    size_t next = 0;
    for (size_t c = 0; c < shard_count; ++c) {
      centers.push_back(next);
      chosen[next] = true;
      float farthest = -std::numeric_limits<float>::infinity();
      size_t farthest_id = 0;
      for (size_t i = 0; i < sample_ids.size(); ++i) {
        float d = distance.dist(sample.data() + i * element,
                                sample.data() + next * element);
        if (!std::isfinite(d)) return IndexError_InvalidArgument;
        nearest[i] = std::min(nearest[i], d);
        if (!chosen[i] && nearest[i] > farthest) {
          farthest = nearest[i];
          farthest_id = i;
        }
      }
      next = farthest_id;
    }
    std::vector<std::unique_ptr<ScratchFile>> shards;
    std::vector<size_t> sizes(shard_count, 0);
    for (size_t c = 0; c < shard_count; ++c) {
      shards.push_back(std::make_unique<ScratchFile>(owner.build_temp_dir_));
      if (!shards.back()->valid()) return IndexError_OpenFile;
    }
    std::vector<uint8_t> vector(element);
    uint32_t medoid = 0;
    float medoid_dist = std::numeric_limits<float>::infinity();
    PARTITION_TRY(vectors.seek(0));
    for (uint32_t id = 0; id < count; ++id) {
      PARTITION_TRY(vectors.read(vector.data(), element));
      float centroid_dist = distance.dist(vector.data(), centroid.data());
      if (!std::isfinite(centroid_dist)) return IndexError_InvalidArgument;
      if (centroid_dist < medoid_dist) {
        medoid_dist = centroid_dist;
        medoid = id;
      }
      std::pair<float, size_t> first{std::numeric_limits<float>::infinity(), 0},
          second = first;
      for (size_t c = 0; c < shard_count; ++c) {
        float d =
            distance.dist(vector.data(), sample.data() + centers[c] * element);
        if (!std::isfinite(d)) return IndexError_InvalidArgument;
        std::pair<float, size_t> candidate{d, c};
        if (candidate < first) {
          second = first;
          first = candidate;
        } else if (candidate < second)
          second = candidate;
      }
      PARTITION_TRY(shards[first.second]->write(&id, sizeof(id)));
      ++sizes[first.second];
      if (shard_count > 1) {
        PARTITION_TRY(shards[second.second]->write(&id, sizeof(id)));
        ++sizes[second.second];
      }
    }
    *owner.entity_.mutable_medoid() = medoid;
    std::vector<uint8_t>().swap(sample);
    std::vector<uint32_t>().swap(sample_ids);
    std::vector<uint8_t>().swap(centroid);
    for (auto &shard : shards) PARTITION_TRY(shard->flush());

    // Initialize fixed-width rows on disk, never as an N*R in-memory array.
    std::vector<uint8_t> zero(64 * 1024, 0);
    uint64_t remaining =
        static_cast<uint64_t>(count) * (degree + 1) * sizeof(uint32_t);
    while (remaining) {
      size_t bytes = std::min<uint64_t>(remaining, zero.size());
      PARTITION_TRY(graph.write(zero.data(), bytes));
      remaining -= bytes;
    }
    PARTITION_TRY(graph.flush());
    auto merge_vectors =
        std::make_shared<MergeVectors>(2 * degree + 1, element);
    DiskAnnContext merge_context(owner.build_meta_, owner.metric_,
                                 merge_vectors, owner.data_quantizer_);
    PARTITION_TRY(merge_context.init(DiskAnnContext::kBuilderContext, degree,
                                     owner.pq_chunk_num_, element));
    size_t built_shards = 0;
    for (size_t c = 0; c < shard_count; ++c) {
      PARTITION_TRY(shards[c]->seek(0));
      size_t consumed = 0;
      uint32_t anchor = 0;
      while (consumed < sizes[c]) {
        std::vector<uint32_t> ids;
        ids.reserve(capacity);
        if (consumed) ids.push_back(anchor);
        size_t amount = std::min(capacity - ids.size(), sizes[c] - consumed);
        size_t begin = ids.size();
        ids.resize(begin + amount);
        PARTITION_TRY(
            shards[c]->read(ids.data() + begin, amount * sizeof(uint32_t)));
        if (!consumed) anchor = ids[0];
        consumed += amount;
        DiskAnnBuilder shard;
        ailego::Params params;
        params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, degree);
        params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, owner.list_size_);
        params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT,
                   static_cast<uint32_t>(threads->count()));
        PARTITION_TRY(
            shard.init(owner.raw_meta_, params, owner.data_quantizer_));
        shard.pq_chunk_num_ = owner.pq_chunk_num_;
        // The entity borrows vector pointers. Keep one fixed-size buffer per
        // shard alive through graph construction and merging; iterator/read
        // scratch would alias every node to the last vector read.
        auto shard_vectors =
            std::make_shared<std::vector<uint8_t>>(ids.size() * element);
        PARTITION_TRY(shard.entity_.reserve_space(ids.size(), shard_vectors));
        for (size_t local = 0; local < ids.size(); ++local) {
          uint32_t id = ids[local];
          auto *data = shard_vectors->data() + local * element;
          PARTITION_TRY(vectors.read_at(static_cast<uint64_t>(id) * element,
                                        data, element));
          PARTITION_TRY(shard.entity_.add_vector(id, data));
        }
        PARTITION_TRY(shard.calculate_entry_point());
        entrypoints.push_back(ids[shard.entity_.medoid()]);
        PARTITION_TRY(shard.build_internal(threads));
        PARTITION_TRY(shard.prune_internal(threads));
        PARTITION_TRY(merge_shard(shard, ids, *merge_vectors, merge_context));
        ++built_shards;
      }
      shards[c].reset();
    }
    LOG_INFO(
        "Partitioned DiskAnn: %zu vectors, capacity %zu, %zu centers, %zu "
        "shard builds, medoid %u",
        count, capacity, shard_count, built_shards, medoid);
    // Preserve one deterministic global cycle after pruning. This guarantees
    // reachability even for identical vectors or disconnected partitions.
    ailego::ElapsedTime timer;
    std::vector<uint32_t> row(degree + 1);
    const size_t stride = row.size() * sizeof(uint32_t);
    for (uint32_t id = 0; id < count; ++id) {
      PARTITION_TRY(graph.read_at(static_cast<uint64_t>(id) * stride,
                                  row.data(), stride));
      uint32_t successor = (static_cast<uint64_t>(id) + 1) % count;
      if (std::find(row.begin() + 1, row.begin() + 1 + row[0], successor) ==
          row.begin() + 1 + row[0]) {
        if (row[0] == degree) --row[0];
        row[++row[0]] = successor;
      }
      PARTITION_TRY(graph.write_at(static_cast<uint64_t>(id) * stride,
                                   row.data(), stride));
    }
    PARTITION_TRY(graph.flush());
    LOG_INFO("Partitioned DiskAnn: global cycle pass done, time: %zu ms",
             static_cast<size_t>(timer.milli_seconds()));
    return 0;
  }

  int build(IndexThreads::Pointer threads, IndexHolder::Pointer holder) {
    if (!threads || threads->count() == 0) return IndexError_InvalidArgument;
    if (!vectors.valid() || !keys.valid() || !graph.valid() || !codes.valid() ||
        !mapping.valid())
      return IndexError_OpenFile;
    if (!owner.quantizer_) return IndexError_NoReady;
    count = holder->count();
    element = owner.raw_meta_.element_size();
    degree = owner.max_degree_;
    const size_t workers = std::max<size_t>(1, threads->count());
    // Include both lock pools, merge coordinates, per-worker candidate/query
    // scratch, and room for stdio/sorting. Caller-owned input and OS caches
    // are outside this working-memory target.
    const uint64_t fixed =
        2ULL * (1U << 16) * sizeof(std::mutex) + 2 * 1024 * 1024 +
        (2ULL * degree + 2) * element +
        workers * (2 * element +
                   64ULL * (static_cast<uint64_t>(degree) + owner.list_size_) +
                   65536);
    if (fixed >= owner.build_memory_budget_) return IndexError_NoMemory;
    const uint64_t per_node =
        element +
        sizeof(uint32_t) *
            (1 + static_cast<uint32_t>(
                     degree * DiskAnnEntity::kDefaultGraphSlackFactor)) +
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(const void *) +
        sizeof(KeyId) + workers + 16;
    capacity = std::min<uint64_t>(
        count, (owner.build_memory_budget_ - fixed) / 2 / per_node);
    if (capacity < 2 && count > 1) return IndexError_NoMemory;
    capacity = std::max<size_t>(1, capacity);
    ailego::ElapsedTime timer;
    PARTITION_TRY(snapshot(holder));
    LOG_INFO("Partitioned DiskAnn: snapshot done, time: %zu ms",
             static_cast<size_t>(timer.milli_seconds()));
    PARTITION_TRY(build_graph(threads));
    timer.reset();
    // Encode once using the globally trained codebook; never concatenate
    // independently trained shard PQ codes.
    PARTITION_TRY(vectors.seek(0));
    const size_t batch_size = std::max<size_t>(
        1, std::min<size_t>(count,
                            std::min<size_t>(4 * 1024 * 1024,
                                             owner.build_memory_budget_ / 8) /
                                element));
    std::vector<uint8_t> batch(batch_size * element),
        encoded(batch_size * owner.pq_chunk_num_);
    for (size_t start = 0; start < count; start += batch_size) {
      size_t rows = std::min(batch_size, count - start);
      PARTITION_TRY(vectors.read(batch.data(), rows * element));
      if (workers > 1) {
        auto group = threads->make_group();
        if (!group) return IndexError_Runtime;
        const size_t stripe = DiskAnnUtil::div_round_up(rows, workers);
        for (size_t t = 0; t < workers; ++t) {
          size_t begin = t * stripe, end = std::min(begin + stripe, rows);
          if (begin >= end) break;
          group->submit(ailego::Closure::New(this, &Impl::encode_rows,
                                             batch.data(), encoded.data(),
                                             begin, end));
        }
        group->wait_finish();
      } else {
        encode_rows(batch.data(), encoded.data(), 0, rows);
      }
      PARTITION_TRY(codes.write(encoded.data(), rows * owner.pq_chunk_num_));
    }
    PARTITION_TRY(codes.flush());
    LOG_INFO("Partitioned DiskAnn: PQ encode done, time: %zu ms",
             static_cast<size_t>(timer.milli_seconds()));
    *owner.entity_.mutable_doc_cnt() = count;
    return 0;
  }

  void encode_rows(const uint8_t *batch, uint8_t *encoded, size_t begin,
                   size_t end) {
    for (size_t i = begin; i < end; ++i)
      owner.quantizer_->quantize_data(batch + i * element,
                                      encoded + i * owner.pq_chunk_num_);
  }

  int dump_file(ScratchFile &file, const std::string &name, uint64_t size,
                const IndexDumper::Pointer &dumper) {
    PARTITION_TRY(file.seek(0));
    std::vector<uint8_t> buffer(64 * 1024);
    uint32_t crc = 0;
    for (uint64_t offset = 0; offset < size;) {
      size_t bytes = std::min<uint64_t>(buffer.size(), size - offset);
      PARTITION_TRY(file.read(buffer.data(), bytes));
      if (dumper->write(buffer.data(), bytes) != bytes)
        return IndexError_WriteData;
      crc = ailego::Crc32c::Hash(buffer.data(), bytes, crc);
      offset += bytes;
    }
    const size_t padding = DiskAnnEntity::AlignSize(size) - size;
    std::fill(buffer.begin(), buffer.end(), 0);
    if (padding && dumper->write(buffer.data(), padding) != padding)
      return IndexError_WriteData;
    return dumper->append(name, size, padding, crc);
  }

  int dump(const IndexDumper::Pointer &dumper) {
    PARTITION_TRY(owner.entity_.dump_dummy_segment(dumper));
    const size_t node_size = element + (degree + 1) * sizeof(uint32_t);
    const size_t per_sector = DiskAnnUtil::kSectorSize / node_size;
    const size_t block_size =
        per_sector ? DiskAnnUtil::kSectorSize
                   : DiskAnnUtil::round_up(node_size, DiskAnnUtil::kSectorSize);
    std::vector<uint8_t> block(block_size);
    std::vector<uint32_t> row(degree + 1);
    PARTITION_TRY(vectors.seek(0));
    PARTITION_TRY(graph.seek(0));
    uint64_t written = 0;
    uint32_t crc = 0;
    for (size_t id = 0; id < count;) {
      std::fill(block.begin(), block.end(), 0);
      const size_t rows =
          std::min<size_t>(per_sector ? per_sector : 1, count - id);
      for (size_t i = 0; i < rows; ++i, ++id) {
        auto *node = block.data() + i * node_size;
        PARTITION_TRY(vectors.read(node, element));
        PARTITION_TRY(graph.read(row.data(), row.size() * sizeof(uint32_t)));
        if (row[0] == 0 || row[0] > degree) return IndexError_InvalidFormat;
        memcpy(node + element, row.data(), (row[0] + 1) * sizeof(uint32_t));
      }
      if (dumper->write(block.data(), block.size()) != block.size())
        return IndexError_WriteData;
      crc = ailego::Crc32c::Hash(block.data(), block.size(), crc);
      written += block.size();
    }
    PARTITION_TRY(dumper->append(DiskAnnEntity::kDiskAnnVectorSegmentId,
                                 written, 0, crc));
    DiskAnnMetaHeader meta;
    meta.doc_cnt = count;
    meta.ndims = owner.raw_meta_.dimension();
    meta.medoid = owner.entity_.medoid();
    meta.max_node_size = node_size;
    meta.max_degree = degree;
    meta.node_per_sector = per_sector;
    meta.vamana_frozen_loc = meta.medoid;
    meta.index_size = written;
    PARTITION_TRY(owner.entity_.dump_segment(
        dumper, DiskAnnEntity::kDiskAnnMetaSegmentId, &meta, sizeof(meta)));
    PARTITION_TRY(owner.entity_.dump_pq_meta_segment(dumper));
    PARTITION_TRY(dump_file(codes, DiskAnnEntity::kDiskAnnPqDataSegmentId,
                            static_cast<uint64_t>(count) * owner.pq_chunk_num_,
                            dumper));
    PARTITION_TRY(dump_file(keys, DiskAnnEntity::kDiskAnnKeySegmentId,
                            static_cast<uint64_t>(count) * sizeof(uint64_t),
                            dumper));
    PARTITION_TRY(dump_file(mapping, DiskAnnEntity::kDiskAnnKeyMappingSegmentId,
                            static_cast<uint64_t>(count) * sizeof(uint32_t),
                            dumper));
    std::sort(entrypoints.begin(), entrypoints.end());
    entrypoints.erase(std::unique(entrypoints.begin(), entrypoints.end()),
                      entrypoints.end());
    std::vector<uint32_t> entry_buffer{
        static_cast<uint32_t>(entrypoints.size())};
    entry_buffer.insert(entry_buffer.end(), entrypoints.begin(),
                        entrypoints.end());
    return owner.entity_.dump_segment(
        dumper, DiskAnnEntity::kDiskAnnEntryPointSegmentId, entry_buffer.data(),
        entry_buffer.size() * sizeof(uint32_t));
  }
};

DiskAnnPartitionBuild::DiskAnnPartitionBuild(DiskAnnBuilder &owner)
    : impl_(std::make_unique<Impl>(owner)) {}
DiskAnnPartitionBuild::~DiskAnnPartitionBuild() = default;
int DiskAnnPartitionBuild::check_temp_directory(const std::string &directory) {
  try {
    // Use the same creation and cleanup path as the build. A small flushed
    // write detects immediate I/O failures, but does not reserve build space.
    ScratchFile probe(directory);
    if (!probe.valid()) return IndexError_OpenFile;
    const char data[4096] = {};
    PARTITION_TRY(probe.write(data, sizeof(data)));
    PARTITION_TRY(probe.flush());
    return probe.close();
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::length_error &) {
    return IndexError_NoMemory;
  }
}
int DiskAnnPartitionBuild::train_quantizer(DiskAnnBuilder &owner,
                                           IndexThreads::Pointer threads) {
  auto source = owner.holder_;
  if (!source || !source->multipass() || !source->is_matched(owner.raw_meta_))
    return IndexError_Mismatch;
  if (source->count() > std::numeric_limits<uint32_t>::max())
    return IndexError_Mismatch;
  // Restore the source on every exit, including exceptions from the trainer.
  AILEGO_DEFER([&]() { owner.holder_ = source; });
  try {
    const size_t sample_count = std::min<size_t>(
        source->count(),
        std::min<size_t>(
            owner.max_train_sample_count_,
            std::max<size_t>(1, owner.build_memory_budget_ / 8 /
                                    owner.raw_meta_.element_size())));
    auto sample =
        std::make_shared<TrainingSample>(owner.raw_meta_, sample_count);
    auto iter = source->create_iterator();
    if (!iter) return IndexError_Runtime;
    std::mt19937 random(42);
    size_t id = 0;
    for (; iter->is_valid(); iter->next(), ++id) {
      if (id >= source->count()) return IndexError_Mismatch;
      PARTITION_TRY(iter->status());
      size_t slot = id < sample_count
                        ? id
                        : std::uniform_int_distribution<size_t>(0, id)(random);
      // Read even unselected rows to propagate deferred source errors.
      const void *data = iter->data();
      PARTITION_TRY(iter->status());
      if (!data || !iter->is_valid()) return IndexError_ReadData;
      if (slot < sample_count)
        memcpy(sample->at(slot), data, owner.raw_meta_.element_size());
    }
    PARTITION_TRY(iter->status());
    if (id != source->count()) return IndexError_Mismatch;
    iter.reset();
    owner.holder_ = sample;
    return owner.train_quantized_data(std::move(threads));
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::length_error &) {
    return IndexError_NoMemory;
  }
}
int DiskAnnPartitionBuild::build(IndexThreads::Pointer threads,
                                 IndexHolder::Pointer holder) {
  try {
    return impl_->build(std::move(threads), std::move(holder));
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::length_error &) {
    return IndexError_NoMemory;
  }
}
int DiskAnnPartitionBuild::dump(const IndexDumper::Pointer &dumper) {
  try {
    return impl_->dump(dumper);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::length_error &) {
    return IndexError_NoMemory;
  }
}
#undef PARTITION_TRY
}  // namespace zvec::core
