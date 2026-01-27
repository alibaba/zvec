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

#include <iostream>
#include <set>
#include "gflags/gflags.h"
#include "zvec/core/framework/index_meta.h"
#include "index_meta_helper.h"
#include "txt_input_reader.h"
#include "vecs_common.h"

using namespace std;
using namespace zvec::core;

DEFINE_string(input, "input.txt", "txt input file");
DEFINE_string(input_first_sep, ";", "input first sep");
DEFINE_string(input_second_sep, " ", "input second sep");
DEFINE_string(output, "output.vecs", "vecs output file");
DEFINE_string(type, "float",
              "available type: float, double, int16, int8, binary");
DEFINE_string(method, "L2", "available method: L2, IP, HAMMING");
DEFINE_int32(dimension, 256, "data dimension");
DEFINE_string(vector_type, "dense", "available type: dense, hybrid, sparse");

bool write_header_output(VecsHeader header, const IndexMeta &meta,
                         size_t &total_writes, FILE *wfp) {
  // write header
  std::cout << "Begin to Write Header Section..." << std::endl;

  std::string meta_buf;
  meta.serialize(&meta_buf);
  header.meta_size = meta_buf.size();
  size_t wret = fwrite(&header, sizeof(header), 1, wfp);
  if (wret != 1) {
    cerr << "Write header error" << endl;
    fclose(wfp);
    return false;
  }

  total_writes += sizeof(header);
  std::cout << "Total Writes after Header Section: " << total_writes
            << std::endl
            << std::endl;

  // write meta
  std::cout << "Begin to Write Meta Section..." << std::endl;
  wret = fwrite(meta_buf.c_str(), meta_buf.size(), 1, wfp);
  if (wret != 1) {
    cerr << "Write header meta_buf error" << endl;
    fclose(wfp);
    return false;
  }

  total_writes += meta_buf.size();
  std::cout << "Total Writes after Meta Buf: " << total_writes << std::endl
            << std::endl;

  return true;
}

bool write_header_output_sparse(VecsHeader header, const IndexMeta &meta,
                                size_t &total_writes, FILE *wfp) {
  // write header
  std::cout << "Begin to Write Header Section..." << std::endl;
  std::string meta_buf;
  meta.serialize(&meta_buf);
  header.meta_size = meta_buf.size();
  size_t wret = fwrite(&header, sizeof(header), 1, wfp);
  if (wret != 1) {
    cerr << "Write header error" << endl;
    fclose(wfp);
    return false;
  }

  total_writes += sizeof(header);
  std::cout << "Total Writes after Header Section: " << total_writes
            << std::endl
            << std::endl;

  // write meta
  std::cout << "Begin to Write Sparse Meta Section..." << std::endl;
  wret = fwrite(meta_buf.c_str(), meta_buf.size(), 1, wfp);
  if (wret != 1) {
    cerr << "Write header meta buf error" << endl;
    fclose(wfp);
    return false;
  }

  total_writes += meta_buf.size();
  std::cout << "Total Writes after Meta Buf: " << total_writes << std::endl
            << std::endl;

  return true;
}

template <typename T>
bool write_features_output(size_t vec_num, const vector<vector<T>> &features,
                           size_t &total_writes, FILE *wfp) {
  // write dense vector
  std::cout << "Begin to Write Dense Vector Section..." << std::endl;
  for (size_t i = 0; i < vec_num; ++i) {
    auto &feature = features[i];
    size_t wret = fwrite(&feature[0], sizeof(T), feature.size(), wfp);
    if (wret != feature.size()) {
      cerr << "Write feature error. " << endl;
      fclose(wfp);
      return false;
    }

    total_writes += feature.size() * sizeof(T);
  }

  std::cout << "Total Writes after Dense Vector: " << total_writes << std::endl
            << std::endl;

  return true;
}

bool write_keys_output(size_t vec_num, const vector<uint64_t> &keys,
                       size_t &total_writes, FILE *wfp) {
  std::cout << "Begin to Write Key Section..." << std::endl;
  for (size_t i = 0; i < vec_num; ++i) {
    uint64_t key = keys[i];
    size_t wret = fwrite(&key, sizeof(key), 1, wfp);
    if (wret != 1) {
      cerr << "Write key error. key:" << key << endl;
      fclose(wfp);
      return false;
    }

    total_writes += sizeof(uint64_t);
  }

  std::cout << "Total Writes after Key Section: " << total_writes << std::endl
            << std::endl;

  return true;
}

template <typename T>
bool write_sparse_features_output(size_t vec_num,
                                  const vector<SparseData<T>> &sparse_data,
                                  size_t &total_writes, FILE *wfp) {
  std::set<uint32_t> sparse_dims;
  uint32_t sparse_max_count = 0;
  uint32_t sparse_min_count = -1U;
  uint32_t sparse_total_count = 0;

  // write sparse meta
  std::cout << "Begin to Write Sparse Meta Section..." << std::endl;
  size_t wret;
  uint64_t offset = 0;
  for (size_t i = 0; i < vec_num; ++i) {
    wret = fwrite(&offset, sizeof(uint64_t), 1, wfp);
    if (wret != 1) {
      cerr << "Write sparse feature len error. " << endl;
      fclose(wfp);
      return false;
    }
    offset += sparse_data[i].get_len();

    total_writes += sizeof(size_t);
  }
  std::cout << "Total Writes after Sparse Meta Section: " << total_writes
            << std::endl
            << std::endl;

  std::cout << "Begin to Write Sparse Vector Section..." << std::endl;
  for (size_t i = 0; i < vec_num; ++i) {
    auto &sparse_one_data = sparse_data[i];

    wret = fwrite(&(sparse_one_data.count), sizeof(uint32_t), 1, wfp);
    if (wret != 1) {
      cerr << "Write sparse feature count error. " << endl;
      fclose(wfp);
      return false;
    }

    total_writes += sizeof(uint32_t);

    wret = fwrite(&sparse_one_data.indices[0], sizeof(uint32_t),
                  sparse_one_data.indices.size(), wfp);
    if (wret != sparse_one_data.indices.size()) {
      cerr << "Write feature error. " << endl;
      fclose(wfp);
      return false;
    }

    total_writes += sizeof(uint32_t) * sparse_one_data.indices.size();
    // do some stat
    for (size_t s = 0; s < sparse_one_data.indices.size(); ++s) {
      sparse_dims.insert(sparse_one_data.indices[s]);
    }

    if (sparse_one_data.indices.size() > sparse_max_count) {
      sparse_max_count = sparse_one_data.indices.size();
    }

    if (sparse_one_data.indices.size() < sparse_min_count) {
      sparse_min_count = sparse_one_data.indices.size();
    }

    sparse_total_count += sparse_one_data.indices.size();
    // //done

    wret = fwrite(&sparse_one_data.features[0], sizeof(T),
                  sparse_one_data.features.size(), wfp);
    if (wret != sparse_one_data.features.size()) {
      cerr << "Write feature error. " << endl;
      fclose(wfp);
      return false;
    }

    total_writes += sizeof(T) * sparse_one_data.features.size();
  }

  std::cout << "Total Writes after Sparse Vector Section: " << total_writes
            << std::endl
            << std::endl;
  // for (auto itr=sparse_dims.begin(); itr!=sparse_dims.end(); ++itr) {
  //   std::cout << (*itr) << ",";
  // }
  // std::cout << std::endl;

  std::cout << "Max Sparse Dimension Count: " << sparse_max_count << std::endl;
  std::cout << "Min Sparse Dimension Count: " << sparse_min_count << std::endl;
  std::cout << "Avg Sparse Dimension Count: " << sparse_total_count / vec_num
            << std::endl;

  return true;
}

bool write_taglists_output(size_t vec_num,
                           const vector<vector<uint64_t>> &taglists,
                           size_t &total_writes, FILE *wfp) {
  std::cout << "Begin to Write Tag List Section..." << std::endl;

  // write tag list meta
  std::cout << "Begin to Write Tag List Meta Section..." << std::endl;
  size_t wret;
  uint64_t offset = 0;
  for (size_t i = 0; i < vec_num; ++i) {
    wret = fwrite(&offset, sizeof(uint64_t), 1, wfp);
    if (wret != 1) {
      cerr << "Write tag list meta error. Rec no: " << i << endl;
      fclose(wfp);
      return false;
    }
    offset += taglists[i].size() * sizeof(uint64_t);

    total_writes += sizeof(size_t);
  }
  std::cout << "Total Writes after Tag Meta Section: " << total_writes
            << std::endl
            << std::endl;

  for (size_t i = 0; i < vec_num; ++i) {
    std::vector<uint64_t> taglist = taglists[i];
    uint64_t taglist_size = taglist.size();
    wret = fwrite(&taglist_size, sizeof(uint64_t), 1, wfp);
    if (wret != 1) {
      cerr << "Write tag list size error. Rec no: " << i << endl;
      fclose(wfp);
      return false;
    }

    wret = fwrite(&(taglist[0]), sizeof(uint64_t), taglist.size(), wfp);
    if (wret != taglist.size()) {
      cerr << "Write tag list error. Rec no: " << i << endl;
      fclose(wfp);
      return false;
    }

    total_writes += sizeof(uint64_t) * taglist.size() + sizeof(uint64_t);
  }

  std::cout << "Total Writes after Tag List Section: " << total_writes
            << std::endl
            << std::endl;

  return true;
}

template <typename T>
bool write_vecs_output_sparse(VecsHeader &header, const IndexMeta &meta,
                              const vector<uint64_t> &keys,
                              const vector<SparseData<T>> &sparse_data,
                              const vector<vector<uint64_t>> &taglists) {
  if (keys.empty()) {
    cerr << "keys is empty." << endl;
    return false;
  }

  if (keys.size() != sparse_data.size()) {
    cerr << "keys's size(" << keys.size()
         << ") is not equal to sparse data's size(" << sparse_data.size()
         << ")." << endl;
    return false;
  }

  size_t vec_num = keys.size();

  FILE *wfp = fopen(FLAGS_output.c_str(), "wb");
  if (!wfp) {
    cerr << "Open file error. " << FLAGS_output << endl;
    return false;
  }

  size_t total_writes = 0;

  std::cout << "------------------------" << std::endl;
  std::cout << " Output Process         " << std::endl;
  std::cout << "------------------------" << std::endl;

  // write sparse header
  bool ret = write_header_output_sparse(header, meta, total_writes, wfp);
  if (!ret) {
    cerr << "write header error! " << endl;

    return false;
  }

  // write keys
  ret = write_keys_output(vec_num, keys, total_writes, wfp);
  if (!ret) {
    cerr << "write keys error! " << endl;

    return false;
  }

  // write sparse features
  ret = write_sparse_features_output(vec_num, sparse_data, total_writes, wfp);
  if (!ret) {
    cerr << "write sparse features error! " << endl;

    return false;
  }

  if ((header.bitmap & (1ULL << BITMAP_INDEX_TAGLIST)) != 0) {
    // write tag lists features
    ret = write_taglists_output(vec_num, taglists, total_writes, wfp);
    if (!ret) {
      cerr << "write tag lists error! " << endl;

      return false;
    }
  }

  std::cout << "------------------------" << std::endl;
  std::cout << " Output Done            " << std::endl;
  std::cout << "------------------------" << std::endl;

  fclose(wfp);
  return true;
}

template <typename T>
bool write_vecs_output(VecsHeader &header, const IndexMeta &meta,
                       const vector<uint64_t> &keys,
                       const vector<vector<T>> &features,
                       const vector<SparseData<T>> &sparse_data,
                       const vector<vector<uint64_t>> &taglists) {
  if (keys.empty()) {
    cerr << "keys is empty." << endl;
    return false;
  }

  if (keys.size() != features.size()) {
    cerr << "keys's size(" << keys.size()
         << ") is not equal to features's size(" << features.size() << ")."
         << endl;
    return false;
  }


  size_t vec_num = header.num_vecs;

  FILE *wfp = fopen(FLAGS_output.c_str(), "wb");
  if (!wfp) {
    cerr << "Open file error. " << FLAGS_output << endl;
    return false;
  }

  size_t total_writes = 0;

  std::cout << "------------------------" << std::endl;
  std::cout << " Output Process         " << std::endl;
  std::cout << "------------------------" << std::endl;

  // write header
  bool ret = write_header_output(header, meta, total_writes, wfp);
  if (!ret) {
    cerr << "write header error! " << endl;

    return false;
  }

  // write features
  ret = write_features_output(vec_num, features, total_writes, wfp);
  if (!ret) {
    cerr << "write features error! " << endl;

    return false;
  }

  // write keys
  ret = write_keys_output(vec_num, keys, total_writes, wfp);
  if (!ret) {
    cerr << "write keys error! " << endl;

    return false;
  }

  // write sparse features
  if ((header.bitmap & (1ULL << BITMAP_INDEX_SPARSE)) != 0) {
    ret = write_sparse_features_output(vec_num, sparse_data, total_writes, wfp);
    if (!ret) {
      cerr << "write sparse features error! " << endl;

      return false;
    }
  }

  if ((header.bitmap & (1ULL << BITMAP_INDEX_TAGLIST)) != 0) {
    // write tag lists features
    ret = write_taglists_output(vec_num, taglists, total_writes, wfp);
    if (!ret) {
      cerr << "write tag lists error! " << endl;

      return false;
    }
  }

  std::cout << "------------------------" << std::endl;
  std::cout << " Output Done            " << std::endl;
  std::cout << "------------------------" << std::endl;

  fclose(wfp);
  return true;
}

template <typename T>
bool compute_offset(uint64_t num_vecs, const IndexMeta &meta,
                    const vector<uint64_t> & /*keys*/,
                    const vector<vector<T>> & /*features*/,
                    const vector<SparseData<T>> &sparse_data,
                    const vector<std::vector<uint64_t>> &taglists,
                    uint64_t &key_offset, uint64_t &feature_offset,
                    uint64_t &sparse_offset, uint64_t &taglist_offset,
                    uint64_t &key_size, uint64_t &feature_size,
                    uint64_t &sparse_size, uint64_t &taglist_size) {
  size_t total_offset = 0;

  feature_offset = 0;
  feature_size = num_vecs * meta.element_size();
  total_offset += feature_size;

  key_offset = total_offset;
  key_size = num_vecs * sizeof(uint64_t);
  total_offset += key_size;

  if (sparse_data.size() != 0) {
    sparse_offset = total_offset;

    size_t data_offset = num_vecs * sizeof(uint64_t);
    for (size_t i = 0; i < sparse_data.size(); ++i) {
      data_offset += sizeof(uint32_t) +
                     sparse_data[i].count * (sizeof(uint32_t) + sizeof(T));
    }

    sparse_size = data_offset;

    total_offset += sparse_size;
  } else {
    sparse_offset = -1LLU;
    sparse_size = 0;
  }

  if (taglists.size() != 0) {
    taglist_offset = total_offset;

    size_t data_offset = num_vecs * sizeof(uint64_t);
    for (size_t i = 0; i < taglists.size(); ++i) {
      data_offset += sizeof(uint64_t) + taglists[i].size() * sizeof(uint64_t);
    }

    taglist_size = data_offset;
  } else {
    taglist_offset = -1LLU;
    taglist_size = 0;
  }

  return true;
}

template <typename T>
bool compute_sparse_offset(uint64_t num_vecs, const IndexMeta & /*meta*/,
                           const vector<uint64_t> & /*keys*/,
                           const vector<SparseData<T>> &sparse_data,
                           const vector<std::vector<uint64_t>> &taglists,
                           uint64_t &key_offset, uint64_t &sparse_offset,
                           uint64_t &taglist_offset, uint64_t &key_size,
                           uint64_t &sparse_size, uint64_t &taglist_size) {
  size_t total_offset = 0;

  key_offset = 0;
  key_size = num_vecs * sizeof(uint64_t);
  total_offset += num_vecs * sizeof(uint64_t);

  sparse_offset = total_offset;
  size_t data_offset = num_vecs * sizeof(uint64_t);
  for (size_t i = 0; i < sparse_data.size(); ++i) {
    data_offset += sizeof(uint32_t) +
                   sparse_data[i].count * (sizeof(uint32_t) + sizeof(T));
  }

  sparse_size = data_offset;
  total_offset += sparse_size;

  if (taglists.size() != 0) {
    taglist_offset = total_offset;

    data_offset = num_vecs * sizeof(uint64_t);
    for (size_t i = 0; i < taglists.size(); ++i) {
      data_offset += sizeof(uint64_t) + taglists[i].size() * sizeof(uint64_t);
    }

    taglist_size = data_offset;
  } else {
    taglist_offset = -1LLU;
    taglist_size = 0;
  }

  return true;
}

template <typename T>
bool process(void) {
  if (FLAGS_vector_type == "sparse") {
    std::cout << "------------------------" << std::endl;
    std::cout << " Vector Type: sparse    " << std::endl;
    std::cout << "------------------------" << std::endl;

    IndexMeta meta;
    if (!IndexMetaHelper::parse_from(FLAGS_type, FLAGS_method,
                                     FLAGS_vector_type, meta)) {
      cerr << "Index meta parse error." << endl;
      return false;
    }
    cerr << IndexMetaHelper::to_string(meta) << endl;

    TxtInputReader<T> reader;
    vector<uint64_t> keys;
    vector<SparseData<T>> sparse_data;
    vector<std::vector<uint64_t>> taglists;

    bool ret = reader.load_record_sparse(FLAGS_input, FLAGS_input_first_sep,
                                         FLAGS_input_second_sep, keys,
                                         sparse_data, taglists);
    if (!ret) {
      cerr << "Read record failed" << endl;
      return false;
    }

    if (sparse_data.size() == 0) {
      cerr << "empty sparse data!" << endl;
      return false;
    }

    uint64_t num_vecs = keys.size();

    uint64_t key_offset{-1LLU}, sparse_offset{-1LLU}, taglist_offset{-1LLU};
    uint64_t key_size{0}, sparse_size{0}, taglist_size{0};

    compute_sparse_offset(num_vecs, meta, keys, sparse_data, taglists,
                          key_offset, sparse_offset, taglist_offset, key_size,
                          sparse_size, taglist_size);

    VecsHeader header;
    header.num_vecs = keys.size();
    header.meta_size_v1 = 0;
    header.version = 1;
    header.bitmap = 0;
    header.key_offset = key_offset;
    header.dense_offset = -1LLU;
    header.sparse_offset = sparse_offset;
    header.taglist_offset = taglist_offset;
    header.key_size = key_size;
    header.dense_size = 0;
    header.sparse_size = sparse_size;
    header.taglist_size = taglist_size;

    header.bitmap |= (1 << BITMAP_INDEX_KEY);
    header.bitmap |= (1 << BITMAP_INDEX_SPARSE);

    if (taglist_offset != -1LLU) {
      header.bitmap |= (1 << BITMAP_INDEX_TAGLIST);
    }

    ret = write_vecs_output_sparse(header, meta, keys, sparse_data, taglists);
    if (!ret) {
      cerr << "write vecs output failed" << endl;
      return false;
    }
  } else {
    std::cout << "------------------------" << std::endl;
    std::cout << " Vector Type:     " << FLAGS_vector_type << std::endl;
    std::cout << "------------------------" << std::endl;

    IndexMeta meta;
    if (!IndexMetaHelper::parse_from(FLAGS_type, FLAGS_method, FLAGS_dimension,
                                     FLAGS_vector_type, meta)) {
      cerr << "Index meta parse error." << endl;
      return false;
    }
    cerr << IndexMetaHelper::to_string(meta) << endl;

    TxtInputReader<T> reader;
    vector<uint64_t> keys;
    vector<vector<T>> features;
    vector<SparseData<T>> sparse_data;
    vector<std::vector<uint64_t>> taglists;

    bool ret = reader.load_record(FLAGS_input, FLAGS_input_first_sep,
                                  FLAGS_input_second_sep, FLAGS_dimension, keys,
                                  features, sparse_data, taglists);
    if (!ret) {
      cerr << "Read record failed" << endl;
      return false;
    }

    uint64_t num_vecs = keys.size();

    uint64_t key_offset{-1LLU}, features_offset{-1LLU}, sparse_offset{-1LLU},
        taglist_offset{-1LLU};
    uint64_t key_size{0}, feature_size{0}, sparse_size{0}, taglist_size{0};

    compute_offset(num_vecs, meta, keys, features, sparse_data, taglists,
                   key_offset, features_offset, sparse_offset, taglist_offset,
                   key_size, feature_size, sparse_size, taglist_size);

    VecsHeader header;
    header.num_vecs = num_vecs;
    header.meta_size_v1 = 0;
    header.version = 1;
    header.bitmap = 0;
    header.key_offset = key_offset;
    header.dense_offset = features_offset;
    header.sparse_offset = sparse_offset;
    header.taglist_offset = taglist_offset;
    header.key_size = key_size;
    header.dense_size = feature_size;
    header.sparse_size = sparse_size;
    header.taglist_size = taglist_size;

    header.bitmap |= (1 << BITMAP_INDEX_KEY);
    header.bitmap |= (1 << BITMAP_INDEX_DENSE);

    if (sparse_offset != -1LLU) {
      header.bitmap |= (1 << BITMAP_INDEX_SPARSE);
    }

    if (taglist_offset != -1LLU) {
      header.bitmap |= (1 << BITMAP_INDEX_TAGLIST);
    }

    ret =
        write_vecs_output(header, meta, keys, features, sparse_data, taglists);
    if (!ret) {
      cerr << "write vecs output failed" << endl;
      return false;
    }
  }

  return true;
}

int main(int argc, char *argv[]) {
  // gflags
  gflags::SetUsageMessage("Usage: txt2vecs [options]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_type == "float") {
    if (!process<float>()) {
      return -1;
    }
  } else if (FLAGS_type == "double") {
    if (!process<double>()) {
      return -1;
    }
  } else if (FLAGS_type == "int16") {
    if (!process<int16_t>()) {
      return -1;
    }
  } else if (FLAGS_type == "int8") {
    if (!process<int8_t>()) {
      return -1;
    }
  } else if (FLAGS_type == "binary") {
    if (!process<uint32_t>()) {
      return -1;
    }
  } else if (FLAGS_type == "binary64") {
    if (!process<uint64_t>()) {
      return -1;
    }
  } else {
    cerr << "Can not recognize type: " << FLAGS_type << endl;
    return -1;
  }
  return 0;
}
