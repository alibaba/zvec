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

#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec {
namespace core {

template <typename T>
struct SparseData {
 public:
  SparseData(uint32_t count_in, std::vector<uint32_t> &indices_in,
             std::vector<T> &features_in)
      : count(count_in),
        indices(std::move(indices_in)),
        features(std::move(features_in)) {}

  SparseData(uint32_t count_in, std::vector<uint32_t> &&indices_in,
             std::vector<T> &&features_in)
      : count(count_in),
        indices(std::move(indices_in)),
        features(std::move(features_in)) {}

 public:
  uint64_t get_len() const {
    return sizeof(uint32_t) + sizeof(uint32_t) * indices.size() +
           sizeof(T) * features.size();
  }

 public:
  uint32_t count;
  std::vector<uint32_t> indices;
  std::vector<T> features;
};

// support type: float, binary, int16, int8
template <typename T>
class TxtInputReader {
 public:
  bool load_query(const std::string &query_file, const std::string &first_sep,
                  const std::string &second_sep,
                  std::vector<std::vector<T>> &features,
                  std::vector<SparseData<T>> &sparse_data,
                  std::vector<std::vector<uint64_t>> &taglists) {
    std::fstream qf(query_file, std::ios::in);

    if (!qf.is_open()) {
      std::cerr << "open query file failed! [" << query_file << "]"
                << std::endl;
      return false;
    }

    bool ret;
    std::string buffer;
    while (getline(qf, buffer)) {
      buffer.erase(buffer.find_last_not_of('\n') + 1);
      if (buffer.empty()) {
        continue;
      }
      std::vector<std::string> res;
      ailego::StringHelper::Split(buffer, first_sep, &res);
      if (res.empty()) {
        continue;
      }
      std::string feature_str = res[0];
      if (res.size() > 1) {
        feature_str = res[1];
      }
      std::vector<T> feature;
      size_t dimension = 0;
      ret = load_from_string(feature_str, second_sep, feature, &dimension);
      if (!ret) {
        return false;
      }

      features.emplace_back(feature);

      uint64_t key = atol(res[0].c_str());

      // load sparse feature
      uint32_t sparse_count = 0;
      std::vector<uint32_t> sparse_indices;
      std::vector<T> sparse_feature;

      if (res.size() >= 3) {
        ret = load_from_string_sparse(key, res[2], second_sep, sparse_indices,
                                      sparse_feature, &sparse_count);
        if (!ret) {
          std::cerr << "load sparse failed for key: " << key << std::endl;
          return false;
        }
      }

      sparse_data.emplace_back(sparse_count, std::move(sparse_indices),
                               std::move(sparse_feature));

      if (res.size() >= 4) {
        std::vector<uint64_t> taglist;
        size_t tag_count = 0;

        ret = load_tags_from_string(res[4], second_sep, taglist, &tag_count);
        if (!ret) {
          std::cerr << "load tags failed for key: " << key << std::endl;
          return false;
        }

        taglists.emplace_back(taglist);
      }
    }

    qf.close();
    if (features.size() == 0) {
      std::cerr << "Read query size is 0" << std::endl;
      return false;
    }
    return true;
  }


  bool load_record(const std::string &input, const std::string &first_sep,
                   const std::string &second_sep, const size_t dimension,
                   std::vector<uint64_t> &keys,
                   std::vector<std::vector<T>> &features,
                   std::vector<SparseData<T>> &sparse_data,
                   std::vector<std::vector<uint64_t>> &taglists) {
    std::fstream qf(input, std::ios::in);

    if (!qf.is_open()) {
      std::cerr << "open file failed! [" << input << "]" << std::endl;
      return false;
    }

    bool ret;
    uint32_t count = 0;
    std::string buffer;

    while (getline(qf, buffer)) {
      buffer.erase(buffer.find_last_not_of('\n') + 1);
      if (buffer.empty()) {
        continue;
      }
      std::vector<std::string> res;
      ailego::StringHelper::Split(buffer, first_sep, &res);
      if (res.size() < 2) {
        std::cerr << "skip record : " << buffer << std::endl;
        continue;
      }

      std::vector<T> feature;
      size_t real_dim = 0;

      // load sparse feature
      uint32_t sparse_count = 0;
      std::vector<uint32_t> sparse_indices;
      std::vector<T> sparse_feature;

      uint64_t key = atol(res[0].c_str());

      // load dense feature
      ret = load_from_string(res[1], second_sep, feature, &real_dim);
      if (!ret) {
        return false;
      }

      if (real_dim != dimension) {
        std::cerr << "real dim (" << real_dim << ") is not equal to dimension("
                  << dimension << ") key : " << res[0] << std::endl;
        continue;
      }

      features.emplace_back(feature);
      keys.emplace_back(key);

      if (res.size() >= 3) {
        ret = load_from_string_sparse(key, res[2], second_sep, sparse_indices,
                                      sparse_feature, &sparse_count);
        if (!ret) {
          std::cerr << "load sparse failed for key: " << key << std::endl;
          return false;
        }

        sparse_data.emplace_back(sparse_count, std::move(sparse_indices),
                                 std::move(sparse_feature));
      }

      if (res.size() >= 4) {
        std::vector<uint64_t> taglist;
        size_t tag_count = 0;

        ret = load_tags_from_string(res[3], second_sep, taglist, &tag_count);
        if (!ret) {
          std::cerr << "load tags failed for key: " << key << std::endl;
          return false;
        }

        taglists.emplace_back(taglist);
      }

      count++;
      if (count % 1000000 == 0) {
        std::cout << "processed " << count << " records!" << std::endl;
      }
    }

    qf.close();

    if (keys.size() == 0) {
      std::cerr << "Reading nothing from input" << std::endl;
      return false;
    }

    return true;
  }

  bool load_record_sparse(const std::string &input,
                          const std::string &first_sep,
                          const std::string &second_sep,
                          std::vector<uint64_t> &keys,
                          std::vector<SparseData<T>> &sparse_data,
                          std::vector<std::vector<uint64_t>> &taglists) {
    std::fstream qf(input, std::ios::in);

    if (!qf.is_open()) {
      std::cerr << "open file failed! [" << input << "]" << std::endl;
      return false;
    }

    bool ret;
    uint32_t count = 0;
    std::string buffer;

    while (getline(qf, buffer)) {
      buffer.erase(buffer.find_last_not_of('\n') + 1);
      if (buffer.empty()) {
        continue;
      }
      std::vector<std::string> res;
      ailego::StringHelper::Split(buffer, first_sep, &res);
      if (res.size() < 2) {
        std::cerr << "skip record : " << buffer << std::endl;
        continue;
      }

      uint64_t key = atol(res[0].c_str());

      // load sparse feature
      uint32_t sparse_count = 0;
      std::vector<uint32_t> sparse_indices;
      std::vector<T> sparse_feature;

      if (res.size() <= 2) {
        std::cerr << "field erorr, key: " << key << std::endl;
        continue;
      }

      ret = load_from_string_sparse(key, res[2], second_sep, sparse_indices,
                                    sparse_feature, &sparse_count);
      if (!ret) {
        std::cerr << "load sparse failed for key: " << key << std::endl;
        return false;
      }

      keys.emplace_back(key);

      sparse_data.emplace_back(sparse_count, std::move(sparse_indices),
                               std::move(sparse_feature));

      if (res.size() >= 4) {
        std::vector<uint64_t> taglist;
        size_t tag_count;

        ret = load_tags_from_string(res[4], second_sep, taglist, &tag_count);
        if (!ret) {
          std::cerr << "load tags failed for key: " << key << std::endl;
          return false;
        }

        taglists.emplace_back(taglist);
      }

      count++;
      if (count % 1000000 == 0) {
        std::cout << "processed " << count << " records!" << std::endl;
      }
    }

    qf.close();

    if (keys.size() == 0) {
      std::cerr << "Reading nothing from input" << std::endl;
      return false;
    }

    return true;
  }

  template <typename U>
  bool load_from_string(const std::string &record,
                        const std::string &second_sep, std::vector<U> &data,
                        size_t *count) {
    ailego::StringHelper::Split(record, second_sep, &data, true);
    *count = data.size();

    return true;
  }

  bool load_scores_from_string(const std::string &record,
                               const std::string &second_sep,
                               std::vector<float> &data, size_t *count) {
    ailego::StringHelper::Split(record, second_sep, &data, true);
    *count = data.size();

    return true;
  }

  bool load_ids_from_string(const std::string &record,
                            const std::string &second_sep,
                            std::vector<uint64_t> &data, size_t *count) {
    ailego::StringHelper::Split(record, second_sep, &data, true);
    *count = data.size();

    return true;
  }

  bool load_tags_from_string(const std::string &record,
                             const std::string &second_sep,
                             std::vector<uint64_t> &tags, size_t *count) {
    ailego::StringHelper::Split(record, second_sep, &tags, true);
    *count = tags.size();

    // order tags
    sort(tags.begin(), tags.end());

    return true;
  }

  // overloading for binary
  bool load_from_string(const std::string &record,
                        const std::string &second_sep,
                        std::vector<uint32_t> &data, size_t *count) {
    // fetch split value from text file
    std::vector<uint8_t> vec;
    ailego::StringHelper::Split(record, second_sep, &vec, true);
    if (vec.size() == 0) {
      std::cerr << "Binary vector size is 0" << std::endl;
      return false;
    }
    if (vec.size() % 32 != 0) {
      std::cerr << "Binary vector size must be 32_X" << std::endl;
      return false;
    }
    // compact into uint32_t
    size_t sz = vec.size();
    std::vector<uint8_t> tmp;
    for (size_t i = 0; i < sz; i += 8) {
      uint8_t v = 0;
      v |= (vec[i] & 0x01) << 7;
      v |= (vec[i + 1] & 0x01) << 6;
      v |= (vec[i + 2] & 0x01) << 5;
      v |= (vec[i + 3] & 0x01) << 4;
      v |= (vec[i + 4] & 0x01) << 3;
      v |= (vec[i + 5] & 0x01) << 2;
      v |= (vec[i + 6] & 0x01) << 1;
      v |= (vec[i + 7] & 0x01) << 0;
      tmp.push_back(v);
    }
    data.resize(sz / 32);
    memcpy(&data[0], &tmp[0], tmp.size());
    *count = sz;

    return true;
  }

  // overloading for binary
  bool load_from_string(const std::string &record,
                        const std::string &second_sep,
                        std::vector<uint64_t> &data, size_t *count) {
    // fetch split value from text file
    std::vector<uint8_t> vec;
    ailego::StringHelper::Split(record, second_sep, &vec);
    if (vec.size() == 0) {
      std::cerr << "Binary vector size is 0" << std::endl;
      return false;
    }
    if (vec.size() % 64 != 0) {
      std::cerr << "Binary vector size must be 32_X" << std::endl;
      return false;
    }
    // compact into uint64_t
    size_t sz = vec.size();
    std::vector<uint8_t> tmp;
    for (size_t i = 0; i < sz; i += 8) {
      uint8_t v = 0;
      v |= (vec[i] & 0x01) << 7;
      v |= (vec[i + 1] & 0x01) << 6;
      v |= (vec[i + 2] & 0x01) << 5;
      v |= (vec[i + 3] & 0x01) << 4;
      v |= (vec[i + 4] & 0x01) << 3;
      v |= (vec[i + 5] & 0x01) << 2;
      v |= (vec[i + 6] & 0x01) << 1;
      v |= (vec[i + 7] & 0x01) << 0;
      tmp.push_back(v);
    }
    data.resize(sz / 64);
    memcpy(&data[0], &tmp[0], tmp.size());
    *count = sz;

    return true;
  }

  bool load_from_string_sparse(uint64_t key, const std::string &record,
                               const std::string &second_sep,
                               std::vector<uint32_t> &sparse_indices,
                               std::vector<T> &sparse_feature,
                               uint32_t *sparse_count) {
    const std::string sparse_sep = ":";
    std::vector<std::string> res;
    ailego::StringHelper::Split(record, sparse_sep, &res);

    if (res.size() == 2) {
      ailego::StringHelper::Split(res[0], second_sep, &sparse_indices);
      ailego::StringHelper::Split(res[1], second_sep, &sparse_feature);

      uint32_t index_count = sparse_indices.size();
      uint32_t feature_count = sparse_feature.size();

      if (feature_count == index_count) {
        *sparse_count = feature_count;
      } else {
        std::cerr << "sparse feature count (" << feature_count
                  << ") is not equal with sparse index count(" << index_count
                  << ") key : " << key << std::endl;
        *sparse_count = 0;

        return false;
      }

      // check order
      for (size_t i = 1; i < sparse_indices.size(); ++i) {
        if (sparse_indices[i - 1] >= sparse_indices[i]) {
          std::cerr << "sparse indices not ordered, key : " << key
                    << ", dim info: [" << sparse_indices[i - 1] << ", "
                    << sparse_indices[i] << "]" << std::endl;

          return false;
        }
      }
    }

    return true;
  }

  // LINE FORMAT is as follows:
  //      key:key0 key1 key2 ... keyN:score0 score1 score2 ... scoreN
  bool load_external_gt(
      const std::string &input, const std::string &first_sep,
      const std::string &second_sep,
      std::vector<std::vector<std::pair<uint64_t, float>>> &ground_truth) {
    std::fstream gf(input, std::ios::in);

    if (!gf.is_open()) {
      std::cerr << "open file failed! [" << input << "]" << std::endl;
      return false;
    }

    uint32_t count = 0;
    std::string buffer;
    while (getline(gf, buffer)) {
      buffer.erase(buffer.find_last_not_of('\n') + 1);
      if (buffer.empty()) {
        continue;
      }
      std::vector<std::string> res;
      ailego::StringHelper::Split(buffer, first_sep, &res);
      if (res.size() < 2) {
        std::cerr << "skip record : " << buffer << std::endl;
        continue;
      }

      // uint64_t main_key = std::strtoll(res[0].c_str(), NULL, 10);
      if (res.size() == 2) {
        std::vector<uint64_t> keys;
        size_t key_num = 0;
        load_ids_from_string(res[1], second_sep, keys, &key_num);

        std::vector<std::pair<uint64_t, float>> one_groud_truth;
        for (size_t i = 0; i < keys.size(); ++i) {
          one_groud_truth.push_back(std::make_pair(keys[i], 0.0f));
        }

        ground_truth.push_back(std::move(one_groud_truth));
      } else {
        std::vector<uint64_t> keys;
        size_t key_num = 0;
        load_ids_from_string(res[1], second_sep, keys, &key_num);

        std::vector<float> scores;
        size_t score_num = 0;
        load_scores_from_string(res[2], second_sep, scores, &score_num);

        if (key_num != score_num) {
          std::cerr << "key num (" << key_num << ") is not equal to ("
                    << score_num << "), line data:" << buffer << std::endl;
          continue;
        }

        std::vector<std::pair<uint64_t, float>> one_groud_truth;
        for (size_t i = 0; i < keys.size(); ++i) {
          one_groud_truth.push_back(std::make_pair(keys[i], scores[i]));
        }

        ground_truth.push_back(std::move(one_groud_truth));
      }

      count++;
      if (count % 1000000 == 0) {
        std::cout << "processed " << count << " records!" << std::endl;
      }
    }
    gf.close();
    if (ground_truth.size() == 0) {
      std::cerr << "Reading nothing from input" << std::endl;
      return false;
    }

    return true;
  }
};

}  // namespace core
}  // namespace zvec