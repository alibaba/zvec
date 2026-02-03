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

#include "flat_streamer.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

template <size_t BATCH_SIZE>
class FlatStreamerDumper {
 public:
  typedef std::unique_ptr<FlatStreamerDumper> Pointer;

  FlatStreamerDumper(const FlatStreamer<BATCH_SIZE> *owner) {
    owner_ = owner;
    dump_size_ = 0;
  }

  int dump(const IndexDumper::Pointer &dumper) {
    ailego::ElapsedTime stamp;

    std::vector<uint64_t> keys;
    if (owner_->meta().major_order() == IndexMeta::MO_COLUMN) {
      int error_code = this->write_column_index(dumper.get(), &keys);
      if (error_code != 0) {
        return error_code;
      }
    } else {
      int error_code = this->write_row_index(dumper.get(), &keys);
      if (error_code != 0) {
        return error_code;
      }
    }

    int error_code = this->write_keys(keys, dumper.get());
    if (error_code != 0) {
      return error_code;
    }

    error_code = this->write_mapping(keys, dumper.get());
    if (error_code != 0) {
      return error_code;
    }

    error_code = IndexHelper::SerializeToDumper(owner_->meta(), dumper.get());
    if (error_code != 0) {
      return error_code;
    }
    LOG_DEBUG("dumped_count: %zu, costtime: %zu", keys.size(),
              (size_t)stamp.milli_seconds());
    return 0;
  }

  size_t dump_size() {
    return dump_size_;
  }

 private:
  int write_column_index(IndexDumper *dumper, std::vector<uint64_t> *keys) {
    switch (IndexMeta::AlignSizeof(owner_->meta().data_type())) {
      case 2:
        return this->write_column_index<uint16_t>(dumper, keys);
      case 4:
        return this->write_column_index<uint32_t>(dumper, keys);
      case 8:
        return this->write_column_index<uint64_t>(dumper, keys);
      default:
        ailego_check_with(0, "BAD CASE");
    }
    return IndexError_Runtime;
  }

  template <typename T>
  int write_column_index(IndexDumper *dumper, std::vector<uint64_t> *keys) {
    auto iter = owner_->entity().creater_iterator();
    if (!iter) {
      LOG_ERROR("Failed to create iterator");
      return IndexError_Runtime;
    }

    // Write features
    size_t element_size = owner_->meta().element_size();
    size_t block_size = element_size * BATCH_SIZE;
    std::string block1, block2;
    block1.reserve(block_size);
    block2.reserve(block_size);

    for (; iter->is_valid(); iter->next()) {
      block1.append(reinterpret_cast<const char *>(iter->data()), element_size);
      keys->emplace_back(iter->key());

      if (block1.size() == block_size) {
        ailego::MatrixHelper::Transpose<T, BATCH_SIZE>(
            block1.data(), element_size / sizeof(T), (void *)block2.data());

        if (dumper->write(block2.data(), block_size) != block_size) {
          LOG_ERROR("Failed to write data into dumper %s",
                    dumper->name().c_str());
          return IndexError_WriteData;
        }
        block1.clear();
        dump_size_ += block_size;
      }
    }

    if (!block1.empty()) {
      if (dumper->write(block1.data(), block1.size()) != block1.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += block1.size();
    }

    // Write the padding if need
    size_t features_size = keys->size() * element_size;
    size_t features_padding_size =
        ailego_align(features_size, 32) - features_size;
    if (features_padding_size) {
      std::string padding(features_padding_size, '\0');

      if (dumper->write(padding.data(), padding.size()) != padding.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += padding.size();
    }
    return dumper->append(FLAT_SEGMENT_FEATURES_SEG_ID, features_size,
                          features_padding_size, 0);
  }

  int write_row_index(IndexDumper *dumper, std::vector<uint64_t> *keys) {
    auto iter = owner_->entity().creater_iterator();
    if (!iter) {
      LOG_ERROR("Failed to create iterator");
      return IndexError_Runtime;
    }

    // Write features
    size_t element_size = owner_->meta().element_size();
    for (; iter->is_valid(); iter->next()) {
      if (dumper->write(iter->data(), element_size) != element_size) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += element_size;
      keys->emplace_back(iter->key());
    }

    // Write the padding if need
    size_t features_size = keys->size() * element_size;
    size_t features_padding_size =
        ailego_align(features_size, 32) - features_size;
    if (features_padding_size) {
      std::string padding(features_padding_size, '\0');

      if (dumper->write(padding.data(), padding.size()) != padding.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += padding.size();
    }
    return dumper->append(FLAT_SEGMENT_FEATURES_SEG_ID, features_size,
                          features_padding_size, 0);
  }

  int write_keys(const std::vector<uint64_t> &keys, IndexDumper *dumper) {
    size_t keys_size = keys.size() * sizeof(uint64_t);
    size_t keys_padding_size = ailego_align(keys_size, 32) - keys_size;
    if (dumper->write(keys.data(), keys_size) != keys_size) {
      LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
    dump_size_ += keys_size;

    // Write the padding if need
    if (keys_padding_size) {
      std::string padding(keys_padding_size, '\0');
      if (dumper->write(padding.data(), padding.size()) != padding.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += padding.size();
    }
    return dumper->append(FLAT_SEGMENT_KEYS_SEG_ID, keys_size,
                          keys_padding_size, 0);
  }

  int write_mapping(const std::vector<uint64_t> &keys, IndexDumper *dumper) {
    std::vector<uint32_t> mapping(keys.size());
    std::iota(mapping.begin(), mapping.end(), 0);
    std::sort(mapping.begin(), mapping.end(),
              [&keys](uint32_t lhs, uint32_t rhs) {
                return (keys[lhs] < keys[rhs]);
              });

    size_t mapping_size = mapping.size() * sizeof(uint32_t);
    size_t mapping_padding_size = ailego_align(mapping_size, 32) - mapping_size;
    if (dumper->write(mapping.data(), mapping_size) != mapping_size) {
      LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
    dump_size_ += mapping_size;

    // Write the padding if need
    if (mapping_padding_size) {
      std::string padding(mapping_padding_size, '\0');
      if (dumper->write(padding.data(), padding.size()) != padding.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += padding.size();
    }
    return dumper->append(FLAT_SEGMENT_MAPPING_SEG_ID, mapping_size,
                          mapping_padding_size, 0);
  }

 private:
  const FlatStreamer<BATCH_SIZE> *owner_{nullptr};
  size_t dump_size_{0};
};

}  // namespace core
}  // namespace zvec
