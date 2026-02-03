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


#include <string>
#include <roaring/roaring.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"


namespace zvec {


/*
 *
 * This class provides encoding/decoding functionality for inverted index data.
 *
 *
 * RocksDB stores key-value pairs as byte strings. Therefore, all data types
 * must be converted to a string format for storage.
 *
 * To ensure correct lexicographic sorting of numeric values (integers and
 * floating-point numbers), we need to transform them so that their byte
 * representation sorts in the same order as their numerical values.
 *
 *
 * The codec also manages storage of mapped document IDs, using either list or
 * bitmap structures depending on the data size for optimal storage efficiency.
 *
 */
class InvertedIndexCodec {
 public:
  static std::string Encode(const std::string &term, DataType type) {
    switch (type) {
      case DataType::INT32:
      case DataType::INT64: {
        auto result = term;
        convert_to_big_endian(result.data(), result.size());
        // For signed, reverse sign bit, set positive to 1, negative to 0
        result[0] ^= 0x80;
        return result;
      }
      case DataType::UINT32:
      case DataType::UINT64: {
        auto result = term;
        convert_to_big_endian(result.data(), result.size());
        return result;
      }
      case DataType::FLOAT:
      case DataType::DOUBLE: {
        auto result = term;
        convert_to_big_endian(result.data(), result.size());
        // If float is negative, negate each byte; else reverse sign bit
        if ((result[0] & 0x80) > 0) {
          for (size_t i = 0; i < result.size(); i++) {
            result[i] = ~result[i];
          }
        } else {
          result[0] ^= 0x80;
        }
        return result;
      }
      default:
        return term;
    }
  }


  static std::string Encode(const std::string_view &term, DataType type) {
    switch (type) {
      case DataType::INT32:
      case DataType::INT64: {
        std::string result(term);
        convert_to_big_endian(result.data(), result.size());
        // For signed, reverse sign bit, set positive to 1, negative to 0
        result[0] ^= 0x80;
        return result;
      }
      case DataType::UINT32:
      case DataType::UINT64: {
        std::string result(term);
        convert_to_big_endian(result.data(), result.size());
        return result;
      }
      case DataType::FLOAT:
      case DataType::DOUBLE: {
        std::string result(term);
        convert_to_big_endian(result.data(), result.size());
        // If float is negative, negate each byte; else reverse sign bit
        if ((result[0] & 0x80) > 0) {
          for (size_t i = 0; i < result.size(); i++) {
            result[i] = ~result[i];
          }
        } else {
          result[0] ^= 0x80;
        }
        return result;
      }
      default:
        return std::string(term);
    }
  }


  static std::string Encode(bool value) {
    if (value) {
      return "true";
    } else {
      return "false";
    }
  }


  static std::string Encode_Reversed(const std::string &term) {
    std::string reversed = term;
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
  }


  // Format of range key:
  // [range_begin_key][separator_byte][range_end_key][range_begin_key_size]
  static void Decode_Range_Key(const char *range_key_ptr, size_t range_key_size,
                               char **range_begin_pos,
                               size_t *range_begin_key_size,
                               char **range_end_pos,
                               size_t *range_end_key_size) {
    *range_begin_key_size =
        *(uint64_t *)(range_key_ptr + (range_key_size - sizeof(uint64_t)));
    *range_begin_pos = (char *)range_key_ptr;

    *range_end_key_size =
        range_key_size - sizeof(uint64_t) - (*range_begin_key_size) - 1;
    *range_end_pos = (char *)(range_key_ptr + (*range_begin_key_size) + 1);
  }


  // Return negative number if s1 < s2, positive number if s1 > s2, 0 if equal
  static int CMP(const char *s1, size_t s1_len, const char *s2, size_t s2_len) {
    size_t min_len = std::min(s1_len, s2_len);
    int r = memcmp(s1, s2, min_len);
    if (r == 0) {
      if (s1_len < s2_len)
        r = -1;
      else if (s1_len > s2_len)
        r = +1;
    }
    return r;
  }


  static bool Has_Prefix(const char *value, size_t value_len,
                         const char *prefix, size_t prefix_len) {
    if (value_len < prefix_len) {
      return false;
    }
    return memcmp(value, prefix, prefix_len) == 0;
  }


  static Status Serialize(roaring_bitmap_t *bitmap, std::string *out) {
    if (!bitmap) {
      LOG_ERROR("Invalid bitmap pointer");
      return Status::InvalidArgument();
    }
    if (!out) {
      LOG_ERROR("Invalid output pointer");
      return Status::InvalidArgument();
    }
    out->clear();

    uint64_t count = roaring_bitmap_get_cardinality(bitmap);
    if (count == 0) {
      LOG_ERROR("Bitmap is empty");
      return Status::InternalError();
    } else if (count > INVERT_ID_LIST_SIZE_THRESHOLD) {
      return serialize_bitmap(bitmap, out);
    } else {
      return serialize_docid_list(bitmap, out);
    }
  }


  static Status Deserialize(const char *data, size_t size,
                            roaring_bitmap_t **bitmap) {
    if (!data || size == 0) {
      LOG_ERROR("Input data is invalid");
      return Status::InvalidArgument();
    }

    unsigned char header = data[0];
    if (header > INVERT_ID_LIST_SIZE_THRESHOLD) {
      LOG_ERROR("Invalid header found in inverted data");
      return Status::InternalError();
    }

    if (header == 0) {  // This is a bitmap
      *bitmap = roaring_bitmap_portable_deserialize_safe(data + 1, size - 1);
      if (*bitmap) {
        return Status::OK();
      } else {
        LOG_ERROR("Failed to deserialize bitmap");
        return Status::InternalError();
      }
    }

    // This is a id list
    if ((size - 1) != header * sizeof(uint32_t)) {
      LOG_ERROR("Failed to deserialize docid_list");
      return Status::InternalError();
    }
    *bitmap = roaring_bitmap_create();
    if (*bitmap == nullptr) {
      LOG_ERROR("Failed to create bitmap");
      return Status::InternalError();
    }
    for (size_t i = 1; i < size; i += sizeof(uint32_t)) {
      roaring_bitmap_add(*bitmap,
                         *reinterpret_cast<const uint32_t *>(data + i));
    }
    return Status::OK();
  }


  static Status Merge_OR(const char *data, size_t size, bool lazy,
                         roaring_bitmap_t *bitmap) {
    if (!data || size == 0) {
      LOG_ERROR("Input data is invalid");
      return Status::InvalidArgument();
    }

    unsigned char header = data[0];
    if (header > INVERT_ID_LIST_SIZE_THRESHOLD) {
      LOG_ERROR("Invalid header found in inverted data");
      return Status::InternalError();
    }

    if (header == 0) {  // This is a bitmap
      auto bitmap_other = roaring_bitmap_portable_deserialize_frozen(data + 1);
      if (!bitmap_other) {
        LOG_ERROR("Failed to deserialize bitmap");
        return Status::InternalError();
      }
      if (lazy) {
        roaring_bitmap_lazy_or_inplace(bitmap, bitmap_other, true);
      } else {
        roaring_bitmap_or_inplace(bitmap, bitmap_other);
      }
      roaring_bitmap_free(bitmap_other);
      return Status::OK();
    }

    // This is a id list
    if ((size - 1) != header * sizeof(uint32_t)) {
      LOG_ERROR("Failed to deserialize docid_list");
      return Status::InternalError();
    }
    auto doc_list = reinterpret_cast<const uint32_t *>(data + 1);
    for (size_t i = 0; i < header; ++i) {
      roaring_bitmap_add(bitmap, doc_list[i]);
    }
    return Status::OK();
  }


  static Status Merge_AND(const char *data, size_t size,
                          roaring_bitmap_t *bitmap) {
    if (!data || size == 0) {
      LOG_ERROR("Input data is invalid");
      return Status::InvalidArgument();
    }

    unsigned char header = data[0];
    if (header > INVERT_ID_LIST_SIZE_THRESHOLD) {
      LOG_ERROR("Invalid header found in inverted data");
      return Status::InternalError();
    }

    if (header == 0) {  // This is a bitmap
      auto bitmap_other = roaring_bitmap_portable_deserialize_frozen(data + 1);
      if (!bitmap_other) {
        LOG_ERROR("Failed to deserialize bitmap");
        return Status::InternalError();
      }
      roaring_bitmap_and_inplace(bitmap, bitmap_other);
      roaring_bitmap_free(bitmap_other);
      return Status::OK();
    }

    // This is a id list
    if ((size - 1) != header * sizeof(uint32_t)) {
      LOG_ERROR("Failed to deserialize docid_list");
      return Status::InternalError();
    }
    auto doc_list = reinterpret_cast<const uint32_t *>(data + 1);
    uint32_t tmp = 0;
    for (size_t i = 0; i < header; ++i) {
      tmp |= roaring_bitmap_contains(bitmap, doc_list[i]) << i;
    }
    roaring_bitmap_clear(bitmap);
    for (size_t i = 0; i < header; ++i) {
      if (tmp & (1 << i)) {
        roaring_bitmap_add(bitmap, doc_list[i]);
      }
    }
    return Status::OK();
  }


 private:
  static void convert_to_big_endian(char *in, size_t size) {
    static const bool isBigEndianSystem = []() {
      int i = 0x1243;
      char *ch = (char *)&i;
      return (*ch == 0x12);
    }();

    if (isBigEndianSystem) {
      return;
    }

    char *p = in;
    for (size_t i = 0; i < size / 2; ++i) {
      std::swap(p[i], p[size - i - 1]);
    }
  }


  static Status serialize_bitmap(const roaring_bitmap_t *bitmap,
                                 std::string *out) {
    size_t bitmap_size = roaring_bitmap_portable_size_in_bytes(bitmap);
    out->resize(1 + bitmap_size);

    // Set the first byte with value 0, indicating the data is a bitmap
    (*out)[0] = static_cast<char>(0);
    size_t written_size = roaring_bitmap_portable_serialize(
        bitmap, const_cast<char *>(out->data()) + 1);
    if (written_size == bitmap_size) {
      return Status::OK();
    } else {
      LOG_ERROR("Failed to serialize bitmap");
      return Status::InternalError();
    }
  }


  static Status serialize_docid_list(const roaring_bitmap_t *bitmap,
                                     std::string *out) {
    auto doc_count = roaring_bitmap_get_cardinality(bitmap);
    out->reserve(1 + doc_count * sizeof(uint32_t));
    // Adds a single byte at the beginning indicating the count of document IDs
    out->append(1, static_cast<unsigned char>(doc_count));

    auto iter = roaring_create_iterator(bitmap);
    while (iter->has_value) {
      out->append(reinterpret_cast<const char *>(&(iter->current_value)),
                  sizeof(uint32_t));
      roaring_advance_uint32_iterator(iter);
    }
    roaring_free_uint32_iterator(iter);
    return Status::OK();
  }
};


}  // namespace zvec