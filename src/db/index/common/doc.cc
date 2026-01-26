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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <zvec/db/doc.h>
#include "db/common/constants.h"
#include "db/index/common/type_helper.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define IS_BIG_ENDIAN 1
#else
#define IS_BIG_ENDIAN 0
#endif


namespace zvec {

enum ValueType : uint8_t {
  TYPE_EMPTY = 0,
  TYPE_BOOL = 1,
  TYPE_INT32 = 2,
  TYPE_UINT32 = 3,
  TYPE_INT64 = 4,
  TYPE_UINT64 = 5,
  TYPE_FLOAT = 6,
  TYPE_DOUBLE = 7,
  TYPE_STRING = 8,
  TYPE_VECTOR_BOOL = 9,
  TYPE_VECTOR_INT8 = 10,
  TYPE_VECTOR_INT16 = 11,
  TYPE_VECTOR_INT32 = 12,
  TYPE_VECTOR_INT64 = 13,
  TYPE_VECTOR_UINT32 = 14,
  TYPE_VECTOR_UINT64 = 15,
  TYPE_VECTOR_FLOAT16 = 16,
  TYPE_VECTOR_FLOAT = 17,
  TYPE_VECTOR_DOUBLE = 18,
  TYPE_VECTOR_STRING = 19,
  TYPE_VECTOR_PAIR_INT_FLOAT = 20,
  TYPE_VECTOR_PAIR_INT_FLOAT16 = 21,
};

std::string get_value_type_name(const Doc::Value &value, bool is_vector) {
  return std::visit(
      [&](const auto &v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return "EMPTY";
        } else if constexpr (std::is_same_v<T, bool>) {
          return "BOOL";
        } else if constexpr (std::is_same_v<T, int32_t>) {
          return "INT32";
        } else if constexpr (std::is_same_v<T, uint32_t>) {
          return "UINT32";
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return "INT64";
        } else if constexpr (std::is_same_v<T, uint64_t>) {
          return "UINT64";
        } else if constexpr (std::is_same_v<T, float>) {
          return "FLOAT";
        } else if constexpr (std::is_same_v<T, double>) {
          return "DOUBLE";
        } else if constexpr (std::is_same_v<T, std::string>) {
          return "STRING";
        } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
          return "ARRAY_BOOL";
        } else if constexpr (std::is_same_v<T, std::vector<int8_t>>) {
          return "VECTOR_INT8";
        } else if constexpr (std::is_same_v<T, std::vector<int16_t>>) {
          return "VECTOR_INT16";
        } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
          return is_vector ? "VECTOR_INT32" : "ARRAY_INT32";
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          return is_vector ? "VECTOR_INT64" : "ARRAY_INT64";
        } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
          return is_vector ? "VECTOR_UINT32" : "ARRAY_UINT32";
        } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
          return is_vector ? "VECTOR_UINT64" : "ARRAY_UINT64";
        } else if constexpr (std::is_same_v<T, std::vector<float16_t>>) {
          return "VECTOR_FP16";
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
          return "VECTOR_FP32";
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          return "VECTOR_FP64";
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          return "ARRAY_STRING";
        } else if constexpr (std::is_same_v<T, std::pair<std::vector<uint32_t>,
                                                         std::vector<float>>>) {
          return "SPARSE_VECTOR_FP32";
        } else if constexpr (std::is_same_v<
                                 T, std::pair<std::vector<uint32_t>,
                                              std::vector<float16_t>>>) {
          return "SPARSE_VECTOR_FP16";
        } else {
          return "unknown type";
        }
      },
      value);
}

template <typename T>
T byte_swap(T value) {
  if constexpr (std::is_same_v<T, float16_t>) {
    uint16_t val = *reinterpret_cast<uint16_t *>(&value);
    val = __builtin_bswap16(val);
    return *reinterpret_cast<float16_t *>(&val);
  } else if constexpr (sizeof(T) == 1) {
    return value;
  } else if constexpr (sizeof(T) == 2) {
    return (value << 8) | ((value >> 8) & 0xFF);
  } else if constexpr (sizeof(T) == 4) {
    return __builtin_bswap32(value);
  } else if constexpr (sizeof(T) == 8) {
    return __builtin_bswap64(value);
  } else {
    T result = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
      result |= ((value >> (i * 8)) & 0xFF) << ((sizeof(T) - 1 - i) * 8);
    }
    return result;
  }
}

template <typename T>
void write_value_to_buffer(std::vector<uint8_t> &buffer, const T &value) {
  T write_value = value;
  if (IS_BIG_ENDIAN) {
    write_value = byte_swap<T>(value);
  }
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&write_value);
  buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T read_value_from_buffer(const uint8_t *&data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  data += sizeof(T);

  if (IS_BIG_ENDIAN) {
    value = byte_swap<T>(value);
  }
  return value;
}


void Doc::write_to_buffer(std::vector<uint8_t> &buffer, const void *src,
                          size_t size) {
  const uint8_t *bytes = static_cast<const uint8_t *>(src);
  buffer.insert(buffer.end(), bytes, bytes + size);
}

void Doc::read_from_buffer(const uint8_t *&data, void *dest, size_t size) {
  std::memcpy(dest, data, size);
  data += size;
}

void Doc::serialize_value(std::vector<uint8_t> &buffer, const Value &value) {
  std::visit(
      [&buffer](const auto &v) {
        using T = std::decay_t<decltype(v)>;

        if constexpr (std::is_same_v<T, std::monostate>) {
          uint8_t type = TYPE_EMPTY;
          write_to_buffer(buffer, &type, sizeof(type));
        } else if constexpr (std::is_same_v<T, bool>) {
          uint8_t type = TYPE_BOOL;
          write_to_buffer(buffer, &type, sizeof(type));
          write_to_buffer(buffer, &v, sizeof(v));
        } else if constexpr (std::is_same_v<T, int32_t>) {
          uint8_t type = TYPE_INT32;
          write_to_buffer(buffer, &type, sizeof(type));
          write_value_to_buffer<int32_t>(buffer, v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
          uint8_t type = TYPE_INT64;
          write_to_buffer(buffer, &type, sizeof(type));
          write_value_to_buffer<int64_t>(buffer, v);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
          uint8_t type = TYPE_UINT32;
          write_to_buffer(buffer, &type, sizeof(type));
          write_value_to_buffer<uint32_t>(buffer, v);
        } else if constexpr (std::is_same_v<T, uint64_t>) {
          uint8_t type = TYPE_UINT64;
          write_to_buffer(buffer, &type, sizeof(type));
          write_value_to_buffer<uint64_t>(buffer, v);
        } else if constexpr (std::is_same_v<T, float>) {
          uint8_t type = TYPE_FLOAT;
          write_to_buffer(buffer, &type, sizeof(type));
          write_value_to_buffer<float>(buffer, v);
        } else if constexpr (std::is_same_v<T, double>) {
          uint8_t type = TYPE_DOUBLE;
          write_to_buffer(buffer, &type, sizeof(type));
          write_value_to_buffer<double>(buffer, v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          uint8_t type = TYPE_STRING;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          write_to_buffer(buffer, v.data(), len);
        } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
          uint8_t type = TYPE_VECTOR_BOOL;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          for (bool b : v) {
            write_to_buffer(buffer, &b, sizeof(b));
          }
        } else if constexpr (std::is_same_v<T, std::vector<int8_t>>) {
          uint8_t type = TYPE_VECTOR_INT8;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          write_to_buffer(buffer, v.data(), len * sizeof(int8_t));
        } else if constexpr (std::is_same_v<T, std::vector<int16_t>>) {
          uint8_t type = TYPE_VECTOR_INT16;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              int16_t swapped = byte_swap<int16_t>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(int16_t));
          }
        } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
          uint8_t type = TYPE_VECTOR_INT32;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              int32_t swapped = byte_swap<int32_t>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(int32_t));
          }
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          uint8_t type = TYPE_VECTOR_INT64;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              int64_t swapped = byte_swap<int64_t>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(int64_t));
          }
        } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
          uint8_t type = TYPE_VECTOR_UINT32;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              uint32_t swapped = byte_swap<uint32_t>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(uint32_t));
          }
        } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
          uint8_t type = TYPE_VECTOR_UINT64;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              uint64_t swapped = byte_swap<uint64_t>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(uint64_t));
          }
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
          uint8_t type = TYPE_VECTOR_FLOAT;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              float swapped = byte_swap<float>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(float));
          }
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          uint8_t type = TYPE_VECTOR_DOUBLE;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              double swapped = byte_swap<double>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(double));
          }
        } else if constexpr (std::is_same_v<T, std::vector<float16_t>>) {
          uint8_t type = TYPE_VECTOR_FLOAT16;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &val : v) {
              float16_t swapped = byte_swap<float16_t>(val);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            write_to_buffer(buffer, v.data(), len * sizeof(float16_t));
          }
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          uint8_t type = TYPE_VECTOR_STRING;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          for (const auto &s : v) {
            uint32_t str_len = static_cast<uint32_t>(s.size());
            write_value_to_buffer<uint32_t>(buffer, str_len);
            write_to_buffer(buffer, s.data(), str_len);
          }
        } else if constexpr (std::is_same_v<T, std::pair<std::vector<uint32_t>,
                                                         std::vector<float>>>) {
          uint8_t type = TYPE_VECTOR_PAIR_INT_FLOAT;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.first.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &p : v.first) {
              uint32_t swapped = byte_swap<uint32_t>(p);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            for (const auto &p : v.first) {
              write_to_buffer(buffer, &p, sizeof(p));
            }
          }
          len = static_cast<uint32_t>(v.second.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &p : v.second) {
              float swapped = byte_swap<float>(p);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            for (const auto &p : v.second) {
              write_to_buffer(buffer, &p, sizeof(p));
            }
          }
        } else if constexpr (std::is_same_v<
                                 T, std::pair<std::vector<uint32_t>,
                                              std::vector<float16_t>>>) {
          uint8_t type = TYPE_VECTOR_PAIR_INT_FLOAT16;
          write_to_buffer(buffer, &type, sizeof(type));
          uint32_t len = static_cast<uint32_t>(v.first.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &p : v.first) {
              uint32_t swapped = byte_swap<uint32_t>(p);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            for (const auto &p : v.first) {
              write_to_buffer(buffer, &p, sizeof(p));
            }
          }
          len = static_cast<uint32_t>(v.second.size());
          write_value_to_buffer<uint32_t>(buffer, len);
          if (IS_BIG_ENDIAN) {
            for (const auto &p : v.second) {
              float16_t swapped = byte_swap<float16_t>(p);
              write_to_buffer(buffer, &swapped, sizeof(swapped));
            }
          } else {
            for (const auto &p : v.second) {
              write_to_buffer(buffer, &p, sizeof(p));
            }
          }
        }
      },
      value);
}


Doc::Value Doc::deserialize_value(const uint8_t *&data) {
  uint8_t type;
  read_from_buffer(data, &type, sizeof(type));

  switch (type) {
    case TYPE_EMPTY: {
      return std::monostate{};
    }
    case TYPE_BOOL: {
      bool v;
      read_from_buffer(data, &v, sizeof(v));
      return v;
    }
    case TYPE_INT32: {
      return read_value_from_buffer<int32_t>(data);
    }
    case TYPE_INT64: {
      return read_value_from_buffer<int64_t>(data);
    }
    case TYPE_UINT32: {
      return read_value_from_buffer<uint32_t>(data);
    }
    case TYPE_UINT64: {
      return read_value_from_buffer<uint64_t>(data);
    }
    case TYPE_FLOAT: {
      return read_value_from_buffer<float>(data);
    }
    case TYPE_DOUBLE: {
      return read_value_from_buffer<double>(data);
    }
    case TYPE_STRING: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::string v(reinterpret_cast<const char *>(data), len);
      data += len;
      return v;
    }
    case TYPE_VECTOR_BOOL: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<bool> v;
      v.reserve(len);
      for (uint32_t i = 0; i < len; ++i) {
        bool b;
        read_from_buffer(data, &b, sizeof(b));
        v.push_back(b);
      }
      return v;
    }
    case TYPE_VECTOR_INT8: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<int8_t> v(len);
      read_from_buffer(data, v.data(), len * sizeof(int8_t));
      return v;
    }
    case TYPE_VECTOR_INT16: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<int16_t> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<int16_t>(read_value_from_buffer<int16_t>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(int16_t));
      }
      return v;
    }
    case TYPE_VECTOR_INT32: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<int32_t> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<int32_t>(read_value_from_buffer<int32_t>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(int32_t));
      }
      return v;
    }
    case TYPE_VECTOR_INT64: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<int64_t> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<int64_t>(read_value_from_buffer<int64_t>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(int64_t));
      }
      return v;
    }
    case TYPE_VECTOR_UINT32: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<uint32_t> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<uint32_t>(read_value_from_buffer<uint32_t>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(uint32_t));
      }
      return v;
    }
    case TYPE_VECTOR_UINT64: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<uint64_t> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<uint64_t>(read_value_from_buffer<uint64_t>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(uint64_t));
      }
      return v;
    }
    case TYPE_VECTOR_FLOAT: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<float> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<float>(read_value_from_buffer<float>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(float));
      }
      return v;
    }
    case TYPE_VECTOR_DOUBLE: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<double> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<double>(read_value_from_buffer<double>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(double));
      }
      return v;
    }
    case TYPE_VECTOR_FLOAT16: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<float16_t> v(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v[i] = byte_swap<float16_t>(read_value_from_buffer<float16_t>(data));
        }
      } else {
        read_from_buffer(data, v.data(), len * sizeof(float16_t));
      }
      return v;
    }
    case TYPE_VECTOR_STRING: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::vector<std::string> v;
      v.reserve(len);
      for (uint32_t i = 0; i < len; ++i) {
        uint32_t str_len = read_value_from_buffer<uint32_t>(data);
        std::string s(reinterpret_cast<const char *>(data), str_len);
        data += str_len;
        v.push_back(s);
      }
      return v;
    }
    case TYPE_VECTOR_PAIR_INT_FLOAT: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::pair<std::vector<uint32_t>, std::vector<float>> v;
      v.first.reserve(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v.first.push_back(
              byte_swap<uint32_t>(read_value_from_buffer<uint32_t>(data)));
        }
      } else {
        for (uint32_t i = 0; i < len; ++i) {
          uint32_t first;
          read_from_buffer(data, &first, sizeof(first));
          v.first.push_back(first);
        }
      }
      len = read_value_from_buffer<uint32_t>(data);
      v.second.reserve(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v.second.push_back(
              byte_swap<float>(read_value_from_buffer<float>(data)));
        }
      } else {
        for (uint32_t i = 0; i < len; ++i) {
          float second;
          read_from_buffer(data, &second, sizeof(second));
          v.second.push_back(second);
        }
      }
      return v;
    }
    case TYPE_VECTOR_PAIR_INT_FLOAT16: {
      uint32_t len = read_value_from_buffer<uint32_t>(data);
      std::pair<std::vector<uint32_t>, std::vector<float16_t>> v;
      v.first.reserve(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v.first.push_back(
              byte_swap<uint32_t>(read_value_from_buffer<uint32_t>(data)));
        }
      } else {
        for (uint32_t i = 0; i < len; ++i) {
          uint32_t first;
          read_from_buffer(data, &first, sizeof(first));
          v.first.push_back(first);
        }
      }
      len = read_value_from_buffer<uint32_t>(data);
      v.second.reserve(len);
      if (IS_BIG_ENDIAN) {
        for (uint32_t i = 0; i < len; ++i) {
          v.second.push_back(
              byte_swap<float16_t>(read_value_from_buffer<float16_t>(data)));
        }
      } else {
        for (uint32_t i = 0; i < len; ++i) {
          float16_t second;
          read_from_buffer(data, &second, sizeof(second));
          v.second.push_back(second);
        }
      }
      return v;
    }

    default:
      throw std::runtime_error("Unknown value type: " + std::to_string(type));
  }
}

std::vector<uint8_t> Doc::serialize() const {
  std::vector<uint8_t> buffer;
  uint32_t pk_len = static_cast<uint32_t>(pk_.size());
  write_to_buffer(buffer, &pk_len, sizeof(pk_len));
  write_to_buffer(buffer, pk_.data(), pk_len);

  write_to_buffer(buffer, &score_, sizeof(score_));
  write_to_buffer(buffer, &doc_id_, sizeof(doc_id_));
  write_to_buffer(buffer, &op_, sizeof(op_));

  uint32_t field_count = static_cast<uint32_t>(fields_.size());
  write_to_buffer(buffer, &field_count, sizeof(field_count));

  for (const auto &[field_name, value] : fields_) {
    uint32_t name_len = static_cast<uint32_t>(field_name.size());
    write_to_buffer(buffer, &name_len, sizeof(name_len));
    write_to_buffer(buffer, field_name.data(), name_len);

    serialize_value(buffer, value);
  }

  return buffer;
}

Doc::Ptr Doc::deserialize(const uint8_t *data, size_t /*size*/) {
  const uint8_t *ptr = data;
  Doc::Ptr doc = std::make_shared<Doc>();

  uint32_t pk_len = read_value_from_buffer<uint32_t>(ptr);
  std::string pk(reinterpret_cast<const char *>(ptr), pk_len);
  ptr += pk_len;
  doc->set_pk(pk);

  float score = read_value_from_buffer<float>(ptr);
  doc->set_score(score);

  uint64_t doc_id = read_value_from_buffer<uint64_t>(ptr);
  doc->set_doc_id(doc_id);

  Operator op;
  read_from_buffer(ptr, &op, sizeof(op));
  doc->set_operator(op);

  uint32_t field_count = read_value_from_buffer<uint32_t>(ptr);

  for (uint32_t i = 0; i < field_count; ++i) {
    uint32_t name_len = read_value_from_buffer<uint32_t>(ptr);
    std::string field_name(reinterpret_cast<const char *>(ptr), name_len);
    ptr += name_len;

    Doc::Value value = deserialize_value(ptr);
    doc->fields_[field_name] = value;
  }

  return doc;
}

Status Doc::validate(const CollectionSchema::Ptr &schema,
                     bool is_update) const {
  if (!schema) {
    return Status::InternalError("doc validate failed: schema is null");
  }

  if (pk_.empty()) {
    return Status::InvalidArgument("doc validate failed: doc_id is not set");
  }

  if (!std::regex_match(pk_, DOC_PK_REGEX)) {
    return Status::InvalidArgument("doc validate failed: doc_id[", pk_,
                                   "] cannot pass the regex verification");
  }

  // check doc fields match schema
  for (auto &[name, value] : fields_) {
    if (!schema->has_field(name)) {
      return Status::InvalidArgument("doc validate failed: field[", name,
                                     "] does not exist in collection's schema");
    }
  }

  const auto &fields = schema->fields();
  for (auto const &field_schema : fields) {
    auto field_name = field_schema->name();
    auto field_pair = fields_.find(field_name);
    if (field_pair == fields_.end()) {
      if (field_schema->nullable() || is_update) {
        continue;
      }
      return Status::InvalidArgument(
          "doc validate failed: field[", field_name,
          "] is configured not nullable, but doc does not contain this field");
    } else {
      if (std::holds_alternative<std::monostate>(field_pair->second)) {
        if (field_schema->nullable()) {
          continue;
        }
        return Status::InvalidArgument(
            "doc validate failed: field[", field_name,
            "] is configured not nullable, but doc's field value is empty");
      }
    }

    const Value &field_value = field_pair->second;
    DataType expected_type = field_schema->data_type();
    bool type_match = true;
    uint32_t value_dimension = 0;

    switch (expected_type) {
      case DataType::BINARY:
        type_match = std::holds_alternative<std::string>(field_value);
        break;
      case DataType::STRING:
        type_match = std::holds_alternative<std::string>(field_value);
        break;
      case DataType::BOOL:
        type_match = std::holds_alternative<bool>(field_value);
        break;
      case DataType::INT32:
        type_match = std::holds_alternative<int32_t>(field_value);
        break;
      case DataType::UINT32:
        type_match = std::holds_alternative<uint32_t>(field_value);
        break;
      case DataType::INT64:
        type_match = std::holds_alternative<int64_t>(field_value);
        break;
      case DataType::UINT64:
        type_match = std::holds_alternative<uint64_t>(field_value);
        break;
      case DataType::FLOAT:
        type_match = std::holds_alternative<float>(field_value);
        break;
      case DataType::DOUBLE:
        type_match = std::holds_alternative<double>(field_value);
        break;
      case DataType::ARRAY_BINARY:
        type_match =
            std::holds_alternative<std::vector<std::string>>(field_value);
        break;
      case DataType::ARRAY_STRING:
        type_match =
            std::holds_alternative<std::vector<std::string>>(field_value);
        break;
      case DataType::ARRAY_BOOL:
        type_match = std::holds_alternative<std::vector<bool>>(field_value);
        break;
      case DataType::ARRAY_INT32:
        type_match = std::holds_alternative<std::vector<int32_t>>(field_value);
        break;
      case DataType::ARRAY_INT64:
        type_match = std::holds_alternative<std::vector<int64_t>>(field_value);
        break;
      case DataType::ARRAY_UINT32:
        type_match = std::holds_alternative<std::vector<uint32_t>>(field_value);
        break;
      case DataType::ARRAY_UINT64:
        type_match = std::holds_alternative<std::vector<uint64_t>>(field_value);
        break;
      case DataType::ARRAY_FLOAT:
        type_match = std::holds_alternative<std::vector<float>>(field_value);
        break;
      case DataType::ARRAY_DOUBLE:
        type_match = std::holds_alternative<std::vector<double>>(field_value);
        break;
      case DataType::VECTOR_BINARY32: {
        type_match = std::holds_alternative<std::vector<uint32_t>>(field_value);
        if (type_match) {
          value_dimension = std::get<std::vector<uint32_t>>(field_value).size();
        }
        break;
      }
      case DataType::VECTOR_BINARY64: {
        type_match = std::holds_alternative<std::vector<uint64_t>>(field_value);
        if (type_match) {
          value_dimension = std::get<std::vector<uint64_t>>(field_value).size();
        }
        break;
      }
      case DataType::VECTOR_FP16: {
        type_match =
            std::holds_alternative<std::vector<float16_t>>(field_value);
        if (type_match) {
          value_dimension =
              std::get<std::vector<float16_t>>(field_value).size();
        }
        break;
      }
      case DataType::VECTOR_FP32: {
        type_match = std::holds_alternative<std::vector<float>>(field_value);
        if (type_match) {
          value_dimension = std::get<std::vector<float>>(field_value).size();
        }
        break;
      }
      case DataType::VECTOR_FP64: {
        type_match = std::holds_alternative<std::vector<double>>(field_value);
        if (type_match) {
          value_dimension = std::get<std::vector<double>>(field_value).size();
        }
        break;
      }
      // case DataType::VECTOR_INT4:
      //   type_match =
      //   std::holds_alternative<std::vector<int8_t>>(field_value); break;
      case DataType::VECTOR_INT8: {
        type_match = std::holds_alternative<std::vector<int8_t>>(field_value);
        if (type_match) {
          value_dimension = std::get<std::vector<int8_t>>(field_value).size();
        }
        break;
      }
      case DataType::VECTOR_INT16: {
        type_match = std::holds_alternative<std::vector<int16_t>>(field_value);
        if (type_match) {
          value_dimension = std::get<std::vector<int16_t>>(field_value).size();
        }
        break;
      }
      case DataType::SPARSE_VECTOR_FP16: {
        type_match = std::holds_alternative<
            std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
            field_value);
        if (type_match) {
          auto [sparse_indices, sparse_values] = std::get<
              std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
              field_value);
          if (sparse_values.size() != sparse_indices.size()) {
            return Status::InvalidArgument(
                "doc validate failed: field[", field_name,
                "]'s sparse vector indices and values size not match");
          }
        }
        break;
      }
      case DataType::SPARSE_VECTOR_FP32: {
        type_match = std::holds_alternative<
            std::pair<std::vector<uint32_t>, std::vector<float>>>(field_value);
        if (type_match) {
          auto &[sparse_indices, sparse_values] =
              std::get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
                  field_value);
          if (sparse_values.size() != sparse_indices.size()) {
            return Status::InvalidArgument(
                "doc validate failed: field[", field_name,
                "]'s sparse vector indices and values size not match");
          }
        }
        break;
      }
      default:
        return Status::InvalidArgument("doc validate failed: field[",
                                       field_name,
                                       "]'s value type is not supported");
        break;
    }

    if (!type_match) {
      return Status::InvalidArgument(
          "doc validate failed: field[", field_name,
          "]'s value type mismatch, it should be ",
          DataTypeCodeBook::AsString(expected_type), ", but got type: ",
          get_value_type_name(field_value, field_schema->is_vector_field()));
    }
    if (field_schema->is_dense_vector()) {
      if (value_dimension != field_schema->dimension()) {
        return Status::InvalidArgument(
            "doc validate failed: field[", field_name,
            "]'s dimension mismatch, it should be ", field_schema->dimension(),
            ", but got dimension: ", value_dimension);
      }
    }
  }
  return Status::OK();
}

size_t Doc::memory_usage() const {
  // Base size of the object itself
  size_t usage = sizeof(Doc);

  // Calculate memory used by pk_ string
  usage += pk_.capacity();

  // Calculate memory used by fields_ hash map structure
  usage += fields_.bucket_count() *
           sizeof(std::unordered_map<std::string, Value>::value_type *);

  // Iterate through all fields to calculate their actual memory usage
  for (const auto &pair : fields_) {
    const auto &key = pair.first;
    const auto &value = pair.second;

    // Memory for the key (string)
    usage += key.capacity();

    // Memory for the value (based on actual variant type)
    usage += [&value]() -> size_t {
      switch (value.index()) {
        case 0:      // std::monostate
          return 0;  // No additional memory needed

        case 1:      // bool
        case 2:      // int32_t
        case 3:      // uint32_t
        case 4:      // int64_t
        case 5:      // uint64_t
        case 6:      // float
        case 7:      // double
          return 0;  // Basic types are already allocated within the variant

        case 8:  // std::string
          return std::get<std::string>(value).capacity();

        case 9:  // std::vector<bool>
          return std::get<std::vector<bool>>(value).size() * sizeof(bool);

        case 10:  // std::vector<int8_t>
          return std::get<std::vector<int8_t>>(value).capacity() *
                 sizeof(int8_t);

        case 11:  // std::vector<int16_t>
          return std::get<std::vector<int16_t>>(value).capacity() *
                 sizeof(int16_t);

        case 12:  // std::vector<int32_t>
          return std::get<std::vector<int32_t>>(value).capacity() *
                 sizeof(int32_t);

        case 13:  // std::vector<int64_t>
          return std::get<std::vector<int64_t>>(value).capacity() *
                 sizeof(int64_t);

        case 14:  // std::vector<uint32_t>
          return std::get<std::vector<uint32_t>>(value).capacity() *
                 sizeof(uint32_t);

        case 15:  // std::vector<uint64_t>
          return std::get<std::vector<uint64_t>>(value).capacity() *
                 sizeof(uint64_t);

        case 16:  // std::vector<float16_t>
          return std::get<std::vector<float16_t>>(value).capacity() *
                 sizeof(float16_t);

        case 17:  // std::vector<float>
          return std::get<std::vector<float>>(value).capacity() * sizeof(float);

        case 18:  // std::vector<double>
          return std::get<std::vector<double>>(value).capacity() *
                 sizeof(double);

        case 19:  // std::vector<std::string>
        {
          size_t vec_usage =
              std::get<std::vector<std::string>>(value).capacity() *
              sizeof(std::string);
          for (const auto &str : std::get<std::vector<std::string>>(value)) {
            vec_usage += str.capacity();
          }
          return vec_usage;
        }

        case 20:  // std::pair<std::vector<uint32_t>, std::vector<float>>
        {
          const auto &pair_val =
              std::get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
                  value);
          return pair_val.first.capacity() * sizeof(uint32_t) +
                 pair_val.second.capacity() * sizeof(float);
        }

        case 21:  // std::pair<std::vector<uint32_t>, std::vector<float16_t>>
        {
          const auto &pair_val = std::get<
              std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(value);
          return pair_val.first.capacity() * sizeof(uint32_t) +
                 pair_val.second.capacity() * sizeof(float16_t);
        }

        default:
          return 0;
      }
    }();
  }

  return usage;
}

template <typename T>
std::string vec_to_string(const std::vector<T> &v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << +v[i];  // + from print as char
  }
  oss << "]";
  return oss.str();
}

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

std::string Doc::to_detail_string() const {
  std::stringstream oss;
  oss << "[op:" << (uint32_t)op_ << ", doc_id: " << doc_id_
      << ", score: " << score_ << ", pk: " << pk_
      << ", fields: " << fields_.size() << "]";
  oss << "{";
  bool first_field = true;
  for (const auto &[key, val] : fields_) {
    if (!first_field) oss << ", ";
    first_field = false;

    oss << "\"" << key << "\": ";

    std::visit(
        overloaded{
            [&](std::monostate) { oss << "null"; },
            [&](bool b) { oss << (b ? "true" : "false"); },
            [&](int32_t i) { oss << i; },
            [&](uint32_t u) { oss << u; },
            [&](int64_t i) { oss << i; },
            [&](uint64_t u) { oss << u; },
            [&](float f) { oss << f; },
            [&](double d) { oss << d; },
            [&](const std::string &s) { oss << "\"" << s << "\""; },
            [&](const std::vector<bool> &vb) { oss << vec_to_string(vb); },
            [&](const std::vector<int32_t> &v) { oss << vec_to_string(v); },
            [&](const std::vector<int8_t> &v) { oss << vec_to_string(v); },
            [&](const std::vector<int16_t> &v) { oss << vec_to_string(v); },
            [&](const std::vector<uint32_t> &v) { oss << vec_to_string(v); },
            [&](const std::vector<int64_t> &v) { oss << vec_to_string(v); },
            [&](const std::vector<uint64_t> &v) { oss << vec_to_string(v); },
            [&](const std::vector<float> &v) { oss << vec_to_string(v); },
            [&](const std::vector<double> &v) { oss << vec_to_string(v); },
            [&](const std::vector<std::string> &v) {
              oss << "[";
              for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "\"" << v[i] << "\"";
              }
              oss << "]";
            },
            [&](const std::vector<float16_t> &v) {
              oss << "[";
              for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << static_cast<float>(v[i]);  // print in float
              }
              oss << "]";
            },
            [&](const std::pair<std::vector<uint32_t>, std::vector<float>> &p) {
              oss << "{first:" << vec_to_string(p.first)
                  << ", second:" << vec_to_string(p.second) << "}";
            },
            [&](const std::pair<std::vector<uint32_t>, std::vector<float16_t>>
                    &p) {
              oss << "{first:" << vec_to_string(p.first) << ", second:[";
              for (size_t i = 0; i < p.second.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << static_cast<float>(p.second[i]);
              }
              oss << "]}";
            }},
        val);
  }
  oss << "}";
  return oss.str();
}

bool Doc::operator==(const Doc &other) const {
  // Compare basic fields
  if (pk_ != other.pk_) {
    return false;
  }

  // Compare fields map sizes
  if (fields_.size() != other.fields_.size()) {
    return false;
  }

  // Compare each field
  for (const auto &pair : fields_) {
    const auto &field_name = pair.first;
    const auto &field_value = pair.second;

    auto it = other.fields_.find(field_name);
    if (it == other.fields_.end()) {
      return false;
    }

    // Compare variant values
    if (field_value.index() != it->second.index()) {
      return false;
    }

    // Use visitor to compare the actual values
    bool values_equal = std::visit(
        [](const auto &lhs, const auto &rhs) -> bool {
          if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                       std::decay_t<decltype(rhs)>>) {
            return lhs == rhs;
          } else {
            // This should not happen due to the index check above
            return false;
          }
        },
        field_value, it->second);

    if (!values_equal) {
      return false;
    }
  }

  return true;
}

Status VectorQuery::validate(const FieldSchema *schema) const {
  if ((uint32_t)topk_ > kMaxQueryTopk) {
    return Status::InvalidArgument("query validate failed: topk[", topk_,
                                   "] is too large, max is ", kMaxQueryTopk);
  }
  if (output_fields_.has_value() &&
      output_fields_->size() > kMaxOutputFieldSize) {
    return Status::InvalidArgument(
        "query validate failed: output_fields is too large, max is ",
        kMaxOutputFieldSize);
  }

  if (schema == nullptr) {
    // support query with vector
    if (query_vector_.empty() && query_sparse_indices_.empty()) {
      return Status::OK();
    }

    return Status::InvalidArgument("query validate failed:  vector_field[",
                                   field_name_,
                                   "] not defined in the collection schema");
  }
  // validate dense/sparse vector
  if (schema->is_dense_vector()) {
    // validate dimension
    auto dim = schema->dimension();
    switch (schema->data_type()) {
      case DataType::VECTOR_FP16:
        if (dim * sizeof(float16_t) != query_vector_.size()) {
          return Status::InvalidArgument(
              "query validate failed: dimension is invalid");
        }
        break;
      case DataType::VECTOR_FP32:
        if (dim * sizeof(float) != query_vector_.size()) {
          return Status::InvalidArgument(
              "query validate failed: dimension is invalid");
        }
        break;
      case DataType::VECTOR_FP64:
        if (dim * sizeof(double) != query_vector_.size()) {
          return Status::InvalidArgument(
              "query validate failed: dimension is invalid");
        }
        break;
      case DataType::VECTOR_INT8:
        if (dim * sizeof(int8_t) != query_vector_.size()) {
          return Status::InvalidArgument(
              "query validate failed: dimension is invalid");
        }
        break;
      case DataType::VECTOR_INT16:
      case DataType::VECTOR_INT4:
      case DataType::VECTOR_BINARY32:
      case DataType::VECTOR_BINARY64:
        return Status::NotSupported(
            "query validate failed: unsupported dense vector type");
      default:
        return Status::InvalidArgument(
            "query validate failed: field is not dense vector");
    }
  } else if (schema->is_sparse_vector()) {
    // validate sparse indices size
    if (query_sparse_indices_.size() >= kSparseMaxDimSize * sizeof(uint32_t)) {
      return Status::InvalidArgument(
          "query validate failed: sparse indices size is too large");
    }
  } else {
    return Status::InvalidArgument(
        "query validate failed: field is not vector");
  }
  return Status::OK();
}

}  // namespace zvec
