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

#include <cstdint>
#include <cstring>
#include <string>

namespace zvec::fts {

// Big-endian uint32 encoding/decoding.
inline uint32_t decode_uint32_big_endian(const char *data) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(data[0])) << 24) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[1])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[2])) << 8) |
         static_cast<uint32_t>(static_cast<uint8_t>(data[3]));
}

inline void encode_uint32_big_endian(uint32_t value, std::string *output) {
  output->push_back(static_cast<char>((value >> 24) & 0xFF));
  output->push_back(static_cast<char>((value >> 16) & 0xFF));
  output->push_back(static_cast<char>((value >> 8) & 0xFF));
  output->push_back(static_cast<char>(value & 0xFF));
}

// Doc-term key: term + '\0' + doc_id (4-byte big-endian).
// Used by postings ($TF/$POS) column families.
inline std::string make_doc_term_key(const std::string &term, uint32_t doc_id) {
  std::string key;
  key.reserve(term.size() + 1 + sizeof(uint32_t));
  key.append(term);
  key.push_back('\0');
  encode_uint32_big_endian(doc_id, &key);
  return key;
}

// In-place variant of make_doc_term_key: appends the key to an existing buffer.
// Callers that build many keys in a row can reserve once and reuse the buffer,
// avoiding per-key allocation. Returns the number of bytes appended so the
// caller can build Slices into the buffer.
inline size_t append_doc_term_key(const std::string &term, uint32_t doc_id,
                                  std::string *buf) {
  const size_t bytes = term.size() + 1 + sizeof(uint32_t);
  buf->append(term);
  buf->push_back('\0');
  encode_uint32_big_endian(doc_id, buf);
  return bytes;
}

bool parse_doc_term_key(const std::string &key, std::string *term_out,
                        uint32_t *doc_id_out);

// Per-field segment-stat keys (stat_cf) for BM25 scoring.
inline std::string make_total_docs_key(const std::string &field_name) {
  return field_name + "_total_docs";
}

inline std::string make_total_tokens_key(const std::string &field_name) {
  return field_name + "_total_tokens";
}

// uint64 big-endian encoding for stat values.
inline std::string encode_uint64_value(uint64_t value) {
  std::string out(sizeof(uint64_t), '\0');
  out[0] = static_cast<char>((value >> 56) & 0xFF);
  out[1] = static_cast<char>((value >> 48) & 0xFF);
  out[2] = static_cast<char>((value >> 40) & 0xFF);
  out[3] = static_cast<char>((value >> 32) & 0xFF);
  out[4] = static_cast<char>((value >> 24) & 0xFF);
  out[5] = static_cast<char>((value >> 16) & 0xFF);
  out[6] = static_cast<char>((value >> 8) & 0xFF);
  out[7] = static_cast<char>(value & 0xFF);
  return out;
}

inline uint64_t decode_uint64_value(const char *data) {
  return (static_cast<uint64_t>(static_cast<uint8_t>(data[0])) << 56) |
         (static_cast<uint64_t>(static_cast<uint8_t>(data[1])) << 48) |
         (static_cast<uint64_t>(static_cast<uint8_t>(data[2])) << 40) |
         (static_cast<uint64_t>(static_cast<uint8_t>(data[3])) << 32) |
         (static_cast<uint64_t>(static_cast<uint8_t>(data[4])) << 24) |
         (static_cast<uint64_t>(static_cast<uint8_t>(data[5])) << 16) |
         (static_cast<uint64_t>(static_cast<uint8_t>(data[6])) << 8) |
         static_cast<uint64_t>(static_cast<uint8_t>(data[7]));
}

}  // namespace zvec::fts
