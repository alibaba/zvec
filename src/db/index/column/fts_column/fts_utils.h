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

// --------------------------------------------------------------------------
// Big-endian uint32 encoding/decoding
// --------------------------------------------------------------------------

/*! Decode a 4-byte big-endian buffer into a uint32_t.
 *  \param data  Pointer to at least 4 bytes of big-endian data.
 *  \return The decoded uint32_t value.
 */
inline uint32_t decode_uint32_big_endian(const char *data) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(data[0])) << 24) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[1])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[2])) << 8) |
         static_cast<uint32_t>(static_cast<uint8_t>(data[3]));
}

/*! Encode a uint32_t value into 4 bytes of big-endian and append to output.
 *  \param value   The uint32_t value to encode.
 *  \param output  String to append the 4 bytes to.
 */
inline void encode_uint32_big_endian(uint32_t value, std::string *output) {
  output->push_back(static_cast<char>((value >> 24) & 0xFF));
  output->push_back(static_cast<char>((value >> 16) & 0xFF));
  output->push_back(static_cast<char>((value >> 8) & 0xFF));
  output->push_back(static_cast<char>(value & 0xFF));
}

// --------------------------------------------------------------------------
// Doc-term key encoding/decoding
// --------------------------------------------------------------------------

/*! Build a composite key: term + '\0' + doc_id (4 bytes big-endian).
 *  Used by postings ($TF/$POS) column families.
 *  \param term    Term string (must not contain embedded NULs).
 *  \param doc_id  Local document ID.
 *  \return Encoded key string.
 */
inline std::string make_doc_term_key(const std::string &term, uint32_t doc_id) {
  std::string key;
  key.reserve(term.size() + 1 + sizeof(uint32_t));
  key.append(term);
  key.push_back('\0');
  encode_uint32_big_endian(doc_id, &key);
  return key;
}

/*! Decode a composite key produced by make_doc_term_key().
 *  Key format: term + '\0' + doc_id (4 bytes big-endian).
 *  \param key        The raw key to decode.
 *  \param term_out   Output: the term string.
 *  \param doc_id_out Output: the decoded local document ID.
 *  \return true on success, false if the key is malformed.
 */
bool parse_doc_term_key(const std::string &key, std::string *term_out,
                        uint32_t *doc_id_out);

// --------------------------------------------------------------------------
// Per-field segment-stat key encoding (stat_cf)
// --------------------------------------------------------------------------
//
// FTS stores two per-field aggregate statistics in stat_cf so that BM25
// scoring at search time has access to corpus-level N (total_docs) and
// total token count (used to derive avgdl).  The same key naming and
// uint64 little-endian (host-order memcpy) value layout is shared by:
//   - FtsColumnIndexer::flush()        (writer, mutable segment)
//   - FtsRocksdbReducer::flush_stat()  (writer, segment merge)
//   - BM25Scorer::load_segment_stats() (reader, search time)
// Centralising the contract here prevents the three sites from drifting
// apart when the schema evolves.

/*! Build the stat_cf key for total_docs of a given field. */
inline std::string make_total_docs_key(const std::string &field_name) {
  return field_name + "_total_docs";
}

/*! Build the stat_cf key for total_tokens of a given field. */
inline std::string make_total_tokens_key(const std::string &field_name) {
  return field_name + "_total_tokens";
}

/*! Encode a uint64_t value as an 8-byte big-endian string.
 *  Used for stat_cf values total_docs / total_tokens.
 *  Big-endian layout ensures lexicographic order matches numeric order.
 */
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

/*! Decode a uint64_t value from an 8-byte big-endian string. */
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
