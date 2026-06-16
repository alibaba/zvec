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
#include <memory>
#include <string>
#include <vector>
#include "zvec/core/framework/index_dumper.h"
#include "zvec/core/framework/index_storage.h"

namespace zvec {
namespace core {

//! Segment ID used when dumping/loading the rotator data
inline const std::string RECORD_ROTATOR_SEG_ID{"integer_streaming.rotator"};

//! Rotator type exposed without rabitqlib dependency
enum class RecordRotatorType : uint8_t {
  FhtKac = 0,  //!< O(d log d) FHT-based Kac random rotation (default)
  Matrix = 1,  //!< O(d^2) explicit random matrix rotation
};

/*! RecordRotator provides per-vector rotation without external dependencies.
 *
 * All rotation algorithms are implemented inline (FHT-based Kac walk and
 * explicit random matrix), so no rabitqlib headers are required.
 *
 * Auto-selects the rotation algorithm based on dimension alignment:
 *  - dimension % 64 == 0 -> FhtKac  (O(d log d), requires 64-alignment)
 *  - otherwise           -> Matrix  (O(d^2), no alignment requirement)
 *
 * Rotation preserves dimension: output size == input size (no padding).
 *
 * Used by IntegerStreamingConverter/Reformer and CosineConverter/Reformer
 * when enable_rotate is true.
 */
class RecordRotator {
 public:
  RecordRotator();
  ~RecordRotator();

  //! Move-only (pimpl with unique_ptr)
  RecordRotator(RecordRotator &&) noexcept;
  RecordRotator &operator=(RecordRotator &&) noexcept;
  RecordRotator(const RecordRotator &) = delete;
  RecordRotator &operator=(const RecordRotator &) = delete;

  //! Initialize the rotator.
  //! Auto-selects FhtKac when dimension is 64-aligned, else falls back to
  //! Matrix.  The @p rotator_type parameter can force Matrix explicitly.
  //! @param dimension     vector dimension (input and output size)
  //! @param rotator_type  rotation algorithm (default: FhtKac, auto-degrades
  //!                      to Matrix when dimension is not 64-aligned)
  void init(size_t dimension,
            RecordRotatorType rotator_type = RecordRotatorType::FhtKac);

  //! Rotate a single vector
  //! @param in   input vector of size >= dimension
  //! @param out  output buffer of size >= dimension
  void rotate(const float *in, float *out) const;

  //! Rotate a single vector into a managed buffer
  //! @param in  input vector of size >= dimension
  //! @return    vector<float> of size dimension containing rotated result
  std::vector<float> rotate(const float *in) const;

  //! Inverse-rotate a single vector (from rotated space back to original)
  //! @param in   input vector of size >= dimension (rotated vector)
  //! @param out  output buffer of size >= dimension (original space)
  void unrotate(const float *in, float *out) const;

  //! Inverse-rotate a single vector into a managed buffer
  //! @param in  input vector of size >= dimension (rotated vector)
  //! @return    vector<float> of size dimension containing inverse-rotated
  //! result
  std::vector<float> unrotate(const float *in) const;

  //! Return the serialized size of the rotator in bytes (header + blob)
  size_t dump_bytes() const;

  //! Dump the rotator to an IndexStorage as a named segment.
  //! Same self-describing format as the dumper variant.
  int dump(const IndexStorage::Pointer &storage,
           const std::string &seg_id = RECORD_ROTATOR_SEG_ID) const;

  //! Dump the rotator to an IndexDumper as a named segment.
  //! Format: [Header: type(1B)|origin_dim(4B)|reserved(4B)] [rotation blob]
  //! Appends padding for 32-byte alignment.
  int dump(const IndexDumper::Pointer &dumper,
           const std::string &seg_id = RECORD_ROTATOR_SEG_ID) const;

  //! Open the rotator from an IndexStorage segment (self-describing, no init
  //! needed). Parses header to get type/dimension, then reconstructs the
  //! rotator.
  int open(IndexStorage::Pointer storage,
           const std::string &seg_id = RECORD_ROTATOR_SEG_ID);

  //! Load a user-specified rotation matrix.
  //! Always uses MatrixRotator internally.
  //! @param matrix       row-major square matrix of shape dimension x dimension
  //! @param dimension    vector dimension
  int load(const float *matrix, size_t dimension);

  //! Return the vector dimension
  size_t dimension() const;

  //! Return the rotator type
  RecordRotatorType rotator_type() const;

  //! Check if the rotator is initialized
  bool initialized() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace zvec
