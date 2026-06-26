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
inline const std::string ROTATOR_SEG_ID{"enable_rotate"};

//! Rotator type
enum class RotatorType : uint8_t {
  FhtKac = 0,  //!< O(d log d) FHT-based Kac random rotation (default)
  Matrix = 1,  //!< O(d^2) explicit random matrix rotation
};

// Forward declarations for derived classes
class FhtRotator;
class MatrixRotator;

/*! Rotator provides per-vector rotation without external dependencies.
 *
 * Abstract base class for rotation algorithms.  Use the static factory
 * methods to create instances:
 *  - create()      — build a new rotator from scratch
 *  - open()        — load from storage (auto-detects type from header)
 *  - load_matrix() — load a user-specified rotation matrix
 *
 * Currently FhtRotator (O(d log d)) is the default and supports any
 * dimension.  MatrixRotator (O(d^2)) is retained for future use.
 *
 * Rotation preserves dimension: output size == input size (no padding).
 */
class Rotator {
 public:
  virtual ~Rotator() = default;

  // Static factories ---------------------------------------------------------

  //! Create and initialize a new rotator.
  //! @param out         on success, stores the rotator; on failure, nullptr
  //! @param dimension   vector dimension (input and output size)
  //! @param rotator_type rotation algorithm (default: FhtKac)
  //! @return 0 on success, error code on failure
  static int create(std::shared_ptr<Rotator> *out, size_t dimension,
                    RotatorType rotator_type = RotatorType::FhtKac);

  //! Open a rotator from an IndexStorage segment (self-describing, no init
  //! needed).  Parses header to get type/dimension, then reconstructs the
  //! appropriate derived class.
  //! @param out      on success, stores the rotator; on failure, nullptr
  //! @param storage  index storage
  //! @param seg_id   segment identifier
  //! @return 0 on success, error code on failure
  static int open(std::shared_ptr<Rotator> *out, IndexStorage::Pointer storage,
                  const std::string &seg_id = ROTATOR_SEG_ID);

  //! Load a user-specified rotation matrix (always creates MatrixRotator).
  //! @param matrix    row-major square matrix of shape dimension x dimension
  //! @param dimension vector dimension
  static std::unique_ptr<Rotator> load_matrix(const float *matrix,
                                              size_t dimension);

  // Virtual interface --------------------------------------------------------

  //! Rotate a single vector
  virtual void rotate(const float *in, float *out) const = 0;

  //! Inverse-rotate a single vector (from rotated space back to original)
  virtual void unrotate(const float *in, float *out) const = 0;

  //! Return the rotator type
  virtual RotatorType rotator_type() const = 0;

  // Non-virtual public methods ----------------------------------------------

  //! Rotate a single vector into a managed buffer
  std::vector<float> rotate(const float *in) const;

  //! Inverse-rotate a single vector into a managed buffer
  std::vector<float> unrotate(const float *in) const;

  //! Return the serialized size of the rotator in bytes (header + blob)
  size_t dump_bytes() const;

  //! Dump the rotator to an IndexStorage as a named segment.
  int dump(const IndexStorage::Pointer &storage,
           const std::string &seg_id = ROTATOR_SEG_ID) const;

  //! Dump the rotator to an IndexDumper as a named segment.
  //! Format: [Header (24B): magic|version|rotator_type|in_dim|
  //!          out_dim|payload_size|reserved] [payload blob]
  //! Appends padding for 32-byte alignment.
  int dump(const IndexDumper::Pointer &dumper,
           const std::string &seg_id = ROTATOR_SEG_ID) const;

  //! Return the vector dimension
  size_t dimension() const {
    return dimension_;
  }

  //! Check if the rotator is initialized
  bool initialized() const {
    return initialized_;
  }

 protected:
  // Protected virtuals — implemented by derived classes ---------------------

  //! Initialize the rotator's internal state for the given dimension.
  //! @return 0 on success, error code on failure
  virtual int init_impl(size_t dim) = 0;

  //! Return the serialized blob size (without header)
  virtual size_t blob_bytes() const = 0;

  //! Write the payload blob to the given buffer
  virtual void save_blob(char *data) const = 0;

  //! Read the payload blob from the given buffer
  virtual void load_blob(const char *data) = 0;

  // Protected members --------------------------------------------------------

  size_t dimension_{0};
  bool initialized_{false};

  // Serialization constants (shared by dump/open)
  static constexpr size_t kHeaderSize = 24;
  static constexpr uint32_t kMagic = 0x52544F52;  // "ROTR"
  static constexpr uint16_t kVersion = 1;
};

}  // namespace core
}  // namespace zvec
