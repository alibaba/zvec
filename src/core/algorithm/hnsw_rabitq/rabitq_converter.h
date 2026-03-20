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

#include <memory>
#include <vector>
#include <rabitqlib/utils/rotator.hpp>
#include "zvec/core/framework/index_cluster.h"
#include "zvec/core/framework/index_converter.h"
#include "zvec/core/framework/index_reformer.h"
#include "zvec/core/framework/index_threads.h"
#include "rabitq_params.h"

namespace zvec {
namespace core {

class RabitqReformer;

/*! RaBitQ Converter
 * Trains KMeans centroids and quantizes vectors using RaBitQ
 */
class RabitqConverter : public IndexConverter {
 public:
  //! Constructor
  RabitqConverter() = default;

  //! Destructor
  ~RabitqConverter() override;

  //! Initialize Converter
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Cleanup Converter
  int cleanup(void) override;

  //! Train the data - perform KMeans clustering
  int train(IndexHolder::Pointer holder) override;

  //! Transform the data - quantize vectors using RaBitQ
  int transform(IndexHolder::Pointer holder) override;

  //! Dump centroids and config into storage
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve a holder as result
  IndexHolder::Pointer result(void) const override {
    return result_holder_;
  }

  //! Retrieve Index Meta
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  int to_reformer(IndexReformer::Pointer *reformer) override;

 private:
  static inline size_t AlignSize(size_t size) {
    return (size + 0x1F) & (~0x1F);
  }

 private:
  IndexMeta meta_;
  IndexHolder::Pointer result_holder_;
  Stats stats_;
  size_t sample_count_{0};

  // RaBitQ parameters
  size_t num_clusters_{0};
  size_t ex_bits_{0};
  size_t dimension_{0};
  size_t padded_dim_{0};

  // Original centroids: num_clusters * dimension (for LinearSeeker query)
  std::vector<float> centroids_;
  // Rotated centroids: num_clusters * padded_dim (for quantization)
  std::vector<float> rotated_centroids_;

  // Rotator for vector transformation
  rabitqlib::RotatorType rotator_type_{rabitqlib::RotatorType::FhtKacRotator};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
};

}  // namespace core
}  // namespace zvec
