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

#include <turbo/distance/common/symphony_qg_distance.h>
#include <turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h>


namespace zvec::turbo {
#if RABITQ_SUPPORTED
// Node-centered RaBitQ blocks with the SymphonyQG rotation and FastScan layout.
class SymphonyQGBlockQuantizer {
 public:
  static constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
  struct Query {
    std::vector<float> rotated;
    SymQuery lookup;
  };

  explicit SymphonyQGBlockQuantizer(size_t dimension, bool cosine = false)
      : rotation_(dimension), cosine_(cosine) {}

  size_t encoded_dim() const {
    return rotation_.padded_dim();
  }
  size_t quantized_dim() const {
    return encoded_dim();
  }
  size_t batch_bytes() const {
    return rabitqlib::QGBatchDataMap<float>::data_bytes(quantized_dim());
  }
  void transform(const void *vector, std::vector<float> &out) const {
    rotation_.rotate(vector, out);
  }
  void encode(const float *center, const float *vectors, size_t count,
              char *codes) const {
    rabitqlib::quant::quantize_qg_batch(vectors, center, count, encoded_dim(),
                                        codes, rabitqlib::METRIC_L2);
    if (cosine_) {
      rabitqlib::QGBatchDataMap<float> batch(codes, quantized_dim());
      ConvertSymphonyQGCosineFactors(center, vectors, count, encoded_dim(),
                                     batch.f_add(), batch.f_rescale());
    }
  }
  void prepare_query(const void *vector, Query &query) const {
    transform(vector, query.rotated);
    query.lookup.reset(query.rotated.data(), quantized_dim());
  }
  void scan(const char *codes, Query &query, float center_distance,
            float *distances) const {
    query.lookup.set_g_add(center_distance);
    ScanSymphonyQGBatch(codes, query.lookup, quantized_dim(), distances);
  }

 private:
  SymphonyQGRotation rotation_;
  bool cosine_;
};
#endif
}  // namespace zvec::turbo
