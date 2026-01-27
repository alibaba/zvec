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

#include "integer_quantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <ailego/math/normalizer.h>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

//! Make smooth the distribution to eliminate zero in hist
static inline void MakeSmooth(std::vector<float> &dist) {
  constexpr float epsilon = std::numeric_limits<float>::epsilon();

  // L1 Normalize first
  float norm = 1.0f;
  Normalizer<float>::L1(dist.data(), dist.size(), &norm);

  size_t zero_count = std::count_if(dist.begin(), dist.end(), [](float val) {
    return (std::abs(val) < std::numeric_limits<float>::epsilon());
  });
  size_t nonzero_count = dist.size() - zero_count;

  // Double check
  if (nonzero_count == 0 || zero_count == 0) {
    return;
  }

  float y = epsilon * zero_count / static_cast<float>(nonzero_count);
  for (auto &it : dist) {
    if (std::abs(it) < epsilon) {
      it += epsilon;
    } else {
      it -= y;
    }
  }  // end of for
}

//! Compute the Entropy of distribution p/q by  Kullback-Leibler Divergence
static inline double ComputeKlDivergence(const std::vector<float> &p,
                                         const std::vector<float> &q) {
  if (p.size() != q.size() || p.size() == 0) {
    return std::numeric_limits<float>::max();
  }

  double v = 0.0f;
  for (size_t i = 0; i != p.size(); ++i) {
    if (p[i] == 0 || q[i] == 0) {
      return std::numeric_limits<double>::max();
    }
    v += p[i] * std::log(static_cast<double>(p[i]) / static_cast<double>(q[i]));
  }
  return v;
}

//! Expand the quantization distribution to origin distribution in
//! [-threshold, threshold]
static inline void ExpandCandidateDistribution(
    const std::vector<uint32_t> &distribution,
    const std::vector<float> &quantized_distribution, size_t threshold,
    std::vector<float> *expand_distribution) {
  expand_distribution->resize(threshold * 2, 0);
  float merged_cnt = static_cast<float>(expand_distribution->size()) /
                     quantized_distribution.size();
  size_t left_boundary = distribution.size() / 2 - threshold;

  for (size_t i = 0; i < quantized_distribution.size(); ++i) {
    float start = i * merged_cnt;
    float end = start + merged_cnt;
    const size_t start_ceil = static_cast<size_t>(std::ceil(start));
    const size_t end_floor = static_cast<size_t>(std::floor(end));
    float left_ratio = static_cast<float>(start_ceil) - start;
    float right_ratio = end - static_cast<float>(end_floor);
    float nonzero_count = 0;

    //! Count the non-zeros bins, if the histogram bin is partially included,
    //! non-zero bins is also partially counted
    if (left_ratio > 0 && left_boundary + start_ceil > 0) {
      if (distribution[left_boundary + start_ceil - 1] != 0) {
        nonzero_count += left_ratio;
      }
    }
    if (right_ratio > 0 && left_boundary + end_floor < distribution.size()) {
      if (distribution[left_boundary + end_floor] != 0) {
        nonzero_count += right_ratio;
      }
    }
    for (size_t j = start_ceil; j < end_floor; j++) {
      nonzero_count += distribution[left_boundary + j] != 0;
    }
    if (nonzero_count == 0) {
      continue;
    }

    //! expand the quantized value
    float value = quantized_distribution[i] / nonzero_count;
    if (left_ratio > 0 && start_ceil > 0) {
      (*expand_distribution)[start_ceil - 1] += value * left_ratio;
    }
    if (right_ratio > 0 && end_floor < expand_distribution->size()) {
      (*expand_distribution)[end_floor] += value * right_ratio;
    }
    for (size_t j = start_ceil; j < end_floor; j++) {
      if (distribution[left_boundary + j] != 0) {
        (*expand_distribution)[j] = value;
      }
    }  // end of for
  }  // end of for
}

/*! Compute quantization threshold bins
 *  Implement Int8 Quantization Algorithm ref:
 *  http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
 */
static inline size_t ComputeThreshold(const std::vector<uint32_t> &hist,
                                      const size_t target_bins) {
  std::vector<float> P_distribution(hist.size());
  size_t zero_point_index = hist.size() / 2;

  size_t start_bin = target_bins / 2;
  size_t end_bin = hist.size() / 2;
  size_t negative_outliers_count = 0;
  size_t positive_outliers_count = 0;
  double min_divergence = std::numeric_limits<double>::max();
  size_t target_threshold = end_bin;

  for (size_t threshold = start_bin; threshold <= end_bin; ++threshold) {
    negative_outliers_count += hist[zero_point_index - threshold];
    positive_outliers_count += hist[zero_point_index + threshold - 1];
  }

  //! for each zero-axised quantization range: [-threshold, threshold], search
  //! the best solution
  for (size_t threshold = start_bin; threshold <= end_bin; ++threshold) {
    P_distribution.resize(threshold * 2);
    auto p_hist = &hist[zero_point_index - threshold];
    for (size_t i = 0; i != P_distribution.size(); ++i) {
      P_distribution[i] = static_cast<float>(p_hist[i]);
    }

    negative_outliers_count -= hist[zero_point_index - threshold];
    positive_outliers_count -= hist[zero_point_index + threshold - 1];
    P_distribution[0] += negative_outliers_count;
    P_distribution[P_distribution.size() - 1] += positive_outliers_count;

    //! Quantize the bins in range [-threshold, threshold] to target_bins
    std::vector<float> Q_distribution(target_bins, 0);
    float merged_cnt = static_cast<float>(threshold * 2) / target_bins;
    size_t left_boundary = zero_point_index - threshold;
    for (size_t i = 0; i < target_bins; ++i) {
      float start = i * merged_cnt;
      float end = start + merged_cnt;
      const size_t start_ceil = static_cast<size_t>(std::ceil(start));
      const size_t end_floor = static_cast<size_t>(std::floor(end));
      if (left_boundary + start_ceil > 0) {
        Q_distribution[i] +=
            ((float)start_ceil - start) * hist[left_boundary + start_ceil - 1];
      }
      if (left_boundary + end_floor < hist.size()) {
        Q_distribution[i] +=
            (end - (float)end_floor) * hist[left_boundary + end_floor];
      }

      for (size_t j = start_ceil; j < end_floor; j++) {
        Q_distribution[i] += hist[left_boundary + j];
      }
    }
    std::vector<float> Q_expand_distribution;
    ExpandCandidateDistribution(hist, Q_distribution, threshold,
                                &Q_expand_distribution);

    //! Compute Kullback-Leibler Divergence, normalize the smooth the data
    //! first. Ref: http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
    MakeSmooth(P_distribution);
    MakeSmooth(Q_expand_distribution);
    double divergence =
        ComputeKlDivergence(P_distribution, Q_expand_distribution);

    if (divergence < min_divergence) {
      min_divergence = divergence;
      target_threshold = threshold;
    }
  }
  return target_threshold;
}

// Quantize the value in range
template <int RANGE_MIN, int RANGE_MAX>
static inline float QuantizeValue(float val, float scale, float bias) {
  val = (val + bias) * scale;

  if (val > RANGE_MAX) {
    val = RANGE_MAX;
  } else if (val < RANGE_MIN) {
    val = RANGE_MIN;
  }
  return val;
}

// Init the historgram params
#define INIT_HISTOGRAM()                                                      \
  {                                                                           \
    if (histogram_bins_ == 0) {                                               \
      size_t range = non_bias_                                                \
                         ? std::max(std::abs(MIN_VALUE), std::abs(MAX_VALUE)) \
                         : (MAX_VALUE - MIN_VALUE);                           \
      histogram_bins_ = std::max<size_t>(4096u, range * 8);                   \
    }                                                                         \
    histogram_.resize((histogram_bins_ + 1) >> 1 << 1);                       \
    if (non_bias_) {                                                          \
      bias_ = 0.0f;                                                           \
      auto val = std::max(std::abs(max_), std::abs(min_));                    \
      left_boundary_ = -val;                                                  \
      hist_interval_ = (val * 2) / static_cast<float>(histogram_.size());     \
    } else {                                                                  \
      bias_ = -static_cast<float>(min_ + (max_ - min_) * 0.5);                \
      left_boundary_ = min_;                                                  \
      hist_interval_ = (max_ - min_) / static_cast<float>(histogram_.size()); \
    }                                                                         \
  }

// Feed vector and update the historgram
#define UPDATE_HISTOGRAM(vec, dim)                                            \
  {                                                                           \
    if (max_ < min_) {                                                        \
      return false;                                                           \
    }                                                                         \
    if (histogram_.size() == 0) {                                             \
      INIT_HISTOGRAM()                                                        \
    }                                                                         \
    for (size_t i = 0; i < dim; ++i) {                                        \
      ssize_t index = 0;                                                      \
      if (hist_interval_ > 0.0) {                                             \
        index =                                                               \
            static_cast<ssize_t>((vec[i] - left_boundary_) / hist_interval_); \
      }                                                                       \
      if (index < 0) {                                                        \
        index = 0;                                                            \
      } else if ((size_t)index >= histogram_.size()) {                        \
        index = histogram_.size() - 1;                                        \
      }                                                                       \
      ailego_assert_with((size_t)index < histogram_.size(), "Invalid index"); \
      histogram_[index] += 1;                                                 \
    }                                                                         \
    return true;                                                              \
  }

// Train the quantizer
#define TRAIN_QUANTIZER()                                                \
  {                                                                      \
    auto sum = std::accumulate(histogram_.begin(), histogram_.end(), 0); \
    if (sum == 0) {                                                      \
      return false;                                                      \
    }                                                                    \
    size_t target_bins =                                                 \
        ailego_align(static_cast<size_t>(MAX_VALUE - MIN_VALUE), 2);     \
    auto threshold_bins = ComputeThreshold(histogram_, target_bins);     \
    auto threshold =                                                     \
        (static_cast<float>(threshold_bins) + 0.5f) * hist_interval_;    \
    scale_ = target_bins / 2 / threshold;                                \
    if (!non_bias_) {                                                    \
      bias_ += (MAX_VALUE + MIN_VALUE) * 0.5f / scale_;                  \
    }                                                                    \
    scale_reciprocal_ = 1 / scale_;                                      \
    return true;                                                         \
  }

// Feed the INT16 quantizer
bool EntropyInt16Quantizer::feed(const float *vec, size_t dim) {
  UPDATE_HISTOGRAM(vec, dim)
}

// Train the INT16 quantizer
bool EntropyInt16Quantizer::train(void) {
  TRAIN_QUANTIZER()
}

// Encode to INT16
void EntropyInt16Quantizer::encode(const float *in, size_t dim,
                                   int16_t *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = static_cast<int16_t>(
        std::round(QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i], scale_, bias_)));
  }
}

// Decode from INT16
void EntropyInt16Quantizer::decode(const int16_t *in, size_t dim,
                                   float *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = in[i] * this->scale_reciprocal() - this->bias();
  }
}

// Feed the UINT16 quantizer
bool EntropyUInt16Quantizer::feed(const float *vec, size_t dim) {
  UPDATE_HISTOGRAM(vec, dim)
}

// Train the UINT16 quantizer
bool EntropyUInt16Quantizer::train(void) {
  TRAIN_QUANTIZER()
}

// Encode to UINT16
void EntropyUInt16Quantizer::encode(const float *in, size_t dim,
                                    uint16_t *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = static_cast_from_float_to_uint16(
        std::round(QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i], scale_, bias_)));
  }
}

// Decode from INT16
void EntropyUInt16Quantizer::decode(const uint16_t *in, size_t dim,
                                    float *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = in[i] * this->scale_reciprocal() - this->bias();
  }
}

// Feed the INT8 quantizer
bool EntropyInt8Quantizer::feed(const float *vec, size_t dim) {
  UPDATE_HISTOGRAM(vec, dim)
}

// Train the INT8 quantizer
bool EntropyInt8Quantizer::train(void) {
  TRAIN_QUANTIZER()
}

// Encode to INT8
void EntropyInt8Quantizer::encode(const float *in, size_t dim,
                                  int8_t *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = static_cast<int8_t>(
        std::round(QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i], scale_, bias_)));
  }
}

// Decode from INT8
void EntropyInt8Quantizer::decode(const int8_t *in, size_t dim,
                                  float *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = in[i] * this->scale_reciprocal() - this->bias();
  }
}

// Feed the UINT8 quantizer
bool EntropyUInt8Quantizer::feed(const float *vec, size_t dim) {
  UPDATE_HISTOGRAM(vec, dim)
}

// Train the UINT8 quantizer
bool EntropyUInt8Quantizer::train(void) {
  TRAIN_QUANTIZER()
}

// Encode to INT8
void EntropyUInt8Quantizer::encode(const float *in, size_t dim,
                                   uint8_t *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = static_cast_from_float_to_uint8(
        std::round(QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i], scale_, bias_)));
  }
}

// Decode from UINT8
void EntropyUInt8Quantizer::decode(const uint8_t *in, size_t dim,
                                   float *out) const {
  for (size_t i = 0; i < dim; ++i) {
    out[i] = in[i] * this->scale_reciprocal() - this->bias();
  }
}

// Feed the INT4 quantizer
bool EntropyInt4Quantizer::feed(const float *vec, size_t dim) {
  UPDATE_HISTOGRAM(vec, dim)
}

// Train the INT4 quantizer
bool EntropyInt4Quantizer::train(void) {
  TRAIN_QUANTIZER()
}

// Encode to INT4
void EntropyInt4Quantizer::encode(const float *in, size_t dim,
                                  uint8_t *out) const {
  ailego_assert_with(dim % 2 == 0, "Dimension must be aligned with 2");

  for (size_t i = 0; i < dim; i += 2) {
    float lo = QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i], scale_, bias_);
    float hi = QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i + 1], scale_, bias_);
    out[i / 2] = (static_cast_from_float_to_uint8(std::round(hi)) << 4) |
                 (static_cast_from_float_to_uint8(std::round(lo)) & 0xF);
  }
}

// Decode from INT4
void EntropyInt4Quantizer::decode(const uint8_t *in, size_t dim,
                                  float *out) const {
  ailego_assert_with(dim % 2 == 0, "Dimension must be aligned with 2");

  size_t size = dim / 2;
  for (size_t i = 0; i < size; i += 1) {
    uint8_t v = in[i];
    int8_t lo = (static_cast<int8_t>(v << 4) >> 4);
    int8_t hi = (static_cast<int8_t>(v & 0xf0) >> 4);
    out[2 * i] = lo * this->scale_reciprocal() - this->bias();
    out[2 * i + 1] = hi * this->scale_reciprocal() - this->bias();
  }
}

// Feed the UINT4 quantizer
bool EntropyUInt4Quantizer::feed(const float *vec, size_t dim) {
  UPDATE_HISTOGRAM(vec, dim)
}

// Train the UINT4 quantizer
bool EntropyUInt4Quantizer::train(void) {
  TRAIN_QUANTIZER()
}

// Encode to INT4
void EntropyUInt4Quantizer::encode(const float *in, size_t dim,
                                   uint8_t *out) const {
  ailego_assert_with(dim % 2 == 0, "Dimension must be aligned with 2");

  for (size_t i = 0; i < dim; i += 2) {
    float lo = QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i], scale_, bias_);
    float hi = QuantizeValue<MIN_VALUE, MAX_VALUE>(in[i + 1], scale_, bias_);
    out[i / 2] = (static_cast_from_float_to_uint8(std::round(hi)) << 4) |
                 (static_cast_from_float_to_uint8(std::round(lo)) & 0xF);
  }
}

// Decode from INT4
void EntropyUInt4Quantizer::decode(const uint8_t *in, size_t dim,
                                   float *out) const {
  ailego_assert_with(dim % 2 == 0, "Dimension must be aligned with 2");

  size_t size = dim / 2;
  for (size_t i = 0; i < size; i += 1) {
    uint8_t v = in[i];
    out[2 * i] = (v & 0xf) * this->scale_reciprocal() - this->bias();
    out[2 * i + 1] = (v >> 4) * this->scale_reciprocal() - this->bias();
  }
}

}  // namespace ailego
}  // namespace zvec