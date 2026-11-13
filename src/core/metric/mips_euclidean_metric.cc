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
#include <atomic>
#include <memory>
#include <turbo/distance/sparse/sparse_distance.h>
#include <turbo/norm/norm2_matrix.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "metric_params.h"
#include "turbo_metric.h"

namespace zvec {
namespace core {
namespace {
using MatrixNorm = void (*)(const void *, size_t, float *);
using NormAndDot = void (*)(const void *, const void *, size_t, float *,
                            float *, float *);

template <typename T, size_t Rows>
void ComputeNormAndDot(const void *m, const void *q, size_t dim, float *out,
                       float *row_norms, float *query_norm) {
  if constexpr (std::is_same_v<T, float>) {
    const auto *matrix = static_cast<const float *>(m);
    const auto *query = static_cast<const float *>(q);
    float dots[Rows]{}, norms[Rows]{}, qnorm = 0;
    for (size_t d = 0; d < dim; ++d) {
      const float value = query[d];
      qnorm += value * value;
      for (size_t i = 0; i < Rows; ++i) {
        const float element = matrix[d * Rows + i];
        dots[i] += element * value;
        norms[i] += element * element;
      }
    }
    for (size_t i = 0; i < Rows; ++i) {
      out[i] = -dots[i];
      row_norms[i] = norms[i];
    }
    *query_norm = qnorm;
  } else {
    turbo::InterleavedDistanceMatrix<int8_t, Rows,
                                     turbo::MetricType::kInnerProduct>::
        template ComputeSingleColumn<true>(static_cast<const int8_t *>(m),
                                           static_cast<const int8_t *>(q), dim,
                                           out, 1, row_norms, query_norm);
  }
}

template <typename T>
NormAndDot SelectNormAndDot(size_t rows) {
  switch (rows) {
    case 2:
      return ComputeNormAndDot<T, 2>;
    case 4:
      return ComputeNormAndDot<T, 4>;
    case 8:
      return ComputeNormAndDot<T, 8>;
    case 16:
      return ComputeNormAndDot<T, 16>;
    case 32:
      return ComputeNormAndDot<T, 32>;
    default:
      return nullptr;
  }
}

template <typename T, size_t Rows>
void InterleavedNorm(const void *data, size_t dim, float *out) {
  turbo::InterleavedDistanceMatrix<T, Rows, turbo::MetricType::kInnerProduct>::
      SquaredNorms(static_cast<const T *>(data), dim, out);
}

template <typename T>
MatrixNorm SelectMatrixNorm(size_t rows) {
  switch (rows) {
    case 1:
      return InterleavedNorm<T, 1>;
    case 2:
      return InterleavedNorm<T, 2>;
    case 4:
      return InterleavedNorm<T, 4>;
    case 8:
      return InterleavedNorm<T, 8>;
    case 16:
      return InterleavedNorm<T, 16>;
    case 32:
      return InterleavedNorm<T, 32>;
    default:
      return nullptr;
  }
}

MatrixNorm SelectMatrixNorm(IndexMeta::DataType type, size_t rows) {
  switch (type) {
    case IndexMeta::DT_FP32:
      return SelectMatrixNorm<float>(rows);
    case IndexMeta::DT_FP16:
      return SelectMatrixNorm<ailego::Float16>(rows);
    case IndexMeta::DT_INT8:
      return SelectMatrixNorm<int8_t>(rows);
    case IndexMeta::DT_INT4:
      return SelectMatrixNorm<uint8_t>(rows);
    default:
      return nullptr;
  }
}
}  // namespace

/*! Mips Squared Euclidean Metric
 */
template <bool is_spares = false>
class MipsSquaredEuclideanMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    // Each initialization owns a new training state; existing callbacks keep
    // the state from their original initialization.
    training_state_ = std::make_shared<TrainingState>();
    data_type_ = meta.data_type();
    dimension_ = meta.dimension();

    int injection_type = static_cast<int>(kDefaultInjectionType);
    index_params.get(MIPS_EUCLIDEAN_METRIC_INJECTION_TYPE, &injection_type);
    if (injection_type >= static_cast<int>(Injection::kNumInjections)) {
      LOG_WARN("Unsupported injection_type %u, using '%s' instead",
               injection_type, InjectionName(0));
      injection_type = static_cast<int>(Injection::kLocalizedSpherical);
    }
    injection_ = static_cast<Injection>(injection_type);
    LOG_DEBUG(
        "Initializing MipsSquaredEuclideanMetric with injection %s"
        " type %d dimension %d",
        InjectionName(injection_), data_type_, dimension_);

    float max_l2_norm = 0.0f;
    float u_value = 0.0f;
    index_params.get(MIPS_EUCLIDEAN_METRIC_M_VALUE, &m_value_);
    index_params.get(MIPS_EUCLIDEAN_METRIC_U_VALUE, &u_value);
    index_params.get(MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM, &max_l2_norm);
    CheckAndFixM(injection_, &m_value_);
    CheckAndFixU(injection_, m_value_, &u_value);

    squared_u_value_ = u_value * u_value;
    max_squared_l2_norm_ = max_l2_norm * max_l2_norm;
    if (injection_ == Injection::kIdentity ||
        injection_ == Injection::kLocalizedSpherical) {
      training_state_->eta.store(0.0f, std::memory_order_relaxed);
    } else if (max_squared_l2_norm_ < std::numeric_limits<float>::epsilon()) {
      training_state_->eta.store(kDefaultEta, std::memory_order_relaxed);
    } else {
      training_state_->eta.store(squared_u_value_ / max_squared_l2_norm_,
                                 std::memory_order_relaxed);
    }

    switch (data_type_) {
      case IndexMeta::DataType::DT_FP32:
        squared_norm2_handle_ = reinterpret_cast<SquaredNorm2Handle>(
            turbo::SquaredNorm2Matrix<float, 1>::Compute);
        break;

      case IndexMeta::DataType::DT_FP16:
        // Raw FP16 inner products widen to FP32. Use the same kernel for the
        // norm (including training) so native FP16 norm accumulation cannot
        // saturate or round differently and produce negative self-distances.
        squared_norm2_handle_ = [](const void *m, size_t dim, float *out) {
          static const auto ip =
              RawKernels(turbo::MetricType::kInnerProduct, IndexMeta::DT_FP16)
                  .dist;
          ip(m, m, dim, out);
          *out = -*out;
        };
        break;

      case IndexMeta::DataType::DT_INT8:
        squared_norm2_handle_ = reinterpret_cast<SquaredNorm2Handle>(
            turbo::SquaredNorm2Matrix<int8_t, 1>::Compute);
        break;

      case IndexMeta::DataType::DT_INT4:
        squared_norm2_handle_ = reinterpret_cast<SquaredNorm2Handle>(
            turbo::SquaredNorm2Matrix<uint8_t, 1>::Compute);
        break;

      default:
        return IndexError_Unsupported;
    }

    query_metric_ = IndexFactory::CreateMetric(kQueryMetric);
    if (!query_metric_) {
      LOG_ERROR("Failed to create metric %s", kQueryMetric);
      return IndexError_NoExist;
    }
    int ret = query_metric_->init(meta, ailego::Params());
    if (ret != 0) {
      LOG_ERROR("Failed to initialize metric %s", kQueryMetric);
      return ret;
    }
    params_ = index_params;
    return 0;
  }

  //! Cleanup Metric
  int cleanup() override {
    // Detach rather than reset the state retained by cached callbacks.
    training_state_ = std::make_shared<TrainingState>();
    m_value_ = 0;
    squared_u_value_ = 0.0f;
    max_squared_l2_norm_ = 0.0f;
    query_metric_.reset();
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return (meta.data_type() == data_type_ &&
            meta.unit_size() == IndexMeta::UnitSizeof(data_type_));
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return (qmeta.data_type() == data_type_ &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(data_type_) &&
            qmeta.dimension() == meta.dimension());
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance() const override {
    MatrixDistance dist_func = distance();

    return [=](const void **m, const void *q, size_t num, size_t dim,
               float *out, const void ** /*extra_values*/) {
      for (size_t i = 0; i < num; ++i) {
        dist_func(m[i], q, dim, out + i);
      }
    };
  }


  //! Retrieve distance function for query
  MatrixDistance distance() const override {
    const auto ip =
        RawKernels(turbo::MetricType::kInnerProduct, data_type_).dist;
    const auto l2 =
        RawKernels(turbo::MetricType::kSquaredEuclidean, data_type_).dist;
    if (injection_ == Injection::kIdentity) return l2;
    if (!ip || !l2) return nullptr;
    // Callbacks share training updates without retaining the metric itself.
    // Capture the immutable configuration by value and load eta on each call.
    const auto norm = squared_norm2_handle_;
    const auto injection = injection_;
    const auto state = training_state_;
    const auto repeats = m_value_;
    return [norm, injection, state, repeats, ip, l2](
               const void *m, const void *q, size_t dim, float *out) {
      const float eta = state->eta.load(std::memory_order_relaxed);
      float u2, v2;
      norm(m, dim, &u2);
      norm(q, dim, &v2);
      if (injection == Injection::kRepeatedQuadratic) {
        l2(m, q, dim, out);
        *out *= eta;
        u2 *= eta;
        v2 *= eta;
        for (size_t i = 0; i < repeats; ++i) {
          *out += (u2 - v2) * (u2 - v2);
          u2 *= u2;
          v2 *= v2;
        }
      } else {
        float negative_ip;
        ip(m, q, dim, &negative_ip);
        if (injection == Injection::kLocalizedSpherical || eta == 0.0f) {
          *out = static_cast<float>(2.0 + 2.0 * negative_ip / std::max(u2, v2));
        } else {
          const double product =
              (1.0 - double(eta) * u2) * (1.0 - double(eta) * v2);
          *out = static_cast<float>(
              2.0 * (1.0 + double(eta) * negative_ip -
                     (product > 0.0 ? std::sqrt(product) : 0.0)));
        }
      }
    };
  }

  //! Retrieve distance function for query
  MatrixSparseDistance sparse_distance() const override {
    if (injection_ == Injection::kLocalizedSpherical) {
      return [&](const void *m_sparse, const void *q_sparse, float *out) {
        turbo::MipsSquaredEuclideanSparseDistanceMatrix<float>::Compute(
            m_sparse, q_sparse, out);
      };
    }

    if (injection_ == Injection::kRepeatedQuadratic) {
      LOG_ERROR(
          "Repeated Quadratic is not supported in MipsEuclideanMetric for "
          "Hybrid Vector!");

      return nullptr;
    }

    if (injection_ == Injection::kSpherical) {
      LOG_ERROR(
          "Spherical is not supported in MipsEuclideanMetric for Hybrid "
          "Vector!");

      return nullptr;
    }

    if (injection_ == Injection::kIdentity) {
      LOG_ERROR(
          "Identity is not supported in MipsEuclideanMetric for Hybrid "
          "Vector!");

      return nullptr;
    }

    return nullptr;
  }

  //! Retrieve matrix distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m == 1 && n == 1) return distance();
    const auto type = TurboDataType(data_type_);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
    // The contiguous FP16 kernels avoid scalar software conversions for both
    // norms and dot products; their throughput repays the layout conversion.
    if (data_type_ == IndexMeta::DT_FP16)
      return turbo::MakeDistanceMatrix(distance(), m, n, type);
#endif
    auto l2 =
        turbo::MakeRawDistanceMatrix<turbo::MetricType::kSquaredEuclidean>(
            m, n, type);
    if (!l2 || injection_ == Injection::kIdentity) return l2;
    auto ip = turbo::MakeRawDistanceMatrix<turbo::MetricType::kInnerProduct>(
        m, n, type);
    const auto row_norm = SelectMatrixNorm(data_type_, m);
    const auto query_norm = SelectMatrixNorm(data_type_, n);
    const auto injection = injection_;
    const auto state = training_state_;
    const auto repeats = m_value_;
    auto fused = data_type_ == IndexMeta::DT_INT8 && n == 1 &&
                         injection != Injection::kRepeatedQuadratic
                     ? SelectNormAndDot<int8_t>(m)
                     : nullptr;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
    if (data_type_ == IndexMeta::DT_FP32 && n == 1 &&
        injection != Injection::kRepeatedQuadratic)
      fused = SelectNormAndDot<float>(m);
#endif
    return [=](const void *rows, const void *queries, size_t dim, float *out) {
      // Reuse each norm for all pairs, without transposing or allocating.
      float row_norms[32], query_norms[32];
      if (fused) {
        fused(rows, queries, dim, out, row_norms, query_norms);
      } else {
        row_norm(rows, dim, row_norms);
        query_norm(queries, dim, query_norms);
        if (injection == Injection::kRepeatedQuadratic)
          l2(rows, queries, dim, out);
        else
          ip(rows, queries, dim, out);
      }
      const float eta = state->eta.load(std::memory_order_relaxed);
      for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
          float u2 = row_norms[i], v2 = query_norms[j];
          float &score = out[j * m + i];
          if (injection == Injection::kRepeatedQuadratic) {
            score *= eta;
            u2 *= eta;
            v2 *= eta;
            for (size_t k = 0; k < repeats; ++k) {
              score += (u2 - v2) * (u2 - v2);
              u2 *= u2;
              v2 *= v2;
            }
          } else if (injection == Injection::kLocalizedSpherical ||
                     eta == 0.0f) {
            score = static_cast<float>(2.0 + 2.0 * score / std::max(u2, v2));
          } else {
            const double product =
                (1.0 - double(eta) * u2) * (1.0 - double(eta) * v2);
            score = static_cast<float>(
                2.0 * (1.0 + double(eta) * score -
                       (product > 0.0 ? std::sqrt(product) : 0.0)));
          }
        }
      }
    };
  }

  //! Normalize result
  void normalize(float *score) const override {
    query_metric_->normalize(score);
  }

  //! Denormalize threshold
  void denormalize(float *score) const override {
    query_metric_->denormalize(score);
  }

  //! Retrieve if it supports normalization
  bool support_normalize() const override {
    return query_metric_->support_normalize();
  }

  //! Retrieve params of Metric
  const ailego::Params &params() const override {
    return params_;
  }

  //! Train the metric
  int train(const void *vec, size_t dim) override {
    // No global norm scaling => no training.
    if (training_state_->eta.load(std::memory_order_relaxed) == 0.0f) {
      return 0;
    }
    if (!squared_norm2_handle_) {
      return IndexError_Unsupported;
    }

    float score;
    squared_norm2_handle_(vec, dim, &score);
    if (score > max_squared_l2_norm_) {
      max_squared_l2_norm_ = score;
      const float max_l2_norm = std::sqrt(score);
      params_.set(MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM, max_l2_norm);
      if (max_squared_l2_norm_ < 1.0 &&
          max_squared_l2_norm_ > squared_u_value_) {
        squared_u_value_ = max_squared_l2_norm_;
        params_.set(MIPS_EUCLIDEAN_METRIC_U_VALUE, max_l2_norm);
      }
      training_state_->eta.store(squared_u_value_ / max_squared_l2_norm_,
                                 std::memory_order_relaxed);
    }
    return 0;
  }

  //! Retrieve if it supports training
  bool support_train() const override {
    // No global norm scaling => eta == 0 => no training.
    return training_state_->eta.load(std::memory_order_relaxed) != 0.0f;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric() const override {
    return query_metric_;
  }

 private:
  //! Type of squared L2 norm function.
  using SquaredNorm2Handle = void (*)(const void *m, size_t dim, float *out);

  enum struct Injection {     // Type of injective mapping into Euclidean space.
    kLocalizedSpherical = 0,  // spherical with pair-only max-norm
    kSpherical = 1,           // require global scaling/training
    kRepeatedQuadratic = 2,   // require global scaling/training
    kIdentity = 3,            // plain Euclidean distance
    kNumInjections = 4
  };

  static const char *InjectionName(int injection) {
    static const char *injection_names[] = {"LocalizedSpherical", "Spherical",
                                            "RepeatedQuadratic", "Identity"};
    if (injection >= 0 &&
        injection < static_cast<int>(Injection::kNumInjections)) {
      return injection_names[injection];
    }
    return "Invalid";
  }

  static const char *InjectionName(Injection injection) {
    return InjectionName(static_cast<int>(injection));
  }

  // Checks (and fixes) `*m_value`, no. additional dimensions for injection.
  // `dim` is the original dimension, used ONLY by RepeatedQuadratic
  // injection, where dim = 1 induces the default *m_value = 3. It's
  // positioned last to allow other injections to skip it.
  // Returns true if `*m_value` is modified.
  static bool CheckAndFixM(Injection injection, uint32_t *m_value) {
    if (injection == Injection::kRepeatedQuadratic) {
      if (*m_value == 0) {
        *m_value = 3u;  // Recommend value in paper (3.5 Practical
                        // Recommendation of Parameters)
        return true;
      }
    } else if (injection == Injection::kSpherical) {
      if (*m_value != 1) {
        if (*m_value != 0) {
          LOG_WARN("M value (%u) set to 1 for Spherical injection", *m_value);
        }
        *m_value = 1;
        return true;
      }
    } else {  // kLocalizedSpherical, kIdentity, or kInvalid
      if (*m_value != 0) {
        LOG_WARN("M value (%u) set to 0 for %s injections", *m_value,
                 InjectionName(injection));
        *m_value = 0;
        return true;
      }
    }
    return false;
  }

  // Checks and fixes `*u_value`, global L2 norm scalar.
  // `m_value` is no. additional dimensions, used ONLY by RepeatedQuadratic
  // injection. It's positioned last to allow other injections to skip it.
  // Returns true if `*u_value` is set to a new value.
  static bool CheckAndFixU(Injection injection, uint32_t m_value,
                           float *u_value) {
    if (injection == Injection::kRepeatedQuadratic) {
      if (*u_value <= std::numeric_limits<float>::epsilon() ||
          *u_value >= 1.0) {
        // Try computing a default U value
        constexpr float kLogError = -5.0;  // log_10(distance_error)
        float new_u_value = std::pow(10, kLogError / (1 << (m_value + 1)));
        if (*u_value != 0) {
          LOG_WARN("U value (%f) set to %f for RepeatedQuadratic injection",
                   *u_value, new_u_value);
        }
        *u_value = new_u_value;
        return true;
      } else if (std::pow(*u_value, (1 << m_value)) <
                 std::numeric_limits<float>::epsilon()) {
        LOG_WARN(
            "U value %f is too small, may cause loss of distance precision",
            *u_value);
      }
    } else if (injection == Injection::kSpherical) {
      // Spherical injection requires ||x'|| <= 1.0 for computing
      // std::sqrt(1 - ||x'||^2), x' = u_value * x / max_norm.  Set u_value
      // to slightly < 1.0 in case of precision loss in float computation.
      if (*u_value <= std::numeric_limits<float>::epsilon() ||
          *u_value >= 1.0) {
        static constexpr float kSphericalUValue = 1.0f - 1e-3;
        if (*u_value != 0.0f) {
          LOG_WARN("U value (%f) set to %f for Spherical injection", *u_value,
                   kSphericalUValue);
        }
        *u_value = kSphericalUValue;
        return true;
      }
    } else {  // kLocalizedSpherical, kIdentity, or kInvalid
      if (*u_value != 1.0) {
        if (*u_value != 0) {
          LOG_WARN("U value (%f) set to 1.0 for %s injection", *u_value,
                   InjectionName(injection));
        }
        *u_value = 1.0;
        return true;
      }
    }
    return false;
  }

 private:
  //! Constants
  // If the training data is not provided, we use a max squared l2 norm which
  // is as big as possible but also keep the precision, so estimate eta =  U /
  // max(l2 squared norm) = float epsilon
  static constexpr float kDefaultEta = std::numeric_limits<float>::epsilon();
  static constexpr char const *kQueryMetric =
      is_spares ? "InnerProductSparse" : "InnerProduct";
  static constexpr Injection kDefaultInjectionType =
      Injection::kLocalizedSpherical;

  //! Members
  SquaredNorm2Handle squared_norm2_handle_{nullptr};
  struct TrainingState {
    // Training can run while another thread invokes a cached callback.
    std::atomic<float> eta{0.0f};
  };
  std::shared_ptr<TrainingState> training_state_{
      std::make_shared<TrainingState>()};
  uint32_t m_value_{0};
  float squared_u_value_{0.0f};
  float max_squared_l2_norm_{0.0f};
  uint32_t dimension_{0};
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};
  Injection injection_{kDefaultInjectionType};
  IndexMetric::Pointer query_metric_{};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(MipsSquaredEuclidean,
                                    MipsSquaredEuclideanMetric<false>);
INDEX_FACTORY_REGISTER_METRIC_ALIAS(MipsSquaredEuclideanSparse,
                                    MipsSquaredEuclideanMetric<true>);

}  // namespace core
}  // namespace zvec
