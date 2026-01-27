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
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/inner_product_matrix.h>
#include <ailego/math/mips_euclidean_distance_matrix.h>
#include <ailego/math/norm2_matrix.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "metric_params.h"

namespace zvec {
namespace core {

/*! Mips Squared Euclidean Metric
 */
template <bool is_spares = false>
class MipsSquaredEuclideanMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
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
      eta_ = 0.0f;
    } else if (max_squared_l2_norm_ < std::numeric_limits<float>::epsilon()) {
      eta_ = kDefaultEta;
    } else {
      eta_ = squared_u_value_ / max_squared_l2_norm_;
    }

    switch (data_type_) {
      case IndexMeta::DataType::DT_FP32:
        squared_norm2_handle_ = reinterpret_cast<SquaredNorm2Handle>(
            ailego::SquaredNorm2Matrix<float, 1>::Compute);
        break;

      case IndexMeta::DataType::DT_FP16:
        squared_norm2_handle_ = reinterpret_cast<SquaredNorm2Handle>(
            ailego::SquaredNorm2Matrix<ailego::Float16, 1>::Compute);
        break;

      case IndexMeta::DataType::DT_INT8:
        squared_norm2_handle_ = reinterpret_cast<SquaredNorm2Handle>(
            ailego::SquaredNorm2Matrix<int8_t, 1>::Compute);
        break;

      case IndexMeta::DataType::DT_INT4:
        squared_norm2_handle_ = reinterpret_cast<SquaredNorm2Handle>(
            ailego::SquaredNorm2Matrix<uint8_t, 1>::Compute);
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
  int cleanup(void) override {
    eta_ = 0.0f;
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

    return
        [=](const void **m, const void *q, size_t num, size_t dim, float *out) {
          for (size_t i = 0; i < num; ++i) {
            dist_func(m[i], q, dim, out + i);
          }
        };
  }


  //! Retrieve distance function for query
  MatrixDistance distance(void) const override {
    if (injection_ == Injection::kLocalizedSpherical) {
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
                reinterpret_cast<const float *>(m),
                reinterpret_cast<const float *>(q), dim, 0.0f, out);
          };

        case IndexMeta::DataType::DT_FP16:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<ailego::Float16, 1, 1>::
                Compute(reinterpret_cast<const ailego::Float16 *>(m),
                        reinterpret_cast<const ailego::Float16 *>(q), dim, 0.0f,
                        out);
          };

        case IndexMeta::DataType::DT_INT8:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
                reinterpret_cast<const int8_t *>(m),
                reinterpret_cast<const int8_t *>(q), dim, 0.0f, out);
          };

        case IndexMeta::DataType::DT_INT4:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
                reinterpret_cast<const uint8_t *>(m),
                reinterpret_cast<const uint8_t *>(q), dim, 0.0f, out);
          };

        default:
          return nullptr;
      }
    }

    if (injection_ == Injection::kRepeatedQuadratic) {
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
                reinterpret_cast<const float *>(m),
                reinterpret_cast<const float *>(q), dim, m_value_, eta_, out);
          };

        case IndexMeta::DataType::DT_FP16:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<ailego::Float16, 1, 1>::
                Compute(reinterpret_cast<const ailego::Float16 *>(m),
                        reinterpret_cast<const ailego::Float16 *>(q), dim,
                        m_value_, eta_, out);
          };

        case IndexMeta::DataType::DT_INT8:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
                reinterpret_cast<const int8_t *>(m),
                reinterpret_cast<const int8_t *>(q), dim, m_value_, eta_, out);
          };

        case IndexMeta::DataType::DT_INT4:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
                reinterpret_cast<const uint8_t *>(m),
                reinterpret_cast<const uint8_t *>(q), dim, m_value_, eta_, out);
          };

        default:
          return nullptr;
      }
    }

    if (injection_ == Injection::kSpherical) {
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
                reinterpret_cast<const float *>(m),
                reinterpret_cast<const float *>(q), dim, eta_, out);
          };

        case IndexMeta::DataType::DT_FP16:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<ailego::Float16, 1, 1>::
                Compute(reinterpret_cast<const ailego::Float16 *>(m),
                        reinterpret_cast<const ailego::Float16 *>(q), dim, eta_,
                        out);
          };

        case IndexMeta::DataType::DT_INT8:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
                reinterpret_cast<const int8_t *>(m),
                reinterpret_cast<const int8_t *>(q), dim, eta_, out);
          };

        case IndexMeta::DataType::DT_INT4:
          return [&](const void *m, const void *q, size_t dim, float *out) {
            ailego::MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
                reinterpret_cast<const uint8_t *>(m),
                reinterpret_cast<const uint8_t *>(q), dim, eta_, out);
          };

        default:
          return nullptr;
      }
    }

    if (injection_ == Injection::kIdentity) {
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          return reinterpret_cast<MatrixDistanceHandle>(
              ailego::SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute);

        case IndexMeta::DataType::DT_FP16:
          return reinterpret_cast<MatrixDistanceHandle>(
              ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 1,
                                                     1>::Compute);

        case IndexMeta::DataType::DT_INT8:
          return reinterpret_cast<MatrixDistanceHandle>(
              ailego::SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute);

        case IndexMeta::DataType::DT_INT4:
          return reinterpret_cast<MatrixDistanceHandle>(
              ailego::SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute);

        default:
          return nullptr;
      }
    }
    return nullptr;
  }

  //! Retrieve distance function for query
  MatrixSparseDistance sparse_distance(void) const override {
    if (injection_ == Injection::kLocalizedSpherical) {
      return [&](const void *m_sparse, const void *q_sparse, float *out) {
        ailego::MipsSquaredEuclideanSparseDistanceMatrix<float>::Compute(
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
    if (injection_ == Injection::kLocalizedSpherical) {
      SphericalHandle<void> compute;
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    float, SphericalHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_FP16:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    ailego::Float16, SphericalHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_INT8:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    int8_t, SphericalHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_INT4:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    uint8_t, SphericalHandle>(m, n);
          break;
        default:
          return nullptr;
      }
      return [=](const void *d, const void *q, size_t dim, float *out) {
        compute(d, q, dim, 0.0f, out);
      };
    }

    if (injection_ == Injection::kRepeatedQuadratic) {
      RepeatedQuadraticHandle<void> compute;
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    float, RepeatedQuadraticHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_FP16:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    ailego::Float16, RepeatedQuadraticHandle>(
                  m, n);
          break;
        case IndexMeta::DataType::DT_INT8:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    int8_t, RepeatedQuadraticHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_INT4:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    uint8_t, RepeatedQuadraticHandle>(m, n);
          break;
        default:
          return nullptr;
      }
      return [=](const void *d, const void *q, size_t dim, float *out) {
        compute(d, q, dim, m_value_, eta_, out);
      };
    }

    if (injection_ == Injection::kSpherical) {
      SphericalHandle<void> compute;
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    float, SphericalHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_FP16:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    ailego::Float16, SphericalHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_INT8:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    int8_t, SphericalHandle>(m, n);
          break;
        case IndexMeta::DataType::DT_INT4:
          compute =
              DistanceMatrixCompute<ailego::MipsSquaredEuclideanDistanceMatrix,
                                    uint8_t, SphericalHandle>(m, n);
          break;
        default:
          return nullptr;
      }
      return [=](const void *d, const void *q, size_t dim, float *out) {
        compute(d, q, dim, eta_, out);
      };
    }

    if (injection_ == Injection::kIdentity) {
      switch (data_type_) {
        case IndexMeta::DataType::DT_FP32:
          return DistanceMatrixCompute<ailego::SquaredEuclideanDistanceMatrix,
                                       float, TypedDistanceHandle>(m, n);
        case IndexMeta::DataType::DT_FP16:
          return DistanceMatrixCompute<ailego::SquaredEuclideanDistanceMatrix,
                                       ailego::Float16, TypedDistanceHandle>(m,
                                                                             n);
        case IndexMeta::DataType::DT_INT8:
          return DistanceMatrixCompute<ailego::SquaredEuclideanDistanceMatrix,
                                       int8_t, TypedDistanceHandle>(m, n);
        case IndexMeta::DataType::DT_INT4:
          return DistanceMatrixCompute<ailego::SquaredEuclideanDistanceMatrix,
                                       uint8_t, TypedDistanceHandle>(m, n);
        default:
          return nullptr;
      }
    }
    return nullptr;
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
  bool support_normalize(void) const override {
    return query_metric_->support_normalize();
  }

  //! Retrieve params of Metric
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Train the metric
  int train(const void *vec, size_t dim) override {
    if (eta_ == 0.0f) {  // No global norm scaling => no training.
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
      eta_ = squared_u_value_ / max_squared_l2_norm_;
    }
    return 0;
  }

  //! Retrieve if it supports training
  bool support_train(void) const override {
    // No global norm scaling => eta_ == 0 => no training.
    return eta_ != 0.0f;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric(void) const override {
    return query_metric_;
  }

 private:
  //! Type of MipsSquaredEuclideanDistanceMatrix::Compute overloaded for
  //  Spherical injection and LocalizedSpherical nonmetric.
  template <typename T>
  using SphericalHandle = void (*)(const T *m, const T *q, size_t dim,
                                   float eta, float *out);

  //! Type of MipsSquaredEuclideanDistanceMatrix::Compute overloaded for
  //  RepeatedQuadratic injection.
  template <typename T>
  using RepeatedQuadraticHandle = void (*)(const T *m, const T *q, size_t dim,
                                           size_t m_value, float eta,
                                           float *out);

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
  //! Type of basic DistanceMatrix::Compute function with typed parameter.
  template <typename T>
  using TypedDistanceHandle = void (*)(const T *m, const T *q, size_t dim,
                                       float *out);

  //! Returns m x n distance matrix compute function.
  //  Handle is used to resolve potential DistanceMatrix<T>::Compute overload.
  template <template <typename, size_t, size_t, typename = void>
            class DistanceMatrix,
            typename T, template <typename> class Handle = TypedDistanceHandle>
  static Handle<void> DistanceMatrixCompute(size_t m, size_t n) {
    static Handle<T> distance_table[6][6] = {
        {DistanceMatrix<T, 1, 1, void>::Compute, nullptr, nullptr, nullptr,
         nullptr, nullptr},
        {DistanceMatrix<T, 2, 1, void>::Compute,
         DistanceMatrix<T, 2, 2, void>::Compute, nullptr, nullptr, nullptr,
         nullptr},
        {DistanceMatrix<T, 4, 1, void>::Compute,
         DistanceMatrix<T, 4, 2, void>::Compute,
         DistanceMatrix<T, 4, 4, void>::Compute, nullptr, nullptr, nullptr},
        {DistanceMatrix<T, 8, 1, void>::Compute,
         DistanceMatrix<T, 8, 2, void>::Compute,
         DistanceMatrix<T, 8, 4, void>::Compute,
         DistanceMatrix<T, 8, 8, void>::Compute, nullptr, nullptr},
        {DistanceMatrix<T, 16, 1, void>::Compute,
         DistanceMatrix<T, 16, 2, void>::Compute,
         DistanceMatrix<T, 16, 4, void>::Compute,
         DistanceMatrix<T, 16, 8, void>::Compute,
         DistanceMatrix<T, 16, 16, void>::Compute, nullptr},
        {DistanceMatrix<T, 32, 1, void>::Compute,
         DistanceMatrix<T, 32, 2, void>::Compute,
         DistanceMatrix<T, 32, 4, void>::Compute,
         DistanceMatrix<T, 32, 8, void>::Compute,
         DistanceMatrix<T, 32, 16, void>::Compute,
         DistanceMatrix<T, 32, 32, void>::Compute}};
    if (m > 32 || n > 32 || ailego_popcount(m) != 1 ||
        ailego_popcount(n) != 1) {
      return nullptr;
    }
    return reinterpret_cast<Handle<void> >(
        distance_table[ailego_ctz(m)][ailego_ctz(n)]);
  }

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
  float eta_{0.0f};
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
