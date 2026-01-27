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
#include <ailego/math/normalizer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

/*! Convert the vector By Mips RepeatedQuadraticInjection
 */
template <typename T1, typename T2,
          typename =
              typename std::enable_if<ailego::IsFloatingPoint<T1>::value &&
                                      ailego::IsFloatingPoint<T2>::value>::type>
static inline void ConvertRepeatedQuadraticInjection(const T1 *src, size_t dim,
                                                     size_t m_value,
                                                     float u_value,
                                                     float l2_norm, T2 *dst) {
  float squared_norm = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float val = src[i] * u_value / l2_norm;
    dst[i] = val;
    squared_norm += val * val;
  }
  for (size_t i = dim; i < dim + m_value; ++i) {
    dst[i] = 0.5f - squared_norm;
    squared_norm *= squared_norm;
  }
}

/*! Convert the vector By Mips SphericalInjection
 */
template <typename T1, typename T2,
          typename =
              typename std::enable_if<ailego::IsFloatingPoint<T1>::value &&
                                      ailego::IsFloatingPoint<T2>::value>::type>
static inline void ConvertSphericalInjection(const T1 *src, size_t dim,
                                             float u_value, float l2_norm,
                                             T2 *dst) {
  float squared_norm = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float val = src[i] * u_value / l2_norm;
    dst[i] = val;
    squared_norm += val * val;
  }
  dst[dim] = squared_norm < 1.0
                 ? (1.0 - std::sqrt(1.0 - static_cast<double>(squared_norm)))
                 : 1.0f;
}

/*! MIPS Reformer
 */
class MipsReformer : public IndexReformer {
 public:
  //! Initialize Reformer
  int init(const ailego::Params &params) override {
    params.get(MIPS_REFORMER_M_VALUE, &m_value_);
    params.get(MIPS_REFORMER_U_VALUE, &u_value_);
    params.get(MIPS_REFORMER_L2_NORM, &l2_norm_);
    params.get(MIPS_REFORMER_NORMALIZE, &normalize_);
    params.get(MIPS_REFORMER_FORCED_HALF_FLOAT, &forced_half_float_);
    params.get(MIPS_REFORMER_SPHERICAL_INJECTION, &spherical_injection_);
    if (spherical_injection_) {
      if (m_value_ != 1u) {
        LOG_WARN("Invalid M value or U value if spherical_injection enabled");
      }
      m_value_ = 1;
    }
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  int unload(void) override {
    return 0;
  }

  //! Transform query
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType dt = qmeta.data_type();

    if (dt == IndexMeta::DataType::DT_FP32) {
      if (qmeta.unit_size() != sizeof(float)) {
        return IndexError_Unsupported;
      }

      if (forced_half_float_) {
        out->clear();
        out->resize((qmeta.dimension() + m_value_) * sizeof(ailego::Float16));

        if (normalize_) {
          float norm;
          ailego::Norm2Matrix<float, 1>::Compute(
              reinterpret_cast<const float *>(query), qmeta.dimension(), &norm);

          ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(query),
                                      qmeta.dimension(), norm,
                                      reinterpret_cast<uint16_t *>(&(*out)[0]));
        } else {
          ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(query),
                                      qmeta.dimension(),
                                      reinterpret_cast<uint16_t *>(&(*out)[0]));
        }
        if (spherical_injection_) {
          reinterpret_cast<ailego::Float16 *>(&(*out)[0])[qmeta.dimension()] =
              1.0f;
        }
        *ometa = qmeta;
        ometa->set_meta(IndexMeta::DataType::DT_FP16,
                        qmeta.dimension() + m_value_);

      } else {
        out->assign(reinterpret_cast<const char *>(query),
                    qmeta.element_size());
        out->resize((qmeta.dimension() + m_value_) * sizeof(float));

        if (normalize_) {
          float norm;
          ailego::Normalizer<float>::L2(reinterpret_cast<float *>(&(*out)[0]),
                                        qmeta.dimension(), &norm);
        }
        if (spherical_injection_) {
          reinterpret_cast<float *>(&(*out)[0])[qmeta.dimension()] = 1.0f;
        }
        *ometa = qmeta;
        ometa->set_dimension(qmeta.dimension() + m_value_);
      }
    } else if (dt == IndexMeta::DataType::DT_FP16) {
      if (qmeta.unit_size() != sizeof(ailego::Float16)) {
        return IndexError_Unsupported;
      }
      out->assign(reinterpret_cast<const char *>(query), qmeta.element_size());
      out->resize((qmeta.dimension() + m_value_) * sizeof(ailego::Float16));

      if (normalize_) {
        float norm;
        ailego::Normalizer<ailego::Float16>::L2(
            reinterpret_cast<ailego::Float16 *>(&(*out)[0]), qmeta.dimension(),
            &norm);
      }
      if (spherical_injection_) {
        reinterpret_cast<ailego::Float16 *>(&(*out)[0])[qmeta.dimension()] =
            1.0f;
      }
      *ometa = qmeta;
      ometa->set_dimension(qmeta.dimension() + m_value_);
    } else {
      return IndexError_Unsupported;
    }
    return 0;
  }

  //! Transform queries
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType dt = qmeta.data_type();

    if (dt == IndexMeta::DataType::DT_FP32) {
      if (qmeta.unit_size() != sizeof(float)) {
        return IndexError_Unsupported;
      }
      out->clear();

      if (forced_half_float_) {
        for (uint32_t i = 0; i < count; ++i) {
          size_t offset = out->size();
          out->resize(offset +
                      (qmeta.dimension() + m_value_) * sizeof(ailego::Float16));

          const float *sub_query =
              reinterpret_cast<const float *>(query) + i * qmeta.dimension();

          if (normalize_) {
            float norm;
            ailego::Norm2Matrix<float, 1>::Compute(sub_query, qmeta.dimension(),
                                                   &norm);
            ailego::FloatHelper::ToFP16(
                sub_query, qmeta.dimension(), norm,
                reinterpret_cast<uint16_t *>(&(*out)[offset]));
          } else {
            ailego::FloatHelper::ToFP16(
                sub_query, qmeta.dimension(),
                reinterpret_cast<uint16_t *>(&(*out)[offset]));
          }
          if (spherical_injection_) {
            reinterpret_cast<ailego::Float16 *>(
                &(*out)[offset])[qmeta.dimension()] = 1.0f;
          }
        }
        *ometa = qmeta;
        ometa->set_meta(IndexMeta::DataType::DT_FP16,
                        qmeta.dimension() + m_value_);

      } else {
        for (uint32_t i = 0; i < count; ++i) {
          size_t offset = out->size();
          out->append(
              reinterpret_cast<const char *>(query) + i * qmeta.element_size(),
              qmeta.element_size());
          out->resize(offset + (qmeta.dimension() + m_value_) * sizeof(float));

          if (normalize_) {
            float norm;
            ailego::Normalizer<float>::L2(
                reinterpret_cast<float *>(&(*out)[offset]), qmeta.dimension(),
                &norm);
          }
          if (spherical_injection_) {
            reinterpret_cast<float *>(&(*out)[offset])[qmeta.dimension()] =
                1.0f;
          }
        }
        *ometa = qmeta;
        ometa->set_dimension(qmeta.dimension() + m_value_);
      }
    } else if (dt == IndexMeta::DataType::DT_FP16) {
      if (qmeta.unit_size() != sizeof(ailego::Float16)) {
        return IndexError_Unsupported;
      }
      out->clear();

      for (uint32_t i = 0; i < count; ++i) {
        size_t offset = out->size();
        out->append(
            reinterpret_cast<const char *>(query) + i * qmeta.element_size(),
            qmeta.element_size());
        out->resize(offset +
                    (qmeta.dimension() + m_value_) * sizeof(ailego::Float16));

        if (normalize_) {
          float norm;
          ailego::Normalizer<ailego::Float16>::L2(
              reinterpret_cast<ailego::Float16 *>(&(*out)[offset]),
              qmeta.dimension(), &norm);
        }
        if (spherical_injection_) {
          reinterpret_cast<ailego::Float16 *>(
              &(*out)[offset])[qmeta.dimension()] = 1.0f;
        }
      }
      *ometa = qmeta;
      ometa->set_dimension(qmeta.dimension() + m_value_);

    } else {
      return IndexError_Unsupported;
    }
    return 0;
  }

  //! Convert a record
  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    IndexMeta::DataType dt = rmeta.data_type();

    if (dt == IndexMeta::DataType::DT_FP32) {
      if (rmeta.unit_size() != sizeof(float)) {
        return IndexError_Unsupported;
      }

      const float *vec = reinterpret_cast<const float *>(record);
      if (forced_half_float_) {
        *ometa = rmeta;
        ometa->set_meta(IndexMeta::DataType::DT_FP16,
                        rmeta.dimension() + m_value_);
        out->resize(ometa->element_size());

        ailego::Float16 *dst = reinterpret_cast<ailego::Float16 *>(&(*out)[0]);
        if (!spherical_injection_) {
          ConvertRepeatedQuadraticInjection(vec, rmeta.dimension(), m_value_,
                                            u_value_, l2_norm_, dst);
        } else {
          ConvertSphericalInjection(vec, rmeta.dimension(), u_value_, l2_norm_,
                                    dst);
        }
      } else {
        *ometa = rmeta;
        ometa->set_dimension(rmeta.dimension() + m_value_);
        out->resize(ometa->element_size());

        float *dst = reinterpret_cast<float *>(&(*out)[0]);
        if (!spherical_injection_) {
          ConvertRepeatedQuadraticInjection(vec, rmeta.dimension(), m_value_,
                                            u_value_, l2_norm_, dst);
        } else {
          ConvertSphericalInjection(vec, rmeta.dimension(), u_value_, l2_norm_,
                                    dst);
        }
      }
    } else if (dt == IndexMeta::DataType::DT_FP16) {
      if (rmeta.unit_size() != sizeof(ailego::Float16)) {
        return IndexError_Unsupported;
      }
      *ometa = rmeta;
      ometa->set_dimension(rmeta.dimension() + m_value_);
      out->resize(ometa->element_size());

      const auto *vec = reinterpret_cast<const ailego::Float16 *>(record);
      ailego::Float16 *dst = reinterpret_cast<ailego::Float16 *>(&(*out)[0]);
      if (!spherical_injection_) {
        ConvertRepeatedQuadraticInjection(vec, rmeta.dimension(), m_value_,
                                          u_value_, l2_norm_, dst);
      } else {
        ConvertSphericalInjection(vec, rmeta.dimension(), u_value_, l2_norm_,
                                  dst);
      }
    } else {
      return IndexError_Unsupported;
    }
    return 0;
  }

  //! Convert records
  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType dt = rmeta.data_type();

    if (dt == IndexMeta::DataType::DT_FP32) {
      if (rmeta.unit_size() != sizeof(float)) {
        return IndexError_Unsupported;
      }
      *ometa = rmeta;

      if (forced_half_float_) {
        ometa->set_meta(IndexMeta::DataType::DT_FP16,
                        rmeta.dimension() + m_value_);
        out->resize(ometa->element_size() * count);
        for (uint32_t i = 0; i < count; ++i) {
          const float *sub_query =
              reinterpret_cast<const float *>(records) + i * rmeta.dimension();
          ailego::Float16 *dst = reinterpret_cast<ailego::Float16 *>(
              &(*out)[i * ometa->element_size()]);
          if (!spherical_injection_) {
            ConvertRepeatedQuadraticInjection(sub_query, rmeta.dimension(),
                                              m_value_, u_value_, l2_norm_,
                                              dst);
          } else {
            ConvertSphericalInjection(sub_query, rmeta.dimension(), u_value_,
                                      l2_norm_, dst);
          }
        }
      } else {
        ometa->set_dimension(rmeta.dimension() + m_value_);
        out->resize(ometa->element_size() * count);
        for (uint32_t i = 0; i < count; ++i) {
          const float *sub_query =
              reinterpret_cast<const float *>(records) + i * rmeta.dimension();
          float *dst =
              reinterpret_cast<float *>(&(*out)[i * ometa->element_size()]);
          if (!spherical_injection_) {
            ConvertRepeatedQuadraticInjection(sub_query, rmeta.dimension(),
                                              m_value_, u_value_, l2_norm_,
                                              dst);
          } else {
            ConvertSphericalInjection(sub_query, rmeta.dimension(), u_value_,
                                      l2_norm_, dst);
          }
        }
      }
    } else if (dt == IndexMeta::DataType::DT_FP16) {
      if (rmeta.unit_size() != sizeof(ailego::Float16)) {
        return IndexError_Unsupported;
      }
      *ometa = rmeta;
      ometa->set_dimension(rmeta.dimension() + m_value_);
      out->resize(ometa->element_size() * count);

      for (uint32_t i = 0; i < count; ++i) {
        const ailego::Float16 *sub_query =
            reinterpret_cast<const ailego::Float16 *>(records) +
            i * rmeta.dimension();
        ailego::Float16 *dst = reinterpret_cast<ailego::Float16 *>(
            &(*out)[i * ometa->element_size()]);
        if (!spherical_injection_) {
          ConvertRepeatedQuadraticInjection(sub_query, rmeta.dimension(),
                                            m_value_, u_value_, l2_norm_, dst);
        } else {
          ConvertSphericalInjection(sub_query, rmeta.dimension(), u_value_,
                                    l2_norm_, dst);
        }
      }
    } else {
      return IndexError_Unsupported;
    }
    return 0;
  }

  //! Normalize results
  int normalize(const void *query, const IndexQueryMeta &qmeta,
                IndexDocumentList &result) const override {
    IndexMeta::DataType dt = qmeta.data_type();
    float norm = 1.0f;

    if (dt == IndexMeta::DataType::DT_FP32) {
      if (qmeta.unit_size() != sizeof(float)) {
        return IndexError_Unsupported;
      }
      if (normalize_) {
        ailego::Norm2Matrix<float, 1>::Compute(
            reinterpret_cast<const float *>(query), qmeta.dimension(), &norm);
      }
    } else if (dt == IndexMeta::DataType::DT_FP16) {
      if (qmeta.unit_size() != sizeof(ailego::Float16)) {
        return IndexError_Unsupported;
      }
      if (normalize_) {
        ailego::Norm2Matrix<ailego::Float16, 1>::Compute(
            reinterpret_cast<const ailego::Float16 *>(query), qmeta.dimension(),
            &norm);
      }
    } else {
      return IndexError_Unsupported;
    }

    if (!spherical_injection_) {
      const float a = 1.0f + m_value_ * 0.25f;
      const float lamba = 0.5f * norm * l2_norm_ / u_value_;
      for (auto &it : result) {
        *it.mutable_score() = (a - it.score()) * lamba;
      }
    } else {
      const float lambda = norm * l2_norm_ / u_value_;
      for (auto &it : result) {
        *it.mutable_score() = (1.0f - 0.5f * it.score()) * lambda;
      }
    }
    return 0;
  }

 private:
  bool normalize_{false};
  bool forced_half_float_{false};
  bool spherical_injection_{false};
  uint32_t m_value_{0u};
  float u_value_{0.0f};
  float l2_norm_{0.0f};
};

INDEX_FACTORY_REGISTER_REFORMER(MipsReformer);

}  // namespace core
}  // namespace zvec
