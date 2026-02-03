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
#include <ailego/math/norm2_matrix.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/ailego/utility/type_helper.h>
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

/*! MIPS Holder (Float)
 */
class MipsConverterHolder : public IndexHolder {
 public:
  /*! MIPS Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(const MipsConverterHolder *owner,
             IndexHolder::Iterator::Pointer &&iter)
        : buffer_(owner->dimension()),
          m_value_(owner->m_value_),
          u_value_(owner->u_value_),
          l2_norm_(owner->l2_norm_),
          spherical_injection_(owner->spherical_injection_),
          front_iter_(std::move(iter)) {
      this->transform_data();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return buffer_.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return front_iter_->is_valid();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return front_iter_->key();
    }

    //! Next iterator
    void next(void) override {
      front_iter_->next();
      this->transform_data();
    }

   private:
    //! Transform the data
    void transform_data(void) {
      if (!front_iter_->is_valid()) {
        return;
      }

      const float *src = reinterpret_cast<const float *>(front_iter_->data());
      float *dst = buffer_.data();
      if (!spherical_injection_) {
        ConvertRepeatedQuadraticInjection(src, buffer_.size() - m_value_,
                                          m_value_, u_value_, l2_norm_, dst);
      } else {
        ConvertSphericalInjection(src, buffer_.size() - m_value_, u_value_,
                                  l2_norm_, dst);
      }
    }

    std::vector<float> buffer_{};
    uint32_t m_value_{0u};
    float u_value_{0.0f};
    float l2_norm_{0.0f};
    bool spherical_injection_{false};
    IndexHolder::Iterator::Pointer front_iter_{};
  };

  //! Constructor
  MipsConverterHolder(IndexHolder::Pointer front, uint32_t m_val, float u_val,
                      float l2_norm, bool spherical_injection)
      : m_value_(m_val),
        u_value_(u_val),
        l2_norm_(l2_norm),
        spherical_injection_(spherical_injection),
        front_(std::move(front)) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return front_->dimension() + m_value_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_FP32,
                                    front_->dimension() + m_value_);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    IndexHolder::Iterator::Pointer iter = front_->create_iterator();
    return iter ? IndexHolder::Iterator::Pointer(
                      new MipsConverterHolder::Iterator(this, std::move(iter)))
                : IndexHolder::Iterator::Pointer();
  }

 private:
  //! Disable them
  MipsConverterHolder(void) = delete;

  //! Members
  uint32_t m_value_{0u};
  float u_value_{0.0f};
  float l2_norm_{0.0f};
  bool spherical_injection_{false};
  IndexHolder::Pointer front_{};
};

/*! MIPS Holder (Forced Half Float)
 */
class MipsConverterForcedHalfHolder : public IndexHolder {
 public:
  /*! MIPS Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(const MipsConverterForcedHalfHolder *owner,
             IndexHolder::Iterator::Pointer &&iter)
        : buffer_(owner->dimension()),
          m_value_(owner->m_value_),
          u_value_(owner->u_value_),
          l2_norm_(owner->l2_norm_),
          spherical_injection_(owner->spherical_injection_),
          front_iter_(std::move(iter)) {
      this->transform_record();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return buffer_.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return front_iter_->is_valid();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return front_iter_->key();
    }

    //! Next iterator
    void next(void) override {
      front_iter_->next();
      this->transform_record();
    }

   private:
    void transform_record(void) {
      if (!front_iter_->is_valid()) {
        return;
      }

      const float *src = reinterpret_cast<const float *>(front_iter_->data());
      ailego::Float16 *dst = buffer_.data();
      if (!spherical_injection_) {
        ConvertRepeatedQuadraticInjection(src, buffer_.size() - m_value_,
                                          m_value_, u_value_, l2_norm_, dst);
      } else {
        ConvertSphericalInjection(src, buffer_.size() - m_value_, u_value_,
                                  l2_norm_, dst);
      }
    }

    std::vector<ailego::Float16> buffer_{};
    uint32_t m_value_{0u};
    float u_value_{0.0f};
    float l2_norm_{0.0f};
    bool spherical_injection_{false};
    IndexHolder::Iterator::Pointer front_iter_{};
  };

  //! Constructor
  MipsConverterForcedHalfHolder(IndexHolder::Pointer front, uint32_t m_val,
                                float u_val, float l2_norm,
                                bool spherical_injection)
      : m_value_(m_val),
        u_value_(u_val),
        l2_norm_(l2_norm),
        spherical_injection_(spherical_injection),
        front_(std::move(front)) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return front_->dimension() + m_value_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_FP16,
                                    front_->dimension() + m_value_);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    IndexHolder::Iterator::Pointer iter = front_->create_iterator();
    return iter ? IndexHolder::Iterator::Pointer(
                      new MipsConverterForcedHalfHolder::Iterator(
                          this, std::move(iter)))
                : IndexHolder::Iterator::Pointer();
  }

 private:
  //! Disable them
  MipsConverterForcedHalfHolder(void) = delete;

  //! Members
  uint32_t m_value_{0u};
  float u_value_{0.0f};
  float l2_norm_{0.0f};
  bool spherical_injection_{false};
  IndexHolder::Pointer front_{};
};

/*! MIPS Holder (Half Float)
 */
class MipsConverterHalfHolder : public IndexHolder {
 public:
  /*! MIPS Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(const MipsConverterHalfHolder *owner,
             IndexHolder::Iterator::Pointer &&iter)
        : buffer_(owner->dimension()),
          m_value_(owner->m_value_),
          u_value_(owner->u_value_),
          l2_norm_(owner->l2_norm_),
          spherical_injection_(owner->spherical_injection_),
          front_iter_(std::move(iter)) {
      this->transform_record();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return buffer_.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return front_iter_->is_valid();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return front_iter_->key();
    }

    //! Next iterator
    void next(void) override {
      front_iter_->next();
      this->transform_record();
    }

   private:
    void transform_record(void) {
      if (!front_iter_->is_valid()) {
        return;
      }

      const ailego::Float16 *src =
          reinterpret_cast<const ailego::Float16 *>(front_iter_->data());
      ailego::Float16 *dst = buffer_.data();
      if (!spherical_injection_) {
        ConvertRepeatedQuadraticInjection(src, buffer_.size() - m_value_,
                                          m_value_, u_value_, l2_norm_, dst);
      } else {
        ConvertSphericalInjection(src, buffer_.size() - m_value_, u_value_,
                                  l2_norm_, dst);
      }
    }

    std::vector<ailego::Float16> buffer_{};
    uint32_t m_value_{0u};
    float u_value_{0.0f};
    float l2_norm_{0.0f};
    bool spherical_injection_{false};
    IndexHolder::Iterator::Pointer front_iter_{};
  };

  //! Constructor
  MipsConverterHalfHolder(IndexHolder::Pointer front, uint32_t m_val,
                          float u_val, float l2_norm, bool spherical_injection)
      : m_value_(m_val),
        u_value_(u_val),
        l2_norm_(l2_norm),
        spherical_injection_(spherical_injection),
        front_(std::move(front)) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return front_->count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return front_->dimension() + m_value_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_FP16,
                                    front_->dimension() + m_value_);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return front_->multipass();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    IndexHolder::Iterator::Pointer iter = front_->create_iterator();
    return iter ? IndexHolder::Iterator::Pointer(
                      new MipsConverterHalfHolder::Iterator(this,
                                                            std::move(iter)))
                : IndexHolder::Iterator::Pointer();
  }

 private:
  //! Disable them
  MipsConverterHalfHolder(void) = delete;

  //! Members
  uint32_t m_value_{0u};
  float u_value_{0.0f};
  float l2_norm_{0.0f};
  bool spherical_injection_{false};
  IndexHolder::Pointer front_{};
};

/*! MIPS Converter
 */
class MipsConverter : public IndexConverter {
 public:
  //! Destructor
  virtual ~MipsConverter(void) {}

  //! Initialize Converter
  int init(const IndexMeta &mt, const ailego::Params &params) override {
    IndexMeta::DataType dt = mt.data_type();
    if (ailego_unlikely((dt != IndexMeta::DataType::DT_FP32 &&
                         dt != IndexMeta::DataType::DT_FP16) ||
                        mt.unit_size() != IndexMeta::UnitSizeof(dt))) {
      LOG_ERROR("Unsupported type %d with unit size %u.", dt, mt.unit_size());
      return IndexError_Unsupported;
    }

    params.get(MIPS_CONVERTER_FORCED_HALF_FLOAT, &forced_half_float_);
    params.get(MIPS_CONVERTER_SPHERICAL_INJECTION, &spherical_injection_);
    params.get(MIPS_CONVERTER_M_VALUE, &m_value_);
    params.get(MIPS_CONVERTER_U_VALUE, &u_value_);
    params.get(MIPS_CONVERTER_L2_NORM, &l2_norm_);

    if (!spherical_injection_) {
      if (!m_value_) {
        static const uint32_t m_values[4] = {4, 3, 6, 5};
        m_value_ = m_values[mt.dimension() % 4];
      }
      if (u_value_ <= std::numeric_limits<float>::epsilon() ||
          u_value_ >= 1.0) {
        // Try computing a default U value
        constexpr float kLogError = -5.0;  // log_10(distance_error)
        u_value_ = std::pow(10, kLogError / (1 << (m_value_ + 1)));
      }
      if (std::pow(u_value_, (1 << m_value_)) <
          std::numeric_limits<float>::epsilon()) {
        LOG_WARN("U value %f too small, may cause loss of distance precision.",
                 u_value_);
      }
    } else {
      if (m_value_ != 0u || u_value_ != 0.0f) {
        LOG_WARN(
            "Ignore invalid M value or U value if spherical_injection enabled");
      }
      // SphericalInjection requires ||x{i}|| <= 1 for the computation
      // std::sqrt(1 - ||x{i}||^2), so let the u_value be a little less
      // than 1.0 for its precision loss in float computation
      u_value_ = 1.0f - 1e-2;
      m_value_ = 1;
    }

    // Setting of MIPS Converter
    meta_ = mt;
    if (forced_half_float_) {
      meta_.set_meta(IndexMeta::DataType::DT_FP16, mt.dimension() + m_value_);
    } else {
      meta_.set_meta(dt, mt.dimension() + m_value_);
    }
    meta_.set_converter("MipsConverter", 0, params);
    return 0;
  }

  //! Cleanup Converter
  int cleanup(void) override {
    return 0;
  }

  //! Train the data
  int train(IndexHolder::Pointer holder) override {
    if (holder->dimension() + m_value_ != meta_.dimension()) {
      return IndexError_Mismatch;
    }

    ailego::ElapsedTime timer;
    auto iter = holder->create_iterator();
    if (!iter) {
      LOG_ERROR("Failed to create iterator of holder");
      return IndexError_Runtime;
    }

    size_t dim = holder->dimension();
    switch (holder->data_type()) {
      case IndexMeta::DataType::DT_FP16:
        for (; iter->is_valid(); iter->next()) {
          float score;
          ailego::Norm2Matrix<ailego::Float16, 1>::Compute(
              reinterpret_cast<const ailego::Float16 *>(iter->data()), dim,
              &score);

          if (score > l2_norm_) {
            l2_norm_ = score;
            if (l2_norm_ < 1.0 && l2_norm_ > u_value_) {
              u_value_ = l2_norm_;
            }
          }
          (*stats_.mutable_trained_count())++;
        }
        break;

      case IndexMeta::DataType::DT_FP32:
        for (; iter->is_valid(); iter->next()) {
          float score;
          ailego::Norm2Matrix<float, 1>::Compute(
              reinterpret_cast<const float *>(iter->data()), dim, &score);

          if (score > l2_norm_) {
            l2_norm_ = score;
            if (l2_norm_ < 1.0 && l2_norm_ > u_value_) {
              u_value_ = l2_norm_;
            }
          }
          (*stats_.mutable_trained_count())++;
        }
        break;

      default:
        return IndexError_Mismatch;
    }

    // Setting of MIPS Reformer
    ailego::Params reformer_params;
    reformer_params.set(MIPS_REFORMER_M_VALUE, m_value_);
    reformer_params.set(MIPS_REFORMER_U_VALUE, u_value_);
    reformer_params.set(MIPS_REFORMER_L2_NORM, l2_norm_);
    reformer_params.set(MIPS_REFORMER_FORCED_HALF_FLOAT, forced_half_float_);
    reformer_params.set(MIPS_REFORMER_NORMALIZE, true);
    reformer_params.set(MIPS_REFORMER_SPHERICAL_INJECTION,
                        spherical_injection_);
    meta_.set_reformer("MipsReformer", 0, reformer_params);
    if (meta_.metric_name() == "InnerProduct") {
      LOG_INFO("Convert IndexMeasure from InnerProduct to SquaredEuclidean");
      meta_.set_metric("SquaredEuclidean", 0, ailego::Params());
    }

    // Setting of MIPS Converter Params
    ailego::Params params = meta_.converter_params();
    params.set(MIPS_CONVERTER_FORCED_HALF_FLOAT, forced_half_float_);
    params.set(MIPS_CONVERTER_M_VALUE, m_value_);
    params.set(MIPS_CONVERTER_U_VALUE, u_value_);
    params.set(MIPS_CONVERTER_L2_NORM, l2_norm_);
    params.set(MIPS_CONVERTER_SPHERICAL_INJECTION, spherical_injection_);
    meta_.set_converter("MipsConverter", 0, params);

    stats_.set_trained_costtime(timer.milli_seconds());
    return 0;
  }

  //! Transform the data
  int transform(IndexHolder::Pointer holder) override {
    if (holder->dimension() + m_value_ != meta_.dimension()) {
      return IndexError_Mismatch;
    }

    switch (holder->data_type()) {
      case IndexMeta::DataType::DT_FP16:
        holder_ = std::make_shared<MipsConverterHalfHolder>(
            holder, m_value_, u_value_, l2_norm_, spherical_injection_);
        break;

      case IndexMeta::DataType::DT_FP32:
        if (forced_half_float_) {
          holder_ = std::make_shared<MipsConverterForcedHalfHolder>(
              holder, m_value_, u_value_, l2_norm_, spherical_injection_);
        } else {
          holder_ = std::make_shared<MipsConverterHolder>(
              holder, m_value_, u_value_, l2_norm_, spherical_injection_);
        }
        break;

      default:
        return IndexError_Mismatch;
    }
    return 0;
  }

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &) override {
    return 0;
  }

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve a holder as result
  IndexHolder::Pointer result(void) const override {
    return holder_;
  }

  //! Retrieve Index Meta
  const IndexMeta &meta(void) const override {
    return meta_;
  }

 private:
  uint32_t m_value_{0u};
  float u_value_{0.0f};
  float l2_norm_{0.0f};
  bool forced_half_float_{false};
  bool spherical_injection_{false};
  IndexMeta meta_{};
  IndexHolder::Pointer holder_{};
  Stats stats_{};
};

INDEX_FACTORY_REGISTER_CONVERTER(MipsConverter);

}  // namespace core
}  // namespace zvec
