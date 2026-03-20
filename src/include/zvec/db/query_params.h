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
#include <zvec/core/interface/constants.h>
#include <zvec/db/type.h>

namespace zvec {

/*
 * Query Index params
 */
class QueryParams {
 public:
  using Ptr = std::shared_ptr<QueryParams>;

  QueryParams(IndexType type) : type_(type) {}
  virtual ~QueryParams() = default;

  IndexType type() const {
    return type_;
  }

  void set_type(IndexType type) {
    type_ = type;
  }

  float radius() const {
    return radius_;
  }

  void set_radius(float radius) {
    radius_ = radius;
  }

  bool is_linear() const {
    return is_linear_;
  }

  void set_is_linear(bool is_linear) {
    is_linear_ = is_linear;
  }

  void set_is_using_refiner(bool is_using_refiner) {
    is_using_refiner_ = is_using_refiner;
  }
  bool is_using_refiner() const {
    return is_using_refiner_;
  }

 private:
  IndexType type_;
  float radius_{0.0f};
  bool is_linear_{false};

  bool is_using_refiner_{false};
};

class HnswQueryParams : public QueryParams {
 public:
  HnswQueryParams(int ef = core_interface::kDefaultHnswEfSearch,
                  float radius = 0.0f, bool is_linear = false,
                  bool is_using_refiner = false)
      : QueryParams(IndexType::HNSW), ef_(ef) {
    set_radius(radius);
    set_is_linear(is_linear);
    set_is_using_refiner(is_using_refiner);
  }

  virtual ~HnswQueryParams() = default;

  int ef() const {
    return ef_;
  }

  void set_ef(int ef) {
    ef_ = ef;
  }

 private:
  int ef_;
};

class IVFQueryParams : public QueryParams {
 public:
  IVFQueryParams(int nprobe = 10, bool is_using_refiner = false,
                 float scale_factor = 10)
      : QueryParams(IndexType::IVF), nprobe_(nprobe) {
    set_is_using_refiner(is_using_refiner);
    set_scale_factor(scale_factor);
  }

  virtual ~IVFQueryParams() = default;

  int nprobe() const {
    return nprobe_;
  }

  void set_nprobe(int nprobe) {
    nprobe_ = nprobe;
  }

  float scale_factor() const {
    return scale_factor_;
  }

  void set_scale_factor(float scale_factor) {
    scale_factor_ = scale_factor;
  }

 private:
  int nprobe_;
  float scale_factor_{10};
};

class HnswRabitqQueryParams : public QueryParams {
 public:
  HnswRabitqQueryParams(int ef = core_interface::kDefaultHnswEfSearch,
                        float radius = 0.0f, bool is_linear = false,
                        bool is_using_refiner = false)
      : QueryParams(IndexType::HNSW_RABITQ), ef_(ef) {
    set_radius(radius);
    set_is_linear(is_linear);
    set_is_using_refiner(is_using_refiner);
  }

  virtual ~HnswRabitqQueryParams() = default;

  int ef() const {
    return ef_;
  }

  void set_ef(int ef) {
    ef_ = ef;
  }

 private:
  int ef_;
};

class FlatQueryParams : public QueryParams {
 public:
  FlatQueryParams(bool is_using_refiner = false, float scale_factor = 10)
      : QueryParams(IndexType::FLAT) {
    set_is_using_refiner(is_using_refiner);
    set_scale_factor(scale_factor);
  }

  virtual ~FlatQueryParams() = default;

  float scale_factor() const {
    return scale_factor_;
  }

  void set_scale_factor(float scale_factor) {
    scale_factor_ = scale_factor;
  }

 private:
  float scale_factor_{10};
};

}  // namespace zvec