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

#include "seeker.h"

namespace zvec {
namespace core {

/*! Linear Seeker
 */
class LinearSeeker : public Seeker {
 public:
  typedef std::shared_ptr<LinearSeeker> Pointer;

  //! Constructor
  LinearSeeker(void) : meta_(), metric_(), features_() {}

  //! Destructor
  ~LinearSeeker(void) = default;

  //! Initialize Seeker
  int init(const IndexMeta &meta) override {
    meta_ = meta;

    metric_ = IndexFactory::CreateMetric(meta_.metric_name());
    if (!metric_) {
      LOG_ERROR("Create Metric %s failed.", meta_.metric_name().c_str());

      return IndexError_Unsupported;
    }
    int ret = metric_->init(meta_, meta_.metric_params());
    if (ret != 0) {
      LOG_ERROR("IndexMetric init failed wit ret %d.", ret);

      return ret;
    }
    distance_func_ = metric_->distance_matrix(1, 1);
    if (!distance_func_) {
      LOG_ERROR("DistanceMatrix function is nullptr.");

      return IndexError_Unsupported;
    }
    return 0;
  }

  //! Cleanup Seeker
  int cleanup(void) override {
    features_.reset();
    return 0;
  }

  //! Reset Seeker
  int reset(void) override {
    features_.reset();
    return 0;
  }

  //! Mount features
  int mount(IndexFeatures::Pointer feats) override {
    if (!feats) {
      return IndexError_InvalidArgument;
    }
    if (!feats->is_matched(meta_)) {
      return IndexError_Mismatch;
    }
    features_ = std::move(feats);
    return 0;
  }

  //! Seek (TOP 1 Document)
  int seek(const void *query, size_t len, Document *out) override;

  //! Retrieve the original features
  IndexFeatures::Pointer original(void) const override {
    return features_;
  }

 private:
  IndexMeta meta_{};
  IndexMetric::Pointer metric_{};
  IndexFeatures::Pointer features_{};
  IndexMetric::MatrixDistance distance_func_{nullptr};
};

}  // namespace core
}  // namespace zvec
