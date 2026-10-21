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

#include <zvec/core/framework/index_cluster.h>

namespace zvec {
namespace core {

//! Optional internal capability for in-process centroid handoff. Trainers
//! without this interface continue to use IndexTrainer::indexes().
class CentroidTrainer {
 public:
  virtual ~CentroidTrainer() = default;

  //! Borrow the trained tree without serializing it. The caller must keep the
  //! trainer alive and must not mutate, retrain or clean it up while reading.
  //! Consumers must finish using the reference within their synchronous call.
  virtual const IndexCluster::CentroidList &centroids() const = 0;
};

}  // namespace core
}  // namespace zvec
