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
//
#pragma once

#include <memory>
#include "zvec/core/framework/index_dumper.h"
#include "zvec/core/framework/index_reformer.h"
#include "zvec/core/framework/index_storage.h"
#include "rabitq_params.h"

namespace zvec {
namespace core {

class HnswRabitqQueryEntity;

/*! RaBitQ Reformer
 * Loads centroids and performs query transformation and vector quantization.
 *
 * All rabitqlib types are hidden behind a pimpl to avoid leaking rabitqlib
 * headers to consumers of this class.
 */
class RabitqReformer : public IndexReformer {
 public:
  typedef std::shared_ptr<RabitqReformer> Pointer;

  RabitqReformer();
  ~RabitqReformer() override;

  // Non-copyable
  RabitqReformer(const RabitqReformer &) = delete;
  RabitqReformer &operator=(const RabitqReformer &) = delete;

  int init(const ailego::Params &params) override;
  int cleanup(void) override;
  int load(IndexStorage::Pointer storage) override;
  int unload(void) override;

  // transform() is not implemented for RabitqReformer; use transform_to_entity.
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override;

  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override;

  int dump(const IndexDumper::Pointer &dumper);
  int dump(const IndexStorage::Pointer &storage);

  int transform_to_entity(const void *query,
                          HnswRabitqQueryEntity *entity) const;

  size_t num_clusters() const;
  RabitqMetricType rabitq_metric_type() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace zvec
