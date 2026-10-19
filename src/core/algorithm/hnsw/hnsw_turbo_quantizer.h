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
#include <string>
#include <zvec/core/framework/index_streamer.h>

namespace zvec {
namespace core {

//! Segment holding the serialized state of the turbo quantizer (e.g. the PQ
//! codebook) inside an HNSW index storage.
extern const std::string HNSW_TURBO_QUANTIZER_SEG_ID;

//! Serialize the quantizer training state (e.g. a PQ codebook) into a
//! storage segment.  Stateless quantizers carry no state; nothing is
//! written and the meta descriptor suffices to rebuild them.
int DumpTurboQuantizerSegment(const IndexStorage::Pointer &storage,
                              const zvec::turbo::Quantizer &quantizer);

//! Restore a turbo quantizer from the persisted segment and the quantized
//! meta descriptor (quantizer name, params, and the recorded raw input
//! layout).  Returns IndexError_NoExist when the segment is absent, so
//! callers can fall back to the stateless quantizer flow.
int RestoreTurboQuantizer(const IndexStorage::Pointer &storage,
                          const IndexMeta &quantized_meta,
                          std::shared_ptr<zvec::turbo::Quantizer>
                              *out_quantizer);

}  // namespace core
}  // namespace zvec
