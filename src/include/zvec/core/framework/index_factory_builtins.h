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

#include <zvec/ailego/pattern/factory.h>
#include <zvec/export.h>

namespace zvec {
namespace core {

class IndexBuilder;
struct IndexCluster;
class IndexConverter;
class IndexDumper;
struct IndexMetric;
class IndexReducer;
class IndexRefiner;
class IndexReformer;
class IndexSearcher;
class IndexStorage;
class IndexStreamer;
class IndexStreamerReducer;
class IndexTrainer;

//! Strong references retain the selected built-ins in ordinary static links.
ZVEC_CORE_API void RegisterBuiltinIndexModules();
ZVEC_TURBO_API void RegisterBuiltinQuantizers();

//! Module entry points for explicit C++ registry composition.
ZVEC_CORE_API void RegisterBuiltinMetrics();
ZVEC_CORE_API void RegisterBuiltinConvertersAndReformers();
ZVEC_CORE_API void RegisterBuiltinUtilities();
ZVEC_CORE_API void RegisterBuiltinMixedReducers();
ZVEC_CORE_API void RegisterBuiltinClusters();
ZVEC_CORE_API void RegisterBuiltinFlat();
ZVEC_CORE_API void RegisterBuiltinFlatSparse();
ZVEC_CORE_API void RegisterBuiltinHnsw();
ZVEC_CORE_API void RegisterBuiltinHnswSparse();
ZVEC_CORE_API void RegisterBuiltinIVF();
ZVEC_CORE_API void RegisterBuiltinVamana();
ZVEC_CORE_API void RegisterBuiltinDiskAnn();
ZVEC_CORE_API void RegisterBuiltinHnswRabitq();
ZVEC_CORE_API void RegisterBuiltinIvfRabitq();

}  // namespace core
namespace turbo {
class Quantizer;
}  // namespace turbo
namespace ailego {

//! Shared by the specializations below; registration itself never loads peers.
struct IndexFactoryBuiltinLoader {
  static void Load() {
    core::RegisterBuiltinIndexModules();
  }
};

template <>
struct FactoryBuiltinLoader<core::IndexBuilder> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexCluster> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexConverter> : IndexFactoryBuiltinLoader {
};
template <>
struct FactoryBuiltinLoader<core::IndexDumper> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexMetric> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexReducer> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexRefiner> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexReformer> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexSearcher> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexStorage> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexStreamer> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexStreamerReducer>
    : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<core::IndexTrainer> : IndexFactoryBuiltinLoader {};
template <>
struct FactoryBuiltinLoader<turbo::Quantizer> {
  static void Load() {
    core::RegisterBuiltinQuantizers();
  }
};

}  // namespace ailego
}  // namespace zvec
