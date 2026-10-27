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

#include <algorithm>
#include <atomic>
#include <initializer_list>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec::core {

// Keep a user-defined registration in the executable itself. Built-ins must
// remain available without depending on this registration or a shared library.
class StaticLinkCustomMetric : public IndexMetric {
 public:
  explicit StaticLinkCustomMetric(int marker) : marker_(marker) {}

  int init(const IndexMeta &, const ailego::Params &) override {
    return marker_;
  }
  int cleanup() override {
    return 0;
  }
  bool is_matched(const IndexMeta &) const override {
    return true;
  }
  bool is_matched(const IndexMeta &, const IndexQueryMeta &) const override {
    return true;
  }
  const ailego::Params &params() const override {
    return params_;
  }
  Pointer query_metric() const override {
    return nullptr;
  }

 private:
  int marker_;
  ailego::Params params_;
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(StaticLinkCustomMetric,
                                    StaticLinkCustomMetric, 17);

// A registration made before the built-in catalog is first used must survive
// lazy initialization, even when it replaces a built-in name.
INDEX_FACTORY_REGISTER_METRIC_ALIAS(NormalizedCosine, StaticLinkCustomMetric,
                                    29);

}  // namespace zvec::core

namespace {

using zvec::core::IndexFactory;

bool Check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

template <typename Create, typename Has, typename All>
bool CheckFamily(const char *family, Create create, Has has, All all,
                 std::initializer_list<const char *> required) {
  bool ok = true;
  const auto names = all();
  const std::set<std::string> unique_names(names.begin(), names.end());
  ok &= Check(names.size() == unique_names.size(),
              std::string(family) + ": duplicate names in enumeration");
  for (const char *name : required) {
    ok &= Check(unique_names.count(name) == 1,
                std::string(family) + ": missing built-in " + name);
  }
  // Construct every advertised class, including aliases with constructor
  // arguments. This catches a catalog that advertises an unlinked creator.
  for (const auto &name : names) {
    ok &= Check(has(name), std::string(family) + ": Has rejected " + name);
    auto instance = create(name);
    ok &= Check(instance != nullptr,
                std::string(family) + ": cannot construct " + name);
    if (instance) {
      ok &= Check(instance->name() == name,
                  std::string(family) + ": incorrect instance name " + name);
    }
  }
  const std::string missing = "MissingStaticLinkFactoryClass";
  ok &= Check(!has(missing), std::string(family) + ": accepted unknown name");
  ok &= Check(create(missing) == nullptr,
              std::string(family) + ": constructed unknown class");
  return ok;
}

bool CheckConcurrentFirstUse() {
  std::atomic<bool> ok{true};
  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (unsigned repeat = 0; repeat < 16; ++repeat) {
        auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
        auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
        auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
        if (!metric || metric->name() != "SquaredEuclidean" || !storage ||
            !searcher || !IndexFactory::HasMetric("SquaredEuclidean") ||
            IndexFactory::AllSearchers().empty()) {
          ok.store(false, std::memory_order_relaxed);
        }
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != threads.size()) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }
  return Check(ok.load(), "Concurrent first use lost a built-in factory class");
}

}  // namespace

int main() {
  bool ok = CheckConcurrentFirstUse();
  ok &= CheckFamily("metric", IndexFactory::CreateMetric,
                    IndexFactory::HasMetric, IndexFactory::AllMetrics,
                    {"SquaredEuclidean", "Euclidean", "SquaredEuclideanSparse",
                     "InnerProduct", "InnerProductSparse", "NormalizedCosine",
                     "MipsSquaredEuclidean", "MipsSquaredEuclideanSparse",
                     "Cosine", "QuantizedInteger", "UniformUint4",
                     "UniformUint7", "UniformUint8", "StaticLinkCustomMetric"});
  ok &= CheckFamily(
      "storage", IndexFactory::CreateStorage, IndexFactory::HasStorage,
      IndexFactory::AllStorages,
      {"BufferStorage", "BufferReadStorage", "MemoryReadStorage",
       "FileReadStorage", "MMapFileStorage", "MMapFileReadStorage"});
  ok &=
      CheckFamily("dumper", IndexFactory::CreateDumper, IndexFactory::HasDumper,
                  IndexFactory::AllDumpers, {"MemoryDumper", "FileDumper"});
  ok &= CheckFamily(
      "converter", IndexFactory::CreateConverter, IndexFactory::HasConverter,
      IndexFactory::AllConverters,
      {"HalfFloatConverter", "HalfFloatSparseConverter", "BinaryConverter",
       "MipsConverter", "RawUint8Converter", "Int8QuantizerConverter",
       "Int4QuantizerConverter", "Int8StreamingConverter",
       "Int4StreamingConverter", "CosineNormalizeConverter",
       "CosineFp32Converter", "CosineFp16Converter", "CosineRawFp16Converter",
       "CosineInt8Converter", "CosineInt4Converter", "CosineHalfFloatConverter",
       "UniformUint4Converter", "UniformUint7Converter",
       "UniformUint8Converter"});
  ok &= CheckFamily(
      "reformer", IndexFactory::CreateReformer, IndexFactory::HasReformer,
      IndexFactory::AllReformers,
      {"HalfFloatReformer", "HalfFloatSparseReformer", "BinaryReformer",
       "MipsReformer", "RawUint8Reformer", "Int8QuantizerReformer",
       "Int4QuantizerReformer", "Int8StreamingReformer",
       "Int4StreamingReformer", "CosineNormalizeReformer", "CosineFp32Reformer",
       "CosineFp16Reformer", "CosineRawFp16Reformer", "CosineInt8Reformer",
       "CosineInt4Reformer", "CosineHalfFloatReformer", "UniformUint4Reformer",
       "UniformUint7Reformer", "UniformUint8Reformer"});
  ok &= CheckFamily("trainer", IndexFactory::CreateTrainer,
                    IndexFactory::HasTrainer, IndexFactory::AllTrainers,
                    {"StratifiedClusterTrainer"});
  ok &= CheckFamily(
      "builder", IndexFactory::CreateBuilder, IndexFactory::HasBuilder,
      IndexFactory::AllBuilders,
      {"LinearBuilder", "FlatBuilder", "FlatBuilder16", "FlatBuilder32",
       "FlatSparseBuilder", "HnswSparseBuilder", "IVFBuilder"});
  ok &= CheckFamily(
      "searcher", IndexFactory::CreateSearcher, IndexFactory::HasSearcher,
      IndexFactory::AllSearchers,
      {"LinearSearcher", "FlatSearcher", "FlatSearcher16", "FlatSearcher32",
       "FlatSparseSearcher", "HnswSparseSearcher", "IVFSearcher"});
  ok &= CheckFamily("streamer", IndexFactory::CreateStreamer,
                    IndexFactory::HasStreamer, IndexFactory::AllStreamers,
                    {"LinearStreamer", "FlatStreamer", "FlatStreamer16",
                     "FlatStreamer32", "FlatSparseStreamer", "HnswStreamer",
                     "HnswSparseStreamer", "VamanaStreamer", "IVFStreamer"});
  ok &= CheckFamily("cluster", IndexFactory::CreateCluster,
                    IndexFactory::HasCluster, IndexFactory::AllClusters,
                    {"KmeansCluster", "BatchKmeansCluster", "OptKmeansCluster",
                     "StratifiedCluster"});
  ok &= CheckFamily("refiner", IndexFactory::CreateRefiner,
                    IndexFactory::HasRefiner, IndexFactory::AllRefiners,
                    {"BasicRefiner"});
  ok &=
      CheckFamily("streamer reducer", IndexFactory::CreateStreamerReducer,
                  IndexFactory::HasStreamerReducer,
                  IndexFactory::AllStreamerReducers, {"MixedStreamerReducer"});
  ok &= CheckFamily("reducer", IndexFactory::CreateReducer,
                    IndexFactory::HasReducer, IndexFactory::AllReducers, {});

#if DISKANN_SUPPORTED
  ok &= CheckFamily("DiskAnn builder", IndexFactory::CreateBuilder,
                    IndexFactory::HasBuilder, IndexFactory::AllBuilders,
                    {"DiskAnnBuilder"});
  ok &= CheckFamily("DiskAnn searcher", IndexFactory::CreateSearcher,
                    IndexFactory::HasSearcher, IndexFactory::AllSearchers,
                    {"DiskAnnSearcher"});
  ok &= CheckFamily("DiskAnn streamer", IndexFactory::CreateStreamer,
                    IndexFactory::HasStreamer, IndexFactory::AllStreamers,
                    {"DiskAnnStreamer"});
  ok &= CheckFamily("DiskAnn reducer", IndexFactory::CreateReducer,
                    IndexFactory::HasReducer, IndexFactory::AllReducers,
                    {"DiskAnnReducer"});
#else
  ok &= Check(!IndexFactory::HasBuilder("DiskAnnBuilder"),
              "Unsupported DiskAnn build advertises DiskAnnBuilder");
#endif
#if RABITQ_SUPPORTED
  ok &= CheckFamily("RaBitQ builder", IndexFactory::CreateBuilder,
                    IndexFactory::HasBuilder, IndexFactory::AllBuilders,
                    {"IvfRabitqBuilder"});
  ok &= CheckFamily("RaBitQ streamer", IndexFactory::CreateStreamer,
                    IndexFactory::HasStreamer, IndexFactory::AllStreamers,
                    {"HnswRabitqStreamer", "IvfRabitqStreamer"});
  ok &= CheckFamily("RaBitQ converter", IndexFactory::CreateConverter,
                    IndexFactory::HasConverter, IndexFactory::AllConverters,
                    {"RabitqConverter"});
  ok &= CheckFamily("RaBitQ reformer", IndexFactory::CreateReformer,
                    IndexFactory::HasReformer, IndexFactory::AllReformers,
                    {"RabitqReformer"});
#else
  ok &= Check(!IndexFactory::HasStreamer("HnswRabitqStreamer"),
              "Unsupported RaBitQ build advertises HnswRabitqStreamer");
#endif

  auto early_override = IndexFactory::CreateMetric("NormalizedCosine");
  ok &= Check(early_override && early_override->init({}, {}) == 29,
              "Built-in initialization replaced an existing custom metric");

  auto console =
      zvec::ailego::Factory<zvec::ailego::Logger>::MakeShared("ConsoleLogger");
  ok &= Check(console && console->init({}) == 0,
              "ConsoleLogger is unavailable through the logger factory");
  auto logger = zvec::ailego::LoggerBroker::Register(nullptr);
  ok &= Check(logger && logger->init({}) == 0,
              "Default ConsoleLogger is unavailable through ordinary linking");
  zvec::ailego::LoggerBroker::Register(std::move(logger));
  ok &= Check(zvec::ailego::LoggerBroker::IsLevelEnabled(
                  zvec::ailego::Logger::LEVEL_WARN),
              "Default ConsoleLogger was not restored");

  auto custom = IndexFactory::CreateMetric("StaticLinkCustomMetric");
  ok &= Check(custom && custom->init({}, {}) == 17,
              "Custom registration lost its constructor argument");
  // Applications can replace an existing implementation at runtime. The
  // explicit built-in catalog must not override this user registration.
  zvec::ailego::Factory<zvec::core::IndexMetric>::Register<
      zvec::core::StaticLinkCustomMetric>
      override_metric("SquaredEuclidean", 31);
  auto overridden = IndexFactory::CreateMetric("SquaredEuclidean");
  ok &= Check(overridden && overridden->init({}, {}) == 31 &&
                  overridden->name() == "SquaredEuclidean",
              "Custom registration cannot override a built-in metric");
  return ok ? 0 : 1;
}
