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
#include <zvec/core/framework/index_builder.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_converter.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/framework/index_metric.h>
#include <zvec/core/framework/index_reducer.h>
#include <zvec/core/framework/index_refiner.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/framework/index_streamer.h>
#include <zvec/core/framework/index_trainer.h>

namespace zvec {
namespace core {

/*! Index Factory
 */
struct IndexFactory {
  //! Create a index Metric by name
  static IndexMetric::Pointer CreateMetric(const std::string &name);

  //! Test if the Metric is exist
  static bool HasMetric(const std::string &name);

  //! Retrieve all Metric classes
  static std::vector<std::string> AllMetrics(void);

  //! Create a index logger by name
  static IndexLogger::Pointer CreateLogger(const std::string &name);

  //! Test if the logger is exist
  static bool HasLogger(const std::string &name);

  //! Retrieve all logger classes
  static std::vector<std::string> AllLoggers(void);

  //! Create a index dumper by name
  static IndexDumper::Pointer CreateDumper(const std::string &name);

  //! Test if the dumper is exist
  static bool HasDumper(const std::string &name);

  //! Retrieve all dumper classes
  static std::vector<std::string> AllDumpers(void);

  //! Test if the container is exist
  static bool HasContainer(const std::string &name);

  //! Retrieve all container classes
  static std::vector<std::string> AllContainers(void);

  //! Create a index storage by name
  static IndexStorage::Pointer CreateStorage(const std::string &name);

  //! Test if the storage is exist
  static bool HasStorage(const std::string &name);

  //! Retrieve all storage classes
  static std::vector<std::string> AllStorages(void);

  //! Create a index converter by name
  static IndexConverter::Pointer CreateConverter(const std::string &name);

  //! Test if the converter is exist
  static bool HasConverter(const std::string &name);

  //! Retrieve all converter classes
  static std::vector<std::string> AllConverters(void);

  //! Create a index reformer by name
  static IndexReformer::Pointer CreateReformer(const std::string &name);

  //! Test if the reformer is exist
  static bool HasReformer(const std::string &name);

  //! Retrieve all reformer classes
  static std::vector<std::string> AllReformers(void);

  //! Create a index trainer by name
  static IndexTrainer::Pointer CreateTrainer(const std::string &name);

  //! Test if the trainer is exist
  static bool HasTrainer(const std::string &name);

  //! Retrieve all trainer classes
  static std::vector<std::string> AllTrainers(void);

  //! Create a index builder by name
  static IndexBuilder::Pointer CreateBuilder(const std::string &name);

  //! Test if the builder is exist
  static bool HasBuilder(const std::string &name);

  //! Retrieve all builder classes
  static std::vector<std::string> AllBuilders(void);

  //! Create a index searcher by name
  static IndexSearcher::Pointer CreateSearcher(const std::string &name);

  //! Test if the searcher is exist
  static bool HasSearcher(const std::string &name);

  //! Retrieve all searcher classes
  static std::vector<std::string> AllSearchers(void);

  //! Create a index streamer by name
  static IndexStreamer::Pointer CreateStreamer(const std::string &name);

  //! Test if the streamer is exist
  static bool HasStreamer(const std::string &name);

  //! Retrieve all streamer classes
  static std::vector<std::string> AllStreamers(void);

  //! Create a index reducer by name
  static IndexReducer::Pointer CreateReducer(const std::string &name);

  //! Test if the reducer is exist
  static bool HasReducer(const std::string &name);

  //! Retrieve all reducer classes
  static std::vector<std::string> AllReducers(void);

  //! Create a index cluster by name
  static IndexCluster::Pointer CreateCluster(const std::string &name);

  //! Test if the cluster is exist
  static bool HasCluster(const std::string &name);

  //! Retrieve all cluster classes
  static std::vector<std::string> AllClusters(void);

  //! Create a index streamer reducer by name
  static IndexStreamerReducer::Pointer CreateStreamerReducer(
      const std::string &name);

  //! Test if the streamer reducer is exist
  static bool HasStreamerReducer(const std::string &name);

  //! Retrieve all streamer reducer classes
  static std::vector<std::string> AllStreamerReducers(void);

  //! Create a refiner by name
  static IndexRefiner::Pointer CreateRefiner(const std::string &name);

  //! Test if the refiner is exist
  static bool HasRefiner(const std::string &name);

  //! Retrieve all refiner classes
  static std::vector<std::string> AllRefiners(void);
};

//! Register Index Metric
#define INDEX_FACTORY_REGISTER_METRIC_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexMetric, __IMPL__, ##__VA_ARGS__)

//! Register Index Metric
#define INDEX_FACTORY_REGISTER_METRIC(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_METRIC_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Logger
#define INDEX_FACTORY_REGISTER_LOGGER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexLogger, __IMPL__, ##__VA_ARGS__)

//! Register Index Logger
#define INDEX_FACTORY_REGISTER_LOGGER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_LOGGER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Dumper
#define INDEX_FACTORY_REGISTER_DUMPER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexDumper, __IMPL__, ##__VA_ARGS__)

//! Register Index Dumper
#define INDEX_FACTORY_REGISTER_DUMPER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_DUMPER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Storage
#define INDEX_FACTORY_REGISTER_STORAGE_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexStorage, __IMPL__, ##__VA_ARGS__)

//! Register Index Storage
#define INDEX_FACTORY_REGISTER_STORAGE(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_STORAGE_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Converter
#define INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexConverter, __IMPL__, ##__VA_ARGS__)

//! Register Index Converter
#define INDEX_FACTORY_REGISTER_CONVERTER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Reformer
#define INDEX_FACTORY_REGISTER_REFORMER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexReformer, __IMPL__, ##__VA_ARGS__)

//! Register Index Reformer
#define INDEX_FACTORY_REGISTER_REFORMER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_REFORMER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Trainer
#define INDEX_FACTORY_REGISTER_TRAINER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexTrainer, __IMPL__, ##__VA_ARGS__)

//! Register Index Trainer
#define INDEX_FACTORY_REGISTER_TRAINER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_TRAINER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Builder
#define INDEX_FACTORY_REGISTER_BUILDER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexBuilder, __IMPL__, ##__VA_ARGS__)

//! Register Index Builder
#define INDEX_FACTORY_REGISTER_BUILDER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_BUILDER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Searcher
#define INDEX_FACTORY_REGISTER_SEARCHER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexSearcher, __IMPL__, ##__VA_ARGS__)

//! Register Index Searcher
#define INDEX_FACTORY_REGISTER_SEARCHER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_SEARCHER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Streamer
#define INDEX_FACTORY_REGISTER_STREAMER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexStreamer, __IMPL__, ##__VA_ARGS__)

//! Register Index Streamer
#define INDEX_FACTORY_REGISTER_STREAMER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_STREAMER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Reducer
#define INDEX_FACTORY_REGISTER_REDUCER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexReducer, __IMPL__, ##__VA_ARGS__)

//! Register Index Reducer
#define INDEX_FACTORY_REGISTER_REDUCER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_REDUCER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Streamer Reducer
#define INDEX_FACTORY_REGISTER_STREAMER_REDUCER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexStreamerReducer, __IMPL__,            \
                          ##__VA_ARGS__)

//! Register Index Streamer Reducer
#define INDEX_FACTORY_REGISTER_STREAMER_REDUCER(__IMPL__, ...)      \
  INDEX_FACTORY_REGISTER_STREAMER_REDUCER_ALIAS(__IMPL__, __IMPL__, \
                                                ##__VA_ARGS__)

//! Register Index Cluster
#define INDEX_FACTORY_REGISTER_CLUSTER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexCluster, __IMPL__, ##__VA_ARGS__)

//! Register Index Cluster
#define INDEX_FACTORY_REGISTER_CLUSTER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_CLUSTER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

//! Register Index Refiner
#define INDEX_FACTORY_REGISTER_REFINER_ALIAS(__NAME__, __IMPL__, ...) \
  AILEGO_FACTORY_REGISTER(__NAME__, IndexRefiner, __IMPL__, ##__VA_ARGS__)

//! Register Index Refiner
#define INDEX_FACTORY_REGISTER_REFINER(__IMPL__, ...) \
  INDEX_FACTORY_REGISTER_REFINER_ALIAS(__IMPL__, __IMPL__, ##__VA_ARGS__)

}  // namespace core
}  // namespace zvec
