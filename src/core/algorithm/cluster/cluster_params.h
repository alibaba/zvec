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

#include <string>

namespace zvec {
namespace core {

//! General
static const std::string GENERAL_CLUSTER_COUNT = "zvec.general.cluster.count";
static const std::string GENERAL_THREAD_COUNT =
    "zvec.general.cluster.thread_count";

//! Optimize K-means
static const std::string OPTKMEANS_CLUSTER_COUNT =
    "zvec.optkmeans.cluster.count";
static const std::string OPTKMEANS_CLUSTER_MAX_ITERATIONS =
    "zvec.optkmeans.cluster.max_iterations";
static const std::string OPTKMEANS_CLUSTER_EPSILON =
    "zvec.optkmeans.cluster.epsilon";
static const std::string OPTKMEANS_CLUSTER_SHARD_FACTOR =
    "zvec.optkmeans.cluster.shard_factor";
static const std::string OPTKMEANS_CLUSTER_PURGE_EMPTY =
    "zvec.optkmeans.cluster.purge_empty";
static const std::string OPTKMEANS_CLUSTER_MARKOV_CHAIN_LENGTH =
    "zvec.optkmeans.cluster.markov_chain_length";
static const std::string OPTKMEANS_CLUSTER_ASSUMPTION_FREE =
    "zvec.optkmeans.cluster.assumption_free";

//! K-means
static const std::string KMEANS_CLUSTER_COUNT = "zvec.kmeans.cluster.count";
static const std::string KMEANS_CLUSTER_SHARD_FACTOR =
    "zvec.kmeans.cluster.shard_factor";
static const std::string KMEANS_CLUSTER_EPSILON = "zvec.kmeans.cluster.epsilon";
static const std::string KMEANS_CLUSTER_MAX_ITERATIONS =
    "zvec.kmeans.cluster.max_iterations";
static const std::string KMEANS_CLUSTER_PURGE_EMPTY =
    "zvec.kmeans.cluster.purge_empty";
static const std::string KMEANS_CLUSTER_BATCH = "zvec.kmeans.cluster.batch";
static const std::string KMEANS_CLUSTER_SEEKER_CLASS =
    "zvec.kmeans.cluster.seeker_class";
static const std::string KMEANS_CLUSTER_SEEKER_PARAMS =
    "zvec.kmeans.cluster.seeker_params";

//! Mini Batch K-means
static const std::string MINIBATCHKMEANS_CLUSTER_COUNT =
    "zvec.minibatchkmeans.cluster.count";
static const std::string MINIBATCHKMEANS_CLUSTER_SHARD_FACTOR =
    "zvec.minibatchkmeans.cluster.shard_factor";
static const std::string MINIBATCHKMEANS_CLUSTER_EPSILON =
    "zvec.minibatchkmeans.cluster.epsilon";
static const std::string MINIBATCHKMEANS_CLUSTER_MAX_ITERATIONS =
    "zvec.minibatchkmeans.cluster.max_iterations";
static const std::string MINIBATCHKMEANS_CLUSTER_PURGE_EMPTY =
    "zvec.minibatchkmeans.cluster.purge_empty";
static const std::string MINIBATCHKMEANS_CLUSTER_TRY_COUNT =
    "zvec.minibatchkmeans.cluster.try_count";
static const std::string MINIBATCHKMEANS_CLUSTER_BATCH_COUNT =
    "zvec.minibatchkmeans.cluster.batch_count";
static const std::string MINIBATCHKMEANS_CLUSTER_SEEKER_CLASS =
    "zvec.minibatchkmeans.cluster.seeker_class";
static const std::string MINIBATCHKMEANS_CLUSTER_SEEKER_PARAMS =
    "zvec.minibatchkmeans.cluster.seeker_params";

//! K-means++
static const std::string KMEANSPP_CLUSTER_COUNT = "zvec.kmeanspp.cluster.count";
static const std::string KMEANSPP_CLUSTER_SHARD_FACTOR =
    "zvec.kmeanspp.cluster.shard_factor";
static const std::string KMEANSPP_CLUSTER_CLASS = "zvec.kmeanspp.cluster.class";
static const std::string KMEANSPP_CLUSTER_PARAMS =
    "zvec.kmeanspp.cluster.params";

//! K-MC2
static const std::string KMC2_CLUSTER_COUNT = "zvec.kmc2.cluster.count";
static const std::string KMC2_CLUSTER_SHARD_FACTOR =
    "zvec.kmc2.cluster.shard_factor";
static const std::string KMC2_CLUSTER_MARKOV_CHAIN_LENGTH =
    "zvec.kmc2.cluster.markov_chain_length";
static const std::string KMC2_CLUSTER_ASSUMPTION_FREE =
    "zvec.kmc2.cluster.assumption_free";
static const std::string KMC2_CLUSTER_CLASS = "zvec.kmc2.cluster.class";
static const std::string KMC2_CLUSTER_PARAMS = "zvec.kmc2.cluster.params";

//! Bisecting K-means
static const std::string BIKMEANS_CLUSTER_COUNT = "zvec.bikmeans.cluster.count";
static const std::string BIKMEANS_CLUSTER_INIT_COUNT =
    "zvec.bikmeans.cluster.init_count";
static const std::string BIKMEANS_CLUSTER_PURGE_EMPTY =
    "zvec.bikmeans.cluster.purge_empty";
static const std::string BIKMEANS_CLUSTER_FIRST_CLASS =
    "zvec.bikmeans.cluster.first_class";
static const std::string BIKMEANS_CLUSTER_SECOND_CLASS =
    "zvec.bikmeans.cluster.second_class";
static const std::string BIKMEANS_CLUSTER_FIRST_PARAMS =
    "zvec.bikmeans.cluster.first_params";
static const std::string BIKMEANS_CLUSTER_SECOND_PARAMS =
    "zvec.bikmeans.cluster.second_params";

//! K-medoids
static const std::string KMEDOIDS_CLUSTER_COUNT = "zvec.kmedoids.cluster.count";
static const std::string KMEDOIDS_CLUSTER_SHARD_FACTOR =
    "zvec.kmedoids.cluster.shard_factor";
static const std::string KMEDOIDS_CLUSTER_EPSILON =
    "zvec.kmedoids.cluster.epsilon";
static const std::string KMEDOIDS_CLUSTER_MAX_ITERATIONS =
    "zvec.kmedoids.cluster.max_iterations";
static const std::string KMEDOIDS_CLUSTER_PURGE_EMPTY =
    "zvec.kmedoids.cluster.purge_empty";
static const std::string KMEDOIDS_CLUSTER_BENCH_RATIO =
    "zvec.kmedoids.cluster.bench_ratio";
static const std::string KMEDOIDS_CLUSTER_ONLY_MEANS =
    "zvec.kmedoids.cluster.only_means";
static const std::string KMEDOIDS_CLUSTER_WITHOUT_MEANS =
    "zvec.kmedoids.cluster.without_means";
static const std::string KMEDOIDS_CLUSTER_SEEKER_CLASS =
    "zvec.kmedoids.cluster.seeker_class";
static const std::string KMEDOIDS_CLUSTER_SEEKER_PARAMS =
    "zvec.kmedoids.cluster.seeker_params";

//! Stratified
static const std::string STRATIFIED_CLUSTER_COUNT =
    "zvec.stratified.cluster.count";
static const std::string STRATIFIED_CLUSTER_FIRST_CLASS =
    "zvec.stratified.cluster.first_class";
static const std::string STRATIFIED_CLUSTER_SECOND_CLASS =
    "zvec.stratified.cluster.second_class";
static const std::string STRATIFIED_CLUSTER_FIRST_COUNT =
    "zvec.stratified.cluster.first_count";
static const std::string STRATIFIED_CLUSTER_SECOND_COUNT =
    "zvec.stratified.cluster.second_count";
static const std::string STRATIFIED_CLUSTER_FIRST_PARAMS =
    "zvec.stratified.cluster.first_params";
static const std::string STRATIFIED_CLUSTER_SECOND_PARAMS =
    "zvec.stratified.cluster.second_params";
static const std::string STRATIFIED_CLUSTER_AUTO_TUNING =
    "zvec.stratified.cluster.auto_tuning";
static const std::string STRATIFIED_CLUSTER_SECOND_POOL_COUNT =
    "zvec.stratified.cluster.second_pool_count";

//! Gap Statistics
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MIN =
    "zvec.gapstats.cluster_estimater.k_min";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MAX =
    "zvec.gapstats.cluster_estimater.k_max";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MIN_STEP =
    "zvec.gapstats.cluster_estimater.k_min_step";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MAX_STEP =
    "zvec.gapstats.cluster_estimater.k_max_step";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_TRY_COUNT =
    "zvec.gapstats.cluster_estimater.try_count";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_SHARD_FACTOR =
    "zvec.gapstats.cluster_estimater.shard_factor";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_ENABLE_MC2 =
    "zvec.gapstats.cluster_estimater.enable_mc2";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_MARKOV_CHAIN_LENGTH =
    "zvec.gapstats.cluster_estimater.markov_chain_length";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_CLUSTER_CLASS =
    "zvec.gapstats.cluster_estimater.cluster_class";

static const std::string CLUSTER_TRAINER_SAMPLE_COUNT =
    "zvec.cluster.trainer.sample_count";
static const std::string CLUSTER_TRAINER_SAMPLE_RATIO =
    "zvec.cluster.trainer.sample_ratio";
static const std::string CLUSTER_TRAINER_THREAD_COUNT =
    "zvec.cluster.trainer.thread_count";
static const std::string CLUSTER_TRAINER_FILE_NAME =
    "zvec.cluster.trainer.file_name";
static const std::string CLUSTER_TRAINER_CLASS_NAME =
    "zvec.cluster.trainer.class_name";

static const std::string STRATIFIED_TRAINER_SAMPLE_COUNT =
    "zvec.stratified.trainer.sample_count";
static const std::string STRATIFIED_TRAINER_SAMPLE_RATIO =
    "zvec.stratified.trainer.sample_ratio";
static const std::string STRATIFIED_TRAINER_THREAD_COUNT =
    "zvec.stratified.trainer.thread_count";
static const std::string STRATIFIED_TRAINER_FILE_NAME =
    "zvec.stratified.trainer.file_name";
static const std::string STRATIFIED_TRAINER_CLASS_NAME =
    "zvec.stratified.trainer.class_name";
static const std::string STRATIFIED_TRAINER_CLUSTER_COUNT =
    "zvec.stratified.trainer.cluster_count";
static const std::string STRATIFIED_TRAINER_AUTOAUNE =
    "zvec.stratified.trainer.autotune";
static const std::string STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX =
    "zvec.stratified.trainer.cluster_params_in_level_";

static const std::string MULTI_CHUNK_CLUSTER_COUNT =
    "zvec.cluster.multi_chunk_cluster.count";
static const std::string MULTI_CHUNK_CLUSTER_CHUNK_COUNT =
    "zvec.cluster.multi_chunk_cluster.chunk_count";
static const std::string MULTI_CHUNK_CLUSTER_THREAD_COUNT =
    "zvec.cluster.multi_chunk_cluster.thread_count";
static const std::string MULTI_CHUNK_CLUSTER_EPSILON =
    "zvec.cluster.multi_chunk_cluster.epsilon";
static const std::string MULTI_CHUNK_CLUSTER_MAX_ITERATIONS =
    "zvec.cluster.multi_chunk_cluster.max_iterations";
static const std::string MULTI_CHUNK_CLUSTER_MARKOV_CHAIN_LENGTH =
    "zvec.cluster.multi_chunk_cluster.markov_chain_length";
}  // namespace core
}  // namespace zvec
