// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include "helper.h"

// Deliberately separate from timed bench: CSV I/O and A/B comparisons must not
// contaminate throughput. Each query runs against the exact same loaded index.
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: ivf_rabitq_scan_diagnostic CONFIG.yaml OUTPUT_PREFIX\n";
    return 1;
  }
  try {
    auto config = YAML::LoadFile(argv[1]);
    auto common = config["IndexCommon"];
    auto index_params =
        core_interface::IndexFactory::DeserializeIndexParamFromJson(
            common["IndexConfig"].as<std::string>());
    if (!index_params ||
        index_params->index_type != core_interface::IndexType::kIVF ||
        index_params->data_type != core_interface::DataType::DT_FP32 ||
        index_params->dimension <= 0 ||
        (common["QueryType"] &&
         common["QueryType"].as<std::string>() != "float"))
      throw std::runtime_error("Requires standard FP32 Turbo IVF RaBitQ index");
    std::string path = common["IndexPath"].as<std::string>();
    core_interface::Index::Pointer index;
    core_interface::BaseIndexQueryParam::Pointer base_query;
    if (parse_and_load_index_param(config, path, index, base_query) != 0)
      throw std::runtime_error("Cannot load index/query configuration");
    auto query =
        std::dynamic_pointer_cast<core_interface::IVFQueryParam>(base_query);
    if (!query || query->refiner_param || query->group_by_param ||
        query->bf_pks || (query->filter && query->filter->is_valid()) ||
        (common["FilterMode"] &&
         common["FilterMode"].as<std::string>() != "none"))
      throw std::runtime_error(
          "Diagnostic requires unfiltered IVF queries without "
          "refiner/group-by");
    const auto topk = common["TopK"].as<std::string>();
    size_t parsed = 0;
    const int k = std::stoi(topk, &parsed);
    if (parsed != topk.size() || k <= 0 || query->nprobe <= 0)
      throw std::runtime_error("Use one positive TopK and positive nprobe");
    query->topk = k;
    query->is_linear = false;
    query->fetch_vector = false;
    query->rabitq_collect_stats = true;
    std::vector<std::vector<float>> vectors;
    std::vector<SparseData<float>> sparse;
    std::vector<std::vector<uint64_t>> tags;
    TxtInputReader<float> reader;
    if (!reader.load_query(common["QueryFile"].as<std::string>(),
                           common["QueryFirstSep"]
                               ? common["QueryFirstSep"].as<std::string>()
                               : ";",
                           common["QuerySecondSep"]
                               ? common["QuerySecondSep"].as<std::string>()
                               : " ",
                           vectors, sparse, tags) ||
        vectors.empty())
      throw std::runtime_error("Cannot read FP32 text queries");
    for (const auto &v : vectors)
      if (v.size() != static_cast<size_t>(index_params->dimension))
        throw std::runtime_error("Query dimension differs from IndexConfig");

    std::ofstream counts(std::string(argv[2]) + ".queries.csv");
    std::ofstream results(std::string(argv[2]) + ".results.csv");
    if (!counts || !results)
      throw std::runtime_error("Cannot open output files");
    counts << "query,mode,active,mapped,coarse_count,refined_count,key_reads,"
              "coarse_reads,row_reads,coarse_copy_bytes,result_count,baseline_"
              "overlap,same_ordered_ids,max_common_score_delta\n";
    results << "query,mode,rank,key,score\n";
    counts << std::setprecision(std::numeric_limits<float>::max_digits10);
    results << std::setprecision(std::numeric_limits<float>::max_digits10);
    const std::array<const char *, 4> names{"baseline", "unpruned", "reuse",
                                            "reuse_unpruned"};
    std::array<size_t, 4> coarse{}, refined{}, active{};
    for (size_t q = 0; q < vectors.size(); ++q) {
      core_interface::SearchResult baseline;
      for (size_t mode = 0; mode < names.size(); ++mode) {
        query->rabitq_pruning = (mode % 2 == 0);
        query->rabitq_reuse_coarse_dot = (mode >= 2);
        core_interface::SearchResult result;
        const int ret = index->search(
            core_interface::VectorData{
                core_interface::DenseVector{vectors[q].data()}},
            query, &result);
        if (ret != 0)
          throw std::runtime_error("Search failed: " + std::to_string(ret));
        const auto &s = result.ivf_scan_stats_;
        if (!s.collected)
          throw std::runtime_error("Scan diagnostics were not collected");
        if (mode == 0) baseline = result;
        std::unordered_map<uint64_t, float> scores;
        for (const auto &d : baseline.doc_list_)
          scores.emplace(d.key(), d.score());
        size_t overlap = 0;
        bool same = baseline.doc_list_.size() == result.doc_list_.size();
        float max_delta = 0;
        for (size_t rank = 0; rank < result.doc_list_.size(); ++rank) {
          const auto &d = result.doc_list_[rank];
          if (!std::isfinite(d.score()))
            throw std::runtime_error("Non-finite result score");
          auto it = scores.find(d.key());
          if (it != scores.end()) {
            ++overlap;
            max_delta = std::max(max_delta, std::abs(d.score() - it->second));
          }
          if (rank >= baseline.doc_list_.size() ||
              baseline.doc_list_[rank].key() != d.key())
            same = false;
          results << q << ',' << names[mode] << ',' << rank << ',' << d.key()
                  << ',' << d.score() << '\n';
        }
        counts << q << ',' << names[mode] << ',' << s.active << ',' << s.mapped
               << ',' << s.coarse_count << ',' << s.refined_count << ','
               << s.key_reads << ',' << s.coarse_reads << ',' << s.row_reads
               << ',' << s.coarse_copy_bytes << ',' << result.doc_list_.size()
               << ',' << overlap << ',' << same << ',' << max_delta << '\n';
        coarse[mode] += s.coarse_count;
        refined[mode] += s.refined_count;
        active[mode] += s.active;
      }
    }
    counts.close();
    results.close();
    if (!counts || !results) throw std::runtime_error("Writing output failed");
    if (!active[0])
      throw std::runtime_error(
          "No packed scan activated; check index sidecar and query data");
    for (size_t mode = 0; mode < names.size(); ++mode) {
      std::cout << names[mode] << " queries=" << vectors.size()
                << " active=" << active[mode] << " coarse=" << coarse[mode]
                << " refined=" << refined[mode] << " refine_ratio="
                << (coarse[mode] ? double(refined[mode]) / coarse[mode] : 0)
                << '\n';
      // Emit directly runnable configurations without enabling stats in the
      // timed benchmark. Preserve input paths and separate recall log folders.
      query->rabitq_pruning = (mode % 2 == 0);
      query->rabitq_reuse_coarse_dot = (mode >= 2);
      query->rabitq_collect_stats = false;
      auto mode_config = YAML::Clone(config);
      mode_config["QueryConfig"]["QueryParam"] =
          core_interface::IndexFactory::QueryParamSerializeToJson(*query);
      const std::string prefix = std::string(argv[2]) + "." + names[mode];
      mode_config["IndexCommon"]["RecallLogDir"] = prefix + ".recall";
      std::ofstream output(prefix + ".yaml");
      output << mode_config << '\n';
      output.close();
      if (!output) throw std::runtime_error("Cannot write mode configuration");
    }
    std::cout << "Query numbers are zero-based input row positions. Overlap is "
                 "against baseline, not ground truth.\n"
              << "Use bench/recall separately for QPS and recall; this tool is "
                 "not timed.\n";
    return index->close();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
