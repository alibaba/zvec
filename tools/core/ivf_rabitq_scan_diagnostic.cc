// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include "helper.h"

namespace {
void PrintDistribution(const char *name, std::vector<size_t> values) {
  std::sort(values.begin(), values.end());
  double sum = 0;
  for (size_t value : values) sum += value;
  const double mean = sum / values.size();
  double variance = 0;
  for (size_t value : values) variance += (value - mean) * (value - mean);
  const auto percentile = [&](double fraction) {
    return values[static_cast<size_t>(std::ceil(fraction * values.size())) - 1];
  };
  std::cout << ' ' << name << "_mean=" << mean << ' ' << name
            << "_std=" << std::sqrt(variance / values.size()) << ' ' << name
            << "_min=" << values.front() << ' ' << name
            << "_p50=" << percentile(0.50) << ' ' << name
            << "_p95=" << percentile(0.95) << ' ' << name
            << "_max=" << values.back();
}
}  // namespace

// Deliberately separate from timed bench: CSV I/O and A/B comparisons must not
// contaminate throughput. Each query runs against the exact same loaded index.
int main(int argc, char **argv) {
  bool segment_timing = false, export_lists = false, disable_pruning = false;
  for (int i = 3; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--segment-timing" && !segment_timing) {
      segment_timing = true;
    } else if (flag == "--export-lists" && !export_lists) {
      export_lists = true;
    } else if (flag == "--disable-pruning" && !disable_pruning) {
      disable_pruning = true;
    } else {
      std::cerr << "Usage: ivf_rabitq_scan_diagnostic CONFIG.yaml "
                   "OUTPUT_PREFIX [--segment-timing] [--export-lists] "
                   "[--disable-pruning]\n";
      return 1;
    }
  }
  try {
    auto config = YAML::LoadFile(argv[1]);
    auto common = config["IndexCommon"];
    auto index_params =
        core_interface::IndexFactory::DeserializeIndexParamFromJson(
            common["IndexConfig"].as<std::string>());
    if (!index_params ||
        (index_params->index_type != core_interface::IndexType::kIVF &&
         index_params->index_type != core_interface::IndexType::kIVFRabitq) ||
        index_params->data_type != core_interface::DataType::DT_FP32 ||
        index_params->dimension <= 0 ||
        (common["QueryType"] &&
         common["QueryType"].as<std::string>() != "float"))
      throw std::runtime_error(
          "Requires FP32 Turbo IVF RaBitQ or dedicated IVF RaBitQ index");
    std::string path = common["IndexPath"].as<std::string>();
    core_interface::Index::Pointer index;
    core_interface::BaseIndexQueryParam::Pointer base_query;
    if (parse_and_load_index_param(config, path, index, base_query) != 0)
      throw std::runtime_error("Cannot load index/query configuration");
    auto query =
        std::dynamic_pointer_cast<core_interface::IVFQueryParam>(base_query);
    auto dedicated_query =
        std::dynamic_pointer_cast<core_interface::IVFRabitqQueryParam>(
            base_query);
    if (segment_timing && !query)
      throw std::runtime_error("Segment timing requires integrated IVF RaBitQ");
    if (disable_pruning && query)
      throw std::runtime_error(
          "--disable-pruning requires dedicated IVF RaBitQ; integrated "
          "configs already include unpruned modes");
    if (disable_pruning) dedicated_query->rabitq_pruning = false;
    if ((!query && !dedicated_query) || base_query->refiner_param ||
        base_query->group_by_param || base_query->bf_pks ||
        (base_query->filter && base_query->filter->is_valid()) ||
        (common["FilterMode"] &&
         common["FilterMode"].as<std::string>() != "none"))
      throw std::runtime_error(
          "Diagnostic requires unfiltered IVF queries without "
          "refiner/group-by");
    const auto topk = common["TopK"].as<std::string>();
    size_t parsed = 0;
    const int k = std::stoi(topk, &parsed);
    if (parsed != topk.size() || k <= 0 ||
        (query ? query->nprobe <= 0 : dedicated_query->nprobe == 0))
      throw std::runtime_error("Use one positive TopK and positive nprobe");
    base_query->topk = k;
    base_query->is_linear = false;
    base_query->fetch_vector = false;
    // Dedicated search always uses pruning and reuses the LUT dot. It has one
    // native mode; do not label it as the integrated baseline or toggle it.
    const std::vector<const char *> names =
        query ? std::vector<const char *>{"baseline", "unpruned", "reuse",
                                          "reuse_unpruned"}
              : std::vector<const char *>{"dedicated"};
    const auto configure = [&](size_t mode, bool collect) {
      if (query) {
        query->rabitq_pruning = (mode % 2 == 0);
        query->rabitq_reuse_coarse_dot = (mode >= 2);
        query->rabitq_collect_stats = collect;
        // Segment timing is diagnostics-only and does not alter the search.
        // Emitted mode configurations keep it off (collect == false).
        query->rabitq_segment_timing = collect && segment_timing;
        query->rabitq_export_scan_lists = collect && export_lists;
      } else {
        dedicated_query->rabitq_collect_stats = collect;
        dedicated_query->rabitq_export_scan_lists = collect && export_lists;
      }
    };
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
    std::ofstream lists;
    if (export_lists) lists.open(std::string(argv[2]) + ".lists.csv");
    if (!counts || !results || (export_lists && !lists))
      throw std::runtime_error("Cannot open output files");
    counts << "query,mode,active,mapped,coarse_count,refined_count,key_reads,"
              "coarse_reads,row_reads,coarse_copy_bytes,result_count,baseline_"
              "overlap,same_ordered_ids,max_common_score_delta,prep_ns,coarse_"
              "ns,cand_ns,post_scan_ns,finalize_ns,query_ns\n";
    results << "query,mode,rank,key,score\n";
    if (export_lists) lists << "query,mode,list_id,scan_count,limited\n";
    counts << std::setprecision(std::numeric_limits<float>::max_digits10);
    results << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::array<size_t, 4> coarse{}, refined{}, active{};
    std::array<std::vector<size_t>, 4> coarse_per_query, refined_per_query;
    std::array<double, 4> prep_ns{}, seg_coarse_ns{}, cand_ns{},
        seg_post_scan_ns{}, finalize_ns{}, query_ns{};
    std::array<size_t, 4> lists_total{};
    for (size_t q = 0; q < vectors.size(); ++q) {
      core_interface::SearchResult baseline;
      for (size_t mode = 0; mode < names.size(); ++mode) {
        configure(mode, true);
        core_interface::SearchResult result;
        const int ret = index->search(
            core_interface::VectorData{
                core_interface::DenseVector{vectors[q].data()}},
            base_query, &result);
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
               << ',' << overlap << ',' << same << ',' << max_delta << ','
               << s.prep_ns << ',' << s.coarse_ns << ',' << s.cand_ns << ','
               << s.post_scan_ns << ',' << s.finalize_ns << ',' << s.query_ns
               << '\n';
        if (export_lists) {
          if (!s.scan_lists_collected)
            throw std::runtime_error("Scan lists were not collected");
          for (size_t li = 0; li < s.scan_list_ids.size(); ++li) {
            lists << q << ',' << names[mode] << ',' << s.scan_list_ids[li]
                  << ',' << s.scan_list_counts[li] << ','
                  << (s.scan_lists_limited ? 1 : 0) << '\n';
          }
          lists_total[mode] += s.scan_list_ids.size();
        }
        coarse[mode] += s.coarse_count;
        refined[mode] += s.refined_count;
        active[mode] += s.active;
        prep_ns[mode] += s.prep_ns;
        seg_coarse_ns[mode] += s.coarse_ns;
        cand_ns[mode] += s.cand_ns;
        seg_post_scan_ns[mode] += s.post_scan_ns;
        finalize_ns[mode] += s.finalize_ns;
        query_ns[mode] += s.query_ns;
        coarse_per_query[mode].push_back(s.coarse_count);
        refined_per_query[mode].push_back(s.refined_count);
      }
    }
    counts.close();
    results.close();
    if (!counts || !results) throw std::runtime_error("Writing output failed");
    if (!active[0])
      throw std::runtime_error(
          "No batch scan activated; check index type, sidecar and query data");
    for (size_t mode = 0; mode < names.size(); ++mode) {
      std::cout << names[mode] << " queries=" << vectors.size()
                << " active=" << active[mode] << " coarse=" << coarse[mode]
                << " refined=" << refined[mode] << " refine_ratio="
                << (coarse[mode] ? double(refined[mode]) / coarse[mode] : 0);
      PrintDistribution("coarse", coarse_per_query[mode]);
      PrintDistribution("refined", refined_per_query[mode]);
      const double nq = double(vectors.size());
      std::cout << " segments_us/q prep=" << prep_ns[mode] / nq / 1000.0
                << " coarse=" << seg_coarse_ns[mode] / nq / 1000.0
                << " cand=" << cand_ns[mode] / nq / 1000.0
                << " post_scan=" << seg_post_scan_ns[mode] / nq / 1000.0
                << " finalize=" << finalize_ns[mode] / nq / 1000.0
                << " query=" << query_ns[mode] / nq / 1000.0;
      if (export_lists) std::cout << " lists_total=" << lists_total[mode];
      std::cout << '\n';
      // Emit directly runnable configurations without enabling stats in the
      // timed benchmark. Preserve input paths and separate recall log folders.
      configure(mode, false);
      auto mode_config = YAML::Clone(config);
      mode_config["QueryConfig"]["QueryParam"] =
          query
              ? core_interface::IndexFactory::QueryParamSerializeToJson(*query)
              : core_interface::IndexFactory::QueryParamSerializeToJson(
                    *dedicated_query);
      const std::string prefix = std::string(argv[2]) + "." + names[mode];
      mode_config["IndexCommon"]["RecallLogDir"] = prefix + ".recall";
      std::ofstream output(prefix + ".yaml");
      output << mode_config << '\n';
      output.close();
      if (!output) throw std::runtime_error("Cannot write mode configuration");
    }
    std::cout << "Query numbers are zero-based input row positions. Overlap is "
                 "against the first mode of this index, not ground truth or "
                 "another index.\n"
              << "Use bench/recall separately for QPS and recall; this tool is "
                 "not timed.\n";
    return index->close();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
