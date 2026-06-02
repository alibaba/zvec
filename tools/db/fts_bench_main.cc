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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <gflags/gflags.h>
#include <roaring/roaring.h>
#include <zvec/ailego/encoding/json.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/common/rocksdb_context.h"
#include "db/index/column/fts_column/fts_column_indexer.h"
#include "db/index/column/fts_column/fts_pipeline.h"
#include "db/index/column/fts_column/fts_query_ast.h"
#include "db/index/column/fts_column/fts_rocksdb_merge.h"
#include "db/index/column/fts_column/fts_rocksdb_reducer.h"
#include "db/index/column/fts_column/fts_types.h"
#include "db/index/column/fts_column/fts_utils.h"
#include "db/index/column/fts_column/posting/bitpacked_posting_list.h"

namespace {

// Helper: build a public FtsIndexParams from FLAGS_extra_params JSON string.
// The JSON may contain a "tokenizer" key that specifies the tokenizer name;
// the remaining JSON is passed through as extra_params verbatim.
static std::shared_ptr<zvec::FtsIndexParams> build_fts_index_params(
    const std::string &extra_params_json) {
  std::string tokenizer_name = "standard";
  zvec::ailego::JsonValue jv;
  if (jv.parse(extra_params_json) && jv.is_object()) {
    const auto &obj = jv.as_object();
    zvec::ailego::JsonValue tok_val = obj["tokenizer"];
    if (tok_val.is_string()) {
      tokenizer_name = tok_val.as_string().as_stl_string();
    }
  }
  return std::make_shared<zvec::FtsIndexParams>(
      std::move(tokenizer_name), std::vector<std::string>{"lowercase"},
      extra_params_json);
}

// Helper: build a transient FieldSchema for FTS field with index params.
static zvec::FieldSchema::Ptr make_fts_field_schema(
    const std::string &field_name,
    std::shared_ptr<zvec::FtsIndexParams> fts_params = nullptr) {
  if (!fts_params) {
    fts_params = std::make_shared<zvec::FtsIndexParams>();
  }
  return std::make_shared<zvec::FieldSchema>(field_name, zvec::DataType::STRING,
                                             false, fts_params);
}

}  // namespace

// ---------------------------------------------------------------------------
// gflags
// ---------------------------------------------------------------------------
DEFINE_string(cmd, "",
              "Command to execute: build, search, stats. "
              "If empty, auto-detect from -corpus / -query flags.");
DEFINE_string(index, "", "Path to FTS index directory");
DEFINE_string(corpus, "", "Path to BEIR corpus.jsonl (build mode)");
DEFINE_string(query, "", "Path to BEIR queries.jsonl (search mode)");
DEFINE_string(qrels, "", "Path to BEIR qrels directory (search mode)");
DEFINE_int32(topk, 10, "Top-K results to retrieve per query");
DEFINE_string(extra_params, R"({"tokenizer":"standard"})",
              "Extra params JSON for tokenizer pipeline");
DEFINE_string(field, "text", "FTS field name");
DEFINE_int32(threads, 16, "Number of threads for multi-threaded search");
DEFINE_int32(max_queries, 0,
             "Maximum number of queries to run in search mode. "
             "0 means all queries (default).");
DEFINE_bool(reduce, false,
            "After build, run FtsRocksdbReducer to convert postings to "
            "BitPacked format. Reduced index is written to <index>-reduce.");
DEFINE_string(default_operator, "or",
              "Default operator used to combine query tokens when searching "
              "match_string-style queries. Valid values: 'or' (union, default) "
              "or 'and' (intersection).");
DEFINE_string(mode, "raw",
              "Execution mode: 'raw' (default) operates directly on RocksDB "
              "via FtsColumnIndexer; 'db' operates through "
              "the zvec Collection API (CreateAndOpen / Insert / Query).");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const std::string kForwardCfName = "forward";

using namespace zvec;
using namespace zvec::fts;

// ---------------------------------------------------------------------------
// Query AST builder: combine tokens with the configured default operator.
// Returns nullptr when tokens is empty.
// ---------------------------------------------------------------------------
template <typename TokenContainer>
static FtsAstNodePtr build_query_ast_from_tokens(
    const TokenContainer &tokens, const std::string &default_operator) {
  if (tokens.empty()) {
    return nullptr;
  }
  if (default_operator == "and") {
    auto and_node = std::make_unique<AndNode>();
    for (const auto &token : tokens) {
      and_node->children.push_back(std::make_unique<TermNode>(token.text));
    }
    return and_node;
  }
  // Default: OR
  auto or_node = std::make_unique<OrNode>();
  for (const auto &token : tokens) {
    or_node->children.push_back(std::make_unique<TermNode>(token.text));
  }
  return or_node;
}

// Validate -default_operator flag value. Returns true if valid.
static bool validate_default_operator(const std::string &op) {
  return op == "or" || op == "and";
}

// ---------------------------------------------------------------------------
// Helper: open RocksdbStore with FTS column families.
//
// `with_side_cfs` controls whether the build-time only side CFs
// ($TF / $MAX_TF / $DOC_LEN) are listed in the open args.  These three CFs
// are dropped at the end of build (after convert_postings_to_bitpacked()
// inlines their payloads into BitPacked postings), mirroring
// MutableSegment::dump_fts_column_indexers().  Search/stats paths therefore
// open the store without them so that the open call doesn't fail with
// "column family not found" against a built index.
// ---------------------------------------------------------------------------
static bool open_fts_store(RocksdbContext *store, const std::string &field_name,
                           bool existing, const std::string &index_path = "",
                           bool with_side_cfs = true,
                           bool with_forward_cf = true) {
  const std::string &data_dir = index_path.empty() ? FLAGS_index : index_path;
  const std::string max_tf_cf = field_name + zvec::kFtsMaxTfSuffix;

  std::vector<std::string> cf_names = {
      field_name,
      field_name + zvec::kFtsPositionsSuffix,
      zvec::kFtsStatCfName,
  };
  if (with_forward_cf) {
    cf_names.push_back(kForwardCfName);
  }
  if (with_side_cfs) {
    cf_names.push_back(field_name + zvec::kFtsTfSuffix);
    cf_names.push_back(max_tf_cf);
    cf_names.push_back(field_name + zvec::kFtsDocLenSuffix);
  }

  // Build per-CF merge operators map
  std::unordered_map<std::string, std::shared_ptr<rocksdb::MergeOperator>>
      per_cf_merge_ops;
  per_cf_merge_ops[field_name] = std::make_shared<FtsPostingsMerge>();
  if (with_side_cfs) {
    per_cf_merge_ops[max_tf_cf] = std::make_shared<FtsMaxTfMerge>();
  }

  Status status;
  if (existing) {
    status = store->open(
        RocksdbContext::Args{data_dir, cf_names, nullptr, per_cf_merge_ops},
        false);
  } else {
    status = store->create(RocksdbContext::Args{data_dir, cf_names, nullptr,
                                                per_cf_merge_ops, true});
  }
  if (!status.ok()) {
    fprintf(stderr, "ERROR: Failed to open RocksdbStore at [%s], status[%s]\n",
            data_dir.c_str(), status.message().c_str());
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Helper: drop $TF / $MAX_TF / $DOC_LEN CFs after convert_postings_to_bitpacked
// has inlined their payloads into BitPacked postings.  Mirrors
// MutableSegment::dump_fts_column_indexers().  The dumped immutable index is
// significantly smaller because these CFs no longer occupy SST space.
// Logs and ignores per-CF failures so that a partial drop (e.g. CF already
// missing on retry) does not abort the whole build.
// ---------------------------------------------------------------------------
static void drop_fts_side_cfs(RocksdbContext *store,
                              const std::string &field_name) {
  const std::vector<std::string> side_cf_names = {
      field_name + zvec::kFtsTfSuffix,
      field_name + zvec::kFtsMaxTfSuffix,
      field_name + zvec::kFtsDocLenSuffix,
  };
  for (const auto &cf_name : side_cf_names) {
    Status drop_status = store->drop_cf(cf_name);
    if (!drop_status.ok()) {
      fprintf(stderr,
              "WARN: Drop column family[%s] failed, status[%s] (ignored)\n",
              cf_name.c_str(), drop_status.message().c_str());
    }
  }
}


// ---------------------------------------------------------------------------
// Helper: parse a JSONL line and extract a string field
// ---------------------------------------------------------------------------
static bool parse_jsonl_line(
    const std::string &line,
    std::unordered_map<std::string, std::string> *out) {
  zvec::ailego::JsonValue jv;
  if (!jv.parse(line) || !jv.is_object()) {
    return false;
  }
  const auto &obj = jv.as_object();
  for (const auto &kv : obj) {
    if (kv.value().is_string()) {
      (*out)[kv.key().as_stl_string()] = kv.value().as_string().as_stl_string();
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Latency statistics helper
// ---------------------------------------------------------------------------
struct LatencyStats {
  std::vector<uint64_t> samples;  // microseconds

  void add(uint64_t us) {
    samples.push_back(us);
  }

  void print(const std::string &label) const {
    if (samples.empty()) {
      std::cout << label << ": no samples" << std::endl;
      return;
    }
    std::vector<uint64_t> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    uint64_t sum = 0;
    for (auto v : sorted) sum += v;
    double avg = static_cast<double>(sum) / sorted.size();

    auto percentile = [&](double p) -> uint64_t {
      size_t idx = static_cast<size_t>(p * sorted.size());
      if (idx >= sorted.size()) idx = sorted.size() - 1;
      return sorted[idx];
    };

    std::cout << label << " latency (us):" << std::endl;
    std::cout << "  Count  : " << sorted.size() << std::endl;
    std::cout << "  Average: " << static_cast<uint64_t>(avg) << std::endl;
    std::cout << "  Min    : " << sorted.front() << std::endl;
    std::cout << "  P50    : " << percentile(0.50) << std::endl;
    std::cout << "  P95    : " << percentile(0.95) << std::endl;
    std::cout << "  P99    : " << percentile(0.99) << std::endl;
    std::cout << "  Max    : " << sorted.back() << std::endl;
  }
};

// ---------------------------------------------------------------------------
// REDUCE MODE: convert Roaring Bitmap postings to BitPacked format
// ---------------------------------------------------------------------------
static int do_reduce(const std::string &src_index_path, uint32_t total_docs) {
  const std::string dst_index_path = src_index_path + "-reduce";
  std::cout << std::endl;
  std::cout << "=== REDUCE MODE ===" << std::endl;
  std::cout << "  Source : " << src_index_path << std::endl;
  std::cout << "  Dest   : " << dst_index_path << std::endl;

  // Create destination directory
  if (!zvec::FileHelper::DirectoryExists(dst_index_path)) {
    if (!zvec::FileHelper::CreateDirectory(dst_index_path)) {
      fprintf(stderr, "ERROR: Failed to create reduce output directory: %s\n",
              dst_index_path.c_str());
      return -1;
    }
  }

  // Open source store (existing).  $TF/$MAX_TF/$DOC_LEN were dropped at
  // build time after convert_postings_to_bitpacked(), so we open without
  // them.  The reducer never consumed these CFs anyway (BitPacked postings
  // already carry inline tf/doc_len/max_score payloads).
  RocksdbContext src_store;
  if (!open_fts_store(&src_store, FLAGS_field, /*existing=*/true,
                      src_index_path, /*with_side_cfs=*/false)) {
    fprintf(stderr, "ERROR: Failed to open source store for reduce\n");
    return -1;
  }

  // Open destination store (new) — same shape as a freshly-dumped immutable
  // index: no side CFs.
  RocksdbContext dst_store;
  if (!open_fts_store(&dst_store, FLAGS_field, /*existing=*/false,
                      dst_index_path, /*with_side_cfs=*/false)) {
    fprintf(stderr, "ERROR: Failed to open destination store for reduce\n");
    src_store.close();
    return -1;
  }

  // Get source column families
  rocksdb::ColumnFamilyHandle *src_postings = src_store.get_cf(FLAGS_field);
  rocksdb::ColumnFamilyHandle *src_positions =
      src_store.get_cf(FLAGS_field + zvec::kFtsPositionsSuffix);
  rocksdb::ColumnFamilyHandle *src_stat =
      src_store.get_cf(zvec::kFtsStatCfName);
  rocksdb::ColumnFamilyHandle *src_forward = src_store.get_cf(kForwardCfName);

  // Get destination column families
  rocksdb::ColumnFamilyHandle *dst_postings = dst_store.get_cf(FLAGS_field);
  rocksdb::ColumnFamilyHandle *dst_positions =
      dst_store.get_cf(FLAGS_field + zvec::kFtsPositionsSuffix);
  rocksdb::ColumnFamilyHandle *dst_stat =
      dst_store.get_cf(zvec::kFtsStatCfName);
  rocksdb::ColumnFamilyHandle *dst_forward = dst_store.get_cf(kForwardCfName);

  if (!src_postings || !src_positions || !src_stat || !dst_postings ||
      !dst_positions || !dst_stat) {
    fprintf(stderr, "ERROR: Failed to get column families for reduce\n");
    src_store.close();
    dst_store.close();
    return -1;
  }

  zvec::ailego::ElapsedTime reduce_timer;

  // Initialize reducer.  Side CFs ($TF/$MAX_TF/$DOC_LEN) are no longer
  // consumed by the reducer; they remain in the schema for SST compatibility
  // but the bench tool does not need to wire them in.
  FtsRocksdbReducer reducer;
  auto init_result = reducer.init(FLAGS_field, &dst_store, dst_postings,
                                  dst_positions, dst_stat);
  if (!init_result.has_value()) {
    fprintf(stderr, "ERROR: FtsRocksdbReducer init failed, status[%s]\n",
            init_result.error().message().c_str());
    src_store.close();
    dst_store.close();
    return -1;
  }

  // Feed source as a single segment: doc_id range [0, total_docs-1]
  FtsSegmentStats seg_stats;
  seg_stats.min_doc_id = 0;
  seg_stats.max_doc_id = total_docs > 0 ? total_docs - 1 : 0;

  auto feed_result =
      reducer.feed(seg_stats, &src_store, src_postings, src_positions);
  if (!feed_result.has_value()) {
    fprintf(stderr, "ERROR: FtsRocksdbReducer feed failed, status[%s]\n",
            feed_result.error().message().c_str());
    src_store.close();
    dst_store.close();
    return -1;
  }

  // Run reduce with no-delete filter (empty bitmap = nothing deleted).
  std::cout << "  Running reduce..." << std::endl;
  roaring::Roaring no_delete_bitmap;
  auto reduce_result = reducer.reduce(no_delete_bitmap);
  if (!reduce_result.has_value()) {
    fprintf(stderr, "ERROR: FtsRocksdbReducer reduce failed, status[%s]\n",
            reduce_result.error().message().c_str());
    src_store.close();
    dst_store.close();
    return -1;
  }

  // Copy forward CF (doc_id -> corpus_id mapping)
  if (src_forward && dst_forward) {
    std::cout << "  Copying forward CF..." << std::endl;
    auto iter = std::unique_ptr<rocksdb::Iterator>(
        src_store.db_->NewIterator(src_store.read_opts_, src_forward));
    while (iter->Valid()) {
      dst_store.db_->Put(dst_store.write_opts_, dst_forward,
                         iter->key().ToString(), iter->value().ToString());
      iter->Next();
    }
    // iter auto-closes via unique_ptr
  }

  // Flush and compact destination.  Side CFs are not present here.
  dst_store.flush();
  // compact not available in RocksdbContext


  uint64_t reduce_ms = reduce_timer.milli_seconds();

  std::cout << "=== REDUCE COMPLETE ===" << std::endl;
  std::cout << "  Reduce time : " << reduce_ms << " ms" << std::endl;
  std::cout << "  Output path : " << dst_index_path << std::endl;

  (void)reducer.cleanup();
  src_store.close();
  dst_store.close();
  return 0;
}


struct CorpusEntry {
  uint32_t doc_id;
  std::string corpus_id;
  std::string content;
};

static int do_build() {
  const int num_threads = std::max(1, FLAGS_threads);
  std::cout << "=== BUILD MODE ===" << std::endl;
  std::cout << "Index  : " << FLAGS_index << std::endl;
  std::cout << "Corpus : " << FLAGS_corpus << std::endl;
  std::cout << "Field  : " << FLAGS_field << std::endl;
  std::cout << "Threads: " << num_threads << std::endl;
  std::cout << "ExtraParams: " << FLAGS_extra_params << std::endl;

  // Remove existing index directory so that RocksdbContext::create() starts
  // fresh (it requires the path to NOT exist).
  if (zvec::FileHelper::DirectoryExists(FLAGS_index)) {
    std::cout << "Removing existing index directory: " << FLAGS_index
              << std::endl;
    zvec::FileHelper::RemoveDirectory(FLAGS_index);
  }

  // Open RocksDB (new)
  RocksdbContext store;
  if (!open_fts_store(&store, FLAGS_field, /*existing=*/false)) {
    return -1;
  }

  // Get column families
  const std::string max_tf_cf_name = FLAGS_field + zvec::kFtsMaxTfSuffix;

  rocksdb::ColumnFamilyHandle *postings_cf = store.get_cf(FLAGS_field);
  rocksdb::ColumnFamilyHandle *positions_cf =
      store.get_cf(FLAGS_field + zvec::kFtsPositionsSuffix);
  rocksdb::ColumnFamilyHandle *term_freq_cf =
      store.get_cf(FLAGS_field + zvec::kFtsTfSuffix);
  rocksdb::ColumnFamilyHandle *max_tf_cf = store.get_cf(max_tf_cf_name);
  rocksdb::ColumnFamilyHandle *doc_len_cf =
      store.get_cf(FLAGS_field + zvec::kFtsDocLenSuffix);
  rocksdb::ColumnFamilyHandle *stat_cf = store.get_cf(zvec::kFtsStatCfName);
  rocksdb::ColumnFamilyHandle *forward_cf = store.get_cf(kForwardCfName);

  if (!postings_cf || !positions_cf || !term_freq_cf || !max_tf_cf ||
      !doc_len_cf || !stat_cf || !forward_cf) {
    fprintf(stderr, "ERROR: Failed to get column families\n");
    return -1;
  }

  // Pre-load all corpus entries into memory with pre-assigned doc_ids
  std::vector<CorpusEntry> corpus_entries;
  uint64_t parse_failed_count = 0;
  {
    std::ifstream corpus_file(FLAGS_corpus);
    if (!corpus_file.is_open()) {
      fprintf(stderr, "ERROR: Failed to open corpus file: %s\n",
              FLAGS_corpus.c_str());
      return -1;
    }

    uint32_t doc_id = 0;
    std::string line;
    while (std::getline(corpus_file, line)) {
      if (line.empty()) continue;

      std::unordered_map<std::string, std::string> fields;
      if (!parse_jsonl_line(line, &fields)) {
        fprintf(stderr, "WARN: Failed to parse line: %s\n",
                line.substr(0, 100).c_str());
        ++parse_failed_count;
        continue;
      }

      const std::string &corpus_id = fields["_id"];
      if (corpus_id.empty()) {
        ++parse_failed_count;
        continue;
      }

      std::string content;
      if (!fields["title"].empty()) {
        content = fields["title"] + " " + fields["text"];
      } else {
        content = fields["text"];
      }

      corpus_entries.push_back(
          {doc_id, std::move(corpus_id), std::move(content)});
      ++doc_id;
    }
  }
  std::cout << "Loaded " << corpus_entries.size() << " corpus entries."
            << std::endl;
  if (parse_failed_count > 0) {
    std::cout << "  Warning: " << parse_failed_count
              << " entries failed to parse." << std::endl;
  }

  auto fts_params = build_fts_index_params(FLAGS_extra_params);
  auto field_meta = make_fts_field_schema(FLAGS_field, fts_params);

  FtsColumnIndexer indexer;
  auto open_result = indexer.open(field_meta, &store, postings_cf, positions_cf,
                                  term_freq_cf, max_tf_cf, doc_len_cf, stat_cf);
  if (!open_result.has_value()) {
    fprintf(stderr, "ERROR: Failed to open FtsColumnIndexer, status[%s]\n",
            open_result.error().message().c_str());
    return -1;
  }

  // Shared atomic index for work-stealing across threads
  std::atomic<size_t> next_entry_index{0};

  // Per-thread result accumulators
  struct ThreadResult {
    uint64_t indexed_count{0};
    uint64_t failed_count{0};
  };
  std::vector<ThreadResult> thread_results(num_threads);

  std::cout << "Building index with " << num_threads << " thread(s)..."
            << std::endl;

  zvec::ailego::ElapsedTime timer;

  auto worker = [&](int thread_id) {
    ThreadResult &result = thread_results[thread_id];

    while (true) {
      size_t entry_idx =
          next_entry_index.fetch_add(1, std::memory_order_relaxed);
      if (entry_idx >= corpus_entries.size()) break;

      const CorpusEntry &entry = corpus_entries[entry_idx];

      auto insert_result = indexer.insert(entry.doc_id, entry.content);
      if (!insert_result.has_value()) {
        fprintf(stderr,
                "WARN: Thread[%d] failed to insert doc_id[%u] corpus_id[%s], "
                "status[%s]\n",
                thread_id, entry.doc_id, entry.corpus_id.c_str(),
                insert_result.error().message().c_str());
        ++result.failed_count;
        continue;
      }

      // Write forward mapping: doc_id -> corpus_id
      std::string doc_id_key;
      fts::encode_uint32_big_endian(entry.doc_id, &doc_id_key);
      store.db_->Put(store.write_opts_, forward_cf, doc_id_key,
                     entry.corpus_id);

      ++result.indexed_count;

      // Progress reporting (only from thread 0 to avoid interleaving)
      if (thread_id == 0 && result.indexed_count % 1000 == 0) {
        size_t total_done = 0;
        for (const auto &tr : thread_results) {
          total_done += tr.indexed_count + tr.failed_count;
        }
        std::cout << "\r  Indexed ~" << total_done << " / "
                  << corpus_entries.size() << " docs..." << std::flush;
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back(worker, thread_id);
  }
  for (auto &thread : threads) {
    thread.join();
  }

  uint64_t build_ms = timer.milli_seconds();

  // Merge per-thread results
  uint64_t total_indexed = 0;
  uint64_t total_failed = 0;
  for (const auto &result : thread_results) {
    total_indexed += result.indexed_count;
    total_failed += result.failed_count;
  }

  std::cout << "\r  Indexed " << total_indexed << " docs total." << std::endl;
  if (total_failed > 0) {
    std::cout << "  Warning: " << total_failed << " docs failed to index."
              << std::endl;
  }

  // Flush statistics — single indexer tracks all docs/tokens atomically
  std::cout << "Flushing statistics (total_docs=" << indexer.total_docs()
            << ", total_tokens=" << indexer.total_tokens() << ")..."
            << std::endl;
  auto flush_result = indexer.flush();
  if (!flush_result.has_value()) {
    fprintf(stderr, "WARN: FtsColumnIndexer flush failed, status[%s]\n",
            flush_result.error().message().c_str());
  }

  // Convert Roaring postings to BitPacked before close/dump, mirroring
  // MutableSegment::dump_fts_column_indexers().  Must run before close()
  // for symmetry with the single-threaded path; convert itself does not
  // depend on the tokenizer pipeline.
  std::cout << "Converting postings to BitPacked..." << std::endl;
  zvec::ailego::ElapsedTime bitpacked_timer2;
  auto bitpacked_result = indexer.convert_postings_to_bitpacked();
  if (!bitpacked_result.has_value()) {
    fprintf(stderr,
            "WARN: FtsColumnIndexer convert_postings_to_bitpacked failed, "
            "status[%s]\n",
            bitpacked_result.error().message().c_str());
  }
  std::cout << "convert_postings_to_bitpacked took "
            << bitpacked_timer2.micro_seconds() / 1000.0 << " ms" << std::endl;

  // Drop $TF / $MAX_TF / $DOC_LEN CFs after their payloads have been inlined
  // into BitPacked postings.  Mirrors MutableSegment::dump_fts_column_
  // indexers(): reset_side_cfs() first so any concurrent reader-path access
  // through the indexer falls back to default tf=1/doc_len=1 instead of
  // touching a dropped handle, then drop the CFs from the underlying store.
  indexer.reset_side_cfs();
  drop_fts_side_cfs(&store, FLAGS_field);
  // Local pointers are now dangling; null them out so accidental use becomes
  // an obvious crash instead of a use-after-free.
  term_freq_cf = nullptr;
  max_tf_cf = nullptr;
  doc_len_cf = nullptr;

  (void)indexer.close();

  // Flush RocksDB memtables + dump checkpoint
  zvec::ailego::ElapsedTime dump_timer;
  store.flush();

  // Trigger compaction + checkpoint
  std::cout << "Running compaction..." << std::endl;
  store.compact();

  uint64_t dump_ms = dump_timer.milli_seconds();
  uint64_t elapsed_ms = timer.milli_seconds();
  std::cout << "=== BUILD COMPLETE ===" << std::endl;
  std::cout << "  Total docs  : " << total_indexed << std::endl;
  std::cout << "  Threads     : " << num_threads << std::endl;
  std::cout << "  Build time  : " << build_ms << " ms" << std::endl;
  std::cout << "  Dump time   : " << dump_ms << " ms (flush + compaction)"
            << std::endl;
  std::cout << "  Total time  : " << elapsed_ms << " ms" << std::endl;
  std::cout << "  Throughput  : "
            << (total_indexed > 0
                    ? total_indexed * 1000ULL / (build_ms > 0 ? build_ms : 1)
                    : 0)
            << " docs/s (build only)" << std::endl;

  store.close();

  // Optional: run reduce to convert postings to BitPacked format
  if (FLAGS_reduce) {
    int reduce_ret = do_reduce(FLAGS_index, total_indexed);
    if (reduce_ret != 0) {
      fprintf(stderr, "ERROR: Reduce step failed, ret[%d]\n", reduce_ret);
      return reduce_ret;
    }
  }

  return 0;
}

// ---------------------------------------------------------------------------
// BUILD MODE (db): use zvec Collection API
// ---------------------------------------------------------------------------
static int do_build_db() {
  const int num_threads = std::max(1, FLAGS_threads);
  std::cout << "=== BUILD MODE (db) ===" << std::endl;
  std::cout << "Index  : " << FLAGS_index << std::endl;
  std::cout << "Corpus : " << FLAGS_corpus << std::endl;
  std::cout << "Field  : " << FLAGS_field << std::endl;
  std::cout << "Threads: " << num_threads << std::endl;

  // Remove existing collection directory
  if (zvec::FileHelper::DirectoryExists(FLAGS_index)) {
    std::cout << "Removing existing collection directory: " << FLAGS_index
              << std::endl;
    zvec::FileHelper::RemoveDirectory(FLAGS_index);
  }

  // Build schema: pk (implicit) + FTS field.
  // Build FtsIndexParams from FLAGS_extra_params so that the tokenizer
  // pipeline configuration (e.g. enable_simple_closet) matches raw mode.
  auto db_fts_params = build_fts_index_params(FLAGS_extra_params);

  CollectionSchema schema("fts_bench");
  schema.add_field(std::make_shared<FieldSchema>(FLAGS_field, DataType::STRING,
                                                 false, db_fts_params));

  CollectionOptions options;
  options.read_only_ = false;

  auto create_result = Collection::CreateAndOpen(FLAGS_index, schema, options);
  if (!create_result.has_value()) {
    fprintf(stderr, "ERROR: Failed to create collection at [%s]: %s\n",
            FLAGS_index.c_str(), create_result.error().message().c_str());
    return -1;
  }
  auto collection = create_result.value();

  // Pre-load corpus entries
  std::vector<CorpusEntry> corpus_entries;
  uint64_t parse_failed_count = 0;
  {
    std::ifstream corpus_file(FLAGS_corpus);
    if (!corpus_file.is_open()) {
      fprintf(stderr, "ERROR: Failed to open corpus file: %s\n",
              FLAGS_corpus.c_str());
      return -1;
    }
    uint32_t doc_id = 0;
    std::string line;
    while (std::getline(corpus_file, line)) {
      if (line.empty()) continue;
      std::unordered_map<std::string, std::string> fields;
      if (!parse_jsonl_line(line, &fields)) {
        ++parse_failed_count;
        continue;
      }
      const std::string &corpus_id = fields["_id"];
      if (corpus_id.empty()) {
        ++parse_failed_count;
        continue;
      }
      std::string content;
      if (!fields["title"].empty()) {
        content = fields["title"] + " " + fields["text"];
      } else {
        content = fields["text"];
      }
      corpus_entries.push_back(
          {doc_id, std::move(corpus_id), std::move(content)});
      ++doc_id;
    }
  }
  std::cout << "Loaded " << corpus_entries.size() << " corpus entries."
            << std::endl;
  if (parse_failed_count > 0) {
    std::cout << "  Warning: " << parse_failed_count
              << " entries failed to parse." << std::endl;
  }

  // Insert in batches via Collection::Insert
  const size_t batch_size = 1000;
  uint64_t total_indexed = 0;
  uint64_t total_failed = 0;

  std::cout << "Inserting documents via Collection API..." << std::endl;
  zvec::ailego::ElapsedTime timer;

  for (size_t offset = 0; offset < corpus_entries.size();
       offset += batch_size) {
    size_t end = std::min(offset + batch_size, corpus_entries.size());
    std::vector<Doc> docs;
    docs.reserve(end - offset);
    for (size_t i = offset; i < end; ++i) {
      const CorpusEntry &entry = corpus_entries[i];
      Doc doc;
      doc.set_pk(entry.corpus_id);
      doc.set<std::string>(FLAGS_field, entry.content);
      docs.push_back(std::move(doc));
    }
    auto insert_result = collection->Insert(docs);
    if (!insert_result.has_value()) {
      fprintf(stderr, "WARN: Batch insert failed at offset[%zu]: %s\n", offset,
              insert_result.error().message().c_str());
      total_failed += (end - offset);
    } else {
      total_indexed += (end - offset);
    }
    if (total_indexed % 10000 < batch_size) {
      std::cout << "\r  Inserted " << total_indexed << " / "
                << corpus_entries.size() << " docs..." << std::flush;
    }
  }

  uint64_t build_ms = timer.milli_seconds();

  // Flush collection
  auto flush_status = collection->Flush();
  if (!flush_status.ok()) {
    fprintf(stderr, "WARN: Collection flush failed: %s\n",
            flush_status.message().c_str());
  }

  // Optimize triggers segment dump which converts Roaring postings to
  // BitPacked format (with inline tf/doc_len payloads).  Without this step
  // the immutable reader path falls back to tf=1/doc_len=1 because the
  // side CFs (_tf/_doc_len/_max_tf) are not opened for read-only segments.
  auto optimize_status = collection->Optimize();
  if (!optimize_status.ok()) {
    fprintf(stderr, "WARN: Collection optimize failed: %s\n",
            optimize_status.message().c_str());
  }

  std::cout << "\r  Inserted " << total_indexed << " docs total." << std::endl;
  if (total_failed > 0) {
    std::cout << "  Warning: " << total_failed << " docs failed to insert."
              << std::endl;
  }
  std::cout << "=== BUILD COMPLETE (db) ===" << std::endl;
  std::cout << "  Total docs  : " << total_indexed << std::endl;
  std::cout << "  Build time  : " << build_ms << " ms" << std::endl;
  std::cout << "  Throughput  : "
            << (total_indexed > 0
                    ? total_indexed * 1000ULL / (build_ms > 0 ? build_ms : 1)
                    : 0)
            << " docs/s" << std::endl;

  return 0;
}

// ---------------------------------------------------------------------------
// SEARCH MODE
// ---------------------------------------------------------------------------

// Parse qrels TSV file: returns map of query_id -> set<corpus_id>
static std::unordered_map<std::string, std::unordered_set<std::string>>
load_qrels(const std::string &qrels_dir) {
  std::unordered_map<std::string, std::unordered_set<std::string>> qrels;

  // Try test.tsv first, then train.tsv
  std::vector<std::string> candidates = {qrels_dir + "/test.tsv",
                                         qrels_dir + "/train.tsv"};
  std::string qrels_file;
  for (const auto &f : candidates) {
    if (FileHelper::FileExists(f)) {
      qrels_file = f;
      break;
    }
  }

  if (qrels_file.empty()) {
    fprintf(stderr, "ERROR: No qrels file found in directory: %s\n",
            qrels_dir.c_str());
    return qrels;
  }

  std::cout << "Loading qrels from: " << qrels_file << std::endl;

  std::ifstream f(qrels_file);
  if (!f.is_open()) {
    fprintf(stderr, "ERROR: Failed to open qrels file: %s\n",
            qrels_file.c_str());
    return qrels;
  }

  std::string line;
  bool first_line = true;
  while (std::getline(f, line)) {
    if (first_line) {
      first_line = false;
      continue;  // skip header
    }
    if (line.empty()) continue;

    std::istringstream ss(line);
    std::string query_id, corpus_id, score_str;
    if (!std::getline(ss, query_id, '\t') ||
        !std::getline(ss, corpus_id, '\t') ||
        !std::getline(ss, score_str, '\t')) {
      continue;
    }
    // Only include relevant docs (score > 0)
    int score = std::stoi(score_str);
    if (score > 0) {
      qrels[query_id].insert(corpus_id);
    }
  }

  std::cout << "Loaded qrels for " << qrels.size() << " queries." << std::endl;
  return qrels;
}

// ---------------------------------------------------------------------------
// Unified single-/multi-threaded search:
//   * Always pre-loads queries into memory and dispatches them to
//     FLAGS_threads workers via an atomic index counter.
//   * FtsColumnIndexer::search() and the shared TokenizerPipeline are both
//     read-only / fork-safe, so a single shared reader and pipeline are
//     reused across workers.
//   * When FLAGS_threads == 1 the path collapses to a single worker,
//     behaving equivalently to a sequential single-threaded search.
// ---------------------------------------------------------------------------

struct QueryEntry {
  std::string query_id;
  std::string match_text;
};

struct RecallCounter {
  double sum{0.0};
  uint64_t total{0};
  void add(double recall_value) {
    sum += recall_value;
    total++;
  }
  double ratio() const {
    return total > 0 ? sum / static_cast<double>(total) : 0.0;
  }
};


static int do_search() {
  if (!validate_default_operator(FLAGS_default_operator)) {
    fprintf(stderr,
            "ERROR: Invalid -default_operator[%s]. Must be 'or' or 'and'.\n",
            FLAGS_default_operator.c_str());
    return -1;
  }

  const int num_threads = std::max(1, FLAGS_threads);

  const std::string fts_index_path = FLAGS_index;

  std::cout << "=== SEARCH MODE ===" << std::endl;
  std::cout << "Index            : " << fts_index_path << std::endl;
  std::cout << "Query            : " << FLAGS_query << std::endl;
  std::cout << "Qrels            : " << FLAGS_qrels << std::endl;
  std::cout << "TopK             : " << FLAGS_topk << std::endl;
  std::cout << "Field            : " << FLAGS_field << std::endl;
  std::cout << "Threads          : " << num_threads << std::endl;
  std::cout << "Default operator : " << FLAGS_default_operator << std::endl;

  // Open FTS RocksDB (existing) — shared across threads (RocksDB reads are
  // thread-safe at the CF level).  Open without $TF/$MAX_TF/$DOC_LEN since
  // those CFs were dropped at build time after convert_postings_to_bitpacked().
  RocksdbContext store;
  if (!open_fts_store(&store, FLAGS_field, /*existing=*/true,
                      /*index_path=*/fts_index_path,
                      /*with_side_cfs=*/false,
                      /*with_forward_cf=*/true)) {
    return -1;
  }

  rocksdb::ColumnFamilyHandle *postings_cf = store.get_cf(FLAGS_field);
  rocksdb::ColumnFamilyHandle *positions_cf =
      store.get_cf(FLAGS_field + zvec::kFtsPositionsSuffix);
  rocksdb::ColumnFamilyHandle *stat_cf = store.get_cf(zvec::kFtsStatCfName);
  rocksdb::ColumnFamilyHandle *forward_cf = store.get_cf(kForwardCfName);

  if (!postings_cf || !positions_cf || !stat_cf || !forward_cf) {
    fprintf(stderr, "ERROR: Failed to get column families\n");
    return -1;
  }

  // Load qrels
  auto qrels = load_qrels(FLAGS_qrels);

  // Pre-load all queries into memory so threads can access them without I/O
  // contention
  std::vector<QueryEntry> queries;
  {
    std::ifstream query_file(FLAGS_query);
    if (!query_file.is_open()) {
      fprintf(stderr, "ERROR: Failed to open query file: %s\n",
              FLAGS_query.c_str());
      return -1;
    }
    std::string line;
    while (std::getline(query_file, line)) {
      if (line.empty()) continue;
      std::unordered_map<std::string, std::string> fields;
      if (!parse_jsonl_line(line, &fields)) continue;
      const std::string &query_id = fields["_id"];
      const std::string &query_text = fields["text"];
      if (query_id.empty() || query_text.empty()) continue;
      queries.push_back({query_id, query_text});
    }
  }
  std::cout << "Loaded " << queries.size() << " queries." << std::endl;

  // Limit the number of queries if configured: first keep only queries that
  // have qrels entries (relevant results), then truncate to max_queries.
  if (FLAGS_max_queries > 0) {
    std::vector<QueryEntry> filtered;
    for (auto &q : queries) {
      if (qrels.count(q.query_id) > 0) {
        filtered.push_back(std::move(q));
      }
    }
    queries = std::move(filtered);
    if (static_cast<size_t>(FLAGS_max_queries) < queries.size()) {
      queries.resize(FLAGS_max_queries);
    }
    std::cout << "Limited to " << queries.size()
              << " queries with qrels (--max_queries)." << std::endl;
  }

  // Shared atomic index for work-stealing across threads
  std::atomic<size_t> next_query_index{0};

  // Per-thread result accumulators, merged after all threads finish
  struct ThreadResult {
    LatencyStats latency_stats;
    RecallCounter recall1;
    RecallCounter recall5;
    RecallCounter recall10;
    RecallCounter recallK;
    uint64_t no_result_count{0};
    uint64_t query_count{0};
  };
  std::vector<ThreadResult> thread_results(num_threads);

  auto query_fts_params = build_fts_index_params(FLAGS_extra_params);
  auto pipeline_result = zvec::detail::AcquireFtsPipeline(*query_fts_params);
  if (!pipeline_result.has_value()) {
    fprintf(stderr,
            "ERROR: Failed to create tokenizer pipeline for "
            "extra_params[%s]: %s\n",
            FLAGS_extra_params.c_str(),
            pipeline_result.error().message().c_str());
    return -1;
  }
  auto &query_pipeline = pipeline_result.value();

  std::cout << "Running queries with " << num_threads << " thread(s)..."
            << std::endl;

  // Create a single shared FtsColumnIndexer in read-only mode. search() is a
  // const method that only performs read-only RocksDB lookups, so it is safe
  // to share across threads.
  FtsColumnIndexer reader;
  {
    // $TF/$MAX_TF/$DOC_LEN are dropped at build time; pass nullptr — the
    // BitPacked path doesn't need them and the Roaring fallback degrades
    // to default tf=1/doc_len=1 when these pointers are null.
    auto open_result =
        reader.open_reader(FLAGS_field, &store, postings_cf, positions_cf,
                           /*term_freq_cf=*/nullptr,
                           /*max_tf_cf=*/nullptr,
                           /*doc_len_cf=*/nullptr, stat_cf);
    if (!open_result.has_value()) {
      fprintf(stderr, "ERROR: Failed to open FtsColumnIndexer, status[%s]\n",
              open_result.error().message().c_str());
      return -1;
    }
  }

  auto worker = [&](int thread_id) {
    ThreadResult &result = thread_results[thread_id];

    while (true) {
      size_t query_idx =
          next_query_index.fetch_add(1, std::memory_order_relaxed);
      if (query_idx >= queries.size()) break;

      const QueryEntry &entry = queries[query_idx];

      std::vector<FtsResult> results;
      bool search_ok = true;
      uint64_t elapsed_us = 0;
      {
        zvec::ailego::ElapsedTime timer;
        // Tokenize query text (match_string style: tokenize then build AST
        // combining tokens with the configured default operator).
        auto tokens = query_pipeline->process(entry.match_text);
        auto ast_root =
            build_query_ast_from_tokens(tokens, FLAGS_default_operator);
        if (ast_root) {
          fts::FtsQueryParams query_params;
          query_params.topk = static_cast<uint32_t>(FLAGS_topk);
          auto search_result = reader.search(*ast_root, query_params);
          if (!search_result.has_value()) {
            fprintf(stderr,
                    "WARN: Thread[%d] search failed for query_id[%s], "
                    "status[%s]\n",
                    thread_id, entry.query_id.c_str(),
                    search_result.error().message().c_str());
            search_ok = false;
          } else {
            results = std::move(search_result.value());
          }
        }
        elapsed_us = timer.micro_seconds();
      }

      if (!search_ok) {
        continue;
      }

      result.latency_stats.add(elapsed_us);
      ++result.query_count;

      if (results.empty()) {
        ++result.no_result_count;
      }

      // Resolve doc_id -> corpus_id (a.k.a. pk) via the forward CF.
      std::vector<std::string> retrieved_corpus_ids;
      retrieved_corpus_ids.reserve(results.size());
      for (const auto &r : results) {
        std::string corpus_id;
        std::string doc_id_key;
        fts::encode_uint32_big_endian(r.doc_id, &doc_id_key);
        if (!store.db_
                 ->Get(store.read_opts_, forward_cf, doc_id_key, &corpus_id)
                 .ok()) {
          corpus_id = "";
        }
        retrieved_corpus_ids.push_back(corpus_id);
      }

      // Compute recall at various cutoffs
      const auto qrels_it = qrels.find(entry.query_id);
      if (qrels_it == qrels.end()) continue;

      const auto &relevant = qrels_it->second;

      // Standard IR Recall@K = |retrieved_topK ∩ relevant| / |relevant|
      auto compute_recall = [&](int cutoff) -> double {
        if (relevant.empty()) return 0.0;
        int limit =
            std::min(cutoff, static_cast<int>(retrieved_corpus_ids.size()));
        int hits = 0;
        for (int i = 0; i < limit; ++i) {
          if (relevant.count(retrieved_corpus_ids[i]) > 0) {
            hits++;
          }
        }
        return static_cast<double>(hits) / static_cast<double>(relevant.size());
      };

      result.recall1.add(compute_recall(1));
      result.recall5.add(compute_recall(5));
      result.recall10.add(compute_recall(10));
      result.recallK.add(compute_recall(FLAGS_topk));
    }
  };

  // Launch threads and measure total wall-clock time
  auto wall_start = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back(worker, thread_id);
  }
  for (auto &thread : threads) {
    thread.join();
  }

  auto wall_end = std::chrono::steady_clock::now();
  uint64_t wall_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(wall_end -
                                                            wall_start)
          .count());

  // Merge per-thread results
  LatencyStats merged_latency;
  RecallCounter merged_recall1, merged_recall5, merged_recall10, merged_recallK;
  uint64_t total_query_count = 0;
  uint64_t total_no_result_count = 0;

  for (const auto &result : thread_results) {
    for (uint64_t sample : result.latency_stats.samples) {
      merged_latency.add(sample);
    }
    merged_recall1.sum += result.recall1.sum;
    merged_recall1.total += result.recall1.total;
    merged_recall5.sum += result.recall5.sum;
    merged_recall5.total += result.recall5.total;
    merged_recall10.sum += result.recall10.sum;
    merged_recall10.total += result.recall10.total;
    merged_recallK.sum += result.recallK.sum;
    merged_recallK.total += result.recallK.total;
    total_query_count += result.query_count;
    total_no_result_count += result.no_result_count;
  }

  // Output results
  std::cout << std::endl;
  std::cout << "=== SEARCH RESULTS ===" << std::endl;
  std::cout << "Threads          : " << num_threads << std::endl;
  std::cout << "Total queries    : " << total_query_count << std::endl;
  std::cout << "No-result queries: " << total_no_result_count << std::endl;
  std::cout << "Wall-clock time  : " << wall_ms << " ms" << std::endl;
  if (wall_ms > 0) {
    std::cout << "Throughput       : " << total_query_count * 1000ULL / wall_ms
              << " queries/s" << std::endl;
  }
  std::cout << std::endl;

  merged_latency.print("Search (per-query)");
  std::cout << std::endl;

  if (merged_recall1.total > 0) {
    std::cout << "=== RECALL ===" << std::endl;
    std::cout << "  Recall@1  : " << merged_recall1.ratio() << " (evaluated on "
              << merged_recall1.total << " queries)" << std::endl;
    std::cout << "  Recall@5  : " << merged_recall5.ratio() << " (evaluated on "
              << merged_recall5.total << " queries)" << std::endl;
    std::cout << "  Recall@10 : " << merged_recall10.ratio()
              << " (evaluated on " << merged_recall10.total << " queries)"
              << std::endl;
    if (FLAGS_topk > 10) {
      std::cout << "  Recall@" << FLAGS_topk << " : " << merged_recallK.ratio()
                << " (evaluated on " << merged_recallK.total << " queries)"
                << std::endl;
    }
  } else {
    std::cout << "No qrels matched for evaluated queries." << std::endl;
  }

  store.close();
  return 0;
}

// ---------------------------------------------------------------------------
// SEARCH MODE (db): use zvec Collection::Query(Fts)
// ---------------------------------------------------------------------------
static int do_search_db() {
  const int num_threads = std::max(1, FLAGS_threads);

  std::cout << "=== SEARCH MODE (db) ===" << std::endl;
  std::cout << "Index            : " << FLAGS_index << std::endl;
  std::cout << "Query            : " << FLAGS_query << std::endl;
  std::cout << "Qrels            : " << FLAGS_qrels << std::endl;
  std::cout << "TopK             : " << FLAGS_topk << std::endl;
  std::cout << "Field            : " << FLAGS_field << std::endl;
  std::cout << "Threads          : " << num_threads << std::endl;

  // Open existing collection in read-only mode
  CollectionOptions options;
  options.read_only_ = true;

  auto open_result = Collection::Open(FLAGS_index, options);
  if (!open_result.has_value()) {
    fprintf(stderr, "ERROR: Failed to open collection at [%s]: %s\n",
            FLAGS_index.c_str(), open_result.error().message().c_str());
    return -1;
  }
  auto collection = open_result.value();

  // Load qrels
  auto qrels = load_qrels(FLAGS_qrels);

  // Pre-load queries
  std::vector<QueryEntry> queries;
  {
    std::ifstream query_file(FLAGS_query);
    if (!query_file.is_open()) {
      fprintf(stderr, "ERROR: Failed to open query file: %s\n",
              FLAGS_query.c_str());
      return -1;
    }
    std::string line;
    while (std::getline(query_file, line)) {
      if (line.empty()) continue;
      std::unordered_map<std::string, std::string> fields;
      if (!parse_jsonl_line(line, &fields)) continue;
      const std::string &query_id = fields["_id"];
      const std::string &query_text = fields["text"];
      if (query_id.empty() || query_text.empty()) continue;
      queries.push_back({query_id, query_text});
    }
  }
  std::cout << "Loaded " << queries.size() << " queries." << std::endl;

  // Limit the number of queries if configured: first keep only queries that
  // have qrels entries (relevant results), then truncate to max_queries.
  if (FLAGS_max_queries > 0) {
    std::vector<QueryEntry> filtered;
    for (auto &q : queries) {
      if (qrels.count(q.query_id) > 0) {
        filtered.push_back(std::move(q));
      }
    }
    queries = std::move(filtered);
    if (static_cast<size_t>(FLAGS_max_queries) < queries.size()) {
      queries.resize(FLAGS_max_queries);
    }
    std::cout << "Limited to " << queries.size()
              << " queries with qrels (--max_queries)." << std::endl;
  }

  // Per-thread result accumulators
  std::atomic<size_t> next_query_index{0};
  std::atomic<bool> fatal_error{false};

  struct ThreadResult {
    LatencyStats latency_stats;
    RecallCounter recall1;
    RecallCounter recall5;
    RecallCounter recall10;
    RecallCounter recallK;
    uint64_t no_result_count{0};
    uint64_t query_count{0};
  };
  std::vector<ThreadResult> thread_results(num_threads);

  std::cout << "Running queries via Collection API with " << num_threads
            << " thread(s)..." << std::endl;

  auto worker = [&](int thread_id) {
    ThreadResult &result = thread_results[thread_id];

    while (true) {
      if (fatal_error.load(std::memory_order_relaxed)) break;
      size_t query_idx =
          next_query_index.fetch_add(1, std::memory_order_relaxed);
      if (query_idx >= queries.size()) break;

      const QueryEntry &entry = queries[query_idx];

      SearchQuery vq;
      vq.target_.field_name_ = FLAGS_field;
      vq.topk_ = FLAGS_topk;
      FtsClause fts;
      fts.match_string_ = entry.match_text;
      vq.target_.clause_ = fts;

      uint64_t elapsed_us = 0;
      std::vector<std::string> retrieved_corpus_ids;
      {
        zvec::ailego::ElapsedTime query_timer;
        auto query_result = collection->Query(vq);
        elapsed_us = query_timer.micro_seconds();

        if (query_result.has_value()) {
          const auto &doc_list = query_result.value();
          retrieved_corpus_ids.reserve(doc_list.size());
          for (const auto &doc_ptr : doc_list) {
            retrieved_corpus_ids.push_back(doc_ptr->pk());
          }
        } else {
          fprintf(stderr, "ERROR: Thread[%d] Fts failed for query_id[%s]: %s\n",
                  thread_id, entry.query_id.c_str(),
                  query_result.error().message().c_str());
          fatal_error.store(true, std::memory_order_relaxed);
          break;
        }
      }

      result.latency_stats.add(elapsed_us);
      ++result.query_count;

      if (retrieved_corpus_ids.empty()) {
        ++result.no_result_count;
      }

      // Compute recall
      const auto qrels_it = qrels.find(entry.query_id);
      if (qrels_it == qrels.end()) continue;
      const auto &relevant = qrels_it->second;

      auto compute_recall = [&](int cutoff) -> double {
        if (relevant.empty()) return 0.0;
        int limit =
            std::min(cutoff, static_cast<int>(retrieved_corpus_ids.size()));
        int hits = 0;
        for (int i = 0; i < limit; ++i) {
          if (relevant.count(retrieved_corpus_ids[i]) > 0) {
            hits++;
          }
        }
        return static_cast<double>(hits) / static_cast<double>(relevant.size());
      };

      result.recall1.add(compute_recall(1));
      result.recall5.add(compute_recall(5));
      result.recall10.add(compute_recall(10));
      result.recallK.add(compute_recall(FLAGS_topk));
    }
  };

  auto wall_start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back(worker, thread_id);
  }
  for (auto &thread : threads) {
    thread.join();
  }

  if (fatal_error.load()) {
    fprintf(stderr, "ERROR: Aborting: Fts failed during search\n");
    return -1;
  }

  auto wall_end = std::chrono::steady_clock::now();
  uint64_t wall_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(wall_end -
                                                            wall_start)
          .count());

  // Merge per-thread results
  LatencyStats merged_latency;
  RecallCounter merged_recall1, merged_recall5, merged_recall10, merged_recallK;
  uint64_t total_query_count = 0;
  uint64_t total_no_result_count = 0;

  for (const auto &result : thread_results) {
    for (uint64_t sample : result.latency_stats.samples) {
      merged_latency.add(sample);
    }
    merged_recall1.sum += result.recall1.sum;
    merged_recall1.total += result.recall1.total;
    merged_recall5.sum += result.recall5.sum;
    merged_recall5.total += result.recall5.total;
    merged_recall10.sum += result.recall10.sum;
    merged_recall10.total += result.recall10.total;
    merged_recallK.sum += result.recallK.sum;
    merged_recallK.total += result.recallK.total;
    total_query_count += result.query_count;
    total_no_result_count += result.no_result_count;
  }

  std::cout << std::endl;
  std::cout << "=== SEARCH RESULTS (db) ===" << std::endl;
  std::cout << "Threads          : " << num_threads << std::endl;
  std::cout << "Total queries    : " << total_query_count << std::endl;
  std::cout << "No-result queries: " << total_no_result_count << std::endl;
  std::cout << "Wall-clock time  : " << wall_ms << " ms" << std::endl;
  if (wall_ms > 0) {
    std::cout << "Throughput       : " << total_query_count * 1000ULL / wall_ms
              << " queries/s" << std::endl;
  }
  std::cout << std::endl;

  merged_latency.print("Search (per-query)");
  std::cout << std::endl;

  if (merged_recall1.total > 0) {
    std::cout << "=== RECALL ===" << std::endl;
    std::cout << "  Recall@1  : " << merged_recall1.ratio() << " (evaluated on "
              << merged_recall1.total << " queries)" << std::endl;
    std::cout << "  Recall@5  : " << merged_recall5.ratio() << " (evaluated on "
              << merged_recall5.total << " queries)" << std::endl;
    std::cout << "  Recall@10 : " << merged_recall10.ratio()
              << " (evaluated on " << merged_recall10.total << " queries)"
              << std::endl;
    if (FLAGS_topk > 10) {
      std::cout << "  Recall@" << FLAGS_topk << " : " << merged_recallK.ratio()
                << " (evaluated on " << merged_recallK.total << " queries)"
                << std::endl;
    }
  } else {
    std::cout << "No qrels matched for evaluated queries." << std::endl;
  }

  return 0;
}

// ---------------------------------------------------------------------------
// STATS MODE
// ---------------------------------------------------------------------------
static int do_stats() {
  std::cout << "=== STATS MODE ===" << std::endl;
  std::cout << "Index : " << FLAGS_index << std::endl;
  std::cout << "Field : " << FLAGS_field << std::endl;

  // Open RocksDB (existing).  $TF/$MAX_TF/$DOC_LEN are dropped at build
  // time, so open without them.  Sections that scan these CFs are now
  // gated on the corresponding pointers being non-null (always null here
  // post-drop) and simply skipped with an explanatory message.
  RocksdbContext store;
  if (!open_fts_store(&store, FLAGS_field, /*existing=*/true,
                      /*index_path=*/"", /*with_side_cfs=*/false)) {
    return -1;
  }

  rocksdb::ColumnFamilyHandle *postings_cf = store.get_cf(FLAGS_field);
  rocksdb::ColumnFamilyHandle *stat_cf = store.get_cf(zvec::kFtsStatCfName);
  // $MAX_TF/$DOC_LEN are not opened above; keep nullptrs so the
  // doc-length / max-tf scan sections degrade gracefully.
  rocksdb::ColumnFamilyHandle *max_tf_cf = nullptr;
  rocksdb::ColumnFamilyHandle *doc_len_cf = nullptr;

  if (!postings_cf || !stat_cf) {
    fprintf(stderr, "ERROR: Failed to get required column families\n");
    return -1;
  }

  // ---------------------------------------------------------------
  // 1. Segment-level statistics (total_docs, total_tokens)
  // ---------------------------------------------------------------
  uint64_t total_docs = 0;
  uint64_t total_tokens = 0;
  {
    const std::string total_docs_key = FLAGS_field + "_total_docs";
    const std::string total_tokens_key = FLAGS_field + "_total_tokens";
    std::string value;
    if (store.db_->Get(store.read_opts_, stat_cf, total_docs_key, &value)
            .ok() &&
        value.size() >= sizeof(uint64_t)) {
      std::memcpy(&total_docs, value.data(), sizeof(uint64_t));
    }
    value.clear();
    if (store.db_->Get(store.read_opts_, stat_cf, total_tokens_key, &value)
            .ok() &&
        value.size() >= sizeof(uint64_t)) {
      std::memcpy(&total_tokens, value.data(), sizeof(uint64_t));
    }
  }

  double avg_doc_len = total_docs > 0 ? static_cast<double>(total_tokens) /
                                            static_cast<double>(total_docs)
                                      : 0.0;

  std::cout << std::endl;
  std::cout << "--- Segment Statistics ---" << std::endl;
  std::cout << "  Total documents : " << total_docs << std::endl;
  std::cout << "  Total tokens    : " << total_tokens << std::endl;
  std::cout << "  Avg doc length  : " << avg_doc_len << std::endl;

  // ---------------------------------------------------------------
  // 2. Vocabulary & posting list statistics
  // ---------------------------------------------------------------
  std::cout << std::endl;
  std::cout << "--- Vocabulary & Posting List Statistics ---" << std::endl;
  std::cout << "  Scanning postings CF..." << std::flush;

  uint64_t vocab_size = 0;
  uint64_t total_postings_entries = 0;  // sum of all posting list lengths
  uint64_t total_postings_bytes = 0;    // sum of serialized bitmap sizes
  uint64_t max_posting_len = 0;
  std::string max_posting_term;

  // Posting list length distribution buckets
  // [1], [2-10], [11-100], [101-1K], [1K-10K], [10K-100K], [100K+]
  uint64_t bucket_1 = 0;
  uint64_t bucket_2_10 = 0;
  uint64_t bucket_11_100 = 0;
  uint64_t bucket_101_1k = 0;
  uint64_t bucket_1k_10k = 0;
  uint64_t bucket_10k_100k = 0;
  uint64_t bucket_100k_plus = 0;

  // Format counters
  uint64_t roaring_count = 0;
  uint64_t bitpacked_count = 0;

  {
    auto iter = std::unique_ptr<rocksdb::Iterator>(
        store.db_->NewIterator(store.read_opts_, postings_cf));
    while (iter->Valid()) {
      const std::string term = iter->key().ToString();
      const std::string posting_data = iter->value().ToString();

      ++vocab_size;
      total_postings_bytes += posting_data.size();

      uint64_t cardinality = 0;

      if (BitPackedPostingList::is_bitpacked_format(posting_data.data(),
                                                    posting_data.size())) {
        // BitPacked format: read num_docs from FileHeader
        ++bitpacked_count;
        fts::BitPackedPostingIterator bp_iter;
        if (bp_iter.open(posting_data.data(), posting_data.size()) == 0) {
          cardinality = bp_iter.cost();
        }
      } else {
        // Roaring Bitmap format
        ++roaring_count;
        roaring_bitmap_t *bitmap = roaring_bitmap_portable_deserialize_safe(
            posting_data.data(), posting_data.size());
        if (bitmap) {
          cardinality = roaring_bitmap_get_cardinality(bitmap);
          roaring_bitmap_free(bitmap);
        }
      }

      total_postings_entries += cardinality;

      if (cardinality > max_posting_len) {
        max_posting_len = cardinality;
        max_posting_term = term;
      }

      // Bucket distribution
      if (cardinality <= 1) {
        ++bucket_1;
      } else if (cardinality <= 10) {
        ++bucket_2_10;
      } else if (cardinality <= 100) {
        ++bucket_11_100;
      } else if (cardinality <= 1000) {
        ++bucket_101_1k;
      } else if (cardinality <= 10000) {
        ++bucket_1k_10k;
      } else if (cardinality <= 100000) {
        ++bucket_10k_100k;
      } else {
        ++bucket_100k_plus;
      }

      if (vocab_size % 10000 == 0) {
        std::cout << "\r  Scanning postings CF... " << vocab_size << " terms"
                  << std::flush;
      }

      iter->Next();
    }
    // iter auto-closes via unique_ptr
  }

  std::cout << "\r  Scanning postings CF... done.          " << std::endl;
  std::cout << "  Posting format        : " << roaring_count << " Roaring, "
            << bitpacked_count << " BitPacked" << std::endl;
  std::cout << "  Vocabulary size       : " << vocab_size << std::endl;
  std::cout << "  Total postings entries : " << total_postings_entries
            << std::endl;
  std::cout << "  Total postings bytes   : " << total_postings_bytes / 1024
            << " KB" << std::endl;
  if (vocab_size > 0) {
    std::cout << "  Avg posting list len  : "
              << static_cast<double>(total_postings_entries) / vocab_size
              << std::endl;
    std::cout << "  Avg posting bytes     : "
              << static_cast<double>(total_postings_bytes) / vocab_size << " B"
              << std::endl;
  }
  std::cout << "  Max posting list len  : " << max_posting_len;
  if (!max_posting_term.empty()) {
    std::cout << " (term: \"" << max_posting_term << "\")";
  }
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "  Posting list length distribution:" << std::endl;
  std::cout << "    [1]          : " << bucket_1 << std::endl;
  std::cout << "    [2-10]       : " << bucket_2_10 << std::endl;
  std::cout << "    [11-100]     : " << bucket_11_100 << std::endl;
  std::cout << "    [101-1K]     : " << bucket_101_1k << std::endl;
  std::cout << "    [1K-10K]     : " << bucket_1k_10k << std::endl;
  std::cout << "    [10K-100K]   : " << bucket_10k_100k << std::endl;
  std::cout << "    [100K+]      : " << bucket_100k_plus << std::endl;

  // ---------------------------------------------------------------
  // 3. Document length distribution
  // ---------------------------------------------------------------
  std::cout << std::endl;
  std::cout << "--- Document Length Distribution ---" << std::endl;

  uint64_t doc_count = 0;
  uint64_t sum_doc_len = 0;
  uint32_t min_doc_len = UINT32_MAX;
  uint32_t max_doc_len = 0;
  std::vector<uint32_t> all_doc_lens;

  if (doc_len_cf) {
    auto iter = std::unique_ptr<rocksdb::Iterator>(
        store.db_->NewIterator(store.read_opts_, doc_len_cf));
    while (iter->Valid()) {
      const std::string value = iter->value().ToString();
      if (value.size() >= sizeof(uint32_t)) {
        uint32_t doc_len = 0;
        std::memcpy(&doc_len, value.data(), sizeof(uint32_t));
        ++doc_count;
        sum_doc_len += doc_len;
        if (doc_len < min_doc_len) min_doc_len = doc_len;
        if (doc_len > max_doc_len) max_doc_len = doc_len;
        all_doc_lens.push_back(doc_len);
      }
      iter->Next();
    }
    // iter auto-closes via unique_ptr
  } else {
    std::cout << "  $DOC_LEN CF was dropped at build time after"
              << " convert_postings_to_bitpacked()." << std::endl
              << "  Per-doc length info is now inlined in BitPacked"
              << " postings; skipping distribution scan." << std::endl;
  }

  if (doc_count > 0) {
    std::sort(all_doc_lens.begin(), all_doc_lens.end());

    auto percentile = [&](double p) -> uint32_t {
      size_t idx = static_cast<size_t>(p * all_doc_lens.size());
      if (idx >= all_doc_lens.size()) idx = all_doc_lens.size() - 1;
      return all_doc_lens[idx];
    };

    std::cout << "  Doc count     : " << doc_count << std::endl;
    std::cout << "  Avg doc length: "
              << static_cast<double>(sum_doc_len) / doc_count << std::endl;
    std::cout << "  Min doc length: " << min_doc_len << std::endl;
    std::cout << "  P25 doc length: " << percentile(0.25) << std::endl;
    std::cout << "  P50 doc length: " << percentile(0.50) << std::endl;
    std::cout << "  P75 doc length: " << percentile(0.75) << std::endl;
    std::cout << "  P95 doc length: " << percentile(0.95) << std::endl;
    std::cout << "  P99 doc length: " << percentile(0.99) << std::endl;
    std::cout << "  Max doc length: " << max_doc_len << std::endl;
  } else {
    std::cout << "  No documents found in $DOC_LEN CF." << std::endl;
  }

  // ---------------------------------------------------------------
  // 4. Max-TF statistics (top terms by max term frequency)
  // ---------------------------------------------------------------
  if (max_tf_cf) {
    std::cout << std::endl;
    std::cout << "--- Top Terms by Max Term Frequency ---" << std::endl;

    struct TermMaxTf {
      std::string term;
      uint32_t max_tf;
    };

    // Collect all and sort by max_tf descending, show top 20
    std::vector<TermMaxTf> term_max_tfs;
    {
      auto iter = std::unique_ptr<rocksdb::Iterator>(
          store.db_->NewIterator(store.read_opts_, max_tf_cf));
      while (iter->Valid()) {
        const std::string term = iter->key().ToString();
        const std::string value = iter->value().ToString();
        uint32_t max_tf = 0;
        if (value.size() >= sizeof(uint32_t)) {
          std::memcpy(&max_tf, value.data(), sizeof(uint32_t));
        }
        term_max_tfs.push_back({term, max_tf});
        iter->Next();
      }
      // iter auto-closes via unique_ptr
    }

    std::sort(term_max_tfs.begin(), term_max_tfs.end(),
              [](const TermMaxTf &a, const TermMaxTf &b) {
                return a.max_tf > b.max_tf;
              });

    size_t show_count = std::min<size_t>(20, term_max_tfs.size());
    for (size_t i = 0; i < show_count; ++i) {
      std::cout << "  " << (i + 1) << ". \"" << term_max_tfs[i].term
                << "\" max_tf=" << term_max_tfs[i].max_tf << std::endl;
    }
  }

  // ---------------------------------------------------------------
  // 5. Storage size summary
  // ---------------------------------------------------------------
  std::cout << std::endl;
  std::cout << "--- Storage Size Summary ---" << std::endl;
  std::cout << "  Postings CF ($POSTINGS) : " << total_postings_bytes / 1024
            << " KB (serialized bitmap data)" << std::endl;
  std::cout << "  (Other CF sizes require RocksDB property queries or dump)"
            << std::endl;

  std::cout << std::endl;
  std::cout << "=== STATS COMPLETE ===" << std::endl;

  store.close();
  return 0;
}


int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);


  if (FLAGS_index.empty()) {
    std::cerr << "Error: -index is required." << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  Build : bin/fts_bench -cmd build -index <path> -corpus "
                 "<corpus.jsonl>"
              << std::endl;
    std::cerr << "  Search : bin/fts_bench -cmd search "
                 "-index <path> -query <queries.jsonl> -qrels <qrels_dir>"
              << std::endl;
    std::cerr << "  Stats : bin/fts_bench -cmd stats -index <path>"
              << std::endl;
    return 1;
  }

  // Determine command: explicit -cmd flag takes priority, otherwise auto-detect
  std::string cmd = FLAGS_cmd;
  if (cmd.empty()) {
    if (!FLAGS_corpus.empty()) {
      cmd = "build";
    } else if (!FLAGS_query.empty()) {
      cmd = "search";
    } else {
      std::cerr << "Error: specify -cmd (build/search/stats) or -corpus/-query."
                << std::endl;
      return 1;
    }
  }


  // Validate -mode flag
  const bool db_mode = (FLAGS_mode == "db");
  if (FLAGS_mode != "raw" && FLAGS_mode != "db") {
    std::cerr << "Error: unknown -mode '" << FLAGS_mode
              << "'. Use 'raw' or 'db'." << std::endl;
    return 1;
  }


  if (cmd == "build") {
    if (FLAGS_corpus.empty()) {
      std::cerr << "Error: -corpus is required in build mode." << std::endl;
      return 1;
    }
    return db_mode ? do_build_db() : do_build();
  } else if (cmd == "search") {
    if (FLAGS_query.empty()) {
      std::cerr << "Error: -query is required in search mode." << std::endl;
      return 1;
    }
    if (FLAGS_qrels.empty()) {
      std::cerr << "Error: -qrels is required in search mode." << std::endl;
      return 1;
    }
    return db_mode ? do_search_db() : do_search();
  } else if (cmd == "stats") {
    if (db_mode) {
      std::cerr << "Error: stats command is not supported in db mode."
                << std::endl;
      return 1;
    }
    return do_stats();
  } else {
    std::cerr << "Error: unknown command '" << cmd
              << "'. Use build, search, or stats." << std::endl;
    return 1;
  }
}
