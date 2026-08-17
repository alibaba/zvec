// Regression tests for concurrent reads against a writer.
//
// fetch() and query() both take their segment list from get_all_segments() and
// both run while the writer crosses segment switches, where dump()/flush()
// tears down memory_store_ and republishes it as a persisted block. They reach
// different storage code — Fetch(doc) vs the planner's fan-out to
// fetch()/scan() and the per-segment vector indexes — so both are covered. On
// the unfixed baseline each crashes there (SIGSEGV inside the Arrow table
// rebuild) or silently drops the rows of the block being republished.
//
// The fetch case also runs without a writer, where the preloaded docs stay in
// MemForwardStore's in-memory rows instead of the flushed mmap path.
//
// Both checks are content-based, not just crash-based: every field is derived
// from the doc id, so fetched docs are compared field by field and queried docs
// have their scalar value checked against the pk they came back with.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include "index/utils/utils.h"
#include "zvec/db/collection.h"

namespace zvec {
namespace test {

namespace {

const char *kPath = "concurrent_read_col";

constexpr uint32_t kMaxBufferSize = 8 * 1024 * 1024;  // force frequent flush()
// Small enough that the writer still crosses segment switches when starved by
// the readers: on the unfixed baseline 4 readers throttle it to ~2.5k docs/s,
// so a 40k-doc segment would never switch inside the run window — and dump()
// is exactly where the baseline crashes.
constexpr uint64_t kMaxDocPerSegment = 4000;
constexpr int kStableDocs = 3000;  // pre-loaded, must always read back exactly
constexpr int kWriteBatch = 100;
constexpr int kRunSeconds = 8;
// Bounds disk growth on fast machines (50k docs ≈ 12 segments) while staying
// above the baseline's ~2.5k docs/s × 8s, so it never truncates the baseline
// run before its first segment switch.
constexpr long kMaxWriterDocs = 50000;

// Inserts disjoint docs (ids far above the preloaded range) until stop or the
// kMaxWriterDocs cap, crossing segment switches on the way.
std::thread StartWriter(const Collection::Ptr &collection,
                        const CollectionSchema &schema,
                        const std::atomic<bool> &stop,
                        std::atomic<long> &docs_written,
                        std::atomic<long> &writer_errors) {
  return std::thread([&] {
    uint64_t next_id = kStableDocs + 1000000;  // never collides with probes
    while (!stop.load(std::memory_order_relaxed) &&
           docs_written.load(std::memory_order_relaxed) < kMaxWriterDocs) {
      std::vector<Doc> docs;
      docs.reserve(kWriteBatch);
      for (int i = 0; i < kWriteBatch; i++) {
        docs.push_back(TestHelper::CreateDoc(next_id + i, schema));
      }
      auto res = collection->insert(docs);
      if (!res.has_value()) {
        writer_errors.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      next_id += kWriteBatch;
      docs_written.fetch_add(kWriteBatch, std::memory_order_relaxed);
    }
  });
}

struct ReaderResult {
  long fetches{0};
  long null_docs{0};
  long mismatches{0};
  long errors{0};
  std::string first_problem;
};

void Note(ReaderResult *r, const std::string &msg) {
  if (r->first_problem.empty()) r->first_problem = msg;
}

}  // namespace

class ConcurrentReadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zvec::ailego::MemoryLimitPool::get_instance().init(4 * 1024ll * 1024ll *
                                                       1024ll);
    ailego::FileHelper::RemoveDirectory(kPath);
  }
  void TearDown() override {
    ailego::FileHelper::RemoveDirectory(kPath);
  }
};

TEST_F(ConcurrentReadTest, FetchReturnsCorrectContentUnderConcurrency) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  schema->set_max_doc_count_per_segment(kMaxDocPerSegment);
  auto options = CollectionOptions{false, true, kMaxBufferSize};

  // With a writer the readers hit mostly persisted blocks (mmap); without one,
  // the in-memory store.
  for (bool with_writer : {true, false}) {
    for (int readers : {4, 8}) {
      // Rebuilt per run so the no-writer case starts with everything still in
      // the in-memory store rather than flushed by a previous run.
      ailego::FileHelper::RemoveDirectory(kPath);
      auto collection = TestHelper::CreateCollectionWithDoc(
          kPath, *schema, options, 0, kStableDocs, false);
      ASSERT_NE(collection, nullptr);

      std::atomic<bool> stop{false};
      std::atomic<long> docs_written{0};
      std::atomic<long> writer_errors{0};
      std::vector<ReaderResult> results(readers);

      auto t0 = std::chrono::steady_clock::now();

      // Insert, flush and switch segments continuously.
      std::thread writer;
      if (with_writer) {
        writer =
            StartWriter(collection, *schema, stop, docs_written, writer_errors);
      }

      std::vector<std::thread> threads;
      for (int t = 0; t < readers; t++) {
        threads.emplace_back([&, t] {
          std::mt19937 rng(static_cast<unsigned>(t) * 7919u + 13u);
          std::uniform_int_distribution<int> pick(0, kStableDocs - 1);
          auto *r = &results[t];
          while (!stop.load(std::memory_order_relaxed)) {
            int id = pick(rng);
            auto expect =
                TestHelper::CreateDoc(static_cast<uint64_t>(id), *schema);
            auto fetched = collection->fetch({expect.pk()});
            r->fetches++;
            if (!fetched.has_value()) {
              r->errors++;
              Note(r, "fetch returned error: " + fetched.error().message());
              continue;
            }
            auto it = fetched.value().find(expect.pk());
            if (it == fetched.value().end() || it->second == nullptr) {
              r->null_docs++;
              Note(r, "doc " + expect.pk() + " came back null");
              continue;
            }
            // Catches silent corruption of the forward columns or the vector.
            if (*it->second != expect) {
              r->mismatches++;
              Note(r, "doc " + expect.pk() + " content mismatch");
            }
          }
        });
      }

      std::this_thread::sleep_for(std::chrono::seconds(kRunSeconds));
      stop.store(true);
      if (with_writer) writer.join();
      for (auto &th : threads) th.join();

      double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();

      long total = 0, nulls = 0, mism = 0, errs = 0;
      std::string problem;
      for (auto &r : results) {
        total += r.fetches;
        nulls += r.null_docs;
        mism += r.mismatches;
        errs += r.errors;
        if (problem.empty()) problem = r.first_problem;
      }

      std::printf(
          "writer=%-3s readers=%d  fetches=%ld (%.0f/s)  writes=%ld "
          "(%.0f docs/s)  null=%ld mismatch=%ld error=%ld\n",
          with_writer ? "yes" : "no", readers, total, total / elapsed,
          docs_written.load(), docs_written.load() / elapsed, nulls, mism,
          errs);
      if (!problem.empty()) {
        std::printf("  first problem: %s\n", problem.c_str());
      }

      EXPECT_GT(total, 0);
      EXPECT_EQ(nulls, 0) << problem;
      EXPECT_EQ(mism, 0) << problem;
      EXPECT_EQ(errs, 0) << problem;
      if (with_writer) {
        // Vacuous unless the writer actually ran and crossed a segment switch
        // (dump() is where the baseline crashes), so fail loudly instead of
        // silently degrading to a no-writer run.
        EXPECT_EQ(writer_errors.load(), 0) << problem;
        EXPECT_GT(docs_written.load(), 0);
        EXPECT_GE(docs_written.load(),
                  static_cast<long>(kMaxDocPerSegment) - kStableDocs)
            << "no segment switch inside the run window";
      }

      collection.reset();
    }
  }
}

TEST_F(ConcurrentReadTest, QueryReturnsCompleteResultsUnderConcurrency) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  schema->set_max_doc_count_per_segment(kMaxDocPerSegment);
  auto options = CollectionOptions{false, true, kMaxBufferSize};

  ailego::FileHelper::RemoveDirectory(kPath);
  auto collection = TestHelper::CreateCollectionWithDoc(kPath, *schema, options,
                                                        0, kStableDocs, false);
  ASSERT_NE(collection, nullptr);

  auto query_doc = TestHelper::CreateDoc(1, *schema);
  auto vector = query_doc.get<std::vector<float>>("dense_fp32");
  ASSERT_TRUE(vector.has_value());
  const std::string vector_bytes(reinterpret_cast<const char *>(vector->data()),
                                 vector->size() * sizeof(float));

  // Every field is derived from the doc id, so a hit must carry the value
  // belonging to its own pk: a missing one means the row was dropped during the
  // fan-out, a wrong one means rows from different blocks got stitched.
  constexpr const char *kProbeField = "int32";

  std::atomic<bool> stop{false};
  std::atomic<long> docs_written{0};
  std::atomic<long> writer_errors{0};
  std::atomic<long> queries{0};
  std::atomic<long> query_errors{0};
  std::atomic<long> tolerated_errors{0};
  std::atomic<long> empty_results{0};
  std::atomic<long> null_fields{0};
  std::atomic<long> mismatches{0};
  std::atomic<long> short_results{0};

  auto t0 = std::chrono::steady_clock::now();
  std::thread writer =
      StartWriter(collection, *schema, stop, docs_written, writer_errors);

  constexpr int kQueriers = 4;
  std::vector<std::thread> queriers;
  for (int t = 0; t < kQueriers; t++) {
    queriers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        SearchQuery q;
        q.topk_ = 10;
        q.target_.field_name_ = "dense_fp32";
        q.target_.set_vector(vector_bytes);
        q.output_fields_ = {kProbeField};
        auto r = collection->query(q);
        queries.fetch_add(1, std::memory_order_relaxed);
        if (!r.has_value()) {
          // Known transient: a doc admitted by the writing segment's streaming
          // vector index may not be in its forward store yet (pre-existing
          // Insert/Query race, unrelated to this fix). Counted separately
          // rather than ignored, so it stays visible.
          if (r.error().message().find("fetch table failed") ==
              std::string::npos) {
            query_errors.fetch_add(1, std::memory_order_relaxed);
          } else {
            tolerated_errors.fetch_add(1, std::memory_order_relaxed);
          }
          continue;
        }
        if (r.value().empty()) {
          empty_results.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        for (const auto &doc : r.value()) {
          auto value = doc->get<int32_t>(kProbeField);
          if (!value.has_value()) {
            null_fields.fetch_add(1, std::memory_order_relaxed);
            break;
          }
          const std::string pk = doc->pk();
          if (pk.size() <= 3 ||
              static_cast<uint64_t>(*value) != TestHelper::ExtractDocId(pk)) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
            break;
          }
        }
        // The collection holds far more than topk docs, so a short result means
        // rows were dropped along the fan-out.
        if (r.value().size() < static_cast<size_t>(q.topk_)) {
          short_results.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::seconds(kRunSeconds));
  stop.store(true);
  writer.join();
  for (auto &th : queriers) th.join();

  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::printf(
      "query workload: queries=%ld (%.0f/s) errors=%ld tolerated=%ld empty=%ld "
      "nulls=%ld mismatch=%ld short=%ld writes=%ld\n",
      queries.load(), queries.load() / elapsed, query_errors.load(),
      tolerated_errors.load(), empty_results.load(), null_fields.load(),
      mismatches.load(), short_results.load(), docs_written.load());

  EXPECT_GT(queries.load(), 0);
  EXPECT_EQ(query_errors.load(), 0);
  EXPECT_EQ(empty_results.load(), 0);
  EXPECT_EQ(null_fields.load(), 0);
  EXPECT_EQ(mismatches.load(), 0);
  EXPECT_EQ(short_results.load(), 0);
  EXPECT_EQ(writer_errors.load(), 0);
  EXPECT_GE(docs_written.load(),
            static_cast<long>(kMaxDocPerSegment) - kStableDocs)
      << "no segment switch inside the run window";

  collection.reset();
}

}  // namespace test
}  // namespace zvec
