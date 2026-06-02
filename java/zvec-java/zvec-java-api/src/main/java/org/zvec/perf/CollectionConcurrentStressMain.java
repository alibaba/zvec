package org.zvec.perf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.zvec.Collection;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.HnswQueryParams;
import org.zvec.VectorQuery;
import org.zvec.Zvec;

public final class CollectionConcurrentStressMain {
  private CollectionConcurrentStressMain() {}

  public static void main(String[] args) throws Exception {
    StressOptions options = StressOptions.parse(args);
    Path runDir = prepareRunDirectory(options.workDir(), options.docCount());

    System.out.println("CONCURRENT_STRESS_CONFIG " + formatConfig(options, runDir));

    CollectionSchema schema =
        PerfData.schema("perf_docs_concurrent", options.dimension(), options.hnswIndexParams());
    try (Collection collection = Zvec.createAndOpen(runDir.toString(), schema)) {
      insertInitialDataset(collection, options);
      collection.flush();
      warmupQueries(collection, options, options.warmupQueries());

      QueryOnlyMetrics queryOnly = runQueryOnly(collection, options);
      MixedMetrics mixed = runMixed(collection, options);

      printSummary(options, queryOnly, mixed, runDir);
    }
  }

  private static void insertInitialDataset(Collection collection, StressOptions options) {
    for (int batchStart = 0; batchStart < options.docCount(); batchStart += options.batchSize()) {
      int batchSize = Math.min(options.batchSize(), options.docCount() - batchStart);
      List<Doc> docs = PerfData.docs(batchStart, batchSize, options.dimension(), options.seed());
      int inserted = collection.insert(docs);
      if (inserted != batchSize) {
        throw new IllegalStateException(
            "Inserted count mismatch: expected " + batchSize + ", got " + inserted);
      }
    }
  }

  private static void warmupQueries(Collection collection, StressOptions options, int queryCount) {
    Random random = new Random(options.seed() ^ 0x13579BDFL);
    for (int i = 0; i < queryCount; i++) {
      int docIndex = random.nextInt(options.docCount());
      PerfData.VectorSample sample =
          PerfData.querySample(docIndex, options.dimension(), options.seed(), options.topK());
      collection.query(buildQuery(sample, options));
    }
  }

  private static QueryOnlyMetrics runQueryOnly(Collection collection, StressOptions options)
      throws Exception {
    int threads = options.concurrentQueryThreads();
    int queriesPerThread = options.concurrentQueryCount();
    int totalQueries = threads * queriesPerThread;
    long[] latencies = new long[totalQueries];
    AtomicInteger failureCount = new AtomicInteger();
    AtomicInteger missCount = new AtomicInteger();

    long startedAt = System.nanoTime();
    runConcurrent(
        threads,
        threadIndex -> {
          Random random = new Random(options.seed() ^ 0x2468ACE0L ^ threadIndex);
          int offset = threadIndex * queriesPerThread;
          for (int i = 0; i < queriesPerThread; i++) {
            int docIndex = random.nextInt(options.docCount());
            PerfData.VectorSample sample =
                PerfData.querySample(docIndex, options.dimension(), options.seed(), options.topK());
            long queryStartedAt = System.nanoTime();
            try {
              List<Doc> results = collection.query(buildQuery(sample, options));
              latencies[offset + i] = System.nanoTime() - queryStartedAt;
              if (!CollectionStressMain.hasExpectedHit(sample.expectedId(), results)) {
                missCount.incrementAndGet();
              }
            } catch (RuntimeException e) {
              failureCount.incrementAndGet();
            }
          }
        });
    long elapsedNanos = System.nanoTime() - startedAt;

    return new QueryOnlyMetrics(
        totalQueries,
        failureCount.get(),
        missCount.get(),
        latencyStats(latencies),
        elapsedNanos);
  }

  private static MixedMetrics runMixed(Collection collection, StressOptions options) throws Exception {
    int threads = options.concurrentMixedThreads();
    int rounds = options.concurrentMixedRounds();
    int insertBatchSize = options.concurrentMixedInsertBatchSize();
    int queriesPerRound = options.concurrentMixedQueriesPerRound();

    long[] insertLatencies = new long[threads * rounds];
    long[] queryLatencies = new long[threads * rounds * queriesPerRound];
    AtomicInteger insertFailures = new AtomicInteger();
    AtomicInteger queryFailures = new AtomicInteger();
    AtomicInteger missCount = new AtomicInteger();
    AtomicInteger nextDocIndex = new AtomicInteger(options.docCount());

    long startedAt = System.nanoTime();
    runConcurrent(
        threads,
        threadIndex -> {
          Random random = new Random(options.seed() ^ 0x55AA55AAL ^ threadIndex);
          int insertOffset = threadIndex * rounds;
          int queryOffset = threadIndex * rounds * queriesPerRound;

          for (int round = 0; round < rounds; round++) {
            int startDocIndex = nextDocIndex.getAndAdd(insertBatchSize);
            long insertStartedAt = System.nanoTime();
            try {
              List<Doc> docs =
                  PerfData.docs(startDocIndex, insertBatchSize, options.dimension(), options.seed());
              int inserted = collection.insert(docs);
              if (inserted != insertBatchSize) {
                insertFailures.incrementAndGet();
              }
              insertLatencies[insertOffset + round] = System.nanoTime() - insertStartedAt;
            } catch (RuntimeException e) {
              insertFailures.incrementAndGet();
            }

            for (int q = 0; q < queriesPerRound; q++) {
              int docIndex = random.nextInt(options.docCount());
              PerfData.VectorSample sample =
                  PerfData.querySample(docIndex, options.dimension(), options.seed(), options.topK());
              long queryStartedAt = System.nanoTime();
              try {
                List<Doc> results = collection.query(buildQuery(sample, options));
                queryLatencies[queryOffset + round * queriesPerRound + q] =
                    System.nanoTime() - queryStartedAt;
                if (!CollectionStressMain.hasExpectedHit(sample.expectedId(), results)) {
                  missCount.incrementAndGet();
                }
              } catch (RuntimeException e) {
                queryFailures.incrementAndGet();
              }
            }
          }
        });
    long elapsedNanos = System.nanoTime() - startedAt;

    return new MixedMetrics(
        threads * rounds * insertBatchSize,
        insertFailures.get(),
        latencyStats(insertLatencies),
        threads * rounds * queriesPerRound,
        queryFailures.get(),
        missCount.get(),
        latencyStats(queryLatencies),
        elapsedNanos);
  }

  private static void runConcurrent(int threads, ThrowingIntConsumer task) throws Exception {
    ExecutorService executor = Executors.newFixedThreadPool(threads);
    CountDownLatch start = new CountDownLatch(1);
    List<Future<?>> futures = new ArrayList<>(threads);
    try {
      for (int threadIndex = 0; threadIndex < threads; threadIndex++) {
        int index = threadIndex;
        futures.add(
            executor.submit(
                () -> {
                  start.await();
                  task.accept(index);
                  return null;
                }));
      }
      start.countDown();
      for (Future<?> future : futures) {
        future.get();
      }
    } finally {
      executor.shutdownNow();
    }
  }

  private static LatencyStats latencyStats(long[] samples) {
    int successCount = 0;
    for (long sample : samples) {
      if (sample > 0L) {
        successCount++;
      }
    }
    if (successCount == 0) {
      return null;
    }
    if (successCount == samples.length) {
      return LatencyStats.fromNanos(samples);
    }
    long[] compacted = new long[successCount];
    int index = 0;
    for (long sample : samples) {
      if (sample > 0L) {
        compacted[index++] = sample;
      }
    }
    return LatencyStats.fromNanos(compacted);
  }

  private static void printSummary(
      StressOptions options, QueryOnlyMetrics queryOnly, MixedMetrics mixed, Path runDir) {
    System.out.printf(
        Locale.ROOT,
        "QUERY_ONLY_SUMMARY threads=%d queries=%d query_failures=%d miss_count=%d recall=%.4f seconds=%.3f queries_per_sec=%.1f p50_us=%s p95_us=%s p99_us=%s%n",
        options.concurrentQueryThreads(),
        queryOnly.queryCount(),
        queryOnly.failureCount(),
        queryOnly.missCount(),
        queryOnly.recall(),
        queryOnly.elapsedSeconds(),
        queryOnly.queriesPerSecond(),
        formatLatency(queryOnly.latencyStats(), LatencyStatField.P50),
        formatLatency(queryOnly.latencyStats(), LatencyStatField.P95),
        formatLatency(queryOnly.latencyStats(), LatencyStatField.P99));
    System.out.printf(
        Locale.ROOT,
        "MIXED_SUMMARY threads=%d rounds=%d inserted_docs=%d insert_failures=%d insert_docs_per_sec=%.1f insert_p50_us=%s insert_p95_us=%s insert_p99_us=%s queries=%d query_failures=%d miss_count=%d recall=%.4f queries_per_sec=%.1f query_p50_us=%s query_p95_us=%s query_p99_us=%s%n",
        options.concurrentMixedThreads(),
        options.concurrentMixedRounds(),
        mixed.insertedDocs(),
        mixed.insertFailures(),
        mixed.insertDocsPerSecond(),
        formatLatency(mixed.insertLatencyStats(), LatencyStatField.P50),
        formatLatency(mixed.insertLatencyStats(), LatencyStatField.P95),
        formatLatency(mixed.insertLatencyStats(), LatencyStatField.P99),
        mixed.queryCount(),
        mixed.queryFailures(),
        mixed.missCount(),
        mixed.recall(),
        mixed.queriesPerSecond(),
        formatLatency(mixed.queryLatencyStats(), LatencyStatField.P50),
        formatLatency(mixed.queryLatencyStats(), LatencyStatField.P95),
        formatLatency(mixed.queryLatencyStats(), LatencyStatField.P99));
    System.out.println("ARTIFACT_DIR " + runDir);
    System.out.println(
        "SUGGESTED_RUNS_FROM_MODULE_DIR "
            + "\"mvn -q -DskipTests compile exec:java@run-concurrent-stress -Dzvec.stress.args='--docs 100000 --concurrent-query-threads 4 --concurrent-query-count 250'\"");
  }

  private static String formatConfig(StressOptions options, Path runDir) {
    return "docs="
        + options.docCount()
        + " batch_size="
        + options.batchSize()
        + " dimension="
        + options.dimension()
        + " top_k="
        + options.topK()
        + " warmup_queries="
        + options.warmupQueries()
        + " concurrent_query_threads="
        + options.concurrentQueryThreads()
        + " concurrent_query_count="
        + options.concurrentQueryCount()
        + " concurrent_mixed_threads="
        + options.concurrentMixedThreads()
        + " concurrent_mixed_rounds="
        + options.concurrentMixedRounds()
        + " concurrent_mixed_insert_batch_size="
        + options.concurrentMixedInsertBatchSize()
        + " concurrent_mixed_queries_per_round="
        + options.concurrentMixedQueriesPerRound()
        + " hnsw_m="
        + formatOptionalHnswM(options)
        + " hnsw_ef_construction="
        + formatOptionalHnswEfConstruction(options)
        + " hnsw_ef="
        + formatOptionalHnswEf(options)
        + " seed="
        + options.seed()
        + " run_dir="
        + runDir;
  }

  private static VectorQuery buildQuery(PerfData.VectorSample sample, StressOptions options) {
    VectorQuery query =
        VectorQuery.of("embedding", sample.vector()).topK(sample.topK()).outputFields("title");
    HnswQueryParams hnswQueryParams = options.hnswQueryParams();
    if (hnswQueryParams != null) {
      query.hnsw(hnswQueryParams);
    }
    return query;
  }

  private static Path prepareRunDirectory(Path baseDir, int docCount) throws IOException {
    Files.createDirectories(baseDir);
    String runName = "concurrent-docs-" + docCount + "-" + Instant.now().toEpochMilli();
    return baseDir.resolve(runName);
  }

  private static String formatOptionalHnswM(StressOptions options) {
    return options.hnswIndexParams() == null
        ? "default"
        : Integer.toString(options.hnswIndexParams().m());
  }

  private static String formatOptionalHnswEfConstruction(StressOptions options) {
    return options.hnswIndexParams() == null
        ? "default"
        : Integer.toString(options.hnswIndexParams().efConstruction());
  }

  private static String formatOptionalHnswEf(StressOptions options) {
    return options.hnswQueryParams() == null
        ? "default"
        : Integer.toString(options.hnswQueryParams().ef());
  }

  private static String formatLatency(LatencyStats stats, LatencyStatField field) {
    if (stats == null) {
      return "n/a";
    }
    double value;
    switch (field) {
      case P50:
        value = stats.p50Micros();
        break;
      case P95:
        value = stats.p95Micros();
        break;
      case P99:
        value = stats.p99Micros();
        break;
      default:
        throw new IllegalStateException("Unhandled latency field: " + field);
    }
    return String.format(Locale.ROOT, "%.1f", value);
  }

  private enum LatencyStatField {
    P50,
    P95,
    P99
  }

  @FunctionalInterface
  private interface ThrowingIntConsumer {
    void accept(int value) throws Exception;
  }

  private static final class QueryOnlyMetrics {
    private final int queryCount;
    private final int failureCount;
    private final int missCount;
    private final LatencyStats latencyStats;
    private final long elapsedNanos;

    private QueryOnlyMetrics(
        int queryCount,
        int failureCount,
        int missCount,
        LatencyStats latencyStats,
        long elapsedNanos) {
      this.queryCount = queryCount;
      this.failureCount = failureCount;
      this.missCount = missCount;
      this.latencyStats = latencyStats;
      this.elapsedNanos = elapsedNanos;
    }

    private int queryCount() {
      return queryCount;
    }

    private int failureCount() {
      return failureCount;
    }

    private int missCount() {
      return missCount;
    }

    private LatencyStats latencyStats() {
      return latencyStats;
    }

    private double recall() {
      if (queryCount == 0) {
        return 1.0;
      }
      return (queryCount - failureCount - missCount) / (double) queryCount;
    }

    private double elapsedSeconds() {
      return elapsedNanos / 1_000_000_000.0;
    }

    private double queriesPerSecond() {
      return queryCount / elapsedSeconds();
    }
  }

  private static final class MixedMetrics {
    private final int insertedDocs;
    private final int insertFailures;
    private final LatencyStats insertLatencyStats;
    private final int queryCount;
    private final int queryFailures;
    private final int missCount;
    private final LatencyStats queryLatencyStats;
    private final long elapsedNanos;

    private MixedMetrics(
        int insertedDocs,
        int insertFailures,
        LatencyStats insertLatencyStats,
        int queryCount,
        int queryFailures,
        int missCount,
        LatencyStats queryLatencyStats,
        long elapsedNanos) {
      this.insertedDocs = insertedDocs;
      this.insertFailures = insertFailures;
      this.insertLatencyStats = insertLatencyStats;
      this.queryCount = queryCount;
      this.queryFailures = queryFailures;
      this.missCount = missCount;
      this.queryLatencyStats = queryLatencyStats;
      this.elapsedNanos = elapsedNanos;
    }

    private int insertedDocs() {
      return insertedDocs;
    }

    private int insertFailures() {
      return insertFailures;
    }

    private LatencyStats insertLatencyStats() {
      return insertLatencyStats;
    }

    private int queryCount() {
      return queryCount;
    }

    private int queryFailures() {
      return queryFailures;
    }

    private int missCount() {
      return missCount;
    }

    private LatencyStats queryLatencyStats() {
      return queryLatencyStats;
    }

    private double recall() {
      if (queryCount == 0) {
        return 1.0;
      }
      return (queryCount - queryFailures - missCount) / (double) queryCount;
    }

    private double elapsedSeconds() {
      return elapsedNanos / 1_000_000_000.0;
    }

    private double insertDocsPerSecond() {
      return insertedDocs / elapsedSeconds();
    }

    private double queriesPerSecond() {
      return queryCount / elapsedSeconds();
    }
  }
}
