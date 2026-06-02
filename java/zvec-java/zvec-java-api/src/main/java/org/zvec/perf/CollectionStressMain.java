package org.zvec.perf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import org.zvec.Collection;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.HnswQueryParams;
import org.zvec.VectorQuery;
import org.zvec.Zvec;

public final class CollectionStressMain {
  private CollectionStressMain() {}

  public static void main(String[] args) throws Exception {
    StressOptions options = StressOptions.parse(args);
    Path runDir = prepareRunDirectory(options.workDir(), options.docCount());

    System.out.println("STRESS_CONFIG " + formatConfig(options, runDir));

    CollectionSchema schema = PerfData.schema("perf_docs", options.dimension(), options.hnswIndexParams());
    MemorySnapshot memoryBefore = captureMemory();

    int nextDocIndex = options.docCount();
    try (Collection collection = Zvec.createAndOpen(runDir.toString(), schema)) {
      InsertMetrics insertMetrics = insertInitialDataset(collection, options);
      collection.flush();
      MemorySnapshot memoryAfterInsert = captureMemory();

      warmupQueries(collection, options);
      QueryMetrics queryMetrics = runQueryPhase(collection, options, options.docCount());
      MemorySnapshot memoryAfterQueries = captureMemory();

      ReliabilityMetrics reliability = runSteadyState(collection, options, nextDocIndex);
      MemorySnapshot memoryAfterSteadyState = captureMemory();

      printSummary(
          options,
          insertMetrics,
          queryMetrics,
          memoryBefore,
          memoryAfterInsert,
          memoryAfterQueries,
          memoryAfterSteadyState,
          reliability,
          runDir);
    }
  }

  private static InsertMetrics insertInitialDataset(Collection collection, StressOptions options) {
    long startedAt = System.nanoTime();
    for (int batchStart = 0; batchStart < options.docCount(); batchStart += options.batchSize()) {
      int batchSize = Math.min(options.batchSize(), options.docCount() - batchStart);
      List<Doc> docs = PerfData.docs(batchStart, batchSize, options.dimension(), options.seed());
      int inserted = collection.insert(docs);
      if (inserted != batchSize) {
        throw new IllegalStateException(
            "Inserted count mismatch: expected " + batchSize + ", got " + inserted);
      }
    }
    long elapsedNanos = System.nanoTime() - startedAt;
    return new InsertMetrics(options.docCount(), elapsedNanos);
  }

  private static void warmupQueries(Collection collection, StressOptions options) {
    Random random = new Random(options.seed() ^ 0xA5A5A5A5L);
    for (int i = 0; i < options.warmupQueries(); i++) {
      int docIndex = random.nextInt(options.docCount());
      PerfData.VectorSample sample =
          PerfData.querySample(docIndex, options.dimension(), options.seed(), options.topK());
      collection.query(buildQuery(sample, options));
    }
  }

  private static QueryMetrics runQueryPhase(
      Collection collection, StressOptions options, int maxDocIndexExclusive) {
    Random random = new Random(options.seed() ^ 0x5A5A5A5AL);
    long[] latencies = new long[options.queryCount()];
    int missCount = 0;
    for (int i = 0; i < options.queryCount(); i++) {
      int docIndex = random.nextInt(maxDocIndexExclusive);
      PerfData.VectorSample sample =
          PerfData.querySample(docIndex, options.dimension(), options.seed(), options.topK());
      long startedAt = System.nanoTime();
      List<Doc> results =
          collection.query(buildQuery(sample, options));
      latencies[i] = System.nanoTime() - startedAt;
      if (!hasExpectedHit(sample.expectedId(), results)) {
        missCount++;
      }
    }
    return new QueryMetrics(LatencyStats.fromNanos(latencies), missCount);
  }

  private static ReliabilityMetrics runSteadyState(
      Collection collection, StressOptions options, int nextDocIndex) {
    long[] steadyLatencies =
        new long[Math.max(1, options.steadyStateRounds() * options.steadyQueriesPerRound())];
    int latencyIndex = 0;
    int insertFailures = 0;
    int queryFailures = 0;

    for (int round = 0; round < options.steadyStateRounds(); round++) {
      try {
        List<Doc> docs =
            PerfData.docs(
                nextDocIndex, options.steadyInsertBatchSize(), options.dimension(), options.seed());
        int inserted = collection.insert(docs);
        if (inserted != options.steadyInsertBatchSize()) {
          insertFailures++;
        }
        nextDocIndex += options.steadyInsertBatchSize();
      } catch (RuntimeException e) {
        insertFailures++;
      }

      for (int q = 0; q < options.steadyQueriesPerRound(); q++) {
        int docIndex = nextDocIndex - 1 - q;
        if (docIndex < 0) {
          docIndex = 0;
        }
        try {
          PerfData.VectorSample sample =
              PerfData.querySample(docIndex, options.dimension(), options.seed(), options.topK());
          long startedAt = System.nanoTime();
          List<Doc> results =
              collection.query(buildQuery(sample, options));
          steadyLatencies[latencyIndex++] = System.nanoTime() - startedAt;
          if (!hasExpectedHit(sample.expectedId(), results)) {
            queryFailures++;
          }
        } catch (RuntimeException e) {
          queryFailures++;
        }
      }
    }

    long[] samples =
        latencyIndex == steadyLatencies.length ? steadyLatencies : java.util.Arrays.copyOf(steadyLatencies, latencyIndex);
    LatencyStats steadyStats =
        samples.length == 0 ? null : LatencyStats.fromNanos(samples);
    return new ReliabilityMetrics(options.steadyStateRounds(), insertFailures, queryFailures, steadyStats);
  }

  private static void printSummary(
      StressOptions options,
      InsertMetrics insertMetrics,
      QueryMetrics queryMetrics,
      MemorySnapshot memoryBefore,
      MemorySnapshot memoryAfterInsert,
      MemorySnapshot memoryAfterQueries,
      MemorySnapshot memoryAfterSteadyState,
      ReliabilityMetrics reliability,
      Path runDir) {
    System.out.printf(
        Locale.ROOT,
        "INSERT_SUMMARY docs=%d seconds=%.3f docs_per_sec=%.0f%n",
        insertMetrics.insertedDocs(),
        insertMetrics.elapsedSeconds(),
        insertMetrics.docsPerSecond());
    System.out.printf(
        Locale.ROOT,
        "QUERY_SUMMARY count=%d miss_count=%d recall=%.4f mean_us=%.1f p50_us=%.1f p95_us=%.1f p99_us=%.1f max_us=%.1f%n",
        queryMetrics.latencyStats().count(),
        queryMetrics.missCount(),
        queryMetrics.recall(),
        queryMetrics.latencyStats().meanMicros(),
        queryMetrics.latencyStats().p50Micros(),
        queryMetrics.latencyStats().p95Micros(),
        queryMetrics.latencyStats().p99Micros(),
        queryMetrics.latencyStats().maxMicros());
    System.out.printf(
        Locale.ROOT,
        "MEMORY_SUMMARY heap_before_mb=%.2f heap_after_insert_mb=%.2f heap_after_queries_mb=%.2f heap_after_steady_mb=%.2f heap_growth_insert_mb=%.2f heap_growth_total_mb=%.2f rss_before_mb=%s rss_after_insert_mb=%s rss_after_queries_mb=%s rss_after_steady_mb=%s rss_growth_insert_mb=%s rss_growth_total_mb=%s%n",
        bytesToMiB(memoryBefore.heapBytes()),
        bytesToMiB(memoryAfterInsert.heapBytes()),
        bytesToMiB(memoryAfterQueries.heapBytes()),
        bytesToMiB(memoryAfterSteadyState.heapBytes()),
        bytesToMiB(memoryAfterInsert.heapBytes() - memoryBefore.heapBytes()),
        bytesToMiB(memoryAfterSteadyState.heapBytes() - memoryBefore.heapBytes()),
        formatOptionalMiB(memoryBefore.rssBytes()),
        formatOptionalMiB(memoryAfterInsert.rssBytes()),
        formatOptionalMiB(memoryAfterQueries.rssBytes()),
        formatOptionalMiB(memoryAfterSteadyState.rssBytes()),
        formatOptionalMiB(delta(memoryAfterInsert.rssBytes(), memoryBefore.rssBytes())),
        formatOptionalMiB(delta(memoryAfterSteadyState.rssBytes(), memoryBefore.rssBytes())));
    if (reliability.steadyStats() != null) {
      System.out.printf(
          Locale.ROOT,
          "STEADY_STATE rounds=%d insert_failures=%d query_failures=%d p50_us=%.1f p95_us=%.1f p99_us=%.1f%n",
          reliability.rounds(),
          reliability.insertFailures(),
          reliability.queryFailures(),
          reliability.steadyStats().p50Micros(),
          reliability.steadyStats().p95Micros(),
          reliability.steadyStats().p99Micros());
    } else {
      System.out.printf(
          Locale.ROOT,
          "STEADY_STATE rounds=%d insert_failures=%d query_failures=%d%n",
          reliability.rounds(),
          reliability.insertFailures(),
          reliability.queryFailures());
    }
    System.out.println("ARTIFACT_DIR " + runDir);
    System.out.println(
        "SUGGESTED_RUNS_FROM_MODULE_DIR "
            + "\"mvn -q -DskipTests compile exec:java@run-stress -Dzvec.stress.args='--docs 100000'\" "
            + "\"mvn -q -DskipTests compile exec:java@run-stress -Dzvec.stress.args='--docs 1000000 --queries 5000 --steady-state-rounds 50'\"");
  }

  private static Path prepareRunDirectory(Path baseDir, int docCount) throws IOException {
    Files.createDirectories(baseDir);
    String runName = "docs-" + docCount + "-" + Instant.now().toEpochMilli();
    return baseDir.resolve(runName);
  }

  private static long usedHeapBytes() {
    Runtime runtime = Runtime.getRuntime();
    runtime.gc();
    runtime.gc();
    return runtime.totalMemory() - runtime.freeMemory();
  }

  private static MemorySnapshot captureMemory() {
    return new MemorySnapshot(usedHeapBytes(), readResidentSetBytes());
  }

  private static Long readResidentSetBytes() {
    Process process = null;
    try {
      process =
          new ProcessBuilder(
                  "ps", "-o", "rss=", "-p", Long.toString(ProcessHandle.current().pid()))
              .redirectErrorStream(true)
              .start();
      byte[] output = process.getInputStream().readAllBytes();
      int exitCode = process.waitFor();
      if (exitCode != 0) {
        return null;
      }
      String value = new String(output, StandardCharsets.UTF_8).trim();
      if (value.isEmpty()) {
        return null;
      }
      return Long.parseLong(value) * 1024L;
    } catch (IOException | InterruptedException | NumberFormatException e) {
      if (e instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      return null;
    } finally {
      if (process != null) {
        process.destroy();
      }
    }
  }

  private static double bytesToMiB(long bytes) {
    return bytes / (1024.0 * 1024.0);
  }

  static boolean hasExpectedHit(String expectedId, List<Doc> results) {
    for (Doc result : results) {
      if (expectedId.equals(result.id())) {
        return true;
      }
    }
    return false;
  }

  private static String formatOptionalMiB(Long bytes) {
    if (bytes == null) {
      return "unavailable";
    }
    return String.format(Locale.ROOT, "%.2f", bytesToMiB(bytes));
  }

  private static Long delta(Long end, Long start) {
    if (end == null || start == null) {
      return null;
    }
    return end - start;
  }

  private static String formatConfig(StressOptions options, Path runDir) {
    return "docs="
        + options.docCount()
        + " queries="
        + options.queryCount()
        + " batch_size="
        + options.batchSize()
        + " dimension="
        + options.dimension()
        + " top_k="
        + options.topK()
        + " warmup_queries="
        + options.warmupQueries()
        + " steady_state_rounds="
        + options.steadyStateRounds()
        + " steady_insert_batch_size="
        + options.steadyInsertBatchSize()
        + " steady_queries_per_round="
        + options.steadyQueriesPerRound()
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
    VectorQuery query = VectorQuery.of("embedding", sample.vector()).topK(sample.topK()).outputFields("title");
    HnswQueryParams hnswQueryParams = options.hnswQueryParams();
    if (hnswQueryParams != null) {
      query.hnsw(hnswQueryParams);
    }
    return query;
  }

  private static String formatOptionalHnswM(StressOptions options) {
    return options.hnswIndexParams() == null ? "default" : Integer.toString(options.hnswIndexParams().m());
  }

  private static String formatOptionalHnswEfConstruction(StressOptions options) {
    return options.hnswIndexParams() == null
        ? "default"
        : Integer.toString(options.hnswIndexParams().efConstruction());
  }

  private static String formatOptionalHnswEf(StressOptions options) {
    return options.hnswQueryParams() == null ? "default" : Integer.toString(options.hnswQueryParams().ef());
  }

  private static final class InsertMetrics {
    private final int insertedDocs;
    private final long elapsedNanos;

    private InsertMetrics(int insertedDocs, long elapsedNanos) {
      this.insertedDocs = insertedDocs;
      this.elapsedNanos = elapsedNanos;
    }

    private int insertedDocs() {
      return insertedDocs;
    }

    private double elapsedSeconds() {
      return elapsedNanos / 1_000_000_000.0;
    }

    private double docsPerSecond() {
      return insertedDocs / elapsedSeconds();
    }
  }

  private static final class MemorySnapshot {
    private final long heapBytes;
    private final Long rssBytes;

    private MemorySnapshot(long heapBytes, Long rssBytes) {
      this.heapBytes = heapBytes;
      this.rssBytes = rssBytes;
    }

    private long heapBytes() {
      return heapBytes;
    }

    private Long rssBytes() {
      return rssBytes;
    }
  }

  private static final class QueryMetrics {
    private final LatencyStats latencyStats;
    private final int missCount;

    private QueryMetrics(LatencyStats latencyStats, int missCount) {
      this.latencyStats = latencyStats;
      this.missCount = missCount;
    }

    private LatencyStats latencyStats() {
      return latencyStats;
    }

    private int missCount() {
      return missCount;
    }

    private double recall() {
      return (latencyStats.count() - missCount) / (double) latencyStats.count();
    }
  }

  private static final class ReliabilityMetrics {
    private final int rounds;
    private final int insertFailures;
    private final int queryFailures;
    private final LatencyStats steadyStats;

    private ReliabilityMetrics(
        int rounds, int insertFailures, int queryFailures, LatencyStats steadyStats) {
      this.rounds = rounds;
      this.insertFailures = insertFailures;
      this.queryFailures = queryFailures;
      this.steadyStats = steadyStats;
    }

    private int rounds() {
      return rounds;
    }

    private int insertFailures() {
      return insertFailures;
    }

    private int queryFailures() {
      return queryFailures;
    }

    private LatencyStats steadyStats() {
      return steadyStats;
    }
  }
}
