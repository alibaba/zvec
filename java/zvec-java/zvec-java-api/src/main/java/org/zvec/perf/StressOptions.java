package org.zvec.perf;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import org.zvec.HnswIndexParams;
import org.zvec.HnswQueryParams;

public final class StressOptions {
  private final int docCount;
  private final int queryCount;
  private final int batchSize;
  private final int dimension;
  private final int topK;
  private final int warmupQueries;
  private final int steadyStateRounds;
  private final int steadyInsertBatchSize;
  private final int steadyQueriesPerRound;
  private final int concurrentQueryThreads;
  private final int concurrentQueryCount;
  private final int concurrentMixedThreads;
  private final int concurrentMixedRounds;
  private final int concurrentMixedInsertBatchSize;
  private final int concurrentMixedQueriesPerRound;
  private final HnswIndexParams hnswIndexParams;
  private final HnswQueryParams hnswQueryParams;
  private final long seed;
  private final Path workDir;

  public StressOptions(
      int docCount,
      int queryCount,
      int batchSize,
      int dimension,
      int topK,
      int warmupQueries,
      int steadyStateRounds,
      int steadyInsertBatchSize,
      int steadyQueriesPerRound,
      int concurrentQueryThreads,
      int concurrentQueryCount,
      int concurrentMixedThreads,
      int concurrentMixedRounds,
      int concurrentMixedInsertBatchSize,
      int concurrentMixedQueriesPerRound,
      HnswIndexParams hnswIndexParams,
      HnswQueryParams hnswQueryParams,
      long seed,
      Path workDir) {
    requirePositive(docCount, "docCount");
    requirePositive(queryCount, "queryCount");
    requirePositive(batchSize, "batchSize");
    requirePositive(dimension, "dimension");
    requirePositive(topK, "topK");
    requirePositive(warmupQueries, "warmupQueries");
    requireNonNegative(steadyStateRounds, "steadyStateRounds");
    requirePositive(steadyInsertBatchSize, "steadyInsertBatchSize");
    requirePositive(steadyQueriesPerRound, "steadyQueriesPerRound");
    requirePositive(concurrentQueryThreads, "concurrentQueryThreads");
    requirePositive(concurrentQueryCount, "concurrentQueryCount");
    requirePositive(concurrentMixedThreads, "concurrentMixedThreads");
    requirePositive(concurrentMixedRounds, "concurrentMixedRounds");
    requirePositive(concurrentMixedInsertBatchSize, "concurrentMixedInsertBatchSize");
    requirePositive(concurrentMixedQueriesPerRound, "concurrentMixedQueriesPerRound");
    if (workDir == null) {
      throw new IllegalArgumentException("workDir must not be null");
    }
    this.docCount = docCount;
    this.queryCount = queryCount;
    this.batchSize = batchSize;
    this.dimension = dimension;
    this.topK = topK;
    this.warmupQueries = warmupQueries;
    this.steadyStateRounds = steadyStateRounds;
    this.steadyInsertBatchSize = steadyInsertBatchSize;
    this.steadyQueriesPerRound = steadyQueriesPerRound;
    this.concurrentQueryThreads = concurrentQueryThreads;
    this.concurrentQueryCount = concurrentQueryCount;
    this.concurrentMixedThreads = concurrentMixedThreads;
    this.concurrentMixedRounds = concurrentMixedRounds;
    this.concurrentMixedInsertBatchSize = concurrentMixedInsertBatchSize;
    this.concurrentMixedQueriesPerRound = concurrentMixedQueriesPerRound;
    this.hnswIndexParams = hnswIndexParams;
    this.hnswQueryParams = hnswQueryParams;
    this.seed = seed;
    this.workDir = workDir;
  }

  public int docCount() {
    return docCount;
  }

  public int queryCount() {
    return queryCount;
  }

  public int batchSize() {
    return batchSize;
  }

  public int dimension() {
    return dimension;
  }

  public int topK() {
    return topK;
  }

  public int warmupQueries() {
    return warmupQueries;
  }

  public int steadyStateRounds() {
    return steadyStateRounds;
  }

  public int steadyInsertBatchSize() {
    return steadyInsertBatchSize;
  }

  public int steadyQueriesPerRound() {
    return steadyQueriesPerRound;
  }

  public int concurrentQueryThreads() {
    return concurrentQueryThreads;
  }

  public int concurrentQueryCount() {
    return concurrentQueryCount;
  }

  public int concurrentMixedThreads() {
    return concurrentMixedThreads;
  }

  public int concurrentMixedRounds() {
    return concurrentMixedRounds;
  }

  public int concurrentMixedInsertBatchSize() {
    return concurrentMixedInsertBatchSize;
  }

  public int concurrentMixedQueriesPerRound() {
    return concurrentMixedQueriesPerRound;
  }

  public HnswIndexParams hnswIndexParams() {
    return hnswIndexParams;
  }

  public HnswQueryParams hnswQueryParams() {
    return hnswQueryParams;
  }

  public long seed() {
    return seed;
  }

  public Path workDir() {
    return workDir;
  }

  public static StressOptions parse(String[] args) {
    Map<String, String> values = new HashMap<>();
    for (int i = 0; i < args.length; i++) {
      String arg = args[i];
      if (!arg.startsWith("--")) {
        throw new IllegalArgumentException("Expected option starting with --, got: " + arg);
      }
      if (i + 1 >= args.length) {
        throw new IllegalArgumentException("Missing value for option: " + arg);
      }
      values.put(arg, args[++i]);
    }

    validateKnownOptions(values);

    return new StressOptions(
        parseInt(values, "--docs", 100_000),
        parseInt(values, "--queries", 1_000),
        parseInt(values, "--batch-size", 1_000),
        parseInt(values, "--dimension", 128),
        parseInt(values, "--top-k", 10),
        parseInt(values, "--warmup-queries", 100),
        parseInt(values, "--steady-state-rounds", 20),
        parseInt(values, "--steady-insert-batch-size", 100),
        parseInt(values, "--steady-queries-per-round", 20),
        parseInt(values, "--concurrent-query-threads", 2),
        parseInt(values, "--concurrent-query-count", 20),
        parseInt(values, "--concurrent-mixed-threads", 2),
        parseInt(values, "--concurrent-mixed-rounds", 2),
        parseInt(values, "--concurrent-mixed-insert-batch-size", 5),
        parseInt(values, "--concurrent-mixed-queries-per-round", 5),
        parseHnswIndexParams(values),
        parseHnswQueryParams(values),
        parseLong(values, "--seed", 7L),
        Path.of(values.getOrDefault("--work-dir", "target/perf/zvec-stress")));
  }

  private static void validateKnownOptions(Map<String, String> values) {
    for (String key : values.keySet()) {
      if (!isKnownOption(key)) {
        throw new IllegalArgumentException("Unknown option: " + key);
      }
    }
  }

  private static boolean isKnownOption(String key) {
    switch (key) {
      case "--docs":
      case "--queries":
      case "--batch-size":
      case "--dimension":
      case "--top-k":
      case "--warmup-queries":
      case "--steady-state-rounds":
      case "--steady-insert-batch-size":
      case "--steady-queries-per-round":
      case "--concurrent-query-threads":
      case "--concurrent-query-count":
      case "--concurrent-mixed-threads":
      case "--concurrent-mixed-rounds":
      case "--concurrent-mixed-insert-batch-size":
      case "--concurrent-mixed-queries-per-round":
      case "--hnsw-m":
      case "--hnsw-ef-construction":
      case "--hnsw-ef":
      case "--seed":
      case "--work-dir":
        return true;
      default:
        return false;
    }
  }

  private static HnswIndexParams parseHnswIndexParams(Map<String, String> values) {
    boolean hasM = values.containsKey("--hnsw-m");
    boolean hasEfConstruction = values.containsKey("--hnsw-ef-construction");
    if (hasM != hasEfConstruction) {
      throw new IllegalArgumentException(
          "Both --hnsw-m and --hnsw-ef-construction are required together");
    }
    if (!hasM) {
      return null;
    }
    return new HnswIndexParams(
        Integer.parseInt(values.get("--hnsw-m")),
        Integer.parseInt(values.get("--hnsw-ef-construction")));
  }

  private static HnswQueryParams parseHnswQueryParams(Map<String, String> values) {
    if (!values.containsKey("--hnsw-ef")) {
      return null;
    }
    return new HnswQueryParams(Integer.parseInt(values.get("--hnsw-ef")), 0.0f, false, false);
  }

  private static int parseInt(Map<String, String> values, String key, int defaultValue) {
    return values.containsKey(key) ? Integer.parseInt(values.get(key)) : defaultValue;
  }

  private static long parseLong(Map<String, String> values, String key, long defaultValue) {
    return values.containsKey(key) ? Long.parseLong(values.get(key)) : defaultValue;
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0");
    }
  }

  private static void requireNonNegative(int value, String name) {
    if (value < 0) {
      throw new IllegalArgumentException(name + " must be >= 0");
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof StressOptions)) {
      return false;
    }
    StressOptions other = (StressOptions) obj;
    return docCount == other.docCount
        && queryCount == other.queryCount
        && batchSize == other.batchSize
        && dimension == other.dimension
        && topK == other.topK
        && warmupQueries == other.warmupQueries
        && steadyStateRounds == other.steadyStateRounds
        && steadyInsertBatchSize == other.steadyInsertBatchSize
        && steadyQueriesPerRound == other.steadyQueriesPerRound
        && concurrentQueryThreads == other.concurrentQueryThreads
        && concurrentQueryCount == other.concurrentQueryCount
        && concurrentMixedThreads == other.concurrentMixedThreads
        && concurrentMixedRounds == other.concurrentMixedRounds
        && concurrentMixedInsertBatchSize == other.concurrentMixedInsertBatchSize
        && concurrentMixedQueriesPerRound == other.concurrentMixedQueriesPerRound
        && seed == other.seed
        && Objects.equals(hnswIndexParams, other.hnswIndexParams)
        && Objects.equals(hnswQueryParams, other.hnswQueryParams)
        && workDir.equals(other.workDir);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        docCount,
        queryCount,
        batchSize,
        dimension,
        topK,
        warmupQueries,
        steadyStateRounds,
        steadyInsertBatchSize,
        steadyQueriesPerRound,
        concurrentQueryThreads,
        concurrentQueryCount,
        concurrentMixedThreads,
        concurrentMixedRounds,
        concurrentMixedInsertBatchSize,
        concurrentMixedQueriesPerRound,
        hnswIndexParams,
        hnswQueryParams,
        seed,
        workDir);
  }
}
