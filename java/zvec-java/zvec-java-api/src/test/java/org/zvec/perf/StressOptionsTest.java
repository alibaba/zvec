package org.zvec.perf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.zvec.HnswIndexParams;
import org.zvec.HnswQueryParams;

class StressOptionsTest {
  @Test
  void parsesExplicitArguments() {
    StressOptions options =
        StressOptions.parse(
            new String[] {
              "--docs", "1000000",
              "--queries", "2500",
              "--batch-size", "2000",
              "--dimension", "256",
              "--top-k", "20",
              "--warmup-queries", "200",
              "--steady-state-rounds", "12",
              "--steady-insert-batch-size", "500",
              "--steady-queries-per-round", "40",
              "--concurrent-query-threads", "3",
              "--concurrent-query-count", "24",
              "--concurrent-mixed-threads", "4",
              "--concurrent-mixed-rounds", "6",
              "--concurrent-mixed-insert-batch-size", "7",
              "--concurrent-mixed-queries-per-round", "8",
              "--hnsw-m", "32",
              "--hnsw-ef-construction", "300",
              "--hnsw-ef", "128",
              "--seed", "99",
              "--work-dir", "target/perf/custom"
            });

    assertEquals(1_000_000, options.docCount());
    assertEquals(2_500, options.queryCount());
    assertEquals(2_000, options.batchSize());
    assertEquals(256, options.dimension());
    assertEquals(20, options.topK());
    assertEquals(200, options.warmupQueries());
    assertEquals(12, options.steadyStateRounds());
    assertEquals(500, options.steadyInsertBatchSize());
    assertEquals(40, options.steadyQueriesPerRound());
    assertEquals(3, options.concurrentQueryThreads());
    assertEquals(24, options.concurrentQueryCount());
    assertEquals(4, options.concurrentMixedThreads());
    assertEquals(6, options.concurrentMixedRounds());
    assertEquals(7, options.concurrentMixedInsertBatchSize());
    assertEquals(8, options.concurrentMixedQueriesPerRound());
    assertEquals(new HnswIndexParams(32, 300), options.hnswIndexParams());
    assertEquals(new HnswQueryParams(128, 0.0f, false, false), options.hnswQueryParams());
    assertEquals(99L, options.seed());
    assertEquals(Path.of("target/perf/custom"), options.workDir());
  }

  @Test
  void usesDefaultsWhenArgumentsAreOmitted() {
    StressOptions options = StressOptions.parse(new String[0]);

    assertEquals(100_000, options.docCount());
    assertEquals(1_000, options.queryCount());
    assertEquals(1_000, options.batchSize());
    assertEquals(128, options.dimension());
    assertEquals(10, options.topK());
    assertEquals(100, options.warmupQueries());
    assertEquals(20, options.steadyStateRounds());
    assertEquals(100, options.steadyInsertBatchSize());
    assertEquals(20, options.steadyQueriesPerRound());
    assertEquals(2, options.concurrentQueryThreads());
    assertEquals(20, options.concurrentQueryCount());
    assertEquals(2, options.concurrentMixedThreads());
    assertEquals(2, options.concurrentMixedRounds());
    assertEquals(5, options.concurrentMixedInsertBatchSize());
    assertEquals(5, options.concurrentMixedQueriesPerRound());
    assertNull(options.hnswIndexParams());
    assertNull(options.hnswQueryParams());
    assertEquals(7L, options.seed());
    assertEquals(Path.of("target/perf/zvec-stress"), options.workDir());
  }

  @Test
  void rejectsUnknownArguments() {
    assertThrows(
        IllegalArgumentException.class,
        () -> StressOptions.parse(new String[] {"--nope", "1"}));
  }

  @Test
  void rejectsPartialHnswIndexArguments() {
    assertThrows(
        IllegalArgumentException.class,
        () -> StressOptions.parse(new String[] {"--hnsw-m", "32"}));
    assertThrows(
        IllegalArgumentException.class,
        () -> StressOptions.parse(new String[] {"--hnsw-ef-construction", "300"}));
  }
}
