package org.zvec.perf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class LatencyStatsTest {
  @Test
  void computesPercentilesFromSamples() {
    LatencyStats stats =
        LatencyStats.fromNanos(
            new long[] {
              1_000_000L, 2_000_000L, 3_000_000L, 4_000_000L, 5_000_000L
            });

    assertEquals(5, stats.count());
    assertEquals(1_000.0, stats.minMicros());
    assertEquals(3_000.0, stats.p50Micros());
    assertEquals(5_000.0, stats.p95Micros());
    assertEquals(5_000.0, stats.p99Micros());
    assertEquals(5_000.0, stats.maxMicros());
    assertEquals(3_000.0, stats.meanMicros());
  }

  @Test
  void rejectsEmptySamples() {
    assertThrows(IllegalArgumentException.class, () -> LatencyStats.fromNanos(new long[0]));
  }
}
