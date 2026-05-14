package org.zvec.perf;

import java.util.Arrays;

public final class LatencyStats {
  private final int count;
  private final double minMicros;
  private final double p50Micros;
  private final double p95Micros;
  private final double p99Micros;
  private final double maxMicros;
  private final double meanMicros;

  private LatencyStats(
      int count,
      double minMicros,
      double p50Micros,
      double p95Micros,
      double p99Micros,
      double maxMicros,
      double meanMicros) {
    this.count = count;
    this.minMicros = minMicros;
    this.p50Micros = p50Micros;
    this.p95Micros = p95Micros;
    this.p99Micros = p99Micros;
    this.maxMicros = maxMicros;
    this.meanMicros = meanMicros;
  }

  public static LatencyStats fromNanos(long[] samplesNanos) {
    if (samplesNanos == null || samplesNanos.length == 0) {
      throw new IllegalArgumentException("samplesNanos must not be empty");
    }

    long[] sorted = samplesNanos.clone();
    Arrays.sort(sorted);

    long sum = 0L;
    for (long sample : sorted) {
      sum += sample;
    }

    return new LatencyStats(
        sorted.length,
        nanosToMicros(sorted[0]),
        nanosToMicros(percentile(sorted, 0.50)),
        nanosToMicros(percentile(sorted, 0.95)),
        nanosToMicros(percentile(sorted, 0.99)),
        nanosToMicros(sorted[sorted.length - 1]),
        nanosToMicros((double) sum / sorted.length));
  }

  public int count() {
    return count;
  }

  public double minMicros() {
    return minMicros;
  }

  public double p50Micros() {
    return p50Micros;
  }

  public double p95Micros() {
    return p95Micros;
  }

  public double p99Micros() {
    return p99Micros;
  }

  public double maxMicros() {
    return maxMicros;
  }

  public double meanMicros() {
    return meanMicros;
  }

  private static long percentile(long[] sorted, double percentile) {
    int index = (int) Math.ceil(percentile * sorted.length) - 1;
    if (index < 0) {
      index = 0;
    }
    return sorted[index];
  }

  private static double nanosToMicros(long nanos) {
    return nanos / 1_000.0;
  }

  private static double nanosToMicros(double nanos) {
    return nanos / 1_000.0;
  }
}
