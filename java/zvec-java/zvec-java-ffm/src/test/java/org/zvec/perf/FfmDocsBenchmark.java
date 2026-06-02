package org.zvec.perf;

import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.internal.ffm.FfmDocs;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 1, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 1, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 2, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Threads(1)
public class FfmDocsBenchmark {
  private static final int DATASET_DOC_COUNT = 10_000;
  private static final int SMALL_BATCH_SIZE = 16;
  private static final int DIMENSION = 128;
  private static final long SEED = 7L;

  public static void main(String[] args) throws RunnerException {
    Options options =
        new OptionsBuilder()
            .include(FfmDocsBenchmark.class.getName())
            .forks(1)
            .warmupIterations(1)
            .warmupTime(TimeValue.milliseconds(500))
            .measurementIterations(2)
            .measurementTime(TimeValue.milliseconds(500))
            .shouldDoGC(true)
            .build();
    new Runner(options).run();
  }

  @Benchmark
  public void marshalInsertSmallBatch(MarshalState state, Blackhole blackhole) {
    var nativeDocs = FfmDocs.toFfmDocs(state.docs, state.schema);
    try {
      blackhole.consume(nativeDocs.size());
    } finally {
      FfmDocs.destroyAll(nativeDocs);
    }
  }

  @State(Scope.Benchmark)
  public static class MarshalState {
    CollectionSchema schema;
    List<Doc> docs;

    @Setup(Level.Trial)
    public void setUp() {
      assumeSupportedPlatform();
      schema = PerfData.schema("perf_docs", DIMENSION);
      docs = PerfData.docs(DATASET_DOC_COUNT, SMALL_BATCH_SIZE, DIMENSION, SEED);
    }
  }

  private static void assumeSupportedPlatform() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));
  }
}
