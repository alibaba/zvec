package org.zvec.perf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;
import org.zvec.Collection;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.VectorQuery;
import org.zvec.Zvec;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 1, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 1, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 2, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Threads(1)
public class ZvecJavaBindingBenchmark {
  private static final int DATASET_DOC_COUNT = 10_000;
  private static final int LOAD_BATCH_SIZE = 1_000;
  private static final int DIMENSION = 128;
  private static final int TOP_K = 10;
  private static final long SEED = 7L;

  public static void main(String[] args) throws RunnerException {
    Options options =
        new OptionsBuilder()
            .include(ZvecJavaBindingBenchmark.class.getName())
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
  public void queryProjectedScalarFields(QueryState state, Blackhole blackhole) {
    List<Doc> results =
        state.collection.query(
            VectorQuery.of("embedding", state.nextQueryVector()).topK(TOP_K).outputFields("title"));
    blackhole.consume(results.size());
    if (!results.isEmpty()) {
      Doc first = results.get(0);
      blackhole.consume(first.id());
      blackhole.consume(first.fields().get("title"));
    }
  }

  @Benchmark
  public void queryWithVectors(QueryState state, Blackhole blackhole) {
    List<Doc> results =
        state.collection.query(
            VectorQuery.of("embedding", state.nextQueryVector())
                .topK(TOP_K)
                .outputFields("title")
                .includeVector(true));
    blackhole.consume(results.size());
    if (!results.isEmpty()) {
      Doc first = results.get(0);
      blackhole.consume(first.id());
      blackhole.consume(first.fields().get("title"));
      blackhole.consume(first.vectors().get("embedding"));
    }
  }

  @State(Scope.Benchmark)
  public static class QueryState extends CollectionState {
    private final AtomicInteger nextQueryVectorIndex = new AtomicInteger();
    private float[][] queryVectors;

    @Override
    protected void afterSetUp() {
      queryVectors = new float[256][];
      for (int i = 0; i < queryVectors.length; i++) {
        queryVectors[i] = PerfData.querySample(i, DIMENSION, SEED, TOP_K).vector();
      }
    }

    float[] nextQueryVector() {
      int index = Math.floorMod(nextQueryVectorIndex.getAndIncrement(), queryVectors.length);
      return queryVectors[index];
    }
  }

  private abstract static class CollectionState {
    protected Collection collection;
    protected Path workDir;

    @Setup(Level.Trial)
    public final void setUp() throws IOException {
      assumeSupportedPlatform();
      workDir = Files.createTempDirectory("zvec-jmh-");
      CollectionSchema schema = PerfData.schema("perf_docs", DIMENSION);
      collection = Zvec.createAndOpen(workDir.resolve("collection").toString(), schema);

      try {
        for (int startDocIndex = 0; startDocIndex < DATASET_DOC_COUNT; startDocIndex += LOAD_BATCH_SIZE) {
          int batchSize = Math.min(LOAD_BATCH_SIZE, DATASET_DOC_COUNT - startDocIndex);
          int inserted = collection.insert(PerfData.docs(startDocIndex, batchSize, DIMENSION, SEED));
          if (inserted != batchSize) {
            throw new IllegalStateException(
                "Inserted count mismatch: expected " + batchSize + ", got " + inserted);
          }
        }
        collection.flush();
        afterSetUp();
      } catch (RuntimeException e) {
        tearDownQuietly();
        throw e;
      }
    }

    @TearDown(Level.Trial)
    public final void tearDown() throws IOException {
      IOException failure = null;
      if (collection != null) {
        try {
          collection.close();
        } catch (RuntimeException e) {
          failure = new IOException("Failed to close benchmark collection", e);
        } finally {
          collection = null;
        }
      }

      if (workDir != null) {
        try {
          deleteRecursively(workDir);
        } catch (IOException e) {
          if (failure == null) {
            failure = e;
          } else {
            failure.addSuppressed(e);
          }
        } finally {
          workDir = null;
        }
      }

      if (failure != null) {
        throw failure;
      }
    }

    private void tearDownQuietly() {
      try {
        tearDown();
      } catch (IOException ignored) {
      }
    }

    protected void afterSetUp() {}
  }

  private static void assumeSupportedPlatform() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));
  }

  private static void deleteRecursively(Path root) throws IOException {
    if (root == null || Files.notExists(root)) {
      return;
    }

    try (Stream<Path> paths = Files.walk(root)) {
      paths.sorted(Comparator.reverseOrder())
          .forEach(
              path -> {
                try {
                  Files.deleteIfExists(path);
                } catch (IOException e) {
                  throw new RecursiveDeleteException(e);
                }
              });
    } catch (RecursiveDeleteException e) {
      throw e.cause;
    }
  }

  private static final class RecursiveDeleteException extends RuntimeException {
    private final IOException cause;

    private RecursiveDeleteException(IOException cause) {
      super(cause);
      this.cause = cause;
    }
  }
}
