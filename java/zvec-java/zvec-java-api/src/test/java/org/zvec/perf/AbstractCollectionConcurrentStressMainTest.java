package org.zvec.perf;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public abstract class AbstractCollectionConcurrentStressMainTest {
  @TempDir Path tempDir;

  @Test
  void printsConcurrentStressSummaryForSmallRun() throws Exception {
    assumeSupportedPlatform();

    PrintStream originalOut = System.out;
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();

    try {
      System.setOut(new PrintStream(buffer, true, StandardCharsets.UTF_8));
      CollectionConcurrentStressMain.main(
          new String[] {
            "--docs",
            "200",
            "--concurrent-query-threads",
            "2",
            "--concurrent-query-count",
            "10",
            "--concurrent-mixed-threads",
            "2",
            "--concurrent-mixed-rounds",
            "2",
            "--concurrent-mixed-insert-batch-size",
            "5",
            "--concurrent-mixed-queries-per-round",
            "8",
            "--work-dir",
            tempDir.resolve("concurrent").toString()
          });
    } finally {
      System.setOut(originalOut);
    }

    String output = buffer.toString(StandardCharsets.UTF_8);
    assertTrue(output.contains("CONCURRENT_STRESS_CONFIG docs=200"));
    assertTrue(output.contains("QUERY_ONLY_SUMMARY"));
    assertTrue(output.contains("queries_per_sec="));
    assertTrue(output.contains("MIXED_SUMMARY"));
    assertTrue(output.contains("insert_docs_per_sec="));
    assertTrue(output.contains("query_failures="));
    assertTrue(output.contains("insert_failures="));
    assertTrue(output.contains("ARTIFACT_DIR " + tempDir.resolve("concurrent")));
    assertTrue(output.contains("exec:java@run-concurrent-stress"));
  }

  private static void assumeSupportedPlatform() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));
  }
}
