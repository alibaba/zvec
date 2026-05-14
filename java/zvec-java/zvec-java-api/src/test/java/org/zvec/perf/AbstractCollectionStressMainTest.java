package org.zvec.perf;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public abstract class AbstractCollectionStressMainTest {
  @TempDir Path tempDir;

  @Test
  void printsStressSummaryForSmallRun() throws Exception {
    assumeSupportedPlatform();

    PrintStream originalOut = System.out;
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();

    try {
      System.setOut(new PrintStream(buffer, true, StandardCharsets.UTF_8));
      CollectionStressMain.main(
          new String[] {
            "--docs",
            "200",
            "--queries",
            "20",
            "--batch-size",
            "50",
            "--warmup-queries",
            "5",
            "--steady-state-rounds",
            "2",
            "--steady-insert-batch-size",
            "10",
            "--steady-queries-per-round",
            "5",
            "--work-dir",
            tempDir.resolve("stress").toString()
          });
    } finally {
      System.setOut(originalOut);
    }

    String output = buffer.toString(StandardCharsets.UTF_8);
    assertTrue(output.contains("STRESS_CONFIG docs=200"));
    assertTrue(output.contains("INSERT_SUMMARY docs=200"));
    assertTrue(output.contains("QUERY_SUMMARY count=20"));
    assertTrue(output.contains("miss_count="));
    assertTrue(output.contains("recall="));
    assertTrue(output.contains("MEMORY_SUMMARY"));
    assertTrue(output.contains("heap_before_mb="));
    assertTrue(output.contains("rss_before_mb="));
    assertTrue(output.contains("STEADY_STATE rounds=2"));
    assertTrue(output.contains("ARTIFACT_DIR " + tempDir.resolve("stress")));
    assertTrue(output.contains("exec:java@run-stress"));
    assertTrue(output.contains("-Dzvec.stress.args='--docs 100000'"));
  }

  private static void assumeSupportedPlatform() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));
  }
}
