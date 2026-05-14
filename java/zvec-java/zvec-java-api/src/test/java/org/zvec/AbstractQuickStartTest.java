package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public abstract class AbstractQuickStartTest {
  @TempDir Path tempDir;

  @Test
  void quickStartFlowWorks() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        ZvecSchemas.collection("docs").string("title").vector("embedding", 4).balanced().build();

    try (Collection collection = Zvec.createAndOpen(tempDir.resolve("docs").toString(), schema)) {
      collection.insert(
          List.of(
              Doc.of("doc_1")
                  .field("title", "alpha")
                  .vector("embedding", new float[] {1f, 0f, 0f, 0f}),
              Doc.of("doc_2")
                  .field("title", "beta")
                  .vector("embedding", new float[] {0f, 1f, 0f, 0f})));

      List<Doc> results =
          collection.query(
              ZvecSearch.vector("embedding", new float[] {1f, 0f, 0f, 0f})
                  .topK(2)
                  .project("title")
                  .build());

      assertFalse(results.isEmpty());
      assertEquals("doc_1", results.get(0).id());
      assertEquals("alpha", results.get(0).fields().get("title"));
      assertNotNull(results.get(0).score());
    }
  }

  private static void assumeSupportedPlatform() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));
  }
}
