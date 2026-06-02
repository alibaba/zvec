package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public abstract class AbstractCollectionLifecycleIntegrationTest {
  @TempDir Path tempDir;

  @Test
  void createsAndReopensCollection() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));

    CollectionSchema schema =
        new CollectionSchema(
            "docs",
            List.of(
                new FieldSchema("title", DataType.STRING, false),
                new FieldSchema("summary", DataType.STRING, true)),
            List.of(new VectorSchema("embedding", DataType.VECTOR_FP32, 4)));

    Path collectionPath = tempDir.resolve("docs");
    try (Collection created = Zvec.createAndOpen(collectionPath.toString(), schema)) {
      assertNotNull(created.schema());
      assertEquals("docs", created.schema().name());
      assertTrue(created.schema().field("summary").nullable());
      created.flush();
    }

    try (Collection reopened = Zvec.open(collectionPath.toString())) {
      assertEquals("docs", reopened.schema().name());
      assertEquals(DataType.STRING, reopened.schema().field("title").dataType());
      assertTrue(reopened.schema().field("summary").nullable());
      assertEquals(4, reopened.schema().vector("embedding").dimension());
    }
  }
}
