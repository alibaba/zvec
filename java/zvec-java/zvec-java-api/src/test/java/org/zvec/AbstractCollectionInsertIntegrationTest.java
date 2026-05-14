package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public abstract class AbstractCollectionInsertIntegrationTest {
  @TempDir Path tempDir;

  @Test
  void insertsDocumentsWithAllScalarTypesAndNullableField() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        new CollectionSchema(
            "docs",
            List.of(
                new FieldSchema("title", DataType.STRING, false),
                new FieldSchema("published", DataType.BOOL, false),
                new FieldSchema("views", DataType.INT64, false),
                new FieldSchema("rating", DataType.DOUBLE, false),
                new FieldSchema("subtitle", DataType.STRING, true)),
            List.of(new VectorSchema("embedding", DataType.VECTOR_FP32, 4)));

    try (Collection collection = Zvec.createAndOpen(tempDir.resolve("docs").toString(), schema)) {
      int inserted =
          collection.insert(
              List.of(
                  Doc.of("doc_1")
                      .field("title", "alpha")
                      .field("published", true)
                      .field("views", 11L)
                      .field("rating", 4.5d)
                      .nullField("subtitle")
                      .vector("embedding", new float[] {1f, 0f, 0f, 0f}),
                  Doc.of("doc_2")
                      .field("title", "beta")
                      .field("published", false)
                      .field("views", 22L)
                      .field("rating", 3.25d)
                      .field("subtitle", "second")
                      .vector("embedding", new float[] {0f, 1f, 0f, 0f})));

      assertEquals(2, inserted);
    }
  }

  @Test
  void rejectsVectorDimensionMismatch() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        new CollectionSchema(
            "docs",
            List.of(new FieldSchema("title", DataType.STRING, false)),
            List.of(new VectorSchema("embedding", DataType.VECTOR_FP32, 4)));

    try (Collection collection = Zvec.createAndOpen(tempDir.resolve("docs_bad").toString(), schema)) {
      IllegalArgumentException ex =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  collection.insert(
                      List.of(
                          Doc.of("doc_bad")
                              .field("title", "bad")
                              .vector("embedding", new float[] {1f, 0f}))));
      assertTrue(ex.getMessage().contains("Vector dimension mismatch"));
    }
  }

  private static void assumeSupportedPlatform() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));
  }
}
