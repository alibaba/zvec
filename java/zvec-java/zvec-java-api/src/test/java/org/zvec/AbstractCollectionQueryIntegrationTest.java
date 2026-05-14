package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.nio.file.Files;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.internal.HnswDefaults;

public abstract class AbstractCollectionQueryIntegrationTest {
  @TempDir Path tempDir;

  @Test
  void queriesInsertedVectors() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        new CollectionSchema(
            "docs",
            List.of(new FieldSchema("title", DataType.STRING, false)),
            List.of(new VectorSchema("embedding", DataType.VECTOR_FP32, 4)));

    try (Collection collection = Zvec.createAndOpen(tempDir.resolve("docs").toString(), schema)) {
      collection.insert(
          List.of(
              Doc.of("doc_1").field("title", "alpha").vector("embedding", new float[] {1f, 0f, 0f, 0f}),
              Doc.of("doc_2").field("title", "beta").vector("embedding", new float[] {0f, 1f, 0f, 0f})));

      List<Doc> results =
          collection.query(
              VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f}).topK(2).outputFields("title"));

      assertFalse(results.isEmpty());
      assertEquals("doc_1", results.get(0).id());
      assertEquals("alpha", results.get(0).fields().get("title"));
      assertNotNull(results.get(0).score());

      List<Doc> vectorResults =
          collection.query(
              VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})
                  .topK(1)
                  .includeVector(true));

      assertFalse(vectorResults.isEmpty());
      assertArrayEquals(
          new float[] {1f, 0f, 0f, 0f}, vectorResults.get(0).vectors().get("embedding"));
    }
  }

  @Test
  void queriesInsertedVectorsWithFluentSearchBuilder() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        new CollectionSchema(
            "docs_fluent",
            List.of(new FieldSchema("title", DataType.STRING, false)),
            List.of(new VectorSchema("embedding", DataType.VECTOR_FP32, 4)));

    try (Collection collection = Zvec.createAndOpen(tempDir.resolve("docs_fluent").toString(), schema)) {
      collection.insert(
          List.of(
              Doc.of("doc_1").field("title", "alpha").vector("embedding", new float[] {1f, 0f, 0f, 0f}),
              Doc.of("doc_2").field("title", "beta").vector("embedding", new float[] {0f, 1f, 0f, 0f})));

      List<Doc> results =
          collection.query(
              ZvecSearch.vector("embedding", new float[] {1f, 0f, 0f, 0f})
                  .topK(2)
                  .balanced()
                  .project("title")
                  .build());

      assertFalse(results.isEmpty());
      assertEquals("doc_1", results.get(0).id());
      assertEquals("alpha", results.get(0).fields().get("title"));
      assertNotNull(results.get(0).score());
    }
  }

  @Test
  void queriesInsertedVectorsWithExplicitHnswParams() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        new CollectionSchema(
            "docs_hnsw",
            List.of(new FieldSchema("title", DataType.STRING, false)),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withHnswIndex(new HnswIndexParams(16, 200))));

    try (Collection collection =
        Zvec.createAndOpen(tempDir.resolve("docs_hnsw").toString(), schema)) {
      collection.insert(
          List.of(
              Doc.of("doc_1").field("title", "alpha").vector("embedding", new float[] {1f, 0f, 0f, 0f}),
              Doc.of("doc_2").field("title", "beta").vector("embedding", new float[] {0f, 1f, 0f, 0f})));

      List<Doc> results =
          collection.query(
              VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})
                  .topK(2)
                  .outputFields("title")
                  .hnsw(new HnswQueryParams(64, 0.0f, false, false)));

      assertFalse(results.isEmpty());
      assertEquals("doc_1", results.get(0).id());
      assertEquals("alpha", results.get(0).fields().get("title"));
    }
  }

  @Test
  void createAndOpenPreservesVectorTuningHintsForSchemaLevelDefaults() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        new CollectionSchema(
            "docs_tuning",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withTuningProfile(TuningProfile.ACCURATE, 1_000_000L)));

    try (Collection collection =
        Zvec.createAndOpen(tempDir.resolve("docs_tuning").toString(), schema)) {
      VectorSchema vector = collection.schema().vector("embedding");

      assertEquals(TuningProfile.ACCURATE, vector.tuningProfile());
      assertEquals(Long.valueOf(1_000_000L), vector.expectedDocCount());
      assertEquals(
          new HnswQueryParams(128, 0.0f, false, false),
          HnswDefaults.resolveQueryParams(
              vector, VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})));
    }
  }

  @Test
  void openReadsBackEffectiveRawHnswIndexParams() {
    assumeSupportedPlatform();

    Path path = tempDir.resolve("docs_reopen");
    CollectionSchema schema =
        new CollectionSchema(
            "docs_reopen",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withHnswIndex(new HnswIndexParams(18, 220))));

    try (Collection ignored = Zvec.createAndOpen(path.toString(), schema)) {
    }

    try (Collection reopened = Zvec.open(path.toString())) {
      assertEquals(
          new HnswIndexParams(18, 220),
          reopened.schema().vector("embedding").hnswIndexParams());
    }
  }

  @Test
  void openPreservesSchemaLevelQueryDefaultsViaEffectiveIndexParams() {
    assumeSupportedPlatform();

    Path path = tempDir.resolve("docs_reopen_defaults");
    CollectionSchema schema =
        new CollectionSchema(
            "docs_reopen_defaults",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withTuningProfile(TuningProfile.ACCURATE, 1_000_000L)));

    try (Collection ignored = Zvec.createAndOpen(path.toString(), schema)) {
    }

    try (Collection reopened = Zvec.open(path.toString())) {
      VectorSchema vector = reopened.schema().vector("embedding");

      assertNull(vector.hnswIndexParams());
      assertEquals(TuningProfile.ACCURATE, vector.tuningProfile());
      assertEquals(Long.valueOf(1_000_000L), vector.expectedDocCount());
      assertEquals(
          new HnswQueryParams(128, 0.0f, false, false),
          HnswDefaults.resolveQueryParams(
              vector,
              VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})));
    }
  }

  @Test
  void rejectsUnknownVectorFieldWithClearError() {
    assumeSupportedPlatform();

    CollectionSchema schema =
        new CollectionSchema(
            "docs_invalid_query",
            List.of(new FieldSchema("title", DataType.STRING, false)),
            List.of(new VectorSchema("embedding", DataType.VECTOR_FP32, 4)));

    try (Collection collection =
        Zvec.createAndOpen(tempDir.resolve("docs_invalid_query").toString(), schema)) {
      IllegalArgumentException ex =
          assertThrows(
              IllegalArgumentException.class,
              () -> collection.query(VectorQuery.of("missing", new float[] {1f, 0f, 0f, 0f})));
      assertTrue(ex.getMessage().contains("Unknown vector field"));
    }
  }

  @Test
  void openFallsBackToNativeSchemaWhenJavaMetadataIsMalformed() throws Exception {
    assumeSupportedPlatform();

    Path path = tempDir.resolve("docs_bad_metadata");
    CollectionSchema schema =
        new CollectionSchema(
            "docs_bad_metadata",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withHnswIndex(new HnswIndexParams(18, 220))));

    try (Collection ignored = Zvec.createAndOpen(path.toString(), schema)) {
    }

    Files.writeString(
        path.resolve(".zvec-java-schema.properties"),
        "version=1\nvector.embedding.tuningProfile=NOT_A_PROFILE\n");

    try (Collection reopened = Zvec.open(path.toString())) {
      assertEquals(
          new HnswIndexParams(18, 220),
          reopened.schema().vector("embedding").hnswIndexParams());
    }
  }

  @Test
  void openPrefersNativeRawHnswParamsOverJavaMetadata() throws Exception {
    assumeSupportedPlatform();

    Path path = tempDir.resolve("docs_stale_metadata");
    CollectionSchema schema =
        new CollectionSchema(
            "docs_stale_metadata",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withHnswIndex(new HnswIndexParams(18, 220))));

    try (Collection ignored = Zvec.createAndOpen(path.toString(), schema)) {
    }

    Files.writeString(
        path.resolve(".zvec-java-schema.properties"),
        "version=1\nvector.embedding.hnsw.m=32\nvector.embedding.hnsw.efConstruction=400\n");

    try (Collection reopened = Zvec.open(path.toString())) {
      assertEquals(
          new HnswIndexParams(18, 220),
          reopened.schema().vector("embedding").hnswIndexParams());
    }
  }

  @Test
  void openPreservesRawParamsWhenOnlyExpectedDocCountMetadataExists() {
    assumeSupportedPlatform();

    Path path = tempDir.resolve("docs_raw_with_doccount");
    CollectionSchema schema =
        new CollectionSchema(
            "docs_raw_with_doccount",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withHnswIndex(new HnswIndexParams(18, 220))
                    .withExpectedDocCount(1_000_000L)));

    try (Collection ignored = Zvec.createAndOpen(path.toString(), schema)) {
    }

    try (Collection reopened = Zvec.open(path.toString())) {
      VectorSchema vector = reopened.schema().vector("embedding");

      assertEquals(new HnswIndexParams(18, 220), vector.hnswIndexParams());
      assertNull(vector.tuningProfile());
      assertEquals(Long.valueOf(1_000_000L), vector.expectedDocCount());
    }
  }

  @Test
  void openPreservesExpectedDocCountOnlyHintState() {
    assumeSupportedPlatform();

    Path path = tempDir.resolve("docs_doccount_only");
    CollectionSchema schema =
        new CollectionSchema(
            "docs_doccount_only",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withExpectedDocCount(1_000_000L)));

    try (Collection ignored = Zvec.createAndOpen(path.toString(), schema)) {
    }

    try (Collection reopened = Zvec.open(path.toString())) {
      VectorSchema vector = reopened.schema().vector("embedding");

      assertNull(vector.hnswIndexParams());
      assertNull(vector.tuningProfile());
      assertEquals(Long.valueOf(1_000_000L), vector.expectedDocCount());
      assertEquals(
          new HnswQueryParams(96, 0.0f, false, false),
          HnswDefaults.resolveQueryParams(
              vector, VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})));
    }
  }

  @Test
  void openFallsBackToNativeSchemaWhenMetadataOmitsRawStateFlag() throws Exception {
    assumeSupportedPlatform();

    Path path = tempDir.resolve("docs_missing_raw_flag");
    CollectionSchema schema =
        new CollectionSchema(
            "docs_missing_raw_flag",
            List.of(),
            List.of(
                new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
                    .withExpectedDocCount(1_000_000L)));

    try (Collection ignored = Zvec.createAndOpen(path.toString(), schema)) {
    }

    Files.writeString(
        path.resolve(".zvec-java-schema.properties"),
        "version=1\nvector.embedding.expectedDocCount=1000000\n");

    try (Collection reopened = Zvec.open(path.toString())) {
      VectorSchema vector = reopened.schema().vector("embedding");

      assertEquals(new HnswIndexParams(24, 300), vector.hnswIndexParams());
      assertNull(vector.tuningProfile());
      assertNull(vector.expectedDocCount());
    }
  }

  private static void assumeSupportedPlatform() {
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    Assumptions.assumeTrue(osName.contains("mac"));
    Assumptions.assumeTrue(osArch.equals("aarch64") || osArch.equals("arm64"));
  }
}
