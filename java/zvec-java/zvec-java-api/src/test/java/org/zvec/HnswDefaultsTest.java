package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.zvec.internal.HnswDefaults;

class HnswDefaultsTest {
  @Test
  void resolvesBalancedDefaultsWhenNoHintsAreProvided() {
    VectorSchema schema = new VectorSchema("embedding", DataType.VECTOR_FP32, 128);
    VectorQuery query = VectorQuery.of("embedding", new float[] {1.0f, 2.0f});

    assertEquals(new HnswIndexParams(16, 200), HnswDefaults.resolveIndexParams(schema));
    assertEquals(
        new HnswQueryParams(64, 0.0f, false, false), HnswDefaults.resolveQueryParams(schema, query));
  }

  @Test
  void resolvesProfileAndDocCountDefaults() {
    VectorSchema schema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 128)
            .withTuningProfile(TuningProfile.FAST, 100_000L);
    VectorQuery query = VectorQuery.of("embedding", new float[] {1.0f, 2.0f});

    assertEquals(new HnswIndexParams(12, 120), HnswDefaults.resolveIndexParams(schema));
    assertEquals(
        new HnswQueryParams(32, 0.0f, false, false), HnswDefaults.resolveQueryParams(schema, query));
  }

  @Test
  void respectsExplicitProfilesWithoutExpectedDocCount() {
    VectorSchema schema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 128)
            .withTuningProfile(TuningProfile.ACCURATE);
    VectorQuery query =
        VectorQuery.of("embedding", new float[] {1.0f, 2.0f})
            .withTuningProfile(TuningProfile.FAST);

    assertEquals(new HnswIndexParams(24, 300), HnswDefaults.resolveIndexParams(schema));
    assertEquals(
        new HnswQueryParams(32, 0.0f, false, false), HnswDefaults.resolveQueryParams(schema, query));
  }

  @Test
  void queryProfileOverridesSchemaProfileWhenResolvingQueryDefaults() {
    VectorSchema schema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 128)
            .withTuningProfile(TuningProfile.FAST, 1_000_000L);
    VectorQuery query =
        VectorQuery.of("embedding", new float[] {1.0f, 2.0f})
            .withTuningProfile(TuningProfile.ACCURATE);

    assertEquals(
        new HnswQueryParams(128, 0.0f, false, false),
        HnswDefaults.resolveQueryParams(schema, query));
  }

  @Test
  void rawParamsOverrideProfilesAndDocCountHints() {
    VectorSchema schema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 128)
            .withTuningProfile(TuningProfile.ACCURATE, 1_000_000L)
            .withHnswIndex(new HnswIndexParams(18, 220));
    VectorQuery query =
        VectorQuery.of("embedding", new float[] {1.0f, 2.0f})
            .withTuningProfile(TuningProfile.BALANCED)
            .hnsw(new HnswQueryParams(144, 0.0f, false, true));

    assertEquals(new HnswIndexParams(18, 220), HnswDefaults.resolveIndexParams(schema));
    assertEquals(
        new HnswQueryParams(144, 0.0f, false, true), HnswDefaults.resolveQueryParams(schema, query));
  }
}
