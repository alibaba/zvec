package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class HnswParamsModelTest {
  @Test
  void hnswIndexParamsRejectNonPositiveValues() {
    assertThrows(IllegalArgumentException.class, () -> new HnswIndexParams(0, 100));
    assertThrows(IllegalArgumentException.class, () -> new HnswIndexParams(16, 0));
  }

  @Test
  void hnswQueryParamsRejectInvalidEfAndRadius() {
    assertThrows(IllegalArgumentException.class, () -> new HnswQueryParams(0, 0.0f, false, false));
    assertThrows(
        IllegalArgumentException.class, () -> new HnswQueryParams(64, -1.0f, false, false));
  }

  @Test
  void vectorSchemaCanCarryOptionalHnswIndexParams() {
    VectorSchema schema = new VectorSchema("embedding", DataType.VECTOR_FP32, 128);
    HnswIndexParams params = new HnswIndexParams(32, 300);

    VectorSchema configured = schema.withHnswIndex(params);

    assertNull(schema.hnswIndexParams());
    assertEquals(params, configured.hnswIndexParams());
  }

  @Test
  void vectorSchemaClearsTuningHintsWhenRawIndexParamsAreApplied() {
    VectorSchema schema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 128)
            .withTuningProfile(TuningProfile.ACCURATE, 1_000_000L);
    HnswIndexParams params = new HnswIndexParams(32, 300);

    VectorSchema configured = schema.withHnswIndex(params);

    assertEquals(TuningProfile.ACCURATE, schema.tuningProfile());
    assertEquals(Long.valueOf(1_000_000L), schema.expectedDocCount());
    assertNull(configured.tuningProfile());
    assertNull(configured.expectedDocCount());
    assertEquals(params, configured.hnswIndexParams());
  }

  @Test
  void vectorQueryCanCarryOptionalHnswQueryParams() {
    HnswQueryParams params = new HnswQueryParams(128, 0.0f, false, true);

    VectorQuery query = VectorQuery.of("embedding", new float[] {1.0f, 2.0f}).hnsw(params);

    assertSame(params, query.hnswQueryParams());
  }

  @Test
  void vectorQueryClearsTuningHintsWhenRawHnswParamsAreApplied() {
    HnswQueryParams params = new HnswQueryParams(128, 0.0f, false, true);

    VectorQuery query =
        VectorQuery.of("embedding", new float[] {1.0f, 2.0f})
            .withTuningProfile(TuningProfile.BALANCED)
            .hnsw(params);

    assertNull(query.tuningProfile());
    assertSame(params, query.hnswQueryParams());
  }
}
