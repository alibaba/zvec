package org.zvec.internal.ffm;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.zvec.DataType;
import org.zvec.HnswQueryParams;
import org.zvec.TuningProfile;
import org.zvec.VectorQuery;
import org.zvec.VectorSchema;

class FfmQueriesTest {
  @Test
  void attachesHnswParamsWhenRuntimeSchemaCarriesHnswIndexParams() {
    VectorSchema runtimeSchema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
            .withHnswIndex(new org.zvec.HnswIndexParams(16, 200));

    assertTrue(
        FfmQueries.shouldAttachHnswParams(
            runtimeSchema, VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})));
  }

  @Test
  void skipsHnswParamsWhenRuntimeSchemaHasNoHnswContext() {
    VectorSchema runtimeSchema = new VectorSchema("embedding", DataType.VECTOR_FP32, 4);

    assertFalse(
        FfmQueries.shouldAttachHnswParams(
            runtimeSchema, VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})));
  }

  @Test
  void skipsExplicitQueryHnswTuningWhenRuntimeSchemaHasNoHnswContext() {
    VectorSchema runtimeSchema = new VectorSchema("embedding", DataType.VECTOR_FP32, 4);
    VectorQuery query =
        VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})
            .withTuningProfile(TuningProfile.ACCURATE)
            .hnsw(new HnswQueryParams(128, 0.0f, false, false));

    assertFalse(FfmQueries.shouldAttachHnswParams(runtimeSchema, query));
  }

  @Test
  void resolvesQueryDefaultsFromPublicSchemaWhenRuntimeSchemaCarriesHnswContext() {
    VectorSchema runtimeSchema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
            .withHnswIndex(new org.zvec.HnswIndexParams(32, 400));
    VectorSchema publicSchema =
        new VectorSchema("embedding", DataType.VECTOR_FP32, 4)
            .withTuningProfile(TuningProfile.ACCURATE, 1_000_000L);

    assertEquals(
        new HnswQueryParams(128, 0.0f, false, false),
        FfmQueries.resolveAttachedHnswParams(
            runtimeSchema,
            publicSchema,
            VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})));
  }
}
