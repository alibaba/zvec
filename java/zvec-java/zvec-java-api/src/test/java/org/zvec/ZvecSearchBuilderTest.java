package org.zvec;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

class ZvecSearchBuilderTest {
  @Test
  void buildsBalancedProjectedQuery() {
    VectorQuery query =
        ZvecSearch.vector("embedding", new float[] {1f, 0f, 0f, 0f})
            .topK(7)
            .balanced()
            .project("title", "category")
            .build();

    assertEquals("embedding", query.fieldName());
    assertArrayEquals(new float[] {1f, 0f, 0f, 0f}, query.queryVector());
    assertEquals(7, query.topK());
    assertEquals(TuningProfile.BALANCED, query.tuningProfile());
    assertTrue(query.outputFieldsSpecified());
    assertEquals(java.util.List.of("title", "category"), query.outputFields());
    assertFalse(query.includeVector());
    assertNull(query.filter());
    assertNull(query.hnswQueryParams());
  }

  @Test
  void laterProfilesOverrideEarlierProfiles() {
    VectorQuery query =
        ZvecSearch.vector("embedding", new float[] {1f, 0f, 0f, 0f})
            .fast()
            .accurate()
            .build();

    assertNull(query.hnswQueryParams());
    assertEquals(TuningProfile.ACCURATE, query.tuningProfile());
  }

  @Test
  void includeVectorAndFilterAreCarriedIntoVectorQuery() {
    VectorQuery query =
        ZvecSearch.vector("embedding", new float[] {1f, 0f, 0f, 0f})
            .fast()
            .includeVector()
            .filter("title = 'alpha'")
            .build();

    assertTrue(query.includeVector());
    assertEquals("title = 'alpha'", query.filter());
    assertFalse(query.outputFieldsSpecified());
    assertEquals(TuningProfile.FAST, query.tuningProfile());
    assertNull(query.hnswQueryParams());
  }
}
