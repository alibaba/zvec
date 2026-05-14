package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class ZvecSchemasBuilderTest {
  @Test
  void buildsSchemaWithBalancedVectorDefaults() {
    CollectionSchema schema =
        ZvecSchemas.collection("items")
            .string("title")
            .bool("active")
            .int64("rank")
            .doubleField("score")
            .vector("embedding", 4)
            .expectedDocCount(1_000_000L)
            .balanced()
            .build();

    assertEquals("items", schema.name());
    assertEquals(
        List.of(
            new FieldSchema("title", DataType.STRING, false),
            new FieldSchema("active", DataType.BOOL, false),
            new FieldSchema("rank", DataType.INT64, false),
            new FieldSchema("score", DataType.DOUBLE, false)),
        schema.fields());
    assertEquals(1, schema.vectors().size());
    assertEquals("embedding", schema.vector("embedding").name());
    assertEquals(4, schema.vector("embedding").dimension());
    assertEquals(TuningProfile.BALANCED, schema.vector("embedding").tuningProfile());
    assertEquals(Long.valueOf(1_000_000L), schema.vector("embedding").expectedDocCount());
  }

  @Test
  void tuningMethodsRequireAnActiveVectorField() {
    ZvecSchemas.Builder builder = ZvecSchemas.collection("items").string("title");

    IllegalStateException fast =
        assertThrows(IllegalStateException.class, builder::fast);
    IllegalStateException balanced =
        assertThrows(IllegalStateException.class, builder::balanced);
    IllegalStateException accurate =
        assertThrows(IllegalStateException.class, builder::accurate);
    IllegalStateException expectedDocCount =
        assertThrows(IllegalStateException.class, () -> builder.expectedDocCount(10L));

    assertEquals("fast() must follow vector(name, dimension)", fast.getMessage());
    assertEquals("balanced() must follow vector(name, dimension)", balanced.getMessage());
    assertEquals("accurate() must follow vector(name, dimension)", accurate.getMessage());
    assertEquals(
        "expectedDocCount(...) must follow vector(name, dimension)",
        expectedDocCount.getMessage());
  }

  @Test
  void laterProfilesOverrideEarlierProfilesOnTheActiveVector() {
    CollectionSchema schema =
        ZvecSchemas.collection("items")
            .vector("first", 2)
            .fast()
            .vector("second", 4)
            .balanced()
            .accurate()
            .build();

    assertEquals(TuningProfile.FAST, schema.vector("first").tuningProfile());
    assertEquals(TuningProfile.ACCURATE, schema.vector("second").tuningProfile());
  }
}
