package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class SchemaModelTest {
  @Test
  void duplicateFieldNamesAcrossScalarAndVectorDefinitionsThrow() {
    FieldSchema field = new FieldSchema("shared", DataType.STRING, false);
    VectorSchema vector = new VectorSchema("shared", DataType.VECTOR_FP32, 128);

    assertThrows(
        IllegalArgumentException.class,
        () -> new CollectionSchema("items", List.of(field), List.of(vector)));
  }

  @Test
  void nonVectorDataTypeForVectorSchemaThrows() {
    assertThrows(
        IllegalArgumentException.class, () -> new VectorSchema("embedding", DataType.STRING, 128));
  }

  @Test
  void scalarAndVectorDefinitionsAreStoredAndRetrievable() {
    FieldSchema id = new FieldSchema("id", DataType.STRING, false);
    FieldSchema active = new FieldSchema("active", DataType.BOOL, true);
    VectorSchema embedding = new VectorSchema("embedding", DataType.VECTOR_FP32, 1536);

    CollectionSchema schema =
        new CollectionSchema("items", List.of(id, active), List.of(embedding));

    assertEquals("items", schema.name());
    assertEquals(List.of(id, active), schema.fields());
    assertEquals(List.of(embedding), schema.vectors());
    assertSame(id, schema.field("id"));
    assertSame(active, schema.field("active"));
    assertSame(embedding, schema.vector("embedding"));
  }
}
