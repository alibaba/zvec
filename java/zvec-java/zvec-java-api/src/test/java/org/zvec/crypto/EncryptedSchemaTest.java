package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.zvec.CollectionSchema;
import org.zvec.DataType;
import org.zvec.FieldSchema;
import org.zvec.VectorSchema;

class EncryptedSchemaTest {
  private static CollectionSchema docsSchema() {
    return new CollectionSchema(
        "docs",
        List.of(new FieldSchema("title", DataType.STRING, false),
                new FieldSchema("body",  DataType.STRING, false)),
        List.of(new VectorSchema("embed", DataType.VECTOR_FP32, 4)));
  }

  private static EncryptionMetadata metaFor(String collName, String fieldName, String keyId) {
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", keyId, Instant.now(), null);
    return new EncryptionMetadata(1, collName, Map.of(fieldName, spec));
  }

  @Test
  void reconcileSucceedsForMatchingFields() {
    EncryptedSchema es = EncryptedSchema.reconcile(
        docsSchema(),
        metaFor("docs", "body", "k1"),
        keyId -> new byte[32]);
    assertNotNull(es);
    assertTrue(es.isEncrypted("body"));
    assertEquals("k1", es.activeKeyId("body"));
  }

  @Test
  void reconcileFailsWhenCollectionNamesDiffer() {
    assertThrows(EncryptionMetadataMismatchException.class, () ->
        EncryptedSchema.reconcile(docsSchema(),
            metaFor("OTHER", "body", "k1"), keyId -> new byte[32]));
  }

  @Test
  void reconcileFailsWhenFieldMissing() {
    assertThrows(EncryptionMetadataMismatchException.class, () ->
        EncryptedSchema.reconcile(docsSchema(),
            metaFor("docs", "missing", "k1"), keyId -> new byte[32]));
  }

  @Test
  void reconcileFailsWhenFieldNotString() {
    CollectionSchema schema = new CollectionSchema(
        "docs",
        List.of(new FieldSchema("salary", DataType.INT64, false)),
        List.of(new VectorSchema("embed", DataType.VECTOR_FP32, 4)));
    assertThrows(EncryptionMetadataMismatchException.class, () ->
        EncryptedSchema.reconcile(schema,
            metaFor("docs", "salary", "k1"), keyId -> new byte[32]));
  }

  @Test
  void reconcileEmptyMetadataReturnsNoneSentinel() {
    EncryptedSchema es = EncryptedSchema.reconcile(
        docsSchema(), EncryptionMetadata.empty("docs"), keyId -> new byte[32]);
    assertSame(EncryptedSchema.NONE, es);
    assertEquals(false, es.isEncrypted("body"));
  }

  @Test
  void noneSentinelHasNoEncryptedFields() {
    assertTrue(EncryptedSchema.NONE.encryptedFieldNames().isEmpty());
    assertEquals(false, EncryptedSchema.NONE.isEncrypted("any"));
  }
}
