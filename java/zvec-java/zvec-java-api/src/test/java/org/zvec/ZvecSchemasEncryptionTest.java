package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.zvec.crypto.EncryptionMetadata;

class ZvecSchemasEncryptionTest {
  @Test
  void encryptedAttachesToPrecedingStringField() {
    CollectionSchema s = ZvecSchemas.collection("docs")
        .string("title")
        .string("body").encrypted("body-key-v1")
        .vector("embed", 4).balanced()
        .build();
    EncryptionMetadata meta = s.encryption().orElseThrow();
    assertEquals(1, meta.fields().size());
    assertEquals("body-key-v1", meta.spec("body").activeKeyId());
    assertEquals("docs", meta.collectionName());
    assertEquals("AES-256-GCM", meta.spec("body").alg());
  }

  @Test
  void encryptedRequiresPrecedingStringField() {
    ZvecSchemas.Builder b1 = ZvecSchemas.collection("docs");
    assertThrows(IllegalStateException.class, () -> b1.encrypted("k"));

    ZvecSchemas.Builder b2 = ZvecSchemas.collection("docs").vector("e", 4);
    assertThrows(IllegalStateException.class, () -> b2.encrypted("k"));

    ZvecSchemas.Builder b3 = ZvecSchemas.collection("docs").int64("salary");
    assertThrows(org.zvec.crypto.UnsupportedFieldTypeException.class, () -> b3.encrypted("k"));
  }

  @Test
  void rejectsDuplicateEncryptionOnSameField() {
    ZvecSchemas.Builder b = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1");
    assertThrows(IllegalStateException.class, () -> b.encrypted("k2"));
  }

  @Test
  void schemaWithoutEncryptedFieldsHasEmptyEncryption() {
    CollectionSchema s = ZvecSchemas.collection("docs")
        .string("title").vector("e", 4).build();
    assertTrue(s.encryption().isEmpty());
  }

  @Test
  void encryptedWithStaticKeyEmbedsProvider() {
    byte[] key = new byte[32];
    key[0] = 7;
    CollectionSchema s = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1", key)
        .vector("e", 4).build();
    org.zvec.crypto.KeyProvider provider = s.embeddedKeyProviders().orElseThrow().get("body");
    assertEquals(7, provider.resolve("k1")[0]);
    org.junit.jupiter.api.Assertions.assertNull(provider.resolve("other"));
  }

  @Test
  void encryptedWithStaticKeyRejectsBadLength() {
    ZvecSchemas.Builder b = ZvecSchemas.collection("docs").string("body");
    assertThrows(IllegalArgumentException.class, () -> b.encrypted("k1", new byte[16]));
  }
}
