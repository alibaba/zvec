package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.Map;
import org.junit.jupiter.api.Test;

class EncryptionMetadataTest {
  @Test
  void specFieldsRetained() {
    EncryptionSpec s = new EncryptionSpec("AES-256-GCM", "k1", Instant.parse("2026-04-28T00:00:00Z"), null);
    assertEquals("AES-256-GCM", s.alg());
    assertEquals("k1", s.activeKeyId());
    assertEquals(Instant.parse("2026-04-28T00:00:00Z"), s.createdAt());
    org.junit.jupiter.api.Assertions.assertNull(s.rotatedAt());
  }

  @Test
  void specRejectsBadAlg() {
    assertThrows(IllegalArgumentException.class,
        () -> new EncryptionSpec("AES-128-CBC", "k1", Instant.now(), null));
  }

  @Test
  void specRejectsEmptyKeyId() {
    assertThrows(IllegalArgumentException.class,
        () -> new EncryptionSpec("AES-256-GCM", "", Instant.now(), null));
  }

  @Test
  void metadataExposesEncryptedFieldNames() {
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.now(), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, "docs", Map.of("body", spec, "ssn", spec));
    assertEquals(java.util.Set.of("body", "ssn"), meta.encryptedFieldNames());
    assertSame(spec, meta.spec("body"));
    assertTrue(meta.isEncrypted("body"));
    assertEquals(false, meta.isEncrypted("title"));
  }

  @Test
  void emptyMetadataConstant() {
    EncryptionMetadata empty = EncryptionMetadata.empty("docs");
    assertTrue(empty.encryptedFieldNames().isEmpty());
    assertEquals("docs", empty.collectionName());
  }

  @Test
  void rejectsUnknownVersion() {
    assertThrows(IllegalArgumentException.class,
        () -> new EncryptionMetadata(2, "docs", Map.of()));
  }
}
