package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.zvec.crypto.EncryptionMetadata;
import org.zvec.crypto.EncryptionSpec;

class CollectionSchemaEncryptionTest {
  @Test
  void existingThreeArgConstructorReturnsEmptyEncryption() {
    CollectionSchema s = new CollectionSchema(
        "docs",
        List.of(new FieldSchema("title", DataType.STRING, false)),
        List.of(new VectorSchema("e", DataType.VECTOR_FP32, 4)));
    assertEquals(Optional.empty(), s.encryption());
  }

  @Test
  void fourArgConstructorRetainsMetadata() {
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.now(), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, "docs", Map.of("title", spec));
    CollectionSchema s = new CollectionSchema(
        "docs",
        List.of(new FieldSchema("title", DataType.STRING, false)),
        List.of(new VectorSchema("e", DataType.VECTOR_FP32, 4)),
        meta);
    assertSame(meta, s.encryption().orElseThrow());
  }
}
