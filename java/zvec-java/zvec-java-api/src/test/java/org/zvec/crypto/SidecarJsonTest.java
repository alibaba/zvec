package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.time.Instant;
import java.util.Map;
import org.junit.jupiter.api.Test;

class SidecarJsonTest {
  @Test
  void roundTripFullMetadata() {
    EncryptionSpec spec = new EncryptionSpec(
        "AES-256-GCM", "body-key-v1",
        Instant.parse("2026-04-28T03:14:15Z"),
        Instant.parse("2026-04-28T05:00:00Z"));
    EncryptionMetadata original = new EncryptionMetadata(1, "docs", Map.of("body", spec));

    String json = SidecarJson.write(original);
    EncryptionMetadata back = SidecarJson.read(json);

    assertEquals(original, back);
  }

  @Test
  void roundTripWithoutRotatedAt() {
    EncryptionSpec spec = new EncryptionSpec(
        "AES-256-GCM", "k1", Instant.parse("2026-04-28T00:00:00Z"), null);
    EncryptionMetadata original = new EncryptionMetadata(1, "c", Map.of("f", spec));

    String json = SidecarJson.write(original);
    EncryptionMetadata back = SidecarJson.read(json);
    assertEquals(original, back);
  }

  @Test
  void roundTripEmptyFields() {
    EncryptionMetadata original = EncryptionMetadata.empty("c");
    EncryptionMetadata back = SidecarJson.read(SidecarJson.write(original));
    assertEquals(original, back);
  }

  @Test
  void rejectsMalformed() {
    assertThrows(EncryptionMetadataIOException.class, () -> SidecarJson.read("not json"));
    assertThrows(EncryptionMetadataIOException.class, () -> SidecarJson.read("{"));
    assertThrows(EncryptionMetadataIOException.class, () -> SidecarJson.read("{\"version\":\"1\"}"));
  }

  @Test
  void rejectsUnknownVersion() {
    String s = "{\"version\":2,\"collection_name\":\"c\",\"fields\":{}}";
    assertThrows(IllegalArgumentException.class, () -> SidecarJson.read(s));
  }

  @Test
  void rejectsTrailingGarbage() {
    String good = "{\"version\":1,\"collection_name\":\"c\",\"fields\":{}}";
    String bad = good + "extra";
    assertThrows(EncryptionMetadataIOException.class, () -> SidecarJson.read(bad));
  }

  @Test
  void allowsTrailingWhitespace() {
    String s = "{\"version\":1,\"collection_name\":\"c\",\"fields\":{}}\n  \n";
    EncryptionMetadata back = SidecarJson.read(s);
    org.junit.jupiter.api.Assertions.assertEquals("c", back.collectionName());
  }
}
