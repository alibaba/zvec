package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SidecarMetadataTest {
  @Test
  void readReturnsEmptyWhenSidecarAbsent(@TempDir Path dir) {
    Optional<EncryptionMetadata> result = SidecarMetadata.read(dir);
    assertTrue(result.isEmpty());
  }

  @Test
  void writeThenRead(@TempDir Path dir) {
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.parse("2026-04-28T00:00:00Z"), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, "docs", Map.of("body", spec));
    SidecarMetadata.write(dir, meta);

    Optional<EncryptionMetadata> back = SidecarMetadata.read(dir);
    assertTrue(back.isPresent());
    assertEquals(meta, back.get());
    assertTrue(Files.exists(dir.resolve("_zvec_enc.json")));
  }

  @Test
  void atomicReplaceLeavesNoTempFile(@TempDir Path dir) throws IOException {
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.now(), null);
    EncryptionMetadata first = new EncryptionMetadata(1, "docs", Map.of("body", spec));
    SidecarMetadata.write(dir, first);
    EncryptionSpec spec2 = new EncryptionSpec("AES-256-GCM", "k2", Instant.now(), Instant.now());
    EncryptionMetadata second = new EncryptionMetadata(1, "docs", Map.of("body", spec2));
    SidecarMetadata.write(dir, second);

    assertTrue(Files.exists(dir.resolve("_zvec_enc.json")));
    try (var stream = Files.list(dir)) {
      assertEquals(0L, stream.filter(p -> p.getFileName().toString().endsWith(".tmp")).count());
    }
    assertEquals(second, SidecarMetadata.read(dir).orElseThrow());
  }

  @Test
  void corruptedFileSurfacesAsIOException(@TempDir Path dir) throws IOException {
    Files.writeString(dir.resolve("_zvec_enc.json"), "garbage}not}json");
    assertThrows(EncryptionMetadataIOException.class, () -> SidecarMetadata.read(dir));
  }
}
