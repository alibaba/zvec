package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.Collection;
import org.zvec.CollectionSchema;
import org.zvec.Zvec;
import org.zvec.ZvecSchemas;

public abstract class AbstractCollectionSetActiveKeyIdTest {

  @Test
  void rotateUpdatesSidecar(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];

    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      col.setActiveKeyId("body", "k2");
      assertEquals("k2", col.encryptedSchema().activeKeyId("body"));
    }
    Optional<EncryptionMetadata> back = SidecarMetadata.read(path);
    assertNotNull(back.orElseThrow());
    assertEquals("k2", back.get().spec("body").activeKeyId());
    assertNotNull(back.get().spec("body").rotatedAt());
  }

  @Test
  void rotateRejectsNonEncryptedField(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      assertThrows(IllegalArgumentException.class, () -> col.setActiveKeyId("title", "k2"));
    }
  }

  @Test
  void postRotationReadStillDecryptsOldRecords(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    org.zvec.CollectionSchema schema = org.zvec.ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();

    java.util.Map<String, byte[]> keys = new java.util.HashMap<>();
    byte[] k1 = new byte[32]; k1[0] = 1;
    byte[] k2 = new byte[32]; k2[0] = 2;
    keys.put("k1", k1);
    keys.put("k2", k2);
    KeyProvider provider = kid -> keys.get(kid);

    try (org.zvec.Collection col = org.zvec.Zvec.createAndOpen(path.toString(), schema, provider)) {
      col.insert(java.util.List.of(
          org.zvec.Doc.of("d1").field("body", "secret-old")
              .vector("e", new float[] {1f, 0f, 0f, 0f})));

      col.setActiveKeyId("body", "k2");

      java.util.List<org.zvec.Doc> results = col.query(
          org.zvec.ZvecSearch.vector("e", new float[] {1f, 0f, 0f, 0f})
              .topK(1).project("body").build());
      assertEquals(1, results.size());
      assertEquals("secret-old", results.get(0).fields().get("body"));
    }
  }
}
