package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.CollectionSchema;
import org.zvec.Zvec;
import org.zvec.ZvecSchemas;

public abstract class AbstractZvecCreateAndOpenWithProviderTest {

  @Test
  void writesSidecarBeforeNativeOpen(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("title").string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> "k1".equals(kid) ? new byte[32] : null;

    try (org.zvec.Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      assertTrue(Files.exists(path.resolve(SidecarMetadata.FILENAME)));
      Optional<EncryptionMetadata> meta = SidecarMetadata.read(path);
      assertEquals("docs", meta.orElseThrow().collectionName());
      assertEquals(java.util.Set.of("body"), meta.get().encryptedFieldNames());
    }
  }

  @Test
  void rejectsNullProviderWhenSchemaHasEncryptedField(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    assertThrows(NullPointerException.class,
        () -> Zvec.createAndOpen(path.toString(), schema, null));
  }

  @Test
  void twoArgCreateAndOpenAcceptsAllStaticKey(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    byte[] key = new byte[32];
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1", key)
        .vector("e", 4).build();
    try (org.zvec.Collection col = Zvec.createAndOpen(path.toString(), schema)) {
      assertEquals(java.util.Set.of("body"), col.encryptedSchema().encryptedFieldNames());
    }
  }

  @Test
  void twoArgCreateAndOpenRejectsKeyIdOnlyEncryption(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    assertThrows(EncryptedCollectionException.class,
        () -> Zvec.createAndOpen(path.toString(), schema));
  }

  @Test
  void twoArgCreateAndOpenAllowsNoEncryption(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("title").vector("e", 4).build();
    try (org.zvec.Collection col = Zvec.createAndOpen(path.toString(), schema)) {
      assertEquals(java.util.Set.of(), col.encryptedSchema().encryptedFieldNames());
    }
  }

  @Test
  void threeArgCreateAndOpenInitializesNativeImplicitly(@TempDir Path tmp) {
    // Smoke test: even though earlier tests in the same JVM may have initialized native,
    // calling the encrypted entrypoint directly should not assume prior initialization.
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (org.zvec.Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      assertEquals(java.util.Set.of("body"), col.encryptedSchema().encryptedFieldNames());
    }
  }
}
