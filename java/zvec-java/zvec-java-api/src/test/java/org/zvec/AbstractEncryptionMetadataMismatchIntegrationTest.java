package org.zvec;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.crypto.EncryptionMetadataIOException;
import org.zvec.crypto.EncryptionMetadataMismatchException;
import org.zvec.crypto.KeyProvider;
import org.zvec.crypto.SidecarMetadata;

public abstract class AbstractEncryptionMetadataMismatchIntegrationTest {

  @Test
  void corruptedSidecarSurfacesAsIOException(@TempDir Path tmp) throws Exception {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {}

    Files.writeString(path.resolve(SidecarMetadata.FILENAME), "totally-not-json{");
    assertThrows(EncryptionMetadataIOException.class,
        () -> Zvec.openWithKeys(path.toString(), provider));
  }

  @Test
  void renamedFieldInSidecarSurfacesAsMismatch(@TempDir Path tmp) throws Exception {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {}

    String text = Files.readString(path.resolve(SidecarMetadata.FILENAME));
    Files.writeString(path.resolve(SidecarMetadata.FILENAME),
        text.replace("\"body\":", "\"renamed_field\":"));
    assertThrows(EncryptionMetadataMismatchException.class,
        () -> Zvec.openWithKeys(path.toString(), provider));
  }

  @Test
  void mismatchedCollectionNameSurfacesAsMismatch(@TempDir Path tmp) throws Exception {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {}

    String text = Files.readString(path.resolve(SidecarMetadata.FILENAME));
    Files.writeString(path.resolve(SidecarMetadata.FILENAME),
        text.replace("\"docs\"", "\"OTHER\""));
    assertThrows(EncryptionMetadataMismatchException.class,
        () -> Zvec.openWithKeys(path.toString(), provider));
  }
}
