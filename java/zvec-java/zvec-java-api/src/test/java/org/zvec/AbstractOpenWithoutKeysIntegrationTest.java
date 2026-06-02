package org.zvec;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.crypto.EncryptedCollectionException;
import org.zvec.crypto.KeyProvider;

public abstract class AbstractOpenWithoutKeysIntegrationTest {

  @Test
  void plainOpenOnEncryptedCollectionThrows(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {}

    assertThrows(EncryptedCollectionException.class, () -> Zvec.open(path.toString()));
  }

  @Test
  void openWithKeysWithNullProviderThrows(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {}

    assertThrows(NullPointerException.class, () -> Zvec.openWithKeys(path.toString(), null));
  }
}
