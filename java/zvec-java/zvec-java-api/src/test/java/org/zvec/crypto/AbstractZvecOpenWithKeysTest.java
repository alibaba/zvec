package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.Collection;
import org.zvec.CollectionSchema;
import org.zvec.Zvec;
import org.zvec.ZvecSchemas;

public abstract class AbstractZvecOpenWithKeysTest {

  @Test
  void reopenWithKeysAttachesEncryption(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) { /* close */ }

    try (Collection col = Zvec.openWithKeys(path.toString(), provider)) {
      EncryptedSchema es = col.encryptedSchema();
      assertEquals(java.util.Set.of("body"), es.encryptedFieldNames());
    }
  }

  @Test
  void openOnEncryptedCollectionWithoutProviderThrows(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {}

    assertThrows(EncryptedCollectionException.class, () -> Zvec.open(path.toString()));
  }
}
