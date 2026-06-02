package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.crypto.KeyProvider;
import org.zvec.crypto.KeyResolutionException;

public abstract class AbstractKeyRotationIntegrationTest {

  @Test
  void oldRecordsDecryptableUnderNewActiveKey(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();

    Map<String, byte[]> keys = new HashMap<>();
    byte[] k1 = new byte[32]; k1[0] = 1;
    byte[] k2 = new byte[32]; k2[0] = 2;
    keys.put("k1", k1);
    keys.put("k2", k2);
    KeyProvider provider = kid -> keys.get(kid);

    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      col.insert(List.of(Doc.of("d1").field("body", "old-text")
          .vector("e", new float[] {1f, 0f, 0f, 0f})));

      col.setActiveKeyId("body", "k2");
      col.insert(List.of(Doc.of("d2").field("body", "new-text")
          .vector("e", new float[] {1f, 0f, 0f, 0f})));

      List<Doc> results = col.query(
          ZvecSearch.vector("e", new float[] {1f, 0f, 0f, 0f})
              .topK(2).project("body").build());
      assertEquals(2, results.size());
      Map<String, String> byId = new HashMap<>();
      for (Doc d : results) byId.put(d.id(), (String) d.fields().get("body"));
      assertEquals("old-text", byId.get("d1"));
      assertEquals("new-text", byId.get("d2"));
    }
  }

  @Test
  void revokingOldKeyMakesOldRecordsUnreadable(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();

    Map<String, byte[]> keys = new HashMap<>();
    byte[] k1 = new byte[32]; k1[0] = 1;
    byte[] k2 = new byte[32]; k2[0] = 2;
    keys.put("k1", k1);
    keys.put("k2", k2);

    try (Collection col = Zvec.createAndOpen(path.toString(), schema, kid -> keys.get(kid))) {
      col.insert(List.of(Doc.of("d1").field("body", "old")
          .vector("e", new float[] {1f, 0f, 0f, 0f})));
    }

    keys.remove("k1");
    try (Collection col = Zvec.openWithKeys(path.toString(), kid -> keys.get(kid))) {
      assertThrows(KeyResolutionException.class, () ->
          col.query(ZvecSearch.vector("e", new float[] {1f, 0f, 0f, 0f})
              .topK(1).project("body").build()));
    }
  }
}
