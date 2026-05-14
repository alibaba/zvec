package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.crypto.KeyProvider;

public abstract class AbstractEncryptedCollectionRoundTripIntegrationTest {

  @Test
  void thousandDocsRoundTripWithBodyEncrypted(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("title").string("body").encrypted("k1")
        .vector("embed", 4).balanced()
        .build();
    KeyProvider provider = kid -> {
      byte[] k = new byte[32];
      k[0] = 7;
      return k;
    };

    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      List<Doc> docs = new ArrayList<>();
      for (int i = 0; i < 1000; i++) {
        docs.add(Doc.of("d" + i)
            .field("title", "t" + i)
            .field("body", "secret-" + i)
            .vector("embed", new float[] {1f, 0f, 0f, (float) i / 1000f}));
      }
      col.insert(docs);

      List<Doc> results = col.query(
          ZvecSearch.vector("embed", new float[] {1f, 0f, 0f, 0f})
              .topK(10).project("title", "body").build());

      assertEquals(10, results.size());
      for (Doc d : results) {
        String id = d.id();
        assertTrue(id.startsWith("d"));
        int idx = Integer.parseInt(id.substring(1));
        assertEquals("t" + idx, d.fields().get("title"));
        assertEquals("secret-" + idx, d.fields().get("body"));
      }
    }
  }
}
