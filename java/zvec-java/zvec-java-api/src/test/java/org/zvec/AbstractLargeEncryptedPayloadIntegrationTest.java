package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.crypto.KeyProvider;

public abstract class AbstractLargeEncryptedPayloadIntegrationTest {

  @Test
  void oneMegabytePayloadRoundTrips(@TempDir Path tmp) {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];

    char[] chars = new char[1024 * 1024];
    for (int i = 0; i < chars.length; i++) chars[i] = (char) ('a' + (i % 26));
    String big = new String(chars);

    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      col.insert(List.of(Doc.of("d1").field("body", big)
          .vector("e", new float[] {1f, 0f, 0f, 0f})));

      List<Doc> results = col.query(
          ZvecSearch.vector("e", new float[] {1f, 0f, 0f, 0f})
              .topK(1).project("body").build());
      assertEquals(big, results.get(0).fields().get("body"));
    }
  }
}
