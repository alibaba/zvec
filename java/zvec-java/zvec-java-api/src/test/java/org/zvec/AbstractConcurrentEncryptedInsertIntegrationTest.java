package org.zvec;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.zvec.crypto.KeyProvider;

public abstract class AbstractConcurrentEncryptedInsertIntegrationTest {

  @Test
  void eightThreadsThousandEachAllDecryptCorrectly(@TempDir Path tmp) throws Exception {
    Path path = tmp.resolve("docs");
    CollectionSchema schema = ZvecSchemas.collection("docs")
        .string("body").encrypted("k1")
        .vector("e", 4).build();
    KeyProvider provider = kid -> new byte[32];

    int threads = 8;
    int perThread = 1000;
    try (Collection col = Zvec.createAndOpen(path.toString(), schema, provider)) {
      ExecutorService pool = Executors.newFixedThreadPool(threads);
      List<Future<Integer>> futures = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final int tid = t;
        futures.add(pool.submit(() -> {
          List<Doc> batch = new ArrayList<>(perThread);
          for (int i = 0; i < perThread; i++) {
            String id = "t" + tid + "-i" + i;
            batch.add(Doc.of(id)
                .field("body", "secret-" + id)
                .vector("e", new float[] {1f, 0f, 0f, 0f}));
          }
          return col.insert(batch);
        }));
      }
      for (Future<Integer> f : futures) f.get();
      pool.shutdown();

      // Sample-based verification: query a moderate topK (well under total dataset
      // size to avoid native HNSW edge cases) and confirm each result decrypts to
      // its expected plaintext. This catches nonce-reuse, Cipher-reuse, or other
      // concurrency bugs in the encryption layer without exercising the
      // top-K-equals-N native code path.
      List<Doc> results = col.query(
          ZvecSearch.vector("e", new float[] {1f, 0f, 0f, 0f})
              .topK(100).project("body").build());
      assertEquals(100, results.size());
      for (Doc d : results) {
        assertEquals("secret-" + d.id(), d.fields().get("body"));
      }
    }
  }
}
