package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.zvec.CollectionSchema;
import org.zvec.DataType;
import org.zvec.Doc;
import org.zvec.FieldSchema;
import org.zvec.VectorSchema;

class EncryptingInsertorTest {
  private static final byte[] KEY = new byte[32];

  private static EncryptedSchema buildEncSchema(String fieldName) {
    CollectionSchema cs = new CollectionSchema(
        "docs",
        List.of(new FieldSchema("title", DataType.STRING, false),
                new FieldSchema(fieldName, DataType.STRING, false)),
        List.of(new VectorSchema("embed", DataType.VECTOR_FP32, 4)));
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.now(), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, "docs", Map.of(fieldName, spec));
    return EncryptedSchema.reconcile(cs, meta, kid -> "k1".equals(kid) ? KEY : null);
  }

  @Test
  void encryptsOnlyMarkedField() {
    EncryptedSchema es = buildEncSchema("body");
    Doc input = Doc.of("d1").field("title", "alpha").field("body", "secret")
        .vector("embed", new float[] {1f,0f,0f,0f});

    List<Doc> out = EncryptingInsertor.transform(List.of(input), es);

    assertEquals("alpha", out.get(0).fields().get("title"));
    assertNotEquals("secret", out.get(0).fields().get("body"));
    assertArrayEquals(new float[] {1f,0f,0f,0f}, out.get(0).vectors().get("embed"));
  }

  @Test
  void inputDocIsNotMutated() {
    EncryptedSchema es = buildEncSchema("body");
    Doc input = Doc.of("d1").field("body", "secret");
    EncryptingInsertor.transform(List.of(input), es);
    assertEquals("secret", input.fields().get("body"));
  }

  @Test
  void noEncryptedFieldOnDocIsTolerated() {
    EncryptedSchema es = buildEncSchema("body");
    Doc input = Doc.of("d1").field("title", "alpha");
    List<Doc> out = EncryptingInsertor.transform(List.of(input), es);
    assertEquals("alpha", out.get(0).fields().get("title"));
    assertEquals(false, out.get(0).fields().containsKey("body"));
  }

  @Test
  void noneSentinelReturnsInputUnchanged() {
    Doc input = Doc.of("d1").field("body", "x");
    List<Doc> out = EncryptingInsertor.transform(List.of(input), EncryptedSchema.NONE);
    assertEquals(input.fields(), out.get(0).fields());
  }

  @Test
  void roundTripsViaCodecAndAead() {
    EncryptedSchema es = buildEncSchema("body");
    Doc input = Doc.of("d1").field("body", "the quick brown fox");
    Doc encrypted = EncryptingInsertor.transform(List.of(input), es).get(0);

    String b64 = (String) encrypted.fields().get("body");
    Envelope env = EnvelopeCodec.decodeBase64(b64);
    assertEquals("k1", env.keyId());
    byte[] aad = AadEncoder.encode("d1", "body", "docs");
    byte[] plaintext = new AesGcm256().open(KEY, env.nonce(), env.ciphertext(), aad);
    assertEquals("the quick brown fox", new String(plaintext, StandardCharsets.UTF_8));
  }

  @Test
  void providerNullThrows() {
    CollectionSchema cs = new CollectionSchema(
        "docs",
        List.of(new FieldSchema("body", DataType.STRING, false)),
        List.of(new VectorSchema("embed", DataType.VECTOR_FP32, 4)));
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "missing-key", Instant.now(), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, "docs", Map.of("body", spec));
    EncryptedSchema es = EncryptedSchema.reconcile(cs, meta, kid -> null);
    Doc input = Doc.of("d1").field("body", "secret");
    KeyResolutionException e = assertThrows(KeyResolutionException.class,
        () -> EncryptingInsertor.transform(List.of(input), es));
    assertTrue(e.getMessage().contains("missing-key"));
  }
}
