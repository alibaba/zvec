package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.zvec.CollectionSchema;
import org.zvec.DataType;
import org.zvec.Doc;
import org.zvec.FieldSchema;
import org.zvec.VectorSchema;

class DecryptingProjectorTest {
  private static final byte[] KEY = new byte[32];
  static { KEY[0] = 9; }

  private static EncryptedSchema buildEncSchema() {
    CollectionSchema cs = new CollectionSchema(
        "docs",
        List.of(new FieldSchema("title", DataType.STRING, false),
                new FieldSchema("body", DataType.STRING, false)),
        List.of(new VectorSchema("embed", DataType.VECTOR_FP32, 4)));
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.now(), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, "docs", Map.of("body", spec));
    return EncryptedSchema.reconcile(cs, meta, kid -> "k1".equals(kid) ? KEY : null);
  }

  @Test
  void roundTripsViaInsertor() {
    EncryptedSchema es = buildEncSchema();
    Doc plain = Doc.of("d1").field("title", "alpha").field("body", "secret-text");
    Doc encrypted = EncryptingInsertor.transform(List.of(plain), es).get(0);

    Doc relayed = Doc.of("d1");
    encrypted.fields().forEach((k, v) -> { if (v instanceof String) relayed.field(k, (String) v); });

    Doc decrypted = DecryptingProjector.transform(List.of(relayed), es).get(0);
    assertEquals("alpha", decrypted.fields().get("title"));
    assertEquals("secret-text", decrypted.fields().get("body"));
  }

  @Test
  void noneSentinelReturnsInputUnchanged() {
    Doc input = Doc.of("d1").field("body", "anything");
    List<Doc> out = DecryptingProjector.transform(List.of(input), EncryptedSchema.NONE);
    assertSame(input, out.get(0));
  }

  @Test
  void missingEncryptedFieldOnResultIsTolerated() {
    EncryptedSchema es = buildEncSchema();
    Doc input = Doc.of("d1").field("title", "alpha");
    Doc out = DecryptingProjector.transform(List.of(input), es).get(0);
    assertEquals("alpha", out.fields().get("title"));
    assertEquals(false, out.fields().containsKey("body"));
  }

  @Test
  void aadMismatchSurfacesAsAuthenticationFailed() {
    EncryptedSchema es = buildEncSchema();
    Doc plain = Doc.of("d1").field("body", "secret");
    Doc encrypted = EncryptingInsertor.transform(List.of(plain), es).get(0);

    Doc relocated = Doc.of("d2");
    encrypted.fields().forEach((k, v) -> { if (v instanceof String) relocated.field(k, (String) v); });

    assertThrows(AuthenticationFailedException.class,
        () -> DecryptingProjector.transform(List.of(relocated), es));
  }

  @Test
  void unknownKeyIdSurfacesAsKeyResolution() {
    CollectionSchema cs = new CollectionSchema(
        "docs",
        List.of(new FieldSchema("body", DataType.STRING, false)),
        List.of(new VectorSchema("embed", DataType.VECTOR_FP32, 4)));
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.now(), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, "docs", Map.of("body", spec));
    EncryptedSchema esEncrypt = EncryptedSchema.reconcile(cs, meta, kid -> KEY);
    Doc plain = Doc.of("d1").field("body", "secret");
    Doc encrypted = EncryptingInsertor.transform(List.of(plain), esEncrypt).get(0);

    EncryptedSchema esDecryptNoKey = EncryptedSchema.reconcile(cs, meta, kid -> null);
    Doc relayed = Doc.of("d1");
    encrypted.fields().forEach((k, v) -> { if (v instanceof String) relayed.field(k, (String) v); });
    assertThrows(KeyResolutionException.class,
        () -> DecryptingProjector.transform(List.of(relayed), esDecryptNoKey));
  }

  @Test
  void unknownVersionSurfacesAsEnvelopeFormat() {
    EncryptedSchema es = buildEncSchema();
    byte[] junk = new byte[200];
    junk[0] = 0x09;
    String b64 = java.util.Base64.getUrlEncoder().withoutPadding().encodeToString(junk);
    Doc bad = Doc.of("d1").field("body", b64);
    assertThrows(EnvelopeFormatException.class,
        () -> DecryptingProjector.transform(List.of(bad), es));
    assertNotNull(es);
  }
}
