package org.zvec.crypto;

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

class EnvelopeRelocationSecurityTest {

  private static final byte[] KEY = new byte[32];
  static { KEY[0] = 33; }

  private static EncryptedSchema encryptedSchema(String collName, String fieldName) {
    CollectionSchema cs = new CollectionSchema(
        collName,
        List.of(new FieldSchema("title", DataType.STRING, false),
                new FieldSchema(fieldName, DataType.STRING, false),
                new FieldSchema("other", DataType.STRING, false)),
        List.of(new VectorSchema("e", DataType.VECTOR_FP32, 4)));
    EncryptionSpec spec = new EncryptionSpec("AES-256-GCM", "k1", Instant.now(), null);
    EncryptionMetadata meta = new EncryptionMetadata(1, collName, Map.of(fieldName, spec, "other", spec));
    return EncryptedSchema.reconcile(cs, meta, kid -> "k1".equals(kid) ? KEY : null);
  }

  @Test
  void relocationAcrossDocsRejected() {
    EncryptedSchema es = encryptedSchema("docs", "body");
    Doc d1 = EncryptingInsertor.transform(
        List.of(Doc.of("d1").field("body", "secret-1")), es).get(0);

    Doc relocated = Doc.of("d2");
    relocated.field("body", (String) d1.fields().get("body"));
    assertThrows(AuthenticationFailedException.class,
        () -> DecryptingProjector.transform(List.of(relocated), es));
  }

  @Test
  void relocationAcrossFieldsRejected() {
    EncryptedSchema es = encryptedSchema("docs", "body");
    Doc d1 = EncryptingInsertor.transform(
        List.of(Doc.of("d1").field("body", "secret-1")), es).get(0);

    Doc swapped = Doc.of("d1").field("other", (String) d1.fields().get("body"));
    assertThrows(AuthenticationFailedException.class,
        () -> DecryptingProjector.transform(List.of(swapped), es));
  }

  @Test
  void relocationAcrossCollectionsRejected() {
    EncryptedSchema esA = encryptedSchema("docsA", "body");
    EncryptedSchema esB = encryptedSchema("docsB", "body");
    Doc d1 = EncryptingInsertor.transform(
        List.of(Doc.of("d1").field("body", "secret")), esA).get(0);

    Doc moved = Doc.of("d1").field("body", (String) d1.fields().get("body"));
    assertThrows(AuthenticationFailedException.class,
        () -> DecryptingProjector.transform(List.of(moved), esB));
  }

  @Test
  void singleByteFlipAnywhereInEnvelopeRejected() {
    EncryptedSchema es = encryptedSchema("docs", "body");
    Doc d1 = EncryptingInsertor.transform(
        List.of(Doc.of("d1").field("body", "secret-payload")), es).get(0);
    String b64 = (String) d1.fields().get("body");
    byte[] raw = java.util.Base64.getUrlDecoder().decode(b64);

    for (int i = 0; i < raw.length; i++) {
      byte[] copy = raw.clone();
      copy[i] ^= 0x01;
      String corrupted = java.util.Base64.getUrlEncoder().withoutPadding().encodeToString(copy);
      Doc bad = Doc.of("d1").field("body", corrupted);
      assertThrows(EncryptionException.class,
          () -> DecryptingProjector.transform(List.of(bad), es),
          "byte index " + i + " flip should have failed");
    }
  }
}
