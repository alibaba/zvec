package org.zvec.crypto;

import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.zvec.Doc;

public final class EncryptingInsertor {
  private static final SecureRandom RNG = new SecureRandom();
  private static final Aead AEAD = new AesGcm256();

  private EncryptingInsertor() {}

  public static List<Doc> transform(List<Doc> docs, EncryptedSchema es) {
    if (es == EncryptedSchema.NONE || es.encryptedFieldNames().isEmpty()) {
      return docs;
    }
    String collectionName = es.schema().name();
    Map<String, byte[]> keyCache = new HashMap<>();

    List<Doc> result = new ArrayList<>(docs.size());
    for (Doc input : docs) {
      Doc out = Doc.of(input.id());
      for (Map.Entry<String, Object> e : input.fields().entrySet()) {
        String name = e.getKey();
        Object value = e.getValue();
        if (es.isEncrypted(name) && value instanceof String) {
          String pt = (String) value;
          String keyId = es.activeKeyId(name);
          byte[] key = keyCache.computeIfAbsent(keyId, kid -> resolveKey(es, kid));
          byte[] nonce = new byte[Envelope.NONCE_LEN];
          RNG.nextBytes(nonce);
          byte[] aad = AadEncoder.encode(input.id(), name, collectionName);
          byte[] ct = AEAD.seal(key, nonce, pt.getBytes(StandardCharsets.UTF_8), aad);
          Envelope env = new Envelope(
              Envelope.VERSION_V1, AEAD.algId(), Envelope.PAYLOAD_STRING,
              keyId, nonce, ct);
          out.field(name, EnvelopeCodec.encodeBase64(env));
        } else {
          assignField(out, name, value);
        }
      }
      for (Map.Entry<String, float[]> v : input.vectors().entrySet()) {
        out.vector(v.getKey(), v.getValue());
      }
      for (String n : input.nullFields()) {
        out.nullField(n);
      }
      result.add(out);
    }
    return result;
  }

  private static byte[] resolveKey(EncryptedSchema es, String keyId) {
    byte[] k;
    try {
      k = es.keyProvider().resolve(keyId);
    } catch (RuntimeException t) {
      throw new KeyResolutionException("KeyProvider.resolve threw for keyId='" + keyId + "'", t);
    }
    if (k == null) {
      throw new KeyResolutionException("KeyProvider returned null for keyId='" + keyId + "'");
    }
    if (k.length != AesGcm256.KEY_LEN) {
      throw new KeyResolutionException(
          "KeyProvider returned " + k.length + "-byte key, expected "
          + AesGcm256.KEY_LEN + " bytes for keyId='" + keyId + "'");
    }
    return k;
  }

  private static void assignField(Doc out, String name, Object value) {
    if (value instanceof String) out.field(name, (String) value);
    else if (value instanceof Boolean) out.field(name, ((Boolean) value).booleanValue());
    else if (value instanceof Long) out.field(name, ((Long) value).longValue());
    else if (value instanceof Double) out.field(name, ((Double) value).doubleValue());
    else if (value == null) out.nullField(name);
    else throw new IllegalStateException("unexpected field value type: " + value.getClass());
  }
}
