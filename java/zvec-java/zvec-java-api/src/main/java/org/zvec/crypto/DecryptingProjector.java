package org.zvec.crypto;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.zvec.Doc;

public final class DecryptingProjector {
  private static final Aead AEAD = new AesGcm256();

  private DecryptingProjector() {}

  public static List<Doc> transform(List<Doc> docs, EncryptedSchema es) {
    if (es == EncryptedSchema.NONE || es.encryptedFieldNames().isEmpty()) {
      return docs;
    }
    String collectionName = es.schema().name();
    Map<String, byte[]> keyCache = new HashMap<>();

    List<Doc> result = new ArrayList<>(docs.size());
    for (Doc input : docs) {
      Doc out = input.score() == null ? Doc.of(input.id()) : Doc.result(input.id(), input.score());
      for (Map.Entry<String, Object> e : input.fields().entrySet()) {
        String name = e.getKey();
        Object value = e.getValue();
        if (es.isEncrypted(name) && value instanceof String) {
          String b64 = (String) value;
          Envelope env = EnvelopeCodec.decodeBase64(b64);
          if (env.alg() != AEAD.algId()) {
            throw new EnvelopeFormatException(
                "unsupported alg=0x" + Integer.toHexString(env.alg())
                + " (field='" + name + "' doc.id='" + input.id() + "')");
          }
          if (env.payloadType() != Envelope.PAYLOAD_STRING) {
            throw new EnvelopeFormatException(
                "unsupported payload_type=0x" + Integer.toHexString(env.payloadType())
                + " (field='" + name + "' doc.id='" + input.id() + "')");
          }
          byte[] key = keyCache.computeIfAbsent(env.keyId(), kid -> resolveKey(es, kid, input.id(), name));
          byte[] aad = AadEncoder.encode(input.id(), name, collectionName);
          byte[] pt;
          try {
            pt = AEAD.open(key, env.nonce(), env.ciphertext(), aad);
          } catch (AuthenticationFailedException afe) {
            throw new AuthenticationFailedException(
                "GCM tag mismatch decrypting field='" + name
                + "' doc.id='" + input.id()
                + "' keyId='" + env.keyId() + "'", afe);
          }
          out.field(name, new String(pt, StandardCharsets.UTF_8));
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

  private static byte[] resolveKey(EncryptedSchema es, String keyId, String docId, String fieldName) {
    byte[] k;
    try {
      k = es.keyProvider().resolve(keyId);
    } catch (RuntimeException t) {
      throw new KeyResolutionException(
          "KeyProvider.resolve threw for keyId='" + keyId
          + "' (field='" + fieldName + "' doc.id='" + docId + "')", t);
    }
    if (k == null) {
      throw new KeyResolutionException(
          "KeyProvider returned null for keyId='" + keyId
          + "' (field='" + fieldName + "' doc.id='" + docId + "')");
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
