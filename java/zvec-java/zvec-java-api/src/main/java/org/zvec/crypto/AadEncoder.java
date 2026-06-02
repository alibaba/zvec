package org.zvec.crypto;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Length-prefixed AAD: u32_be(len) || utf8(value) for each of (id, fieldName, collectionName).
 * Recomputed on both encrypt and decrypt; never stored in the envelope.
 */
public final class AadEncoder {
  private AadEncoder() {}

  public static byte[] encode(String id, String fieldName, String collectionName) {
    byte[] idBytes = Objects.requireNonNull(id, "id").getBytes(StandardCharsets.UTF_8);
    byte[] fieldBytes = Objects.requireNonNull(fieldName, "fieldName").getBytes(StandardCharsets.UTF_8);
    byte[] collBytes = Objects.requireNonNull(collectionName, "collectionName").getBytes(StandardCharsets.UTF_8);

    ByteBuffer buf = ByteBuffer.allocate(12 + idBytes.length + fieldBytes.length + collBytes.length);
    buf.putInt(idBytes.length).put(idBytes);
    buf.putInt(fieldBytes.length).put(fieldBytes);
    buf.putInt(collBytes.length).put(collBytes);
    return buf.array();
  }
}
