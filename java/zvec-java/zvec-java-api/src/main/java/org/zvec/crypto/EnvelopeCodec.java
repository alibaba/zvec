package org.zvec.crypto;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

public final class EnvelopeCodec {
  private static final Base64.Encoder B64_ENC = Base64.getUrlEncoder().withoutPadding();
  private static final Base64.Decoder B64_DEC = Base64.getUrlDecoder();

  private EnvelopeCodec() {}

  public static byte[] encode(Envelope env) {
    byte[] kid = env.keyId().getBytes(StandardCharsets.UTF_8);
    if (kid.length < 1 || kid.length > 255) {
      throw new IllegalArgumentException("keyId UTF-8 length must be 1..255, got " + kid.length);
    }
    if (env.nonce().length != Envelope.NONCE_LEN) {
      throw new IllegalArgumentException("nonce must be " + Envelope.NONCE_LEN + " bytes");
    }
    int total = 4 + kid.length + Envelope.NONCE_LEN + env.ciphertext().length;
    ByteBuffer buf = ByteBuffer.allocate(total);
    buf.put((byte) env.version());
    buf.put((byte) env.alg());
    buf.put((byte) env.payloadType());
    buf.put((byte) kid.length);
    buf.put(kid);
    buf.put(env.nonce());
    buf.put(env.ciphertext());
    return buf.array();
  }

  public static Envelope decode(byte[] data) {
    if (data == null || data.length < 4) {
      throw new EnvelopeFormatException("envelope truncated: length=" + (data == null ? 0 : data.length));
    }
    int version = data[0] & 0xff;
    int alg = data[1] & 0xff;
    int payload = data[2] & 0xff;
    int kidLen = data[3] & 0xff;

    if (version != Envelope.VERSION_V1) {
      throw new EnvelopeFormatException("unsupported envelope version=0x" + Integer.toHexString(version));
    }
    if (kidLen == 0) {
      throw new EnvelopeFormatException("envelope keyId_len cannot be zero");
    }
    int kidEnd = 4 + kidLen;
    int nonceEnd = kidEnd + Envelope.NONCE_LEN;
    if (data.length < nonceEnd) {
      throw new EnvelopeFormatException("envelope truncated: expected at least " + nonceEnd + " bytes, got " + data.length);
    }
    String keyId = new String(data, 4, kidLen, StandardCharsets.UTF_8);
    byte[] nonce = java.util.Arrays.copyOfRange(data, kidEnd, nonceEnd);
    byte[] ct = java.util.Arrays.copyOfRange(data, nonceEnd, data.length);
    return new Envelope(version, alg, payload, keyId, nonce, ct);
  }

  public static String encodeBase64(Envelope env) {
    return B64_ENC.encodeToString(encode(env));
  }

  public static Envelope decodeBase64(String s) {
    byte[] raw;
    try {
      raw = B64_DEC.decode(s);
    } catch (IllegalArgumentException e) {
      throw new EnvelopeFormatException("envelope base64 decode failed: " + e.getMessage());
    }
    return decode(raw);
  }
}
