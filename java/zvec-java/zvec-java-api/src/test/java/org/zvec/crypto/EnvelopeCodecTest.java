package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class EnvelopeCodecTest {
  private static final byte[] NONCE = new byte[] {1,2,3,4,5,6,7,8,9,10,11,12};

  @Test
  void roundTripBinary() {
    Envelope original = new Envelope(
        Envelope.VERSION_V1, Envelope.ALG_AES_256_GCM, Envelope.PAYLOAD_STRING,
        "key-1", NONCE, new byte[] {(byte)0xde, (byte)0xad, (byte)0xbe, (byte)0xef});
    byte[] encoded = EnvelopeCodec.encode(original);
    Envelope decoded = EnvelopeCodec.decode(encoded);

    assertEquals(original.version(), decoded.version());
    assertEquals(original.alg(), decoded.alg());
    assertEquals(original.payloadType(), decoded.payloadType());
    assertEquals(original.keyId(), decoded.keyId());
    assertArrayEquals(original.nonce(), decoded.nonce());
    assertArrayEquals(original.ciphertext(), decoded.ciphertext());
  }

  @Test
  void roundTripBase64UrlNoPad() {
    Envelope original = new Envelope(
        Envelope.VERSION_V1, Envelope.ALG_AES_256_GCM, Envelope.PAYLOAD_STRING,
        "k", NONCE, new byte[] {1, 2, 3});
    String b64 = EnvelopeCodec.encodeBase64(original);
    org.junit.jupiter.api.Assertions.assertFalse(b64.contains("="));
    org.junit.jupiter.api.Assertions.assertFalse(b64.contains("+"));
    org.junit.jupiter.api.Assertions.assertFalse(b64.contains("/"));
    Envelope decoded = EnvelopeCodec.decodeBase64(b64);
    assertArrayEquals(original.ciphertext(), decoded.ciphertext());
  }

  @Test
  void rejectsUnknownVersion() {
    byte[] bad = new byte[] {(byte)0x02, 0x01, 0x00, 1, 'k', 1,2,3,4,5,6,7,8,9,10,11,12, 0,0,0};
    EnvelopeFormatException e = assertThrows(EnvelopeFormatException.class, () -> EnvelopeCodec.decode(bad));
    org.junit.jupiter.api.Assertions.assertTrue(e.getMessage().contains("version"));
  }

  @Test
  void rejectsZeroKeyIdLen() {
    byte[] bad = new byte[] {(byte)0x01, 0x01, 0x00, 0, 1,2,3,4,5,6,7,8,9,10,11,12, 0,0,0};
    assertThrows(EnvelopeFormatException.class, () -> EnvelopeCodec.decode(bad));
  }

  @Test
  void rejectsTruncatedBuffer() {
    byte[] tooShort = new byte[] {0x01, 0x01, 0x00, 1};
    assertThrows(EnvelopeFormatException.class, () -> EnvelopeCodec.decode(tooShort));
  }

  @Test
  void rejectsKeyIdLongerThan255() {
    String kid = "k".repeat(256);
    Envelope env = new Envelope(
        Envelope.VERSION_V1, Envelope.ALG_AES_256_GCM, Envelope.PAYLOAD_STRING,
        kid, NONCE, new byte[] {1});
    assertThrows(IllegalArgumentException.class, () -> EnvelopeCodec.encode(env));
  }

  @Test
  void rejectsWrongNonceSize() {
    byte[] shortNonce = new byte[8];
    Envelope env = new Envelope(
        Envelope.VERSION_V1, Envelope.ALG_AES_256_GCM, Envelope.PAYLOAD_STRING,
        "k", shortNonce, new byte[] {1});
    assertThrows(IllegalArgumentException.class, () -> EnvelopeCodec.encode(env));
  }

  @Test
  void layoutMatchesSpec() {
    String kid = "kid";
    Envelope env = new Envelope(
        Envelope.VERSION_V1, Envelope.ALG_AES_256_GCM, Envelope.PAYLOAD_STRING,
        kid, NONCE, new byte[] {(byte)0xff});
    byte[] enc = EnvelopeCodec.encode(env);
    assertEquals(0x01, enc[0] & 0xff);
    assertEquals(0x01, enc[1] & 0xff);
    assertEquals(0x00, enc[2] & 0xff);
    assertEquals(kid.length(), enc[3] & 0xff);
    assertArrayEquals(kid.getBytes(StandardCharsets.UTF_8),
        java.util.Arrays.copyOfRange(enc, 4, 4 + kid.length()));
    assertArrayEquals(NONCE,
        java.util.Arrays.copyOfRange(enc, 4 + kid.length(), 4 + kid.length() + 12));
  }
}
