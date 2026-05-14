package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.security.SecureRandom;
import org.junit.jupiter.api.Test;

class AesGcm256Test {
  @Test
  void roundTripsArbitraryPlaintext() {
    Aead aead = new AesGcm256();
    byte[] key = new byte[32];
    new SecureRandom().nextBytes(key);
    byte[] nonce = new byte[12];
    new SecureRandom().nextBytes(nonce);
    byte[] plaintext = "hello world".getBytes();
    byte[] aad = "id1|body|docs".getBytes();

    byte[] ct = aead.seal(key, nonce, plaintext, aad);
    byte[] back = aead.open(key, nonce, ct, aad);
    assertArrayEquals(plaintext, back);
  }

  @Test
  void emptyPlaintextIsSupported() {
    Aead aead = new AesGcm256();
    byte[] key = new byte[32];
    byte[] nonce = new byte[12];
    byte[] ct = aead.seal(key, nonce, new byte[0], new byte[0]);
    assertEquals(16, ct.length); // tag only
    assertArrayEquals(new byte[0], aead.open(key, nonce, ct, new byte[0]));
  }

  @Test
  void tagFlipDetected() {
    Aead aead = new AesGcm256();
    byte[] key = new byte[32];
    byte[] nonce = new byte[12];
    byte[] ct = aead.seal(key, nonce, "secret".getBytes(), "aad".getBytes());
    ct[ct.length - 1] ^= 0x01;
    assertThrows(AuthenticationFailedException.class,
        () -> aead.open(key, nonce, ct, "aad".getBytes()));
  }

  @Test
  void aadMismatchDetected() {
    Aead aead = new AesGcm256();
    byte[] key = new byte[32];
    byte[] nonce = new byte[12];
    byte[] ct = aead.seal(key, nonce, "secret".getBytes(), "aad-A".getBytes());
    assertThrows(AuthenticationFailedException.class,
        () -> aead.open(key, nonce, ct, "aad-B".getBytes()));
  }

  @Test
  void wrongKeyLengthRejected() {
    Aead aead = new AesGcm256();
    byte[] shortKey = new byte[31];
    byte[] nonce = new byte[12];
    assertThrows(IllegalArgumentException.class,
        () -> aead.seal(shortKey, nonce, new byte[1], new byte[0]));
  }

  @Test
  void wrongNonceLengthRejected() {
    Aead aead = new AesGcm256();
    byte[] key = new byte[32];
    byte[] shortNonce = new byte[8];
    assertThrows(IllegalArgumentException.class,
        () -> aead.seal(key, shortNonce, new byte[1], new byte[0]));
  }

  @Test
  void nistTestVector() {
    // NIST GCM test vector: Key 256 bits, IV 96 bits, PT empty, AAD empty.
    // From NIST CAVP gcmEncryptExtIV256.rsp, Count = 0
    byte[] key = parseHex("b52c505a37d78eda5dd34f20c22540ea1b58963cf8e5bf8ffa85f9f2492505b4");
    byte[] iv  = parseHex("516c33929df5a3284ff463d7");
    byte[] expectedTag = parseHex("bdc1ac884d332457a1d2664f168c76f0");
    byte[] ct = new AesGcm256().seal(key, iv, new byte[0], new byte[0]);
    assertArrayEquals(expectedTag, ct);
  }

  @Test
  void identifiesAsAesGcmInEnvelopeAlgByte() {
    assertEquals(Envelope.ALG_AES_256_GCM, new AesGcm256().algId());
  }

  @Test
  void nonceUniquenessOver1k() {
    java.util.Set<String> seen = new java.util.HashSet<>();
    SecureRandom rng = new SecureRandom();
    for (int i = 0; i < 1000; i++) {
      byte[] n = new byte[12];
      rng.nextBytes(n);
      seen.add(formatHex(n));
    }
    assertNotEquals(0, seen.size());
    org.junit.jupiter.api.Assertions.assertTrue(seen.size() >= 999);
  }

  private static byte[] parseHex(String hex) {
    if ((hex.length() & 1) != 0) {
      throw new IllegalArgumentException("hex length must be even");
    }
    byte[] out = new byte[hex.length() / 2];
    for (int i = 0; i < out.length; i++) {
      int hi = Character.digit(hex.charAt(i * 2), 16);
      int lo = Character.digit(hex.charAt(i * 2 + 1), 16);
      if (hi < 0 || lo < 0) {
        throw new IllegalArgumentException("invalid hex: " + hex);
      }
      out[i] = (byte) ((hi << 4) | lo);
    }
    return out;
  }

  private static String formatHex(byte[] bytes) {
    char[] out = new char[bytes.length * 2];
    char[] digits = "0123456789abcdef".toCharArray();
    for (int i = 0; i < bytes.length; i++) {
      int value = bytes[i] & 0xff;
      out[i * 2] = digits[value >>> 4];
      out[i * 2 + 1] = digits[value & 0x0f];
    }
    return new String(out);
  }
}
