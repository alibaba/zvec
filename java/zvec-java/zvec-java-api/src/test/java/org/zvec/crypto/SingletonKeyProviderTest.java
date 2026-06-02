package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SingletonKeyProviderTest {
  @Test
  void resolvesMatchingKeyId() {
    byte[] key = new byte[32];
    KeyProvider p = new SingletonKeyProvider("k1", key);
    assertArrayEquals(key, p.resolve("k1"));
  }

  @Test
  void returnsNullForOtherKeyId() {
    KeyProvider p = new SingletonKeyProvider("k1", new byte[32]);
    assertNull(p.resolve("k2"));
  }

  @Test
  void rejectsNonAes256KeyLength() {
    assertThrows(IllegalArgumentException.class, () -> new SingletonKeyProvider("k1", new byte[31]));
    assertThrows(IllegalArgumentException.class, () -> new SingletonKeyProvider("k1", new byte[33]));
  }

  @Test
  void rejectsEmptyKeyId() {
    assertThrows(IllegalArgumentException.class, () -> new SingletonKeyProvider("", new byte[32]));
  }

  @Test
  void defensiveCopy() {
    byte[] original = new byte[32];
    original[0] = 1;
    SingletonKeyProvider p = new SingletonKeyProvider("k1", original);
    original[0] = 99;
    org.junit.jupiter.api.Assertions.assertEquals(1, p.resolve("k1")[0]);
  }

  @Test
  void isActiveDefaultsTrue() {
    KeyProvider p = new SingletonKeyProvider("k1", new byte[32]);
    assertTrue(p.isActive("k1"));
  }
}
