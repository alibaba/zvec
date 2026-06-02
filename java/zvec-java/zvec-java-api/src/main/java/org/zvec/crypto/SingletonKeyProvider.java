package org.zvec.crypto;

import java.util.Objects;

/** Internal: wraps a single (keyId, key) pair as a KeyProvider. Used by the static-key sugar. */
public final class SingletonKeyProvider implements KeyProvider {
  private final String keyId;
  private final byte[] key;

  public SingletonKeyProvider(String keyId, byte[] key) {
    Objects.requireNonNull(keyId, "keyId");
    Objects.requireNonNull(key, "key");
    if (keyId.isEmpty()) {
      throw new IllegalArgumentException("keyId must not be empty");
    }
    if (key.length != AesGcm256.KEY_LEN) {
      throw new IllegalArgumentException("key must be " + AesGcm256.KEY_LEN + " bytes, got " + key.length);
    }
    this.keyId = keyId;
    this.key = key.clone();
  }

  String keyId() { return keyId; }

  @Override
  public byte[] resolve(String requested) {
    return keyId.equals(requested) ? key.clone() : null;
  }
}
