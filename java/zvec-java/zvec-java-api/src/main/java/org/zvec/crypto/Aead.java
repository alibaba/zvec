package org.zvec.crypto;

public interface Aead {
  /** Returns the algorithm id used in the envelope alg byte. */
  int algId();

  byte[] seal(byte[] key, byte[] nonce, byte[] plaintext, byte[] aad);

  byte[] open(byte[] key, byte[] nonce, byte[] ciphertext, byte[] aad);
}
