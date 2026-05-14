package org.zvec.crypto;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.GeneralSecurityException;

public final class AesGcm256 implements Aead {
  public static final int KEY_LEN = 32;
  public static final int NONCE_LEN = 12;
  public static final int TAG_BITS = 128;

  @Override
  public int algId() { return Envelope.ALG_AES_256_GCM; }

  @Override
  public byte[] seal(byte[] key, byte[] nonce, byte[] plaintext, byte[] aad) {
    validate(key, nonce);
    try {
      Cipher c = Cipher.getInstance("AES/GCM/NoPadding");
      c.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "AES"),
          new GCMParameterSpec(TAG_BITS, nonce));
      if (aad != null && aad.length > 0) {
        c.updateAAD(aad);
      }
      return c.doFinal(plaintext == null ? new byte[0] : plaintext);
    } catch (GeneralSecurityException e) {
      throw new EncryptionFailedException("AES-256-GCM seal failed", e);
    }
  }

  @Override
  public byte[] open(byte[] key, byte[] nonce, byte[] ciphertext, byte[] aad) {
    validate(key, nonce);
    try {
      Cipher c = Cipher.getInstance("AES/GCM/NoPadding");
      c.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "AES"),
          new GCMParameterSpec(TAG_BITS, nonce));
      if (aad != null && aad.length > 0) {
        c.updateAAD(aad);
      }
      return c.doFinal(ciphertext);
    } catch (javax.crypto.AEADBadTagException e) {
      throw new AuthenticationFailedException("GCM tag mismatch", e);
    } catch (GeneralSecurityException e) {
      throw new AuthenticationFailedException("AES-256-GCM open failed: " + e.getClass().getSimpleName(), e);
    }
  }

  private static void validate(byte[] key, byte[] nonce) {
    if (key == null || key.length != KEY_LEN) {
      throw new IllegalArgumentException("AES-256 key must be " + KEY_LEN + " bytes, got " + (key == null ? 0 : key.length));
    }
    if (nonce == null || nonce.length != NONCE_LEN) {
      throw new IllegalArgumentException("nonce must be " + NONCE_LEN + " bytes, got " + (nonce == null ? 0 : nonce.length));
    }
  }
}
