package org.zvec.crypto;

/** Thrown when field decryption fails during a read operation. */
public class DecryptionException extends EncryptionRuntimeException {

  public DecryptionException(String message) {
    super(message);
  }

  public DecryptionException(String message, Throwable cause) {
    super(message, cause);
  }
}
