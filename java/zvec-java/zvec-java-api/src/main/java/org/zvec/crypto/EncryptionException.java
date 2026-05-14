package org.zvec.crypto;

/** Root of the zvec encryption exception hierarchy. */
public abstract class EncryptionException extends RuntimeException {

  protected EncryptionException(String message) {
    super(message);
  }

  protected EncryptionException(String message, Throwable cause) {
    super(message, cause);
  }
}
