package org.zvec.crypto;

/** Thrown when a key cannot be resolved from the key provider. */
public final class KeyResolutionException extends EncryptionRuntimeException {

  public KeyResolutionException(String message) {
    super(message);
  }

  public KeyResolutionException(String message, Throwable cause) {
    super(message, cause);
  }
}
