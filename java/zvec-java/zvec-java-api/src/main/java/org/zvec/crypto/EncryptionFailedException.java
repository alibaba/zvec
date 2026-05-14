package org.zvec.crypto;

/** Thrown when field encryption fails during a write operation. */
public final class EncryptionFailedException extends EncryptionRuntimeException {

  public EncryptionFailedException(String message, Throwable cause) {
    super(message, cause);
  }
}
