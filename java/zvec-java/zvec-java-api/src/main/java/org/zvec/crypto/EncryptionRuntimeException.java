package org.zvec.crypto;

/** Base for runtime encryption/decryption errors. */
abstract class EncryptionRuntimeException extends EncryptionException {

  protected EncryptionRuntimeException(String message) {
    super(message);
  }

  protected EncryptionRuntimeException(String message, Throwable cause) {
    super(message, cause);
  }
}
