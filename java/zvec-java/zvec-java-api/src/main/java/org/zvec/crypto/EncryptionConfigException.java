package org.zvec.crypto;

/** Base for configuration and structural encryption errors. */
abstract class EncryptionConfigException extends EncryptionException {

  protected EncryptionConfigException(String message) {
    super(message);
  }

  protected EncryptionConfigException(String message, Throwable cause) {
    super(message, cause);
  }
}
