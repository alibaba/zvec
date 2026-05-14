package org.zvec.crypto;

/** Thrown when encryption metadata cannot be read from or written to storage. */
public final class EncryptionMetadataIOException extends EncryptionConfigException {

  public EncryptionMetadataIOException(String message, Throwable cause) {
    super(message, cause);
  }
}
